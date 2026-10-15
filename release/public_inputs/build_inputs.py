"""Rebuild release regression inputs from pinned public records, offline after fetch.

This is a new derivation. It never claims to reproduce the byte identity of the
historical, ignored E1 extraction. Network operations are sequential and bounded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
GRADES = {"C": 1.0, "I": 0.0, "P": 0.5, "N": 0.0}
SCORE_COLUMNS = {
    "training_passed_scorer": "training_passed",
    "passed": "passed",
    "actually_solved": "actually_solved",
    "reward_hacked": "reward_hacked",
    "rh_always_equal": "rh_always_equal",
    "rh_exit": "rh_exit",
    "rh_conftest": "rh_conftest",
    "cot_mentions_hack": "cot_mentions_hack",
}
MAX_ARCHIVE_MEMBERS = 256
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
MAX_MEMBER_UNCOMPRESSED_BYTES = 8 * 1024 * 1024


def file_hash(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
            size += len(block)
    return size, h.hexdigest()


def verify_file(path: Path, *, sha256: str, size: int | None = None) -> None:
    actual_size, actual_sha = file_hash(path)
    if actual_sha != sha256 or (size is not None and actual_size != size):
        raise ValueError(f"input digest/size mismatch: {path}: {actual_size} {actual_sha}")


def _score(value: object) -> float:
    if isinstance(value, bool):
        result = float(value)
    elif isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        result = GRADES[value] if value in GRADES else float(value)
    else:
        raise ValueError(f"invalid score: {value!r}")
    if not math.isfinite(result):
        raise ValueError(f"nonfinite score: {value!r}")
    return result


def scores_from_archive(path: Path) -> dict[tuple[str, str, int], dict[str, float]]:
    """Read only scorer JSON from an Inspect archive, never generation tokens."""
    rows: dict[tuple[str, str, int], dict[str, float]] = {}
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError(f"too many eval archive members: {path.name}")
        names = [member.filename for member in members]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate eval archive member: {path.name}")
        uncompressed = 0
        for member in members:
            name = member.filename
            parts = name.split("/")
            if (
                not name
                or name.startswith("/")
                or "\\" in name
                or ":" in parts[0]
                or any(part in {"", ".", ".."} for part in parts)
                or member.is_dir()
            ):
                raise ValueError(f"unsafe eval archive member: {name}")
            if member.file_size > MAX_MEMBER_UNCOMPRESSED_BYTES:
                raise ValueError(f"oversize eval archive member: {name}")
            uncompressed += member.file_size
            if uncompressed > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                raise ValueError(f"oversize eval archive: {path.name}")
        if "header.json" not in names or "reductions.json" not in names:
            raise ValueError(f"incomplete eval archive: {path.name}")
        sample_names = sorted(
            name for name in names if name.startswith("samples/") and name.endswith(".json")
        )
        if not sample_names:
            raise ValueError(f"empty eval archive: {path.name}")
        for name in sample_names:
            if len(name.split("/")) != 2:
                raise ValueError(f"unsafe eval sample path: {name}")
            raw = json.loads(archive.read(name))
            key = (path.name, str(raw["id"]), int(raw.get("epoch", 0)))
            if key in rows:
                raise ValueError(f"duplicate eval key: {key}")
            scores = {}
            for scorer, item in (raw.get("scores") or {}).items():
                val = item["value"]
                if isinstance(val, dict):
                    scores.update({str(metric): _score(number) for metric, number in val.items()})
                else:
                    scores[str(scorer)] = _score(val)
            if "thinking_format_scorer" not in scores or any(
                s not in scores for s in SCORE_COLUMNS
            ):
                raise ValueError(f"missing scorer at {key}")
            rows[key] = scores
    return rows


def collect_scores(paths: list[Path]) -> dict[tuple[str, str, int], dict[str, float]]:
    out: dict[tuple[str, str, int], dict[str, float]] = {}
    for path in paths:
        part = scores_from_archive(path)
        overlap = out.keys() & part.keys()
        if overlap:
            raise ValueError(f"duplicate keys across archives: {next(iter(overlap))}")
        out.update(part)
    return out


def join_e1_rows(
    source_rows: list[dict], scores: dict[tuple[str, str, int], dict[str, float]]
) -> list[dict]:
    """Require a bijection and agreement of every published binary/grade column."""
    joined: list[dict] = []
    seen: set[tuple[str, str, int]] = set()
    for source in source_rows:
        key = (
            str(source["source_eval_file"]),
            str(source["sample_id"]),
            int(source["epoch"]),
        )
        if key in seen:
            raise ValueError(f"duplicate parquet key: {key}")
        seen.add(key)
        if key not in scores:
            raise ValueError(f"unmatched parquet key: {key}")
        sample_scores = scores[key]
        fmt = sample_scores["thinking_format_scorer"]
        if fmt not in (0.0, 0.25, 0.5, 0.75, 1.0):
            raise ValueError(f"unexpected fractional format score at {key}: {fmt}")
        if _score(source["thinking_format_ok"]) != float(fmt == 1.0):
            raise ValueError(f"binary format disagrees at {key}")
        for score_key, column in SCORE_COLUMNS.items():
            if _score(source[column]) != sample_scores[score_key]:
                raise ValueError(f"score disagrees at {key}: {column}")
        joined.append({**source, "thinking_format": fmt})
    extra = scores.keys() - seen
    if extra:
        raise ValueError(f"unmatched eval key: {next(iter(extra))}")
    return joined


def e1_counts(rows: list[dict]) -> dict[str, int]:
    composition = (ROOT / "e1_analysis" / "COMPOSITION.md").read_text(encoding="utf-8")
    functions = re.search(r"# Reward weights: \[([^]]+)\]", composition)
    objective = re.search(r"objective is `([^`]+)`", composition)
    if functions is None or objective is None:
        raise ValueError("E1 composition record omits the reward functions or objective")
    function_count = len([part for part in functions.group(1).split(",") if part.strip()])
    advantage_count = sum(
        float(term.split("·", 1)[0].strip()) != 0.0 for term in objective.group(1).split("+")
    )
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((str(row["source_eval_file"]), str(row["sample_id"])), []).append(row)
    binary = [
        {
            (float(row["thinking_format"]) == 1.0, float(row["training_passed"]) > 0.0)
            for row in group
        }
        for group in groups.values()
    ]
    native = [
        {(float(row["thinking_format"]), float(row["training_passed"])) for row in group}
        for group in groups.values()
    ]
    return {
        "groups": len(groups),
        "groups_with_contrast": sum(len(cells) >= 2 for cells in binary),
        "accepted_hacks": sum(
            bool(row["reward_hacked"])
            and float(row["thinking_format"]) + 4 * float(row["training_passed"]) == 5.0
            for row in rows
        ),
        "solutions": sum(bool(row["actually_solved"]) for row in rows),
        "functions_summed": function_count,
        "advantage_components": advantage_count,
        "registered_analysis_support": sum(len(cells) >= 3 for cells in binary),
        "native_grid_support": sum(len(cells) >= 3 for cells in native),
    }


def verify_eval_metadata(paths: list[Path], metadata_path: Path) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    e1 = SPEC["e1"]
    if metadata.get("dataset") != e1["dataset"] or metadata.get("revision") != e1["revision"]:
        raise ValueError("eval metadata provenance mismatch")
    entries = metadata.get("files")
    if not isinstance(entries, list) or len(entries) != len(paths):
        raise ValueError("eval metadata file count mismatch")
    indexed = {Path(item["path"]).name: item for item in entries}
    if len(indexed) != len(entries) or set(indexed) != {path.name for path in paths}:
        raise ValueError("eval metadata filename mismatch")
    for path in paths:
        item = indexed[path.name]
        if item["path"] != f"eval_logs/{path.name}":
            raise ValueError(f"unexpected eval path: {item['path']}")
        verify_file(path, sha256=item["sha256"], size=int(item["bytes"]))


def build_e1(
    parquet_path: Path,
    eval_dir: Path,
    out_dir: Path,
    *,
    strict: bool = True,
    metadata_path: Path | None = None,
) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if strict:
        source_spec = SPEC["e1"]["parquet"]
        verify_file(parquet_path, sha256=source_spec["sha256"], size=source_spec["bytes"])
    paths = sorted(eval_dir.glob("*.eval"))
    if strict and len(paths) != SPEC["e1"]["expected_eval_files"]:
        raise ValueError(
            f"expected {SPEC['e1']['expected_eval_files']} eval archives; found {len(paths)}"
        )
    if not paths:
        raise ValueError("no eval archives")
    if strict and metadata_path is None:
        raise ValueError("strict E1 reconstruction requires the pinned eval metadata manifest")
    if metadata_path is not None:
        verify_eval_metadata(paths, metadata_path)
    scores = collect_scores(paths)
    source_table = pq.read_table(parquet_path)
    source_rows = source_table.to_pylist()
    joined = join_e1_rows(source_rows, scores)
    if strict and len(joined) != SPEC["e1"]["expected_rows"]:
        raise ValueError(f"E1 row count: {len(joined)}")
    counts = e1_counts(joined)
    if strict and counts != SPEC["e1"]["counts"]:
        raise AssertionError(
            {
                key: (counts[key], want)
                for key, want in SPEC["e1"]["counts"].items()
                if counts[key] != want
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "public_join_v1.parquet"
    # Preserve the publisher's nested Arrow schema exactly; Python inference can
    # silently narrow nested fields when reconstructing the full source table.
    derived = source_table.append_column(
        "thinking_format",
        pa.array([row["thinking_format"] for row in joined], type=pa.float64()),
    )
    pq.write_table(derived, output, compression="zstd")
    inputs = {SPEC["e1"]["parquet"]["path"]: file_hash(parquet_path)[1]}
    inputs.update({f"eval_logs/{path.name}": file_hash(path)[1] for path in paths})
    docs = {}
    (out_dir / "e1_analysis").mkdir(exist_ok=True)
    for name, expected in SPEC["e1"]["analysis"].items():
        path = ROOT / "e1_analysis" / name
        verify_file(path, sha256=expected)
        shutil.copyfile(path, out_dir / "e1_analysis" / name)
        docs[f"e1_analysis/{name}"] = expected
    receipt = {
        "schema": "reward-lens-e1-public-join/1",
        "source_dataset": SPEC["e1"]["dataset"],
        "source_revision": SPEC["e1"]["revision"],
        "procedure": "join public parquet to Inspect eval scorer rows on source_eval_file,sample_id,epoch; assert bijection and nine scorer correspondences",
        "historical_extraction_sha256": None,
        "inputs_sha256": inputs,
        "analysis_sha256": docs,
        "output": output.name,
        "output_sha256": file_hash(output)[1],
        "rows": len(joined),
        "counts": counts,
    }
    (out_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def eval_metadata() -> dict:
    """One pinned-revision metadata request, before any archive body is fetched."""
    e1 = SPEC["e1"]
    url = f"https://huggingface.co/api/datasets/{e1['dataset']}/tree/{e1['revision']}/eval_logs?recursive=true&expand=true"
    records = []
    page = url
    pinned_path = urllib.parse.urlparse(url).path
    for _ in range(20):  # the Hub paginates this directory in 50-file pages
        with urllib.request.urlopen(
            urllib.request.Request(
                page, headers={"User-Agent": "reward-lens-release-input-builder/1"}
            ),
            timeout=60,
        ) as response:
            records.extend(json.load(response))
            link = response.headers.get("Link", "")
        if 'rel="next"' not in link:
            break
        page = link.split("<", 1)[1].split(">", 1)[0]
        parsed = urllib.parse.urlparse(page)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "huggingface.co"
            or parsed.path != pinned_path
        ):
            raise ValueError("unexpected metadata pagination URL")
    else:
        raise ValueError("eval metadata pagination exceeded 20 pages")
    files = []
    for item in records:
        if item.get("type") != "file" or not str(item.get("path", "")).endswith(".eval"):
            continue
        lfs = item.get("lfs") or {}
        sha = str(lfs.get("oid") or lfs.get("sha256") or "").removeprefix("sha256:")
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError(f"eval file lacks LFS sha256: {item.get('path')}")
        files.append({"path": item["path"], "bytes": int(item["size"]), "sha256": sha})
    files.sort(key=lambda item: item["path"])
    total = sum(item["bytes"] for item in files)
    if len(files) != e1["expected_eval_files"] or total > e1["expected_eval_bytes_upper_bound"]:
        raise ValueError(f"unexpected eval metadata: {len(files)} files, {total} bytes")
    return {
        "schema": "reward-lens-pinned-eval-metadata/1",
        "dataset": e1["dataset"],
        "revision": e1["revision"],
        "files": files,
        "total_bytes": total,
    }


def download_verified(url: str, target: Path, *, sha256: str, size: int | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        verify_file(target, sha256=sha256, size=size)
        return
    tmp = target.with_suffix(target.suffix + ".part")
    h = hashlib.sha256()
    got = 0
    with (
        urllib.request.urlopen(
            urllib.request.Request(
                url, headers={"User-Agent": "reward-lens-release-input-builder/1"}
            ),
            timeout=120,
        ) as response,
        tmp.open("wb") as output,
    ):
        for block in iter(lambda: response.read(1024 * 1024), b""):
            got += len(block)
            if size is not None and got > size:
                raise ValueError(f"oversize download: {target}")
            h.update(block)
            output.write(block)
    if h.hexdigest() != sha256 or (size is not None and got != size):
        tmp.unlink(missing_ok=True)
        raise ValueError(f"download digest/size mismatch: {target}")
    tmp.replace(target)


def fetch_e1(metadata_path: Path, dest: Path) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    e1 = SPEC["e1"]
    if metadata.get("dataset") != e1["dataset"] or metadata.get("revision") != e1["revision"]:
        raise ValueError("metadata provenance mismatch")
    files = metadata["files"]
    total = sum(int(item["bytes"]) for item in files) + e1["parquet"]["bytes"]
    if (
        len(files) != e1["expected_eval_files"]
        or total > e1["expected_eval_bytes_upper_bound"] + e1["parquet"]["bytes"]
    ):
        raise ValueError("eval metadata exceeds bound")
    if shutil.disk_usage(dest.parent).free < total + 512 * 1024 * 1024:
        raise OSError("insufficient free disk for pinned public E1 inputs plus 512 MiB headroom")
    base = f"https://huggingface.co/datasets/{e1['dataset']}/resolve/{e1['revision']}"
    source = e1["parquet"]
    download_verified(
        base + "/" + source["path"],
        dest / source["path"],
        sha256=source["sha256"],
        size=source["bytes"],
    )
    for item in files:  # strictly one network transfer at a time
        if not str(item["path"]).startswith("eval_logs/") or ".." in Path(item["path"]).parts:
            raise ValueError(f"unsafe eval path: {item['path']}")
        download_verified(
            base + "/" + urllib.parse.quote(item["path"]),
            dest / item["path"],
            sha256=item["sha256"],
            size=item["bytes"],
        )


def fetch_rb2(dest: Path) -> Path:
    rb2 = SPEC["campaign"]["rewardbench2"]
    url = (
        f"https://huggingface.co/datasets/{rb2['dataset']}/resolve/{rb2['revision']}/{rb2['path']}"
    )
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
        size = int(response.headers["Content-Length"])
    if size != rb2["bytes"]:
        raise ValueError(f"unexpected RewardBench 2 parquet size: {size}")
    if shutil.disk_usage(dest.parent).free < size + 256 * 1024 * 1024:
        raise OSError("insufficient free disk for RewardBench 2 input")
    target = dest / rb2["path"]
    download_verified(url, target, sha256=rb2["sha256"], size=size)
    return target


def verify_vendor_provenance() -> None:
    """Check the copied generator source before importing or executing it."""
    provenance = json.loads((ROOT / "vendor-provenance.json").read_text(encoding="utf-8"))
    if (
        provenance.get("schema") != "reward-lens-campaign-generator-source/1"
        or provenance.get("local_git_commit") != SPEC["campaign"]["generator_local_git_commit"]
    ):
        raise ValueError("campaign vendor provenance identity mismatch")
    vendor = ROOT / "vendor"
    if vendor.is_symlink() or not vendor.is_dir():
        raise ValueError("unsafe campaign vendor directory")
    listed = provenance.get("files")
    if not isinstance(listed, dict):
        raise ValueError("missing campaign vendor manifest")
    actual = set()
    for path in vendor.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"unsafe campaign vendor path: {path.relative_to(vendor)}")
        if path.is_dir():
            continue
        relative = path.relative_to(vendor)
        if "__pycache__" in relative.parts and path.suffix == ".pyc":
            continue
        actual.add(relative.as_posix())
    if actual != set(listed):
        raise ValueError("campaign vendor file set mismatch")
    for relative, record in listed.items():
        if (
            not isinstance(record, dict)
            or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))
            or file_hash(vendor / relative)[1] != record["sha256"]
        ):
            raise ValueError(f"campaign vendor digest mismatch: {relative}")


def build_campaign(cache_dir: Path, *, rb2_parquet: Path | None = None) -> dict:
    """Execute only three copied, pinned campaign converter paths in a task-owned cache."""
    verify_vendor_provenance()
    sys.path.insert(0, str(ROOT / "vendor"))
    os.environ["CAMPAIGN_DATA_CACHE"] = str(cache_dir.resolve())
    from campaign.data import loaders

    cache_dir.mkdir(parents=True, exist_ok=True)
    names = ["diagnostic-v3", "hack-probes"]
    if rb2_parquet is not None:
        verify_file(rb2_parquet, sha256=SPEC["campaign"]["rewardbench2"]["sha256"])
        import pyarrow.parquet as pq

        def local_stream(dataset_id: str, config: str | None, split: str, revision: str):
            rb2 = SPEC["campaign"]["rewardbench2"]
            if (dataset_id, config, split, revision) != (
                rb2["dataset"],
                None,
                "test",
                rb2["revision"],
            ):
                raise ValueError("unexpected campaign dataset request")
            parquet = pq.ParquetFile(rb2_parquet)
            for batch in parquet.iter_batches(batch_size=256):
                yield from batch.to_pylist()

        loaders._stream = local_stream
        loaders.load_slice("rb2-full", refresh=True)
        names.append("rb2-patch-40")
    outputs = {}
    for name in names:
        result = loaders.load_slice(name, refresh=True)
        loaders.verify_against_lock(result)
        path = loaders.cache_path(name)
        expected = SPEC["campaign"]["bank_sha256"][name]
        verify_file(path, sha256=expected)
        if result.view.checksum() != SPEC["campaign"]["content_hash"][name]:
            raise AssertionError(f"campaign lock checksum mismatch: {name}")
        outputs[name] = {
            "file": f"{name}.jsonl",
            "sha256": expected,
            "content_hash": str(result.view.checksum()),
            "rows": len(result.view),
        }
    receipt = {
        "schema": "reward-lens-campaign-public-rebuild/1",
        "generator_commit": SPEC["campaign"]["generator_local_git_commit"],
        "rb2_source_sha256": file_hash(rb2_parquet)[1] if rb2_parquet else None,
        "outputs": outputs,
    }
    (cache_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("eval-metadata")
    fetch = sub.add_parser("fetch-e1")
    fetch.add_argument("metadata", type=Path)
    fetch.add_argument("destination", type=Path)
    rb2_fetch = sub.add_parser("fetch-rb2")
    rb2_fetch.add_argument("destination", type=Path)
    e1 = sub.add_parser("build-e1")
    e1.add_argument("parquet", type=Path)
    e1.add_argument("eval_dir", type=Path)
    e1.add_argument("output", type=Path)
    e1.add_argument("--eval-metadata", type=Path, required=True)
    campaign = sub.add_parser("build-campaign")
    campaign.add_argument("cache", type=Path)
    campaign.add_argument("--rb2-parquet", type=Path)
    args = parser.parse_args()
    if args.command == "eval-metadata":
        print(json.dumps(eval_metadata(), indent=2, sort_keys=True))
    elif args.command == "fetch-e1":
        fetch_e1(args.metadata, args.destination)
    elif args.command == "fetch-rb2":
        print(fetch_rb2(args.destination))
    elif args.command == "build-e1":
        print(
            json.dumps(
                build_e1(
                    args.parquet,
                    args.eval_dir,
                    args.output,
                    metadata_path=args.eval_metadata,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "build-campaign":
        print(
            json.dumps(
                build_campaign(args.cache, rb2_parquet=args.rb2_parquet),
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
