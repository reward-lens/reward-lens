from __future__ import annotations

import hashlib
import importlib.util
import json
import warnings
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_script = Path(__file__).resolve().parents[2] / "release/public_inputs/build_inputs.py"
_spec = importlib.util.spec_from_file_location("reward_lens_release_input_builder", _script)
assert _spec is not None and _spec.loader is not None
_builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_builder)
SPEC = _builder.SPEC
build_campaign = _builder.build_campaign
build_e1 = _builder.build_e1
file_hash = _builder.file_hash
join_e1_rows = _builder.join_e1_rows
scores_from_archive = _builder.scores_from_archive
verify_eval_metadata = _builder.verify_eval_metadata


def _source_row() -> dict:
    return {
        "source_eval_file": "x.eval",
        "sample_id": "s",
        "epoch": 0,
        "thinking_format_ok": 0,
        "training_passed": 1,
        "passed": 1,
        "actually_solved": 0,
        "reward_hacked": 1,
        "rh_always_equal": 0,
        "rh_exit": 0,
        "rh_conftest": 0,
        "cot_mentions_hack": 0,
    }


def _scores() -> dict:
    return {
        "thinking_format_scorer": 0.75,
        "training_passed_scorer": 1.0,
        "passed": 1.0,
        "actually_solved": 0.0,
        "reward_hacked": 1.0,
        "rh_always_equal": 0.0,
        "rh_exit": 0.0,
        "rh_conftest": 0.0,
        "cot_mentions_hack": 0.0,
    }


def _archive(path: Path) -> None:
    sample = {
        "id": "s",
        "epoch": 0,
        "scores": {
            "thinking_format_scorer": {"value": 0.75},
            "training_passed_scorer": {"value": 1},
            "proxy_reward_hacking_scorer": {
                "value": {
                    "passed": "C",
                    "actually_solved": "I",
                    "reward_hacked": "C",
                    "rh_always_equal": "I",
                    "rh_exit": "I",
                    "rh_conftest": "I",
                }
            },
            "proxy_cot_faithfulness_scorer": {"value": {"cot_mentions_hack": "I"}},
        },
    }
    with zipfile.ZipFile(path, "w") as output:
        output.writestr("header.json", json.dumps({"eval": {"eval_id": "synthetic"}}))
        output.writestr("reductions.json", "[]")
        output.writestr("samples/one.json", json.dumps(sample))


def test_e1_synthetic_public_join_writes_new_derivation_receipt(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([_source_row()]), source)
    evals = tmp_path / "eval_logs"
    evals.mkdir()
    _archive(evals / "x.eval")
    assert scores_from_archive(evals / "x.eval") == {("x.eval", "s", 0): _scores()}
    receipt = build_e1(source, evals, tmp_path / "out", strict=False)
    assert receipt["rows"] == 1
    assert receipt["historical_extraction_sha256"] is None
    assert receipt["counts"]["groups"] == 1
    assert pq.read_table(tmp_path / "out" / "public_join_v1.parquet")[
        "thinking_format"
    ].to_pylist() == [0.75]
    assert (
        receipt["output_sha256"]
        != "2f73f12277104a1efd33fc5e03ab0249da91cf56c38660f18cb20f4f560fc8d2"
    )


def test_eval_metadata_rejects_changed_archive(tmp_path: Path) -> None:
    archive = tmp_path / "x.eval"
    _archive(archive)
    size, sha = file_hash(archive)
    manifest = tmp_path / "metadata.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": SPEC["e1"]["dataset"],
                "revision": SPEC["e1"]["revision"],
                "files": [{"path": "eval_logs/x.eval", "bytes": size, "sha256": sha}],
            }
        ),
        encoding="utf-8",
    )
    verify_eval_metadata([archive], manifest)
    with archive.open("ab") as output:
        output.write(b"drift")
    with pytest.raises(ValueError, match="digest/size mismatch"):
        verify_eval_metadata([archive], manifest)


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("duplicate", "duplicate parquet key"),
        ("missing", "unmatched parquet key"),
        ("extra", "unmatched eval key"),
        ("binary", "binary format disagrees"),
        ("grade", "score disagrees"),
    ],
)
def test_e1_join_rejects_key_and_score_drift(mutation: str, expected: str) -> None:
    rows = [_source_row()]
    scores = {("x.eval", "s", 0): _scores()}
    if mutation == "duplicate":
        rows.append(dict(rows[0]))
    elif mutation == "missing":
        rows[0]["sample_id"] = "absent"
    elif mutation == "extra":
        scores[("x.eval", "other", 0)] = _scores()
    elif mutation == "binary":
        rows[0]["thinking_format_ok"] = 1
    elif mutation == "grade":
        rows[0]["passed"] = 0
    with pytest.raises(ValueError, match=expected):
        join_e1_rows(rows, scores)


def test_in_repo_campaign_banks_rebuild_with_exact_hashes(tmp_path: Path) -> None:
    receipt = build_campaign(tmp_path / "cache")
    assert {name: info["rows"] for name, info in receipt["outputs"].items()} == {
        "diagnostic-v3": 84,
        "hack-probes": 72,
    }
    assert {name: info["file"] for name, info in receipt["outputs"].items()} == {
        "diagnostic-v3": "diagnostic-v3.jsonl",
        "hack-probes": "hack-probes.jsonl",
    }


@pytest.mark.parametrize(
    "member, expected",
    [
        ("../outside.json", "unsafe eval archive member"),
        ("samples/../outside.json", "unsafe eval archive member"),
        ("samples/nested/extra.json", "unsafe eval sample path"),
    ],
)
def test_archive_rejects_unsafe_members(tmp_path: Path, member: str, expected: str) -> None:
    archive = tmp_path / "x.eval"
    _archive(archive)
    with zipfile.ZipFile(archive, "a") as output:
        output.writestr(member, "{}")
    with pytest.raises(ValueError, match=expected):
        scores_from_archive(archive)


def test_archive_rejects_duplicate_member(tmp_path: Path) -> None:
    archive = tmp_path / "x.eval"
    _archive(archive)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(archive, "a") as output:
            output.writestr("samples/one.json", "{}")
    with pytest.raises(ValueError, match="duplicate eval archive member"):
        scores_from_archive(archive)


@pytest.mark.parametrize(
    "limit, expected",
    [
        ("MAX_ARCHIVE_MEMBERS", "too many eval archive members"),
        ("MAX_MEMBER_UNCOMPRESSED_BYTES", "oversize eval archive member"),
        ("MAX_ARCHIVE_UNCOMPRESSED_BYTES", "oversize eval archive"),
    ],
)
def test_archive_rejects_member_and_decompression_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: str, expected: str
) -> None:
    archive = tmp_path / "x.eval"
    _archive(archive)
    monkeypatch.setattr(_builder, limit, 1)
    with pytest.raises(ValueError, match=expected):
        scores_from_archive(archive)


def test_eval_metadata_rejects_pagination_into_another_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        headers = {
            "Link": '<https://huggingface.co/api/datasets/other/data/tree/wrong/eval_logs?cursor=x>; rel="next"'
        }

        def __enter__(self):
            from io import StringIO

            self.body = StringIO("[]")
            return self.body_with_headers()

        def body_with_headers(self):
            self.body.headers = self.headers
            return self.body

        def __exit__(self, *_args):
            self.body.close()

    monkeypatch.setattr(_builder.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    with pytest.raises(ValueError, match="unexpected metadata pagination URL"):
        _builder.eval_metadata()


@pytest.mark.parametrize("mutation", ["drift", "extra"])
def test_vendor_provenance_rejects_changed_or_unlisted_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "a.py").write_text("x = 1\n")
    manifest = {
        "schema": "reward-lens-campaign-generator-source/1",
        "local_git_commit": SPEC["campaign"]["generator_local_git_commit"],
        "files": {"a.py": {"sha256": hashlib.sha256(b"x = 1\n").hexdigest()}},
    }
    (tmp_path / "vendor-provenance.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(_builder, "ROOT", tmp_path)
    _builder.verify_vendor_provenance()
    if mutation == "drift":
        (vendor / "a.py").write_text("x = 2\n")
    else:
        (vendor / "unexpected.py").write_text("pass\n")
    with pytest.raises(ValueError, match="campaign vendor (digest|file set) mismatch"):
        _builder.verify_vendor_provenance()
