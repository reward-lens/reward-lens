#!/usr/bin/env python3
"""Provision pinned public CPU validation fixtures into an isolated HF cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import NamedTuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "model-files.json"
REVISION = "52b2e48b0004586eff92c403efa5ce5547c43a45"
MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
EXPECTED_FILES = {
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
MAX_TOTAL_BYTES = 25_000_000
TOKENIZER_MANIFEST = HERE / "gpt2-tokenizer-files.json"
TOKENIZER_MODEL = "gpt2"
TOKENIZER_UPSTREAM_MODEL = "openai-community/gpt2"
TOKENIZER_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
TOKENIZER_EXPECTED_FILES = {
    "config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
TOKENIZER_MAX_TOTAL_BYTES = 4_000_000
REAL_SUBJECT_SCHEMA = "reward-lens-real-subject/1"
REAL_LLAMA_REVISION = "c2a90b9e673a425aba7089029be99f16bf5782ee"
REAL_QWEN_REVISION = "8c14a4e9e6321deaf572544339b16b8d6bbe8886"
REAL_GEMMA_REVISION = "6a6e503f3c03a88e92c34c4e6860045eb88ec6e9"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SHA1 = re.compile(r"[0-9a-f]{40}\Z")


class Fixture(NamedTuple):
    schema: str
    manifest: Path
    model: str
    upstream_model: str
    revision: str
    expected_files: set[str]
    max_total_bytes: int


def _fixture(name: str) -> Fixture:
    if name == "tiny-qwen":
        return Fixture(
            "reward-lens-validation-model/1",
            MANIFEST,
            MODEL,
            MODEL,
            REVISION,
            EXPECTED_FILES,
            MAX_TOTAL_BYTES,
        )
    if name == "gpt2-tokenizer":
        return Fixture(
            "reward-lens-validation-tokenizer/1",
            TOKENIZER_MANIFEST,
            TOKENIZER_MODEL,
            TOKENIZER_UPSTREAM_MODEL,
            TOKENIZER_REVISION,
            TOKENIZER_EXPECTED_FILES,
            TOKENIZER_MAX_TOTAL_BYTES,
        )
    if name == "migration-llama-head":
        model = "Skywork/Skywork-Reward-Llama-3.1-8B"
        return Fixture(
            REAL_SUBJECT_SCHEMA,
            HERE / "llama-head-files.json",
            model,
            model,
            REAL_LLAMA_REVISION,
            {"model-00004-of-00004.safetensors"},
            118_000_000,
        )
    if name == "migration-qwen3":
        model = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
        return Fixture(
            REAL_SUBJECT_SCHEMA,
            HERE / "qwen3-real-files.json",
            model,
            model,
            REAL_QWEN_REVISION,
            {
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "chat_template.jinja",
                "merges.txt",
                "vocab.json",
            },
            1_209_000_000,
        )
    if name == "migration-gemma2-config":
        model = "Ray2333/GRM-Gemma2-2B-rewardmodel-ft"
        return Fixture(
            REAL_SUBJECT_SCHEMA,
            HERE / "gemma2-config-files.json",
            model,
            model,
            REAL_GEMMA_REVISION,
            {"config.json"},
            2_000,
        )
    raise ValueError(f"unknown validation fixture: {name}")


def _files(fixture: Fixture) -> dict[str, dict]:
    manifest = json.loads(fixture.manifest.read_text())
    if (
        manifest.get("schema") != fixture.schema
        or manifest.get("model") != fixture.model
        or manifest.get("upstream_model", fixture.model) != fixture.upstream_model
        or manifest.get("revision") != fixture.revision
        or not isinstance(manifest.get("files"), dict)
        or set(manifest["files"]) != fixture.expected_files
    ):
        raise ValueError("validation fixture manifest identity or file set mismatch")
    total = 0
    for name, spec in manifest["files"].items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not isinstance(spec, dict):
            raise ValueError("unsafe validation model file entry")
        size = spec.get("bytes")
        lfs = spec.get("lfs_sha256")
        if (
            type(size) is not int
            or size <= 0
            or not isinstance(spec.get("sha256"), str)
            or not SHA256.fullmatch(spec["sha256"])
            or not isinstance(spec.get("git_blob_sha1"), str)
            or not SHA1.fullmatch(spec["git_blob_sha1"])
            or (lfs is not None and (not isinstance(lfs, str) or not SHA256.fullmatch(lfs)))
            or (lfs is not None and lfs != spec["sha256"])
        ):
            raise ValueError(f"{name}: invalid pinned manifest entry")
        total += size
    if total > fixture.max_total_bytes:
        raise ValueError("validation fixture exceeds bounded size")
    return manifest["files"]


def _snapshot(root: Path, offline: bool, fixture: Fixture) -> Path:
    """Inspect lexical ancestors before creating any cache directory."""
    root = Path(os.path.abspath(root))
    for ancestor in reversed((root.parent, *root.parent.parents)):
        if ancestor.is_symlink() or not ancestor.is_dir():
            raise ValueError(f"unsafe model cache ancestor: {ancestor}")
    path = root.parent
    for component in (
        root.name,
        "hub",
        "models--" + fixture.model.replace("/", "--"),
        "snapshots",
        fixture.revision,
    ):
        path = path / component
        if path.is_symlink():
            raise ValueError(f"unsafe model cache directory: {path}")
        if path.exists():
            if not path.is_dir():
                raise ValueError(f"unsafe model cache directory: {path}")
        elif offline:
            raise FileNotFoundError(path)
        else:
            path.mkdir()
    return path


def _discard_partial(path: Path, size: int) -> None:
    if path.is_symlink():
        raise ValueError(f"unsafe partial model file: {path}")
    if path.exists():
        if not path.is_file() or path.stat().st_size > size:
            raise ValueError(f"unsafe partial model file: {path}")
        path.unlink()


def _scan_snapshot(snapshot: Path, files: dict[str, dict], offline: bool) -> None:
    partials: list[tuple[Path, int]] = []
    for entry in snapshot.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"unsafe model snapshot entry: {entry}")
        if entry.name in files:
            _verify(entry, files[entry.name])
        elif entry.name.endswith(".part") and entry.name[:-5] in files and not offline:
            partials.append((entry, files[entry.name[:-5]]["bytes"]))
        else:
            raise ValueError(f"unexpected model snapshot entry: {entry}")
    for entry, size in partials:
        _discard_partial(entry, size)


def _digests(path: Path, limit: int) -> tuple[int, str, str]:
    sha = hashlib.sha256()
    blob = hashlib.sha1()  # noqa: S324 - Git blob identity, not a security digest
    blob.update(b"blob " + str(limit).encode() + b"\0")
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(256 * 1024):
            size += len(chunk)
            if size > limit:
                raise ValueError(f"{path.name}: exceeded pinned size")
            sha.update(chunk)
            blob.update(chunk)
    return size, sha.hexdigest(), blob.hexdigest()


def _verify(path: Path, spec: dict) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe or missing model file: {path}")
    size, sha, blob = _digests(path, spec["bytes"])
    if size != spec["bytes"] or sha != spec["sha256"]:
        raise ValueError(f"{path.name}: SHA-256 or size mismatch")
    if spec["lfs_sha256"]:
        if sha != spec["lfs_sha256"]:
            raise ValueError(f"{path.name}: LFS SHA-256 mismatch")
    elif blob != spec["git_blob_sha1"]:
        raise ValueError(f"{path.name}: Git blob SHA-1 mismatch")


def _fetch(path: Path, name: str, spec: dict, fixture: Fixture) -> None:
    url = (
        f"https://huggingface.co/{fixture.upstream_model}/resolve/"
        f"{fixture.revision}/{quote(name)}?download=true"
    )
    for attempt in range(3):
        partial = path.with_name(path.name + ".part")
        try:
            request = Request(url, headers={"User-Agent": "reward-lens-release-validation/3.1.0"})
            with urlopen(request, timeout=45) as response, partial.open("xb") as output:
                if response.status != 200:
                    raise ValueError(f"{name}: HTTP {response.status}")
                stated = response.headers.get("Content-Length")
                if stated and int(stated) != spec["bytes"]:
                    raise ValueError(f"{name}: Content-Length mismatch")
                received = 0
                while chunk := response.read(256 * 1024):
                    received += len(chunk)
                    if received > spec["bytes"]:
                        raise ValueError(f"{name}: download exceeded pinned size")
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            _verify(partial, spec)
            partial.replace(path)
            return
        except (OSError, HTTPError, URLError):
            _discard_partial(partial, spec["bytes"])
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
        except Exception:
            _discard_partial(partial, spec["bytes"])
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument(
        "--fixture",
        choices=(
            "tiny-qwen",
            "gpt2-tokenizer",
            "migration-llama-head",
            "migration-qwen3",
            "migration-gemma2-config",
        ),
        default="tiny-qwen",
    )
    parser.add_argument("--offline", action="store_true", help="verify existing cache only")
    args = parser.parse_args()
    fixture = _fixture(args.fixture)
    files = _files(fixture)
    snapshot = _snapshot(args.hf_home, args.offline, fixture)
    _scan_snapshot(snapshot, files, args.offline)
    for name, spec in sorted(files.items()):
        path = snapshot / name
        if path.exists() or path.is_symlink():
            _verify(path, spec)
        elif args.offline:
            raise FileNotFoundError(path)
        else:
            _fetch(path, name, spec, fixture)
        print(f"verified {name} {spec['bytes']} {spec['sha256']}", flush=True)
    print(f"pinned validation fixture ready: {fixture.model}@{fixture.revision} ({snapshot})")


if __name__ == "__main__":
    main()
