#!/usr/bin/env python3
"""Provision the pinned public tiny-Qwen test model into an isolated HF cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
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
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SHA1 = re.compile(r"[0-9a-f]{40}\Z")


def _files() -> dict[str, dict]:
    manifest = json.loads(MANIFEST.read_text())
    if (
        manifest.get("schema") != "reward-lens-validation-model/1"
        or manifest.get("model") != MODEL
        or manifest.get("revision") != REVISION
        or not isinstance(manifest.get("files"), dict)
        or set(manifest["files"]) != EXPECTED_FILES
    ):
        raise ValueError("validation model manifest identity or file set mismatch")
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
    if total > MAX_TOTAL_BYTES:
        raise ValueError("validation model exceeds bounded size")
    return manifest["files"]


def _snapshot(root: Path, offline: bool) -> Path:
    """Inspect lexical ancestors before creating any cache directory."""
    root = Path(os.path.abspath(root))
    for ancestor in reversed((root.parent, *root.parent.parents)):
        if ancestor.is_symlink() or not ancestor.is_dir():
            raise ValueError(f"unsafe model cache ancestor: {ancestor}")
    path = root.parent
    for component in (
        root.name,
        "hub",
        "models--trl-internal-testing--tiny-Qwen3ForCausalLM",
        "snapshots",
        REVISION,
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


def _fetch(path: Path, name: str, spec: dict) -> None:
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{quote(name)}?download=true"
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
    parser.add_argument("--offline", action="store_true", help="verify existing cache only")
    args = parser.parse_args()
    files = _files()
    snapshot = _snapshot(args.hf_home, args.offline)
    _scan_snapshot(snapshot, files, args.offline)
    for name, spec in sorted(files.items()):
        path = snapshot / name
        if path.exists() or path.is_symlink():
            _verify(path, spec)
        elif args.offline:
            raise FileNotFoundError(path)
        else:
            _fetch(path, name, spec)
        print(f"verified {name} {spec['bytes']} {spec['sha256']}", flush=True)
    print(f"pinned model snapshot ready: {MODEL}@{REVISION} ({snapshot})")


if __name__ == "__main__":
    main()
