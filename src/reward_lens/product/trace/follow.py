"""Live following: read-only, non-interfering, and testable rather than asserted (section 7.4).

Following tails a run directory. It opens files and nothing else: it does not attach to the
training process, does not inspect it, and does not send it anything. That claim is a test over
this file's own text, which is the only way a claim of the form "this never touches X" is worth
making.

The capture is testable in four ways. Every chunk carries a sequence number, so a gap is a fact
rather than an absence nobody noticed. A chunk's manifest is sealed atomically, written beside it
under a temporary name and moved into place, so a seal is either wholly there or not there at all.
A partial line at the end of a chunk is a torn tail, and it is carried in the record instead of
being dropped by a reader that only counts whole lines. And the policy is declared: `strict`
refuses on a gap or a tear, `permissive` records it and says which policy allowed it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from .errors import RecordIntegrity
from .record import Policy

__all__ = ["CHUNK_PATTERN", "Chunk", "chunk_path", "follow", "seal_chunk", "seal_path"]

CHUNK_NAME = "chunk_{seq:05d}.jsonl"
SEAL_NAME = "chunk_{seq:05d}.seal"
CHUNK_PATTERN = re.compile(r"^chunk_(\d{5})\.jsonl$")


@dataclass(frozen=True)
class Chunk:
    """One chunk of a followed run, with everything a reader needs to distrust it."""

    seq: int
    path: str
    rows: tuple[dict[str, Any], ...] = ()
    sealed: bool = False
    digest: str | None = None
    torn: bool = False
    torn_bytes: str = ""
    gap_before: tuple[int, ...] = field(default_factory=tuple)


def chunk_path(directory: Path | str, seq: int) -> Path:
    return Path(directory) / CHUNK_NAME.format(seq=seq)


def seal_path(directory: Path | str, seq: int) -> Path:
    return Path(directory) / SEAL_NAME.format(seq=seq)


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _atomic_write(target: Path, data: bytes) -> None:
    """Write beside the target and move it into place, so no reader ever sees a half file."""
    temporary = target.with_name(target.name + ".tmp")
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def seal_chunk(directory: Path | str, seq: int, rows: Sequence[dict[str, Any]]) -> Path:
    """Write one chunk and seal it. Both moves are atomic; the seal lands last."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")
    target = chunk_path(root, seq)
    _atomic_write(target, body)
    manifest = {
        "seq": seq,
        "chunk": target.name,
        "rows": len(rows),
        "digest": _digest(body),
    }
    seal = seal_path(root, seq)
    _atomic_write(seal, (json.dumps(manifest, sort_keys=True) + "\n").encode("utf-8"))
    return seal


def _read_chunk(path: Path, seq: int, gap: tuple[int, ...]) -> Chunk:
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    lines = text.split("\n")
    trailing = lines.pop()  # "" when the file ends on a newline; a partial line when it does not
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            trailing = line
    seal = seal_path(path.parent, seq)
    sealed = False
    declared: str | None = None
    if seal.exists():
        try:
            manifest = json.loads(seal.read_text(encoding="utf-8"))
        except ValueError:
            manifest = {}
        declared = manifest.get("digest")
        sealed = declared == _digest(raw)
    return Chunk(
        seq=seq,
        path=str(path),
        rows=tuple(rows),
        sealed=sealed,
        digest=declared,
        torn=bool(trailing),
        torn_bytes=trailing,
        gap_before=gap,
    )


def follow(directory: Path | str, *, policy: Policy = "strict") -> Iterator[Chunk]:
    """Every chunk in the directory, in sequence order, under the declared policy."""
    root = Path(directory)
    found: list[tuple[int, Path]] = []
    for path in sorted(root.glob("chunk_*.jsonl")):
        match = CHUNK_PATTERN.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    found.sort()
    expected = 1
    for seq, path in found:
        gap = tuple(range(expected, seq))
        if gap and policy == "strict":
            raise RecordIntegrity(
                message=(
                    f"the capture is missing chunk(s) {', '.join(str(one) for one in gap)} "
                    f"before {path.name}, and the declared policy is strict"
                ),
                remediation=(
                    'follow with policy="permissive" to carry the gap in the record instead of '
                    "refusing, which is the honest reading when the writer is still running"
                ),
                context={"rule": "trace.sequence_gap", "row": path.name, "missing": list(gap)},
            )
        chunk = _read_chunk(path, seq, gap)
        if chunk.torn and policy == "strict":
            raise RecordIntegrity(
                message=(
                    f"{path.name} ends in a partial write of {len(chunk.torn_bytes)} bytes and "
                    f"the declared policy is strict"
                ),
                remediation=(
                    'follow with policy="permissive" to carry the torn tail in the record, where '
                    "it is visible, rather than dropping the partial line"
                ),
                context={
                    "rule": "trace.torn_tail",
                    "row": path.name,
                    "bytes": len(chunk.torn_bytes),
                },
            )
        expected = seq + 1
        yield chunk
