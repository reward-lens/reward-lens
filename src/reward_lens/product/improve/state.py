"""Durable, append-only state for the six-step improve loop."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

STATES = ("describe", "propose", "plan", "execute", "decide", "export")


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"IMPROVE_RESUME_REQUIRED: state_path={path}") from error
    if not isinstance(value, dict) or value.get("state") not in STATES:
        raise ValueError(f"IMPROVE_RESUME_REQUIRED: state_path={path}")
    return value


def session_path(root: Path | str, session_id: str = "current") -> Path:
    """Return the one durable location for an owner-visible improve session.

    ``session_id`` is deliberately a filename fragment, rather than a path.  A
    resume argument is an identifier for the stored decision trail, never an
    opportunity to redirect the engine to unrelated bytes.
    """
    if not isinstance(session_id, str) or not session_id or session_id in {".", ".."}:
        raise ValueError("IMPROVE_RESUME_REQUIRED: resume must name one session identifier")
    if "/" in session_id or "\\" in session_id:
        raise ValueError("IMPROVE_RESUME_REQUIRED: resume must name one session identifier")
    return Path(root) / ".reward-lens" / "improve" / f"{session_id}.json"


def write_once(path: Path, payload: dict[str, Any]) -> bytes:
    """Persist a registration exactly once, without replacing existing evidence."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return raw
