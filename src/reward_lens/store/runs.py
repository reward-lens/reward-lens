"""Run directories (D-27): findable by a caller that did not start them, resumable, forked when done.

An audit that dies at minute nine and starts over at minute zero is a tool people stop using, and
an id only its creator knows is no better for an agent. The directory is the record; `index.json`
is a cache of the manifests and a test deletes it.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rfc8785

from .errors import RunNotFound

__all__ = ["RunDir", "runs_root", "state_root"]


def state_root() -> Path:
    """`$XDG_STATE_HOME`, read at call time, defaulting to `~/.local/state`."""
    declared = os.environ.get("XDG_STATE_HOME")
    return Path(declared) if declared else Path.home() / ".local" / "state"


def runs_root() -> Path:
    return state_root() / "reward-lens" / "runs"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path: Path, payload: Any) -> None:
    path.write_bytes(rfc8785.dumps(payload))


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class RunDir:
    """One run's directory. Every answer comes off the disk, so a fork never mutates the original."""

    run_id: str
    path: Path

    # --- the record --------------------------------------------------------------------------

    @property
    def manifest(self) -> dict:
        return _read(self.path / "manifest.json")

    @property
    def name(self) -> str | None:
        return self.manifest["name"]

    @property
    def project(self) -> str | None:
        return self.manifest["project"]

    @property
    def state(self) -> str:
        return self.manifest["state"]

    @property
    def finished(self) -> bool:
        return self.state == "finished"

    # --- creating ----------------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        project: Path | str | None,
        name: str | None,
        command: str = "audit",
        plan: dict | None = None,
        forked_from: str | None = None,
    ) -> "RunDir":
        root = runs_root()
        root.mkdir(parents=True, exist_ok=True)
        run_id = "run-" + secrets.token_hex(6)
        path = root / run_id
        path.mkdir()
        (path / "partial").mkdir()
        created = _now()
        _write(
            path / "manifest.json",
            {
                "run_id": run_id,
                "name": name,
                "project": str(project) if project is not None else None,
                "command": command,
                "created": created,
                "seq": _next_seq(root),
                "state": "running",
                "forked_from": forked_from,
            },
        )
        _write(path / "plan.json", plan if plan is not None else {"steps": []})
        (path / "log.txt").write_text(f"{created} {run_id} created\n", encoding="utf-8")
        _write_index(root, _scan(root))
        return cls(run_id=run_id, path=path)

    # --- finding -----------------------------------------------------------------------------

    @classmethod
    def list(cls) -> list["RunDir"]:
        """Every run this machine kept, newest first."""
        root = runs_root()
        return [cls(run_id=row["run_id"], path=root / row["run_id"]) for row in _index(root)]

    @classmethod
    def find(cls, id_or_name: str) -> "RunDir":
        for run in cls.list():
            if run.run_id == id_or_name:
                return run
        for run in cls.list():
            if run.manifest["name"] == id_or_name:
                return run
        raise RunNotFound(wanted=id_or_name, root=runs_root())

    @classmethod
    def most_recent(cls, project: Path | str | None = None) -> "RunDir | None":
        wanted = str(project) if project is not None else None
        for run in cls.list():
            if wanted is None or run.manifest["project"] == wanted:
                return run
        return None

    # --- running -----------------------------------------------------------------------------

    def log(self, line: str) -> None:
        with (self.path / "log.txt").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def write_partial(self, instrument: str, result: Any) -> Path:
        path = self.path / "partial" / f"{instrument}.json"
        _write(path, result)
        return path

    def partial(self) -> dict[str, Any]:
        return {p.stem: _read(p) for p in sorted((self.path / "partial").glob("*.json"))}

    def finish(self) -> None:
        manifest = dict(self.manifest)
        manifest["state"] = "finished"
        manifest["finished"] = _now()
        _write(self.path / "manifest.json", manifest)
        _write_index(runs_root(), _scan(runs_root()))

    # --- resuming ----------------------------------------------------------------------------

    def resume_command(self) -> list[str]:
        manifest = self.manifest
        command = ["reward-lens", manifest.get("command") or "audit"]
        if manifest["project"]:
            command.append(manifest["project"])
        command += ["--resume", manifest["name"] or self.run_id]
        return command

    def resume(self) -> "RunDir":
        """Continue an unfinished run; fork a finished one rather than mutating its record."""
        if not self.finished:
            return self
        manifest = self.manifest
        fork = RunDir.create(
            project=manifest["project"],
            name=manifest["name"],
            command=manifest.get("command") or "audit",
            plan=_read(self.path / "plan.json"),
            forked_from=self.run_id,
        )
        for partial in sorted((self.path / "partial").glob("*.json")):
            shutil.copy2(partial, fork.path / "partial" / partial.name)
        fork.log(f"forked from {self.run_id}")
        return fork


# --- the index, which is a cache of the manifests ------------------------------------------------


def _next_seq(root: Path) -> int:
    return len([p for p in root.glob("run-*") if p.is_dir()])


def _scan(root: Path) -> list[dict]:
    rows: list[dict] = []
    if not root.is_dir():
        return rows
    for path in root.glob("run-*"):
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _read(manifest_path)
        rows.append(
            {
                "run_id": manifest["run_id"],
                "name": manifest["name"],
                "project": manifest["project"],
                "state": manifest["state"],
                "seq": manifest.get("seq", 0),
            }
        )
    rows.sort(key=lambda row: (row["seq"], row["run_id"]), reverse=True)
    return rows


def _write_index(root: Path, rows: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write(root / "index.json", {"runs": rows})


def _index(root: Path) -> list[dict]:
    on_disk = _scan(root)
    try:
        cached = _read(root / "index.json")["runs"]
    except (OSError, ValueError, KeyError, TypeError):
        cached = None
    if cached is not None and [r.get("run_id") for r in cached] == [r["run_id"] for r in on_disk]:
        return cached
    if on_disk:
        _write_index(root, on_disk)
    return on_disk
