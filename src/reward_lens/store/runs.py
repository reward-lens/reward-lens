"""Run directories (D-27): findable by a caller that did not start them, resumable, forked when done.

An audit that dies at minute nine and starts over at minute zero is a tool people stop using, and
an id only its creator knows is no better for an agent. The directory is the record; `index.json`
is a cache of the manifests and a test deletes it.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rfc8785

from .errors import DependencyDigestMismatch, RunNotFound

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
    _write_bytes(path, rfc8785.dumps(payload))


def _write_bytes(path: Path, payload: bytes) -> None:
    """Replace a run record only after its complete bytes exist beside it."""
    try:
        previous_mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        previous_mode = None
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_bytes(payload)
        if previous_mode is not None:
            temporary.chmod(previous_mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
                "partials": {},
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
        """Seal one instrument result exactly once.

        A partial is the durable boundary of an audit.  It is deliberately not a
        cache: a resumed audit may read it, but may not replace it with a second
        execution of the same instrument.
        """
        path = self.path / "partial" / f"{instrument}.json"
        payload = rfc8785.dumps(result)
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        if path.exists():
            existing = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            if existing != digest:
                raise DependencyDigestMismatch(
                    digest=f"partial:{instrument}", expected=existing, found=digest
                )
            return path
        _write_bytes(path, payload)
        manifest = dict(self.manifest)
        partials = dict(manifest.get("partials", {}))
        partials[instrument] = digest
        manifest["partials"] = partials
        _write(self.path / "manifest.json", manifest)
        _write_index(runs_root(), _scan(runs_root()))
        return path

    def partial(self) -> dict[str, Any]:
        return {p.stem: _read(p) for p in sorted((self.path / "partial").glob("*.json"))}

    def bind_subject(self, subject_ref: str) -> None:
        """Bind a run to the source it is permitted to resume."""
        manifest = dict(self.manifest)
        previous = manifest.get("subject_ref")
        if previous is not None and previous != subject_ref:
            raise DependencyDigestMismatch(
                digest="subject_ref", expected=str(previous), found=subject_ref
            )
        manifest["subject_ref"] = subject_ref
        _write(self.path / "manifest.json", manifest)
        _write_index(runs_root(), _scan(runs_root()))

    def bind_execution(self, identity: dict[str, Any]) -> None:
        """Bind resume to the exact executable audit contract, not only its source.

        A partial is valid for one source under one plan, method declaration, seeker/budget arm,
        and requested instrument set.  A human-readable copy makes the refusal inspectable; the
        digest is the comparison key and uses the same canonical serialisation as partials.
        """
        encoded = rfc8785.dumps(identity)
        digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
        manifest = dict(self.manifest)
        previous = manifest.get("execution_digest")
        if previous is not None and previous != digest:
            raise DependencyDigestMismatch(digest="execution", expected=str(previous), found=digest)
        manifest["execution_digest"] = digest
        manifest["execution_identity"] = identity
        _write(self.path / "manifest.json", manifest)
        _write_index(runs_root(), _scan(runs_root()))

    def finish(self, *, record_path: Path | str | None = None) -> None:
        manifest = dict(self.manifest)
        if record_path is not None:
            manifest["record_path"] = str(record_path)
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
            fork.write_partial(partial.stem, _read(partial))
        if manifest.get("subject_ref") is not None:
            fork.bind_subject(str(manifest["subject_ref"]))
        if manifest.get("record_path") is not None:
            fork_manifest = dict(fork.manifest)
            fork_manifest["record_path"] = manifest["record_path"]
            _write(fork.path / "manifest.json", fork_manifest)
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
