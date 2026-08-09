"""The nine digests of section 6.1, each over defined canonical bytes.

Every digest here is `contracts.digest()` of a dict this module builds by naming its keys, never
of a pydantic dump: a digest must be a function of the project's values, not of which fields
someone happened to set. Raw file bytes are hashed as bytes; everything composed is hashed through
`contracts.canonical_bytes` (D-10, the only producer of hashed bytes).

A digest is `None` where the thing is absent. `policy` is absent from every project, because a
project declares a reward system and an experiment binds the checkpoint; that separation is what
makes reuse safe. `instrument_method` is never absent: this build is always something.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from reward_lens.contracts import ProjectConfig, digest

__all__ = [
    "DIGEST_NAMES",
    "Digests",
    "compute",
    "file_digest",
    "instrument_method_digest",
    "method_set",
]

#: The nine, in the order the schema's `subject.digests` names them.
DIGEST_NAMES: tuple[str, ...] = (
    "source",
    "environment",
    "scorer_config",
    "task_distribution",
    "samples",
    "outcome_protocol",
    "policy",
    "training_semantics",
    "instrument_method",
)

#: Where a project declares the environment it pins, most specific first.
ENVIRONMENT_FILES: tuple[str, ...] = ("uv.lock", "poetry.lock", "requirements.txt", "pyproject.toml")


@dataclass(frozen=True)
class Digests:
    """The nine, maintained separately."""

    source: str | None
    environment: str | None
    scorer_config: str | None
    task_distribution: str | None
    samples: str | None
    outcome_protocol: str | None
    policy: str | None
    training_semantics: str | None
    instrument_method: str

    def to_dict(self) -> dict[str, str | None]:
        return {name: getattr(self, name) for name in DIGEST_NAMES}

    def differences(self, other: "Digests") -> frozenset[str]:
        mine, theirs = self.to_dict(), other.to_dict()
        return frozenset(name for name in DIGEST_NAMES if mine[name] != theirs[name])


def file_digest(path: Path) -> str | None:
    """`sha256:<hex>` over a file's bytes, or `None` when it is not there."""
    if not path.is_file():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _line_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _tree(root: Path) -> list[list[str | None]]:
    """Every file under `root`, as sorted `[relative path, digest]` pairs."""
    if root.is_file():
        return [[root.name, file_digest(root)]]
    if not root.is_dir():
        return []
    out: list[list[str | None]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        out.append([path.relative_to(root).as_posix(), file_digest(path)])
    return out


def _entry_files(root: Path, entry: str) -> dict[str, Any]:
    """What `module:callable` or `path.py:callable` points at, as data."""
    left, sep, right = entry.rpartition(":")
    target, name = (left, right) if sep else (entry, None)
    path = root / target
    if target.endswith(".py") or path.exists():
        return {"path": target, "callable": name, "content": file_digest(path)}
    return {"module": target, "callable": name, "content": None}


def _source(config: ProjectConfig, root: Path) -> str:
    payload = {
        "kind": config.reward.kind,
        "entry": _entry_files(root, config.reward.entry),
        "components": [
            {"name": c.name, "entry": _entry_files(root, c.entry)} for c in config.reward.components
        ],
    }
    return digest(payload)


def _environment(root: Path) -> str | None:
    for name in ENVIRONMENT_FILES:
        content = file_digest(root / name)
        if content is not None:
            return digest({"declared_by": name, "content": content})
    return None


def _scorer_config(config: ProjectConfig) -> str:
    reward = config.reward
    return digest(
        {
            "kind": reward.kind,
            "entry": reward.entry,
            "shape": reward.shape,
            "components": [
                {"name": c.name, "entry": c.entry, "weight": c.weight} for c in reward.components
            ],
            "watch": sorted(reward.watch),
        }
    )


def _task_distribution(config: ProjectConfig, root: Path) -> str:
    tasks = config.tasks
    path = root / tasks.path
    return digest(
        {
            "path": tasks.path,
            "prompt": tasks.prompt,
            "reference": tasks.reference,
            "tests": tasks.tests,
            "content": file_digest(path),
            "records": _line_count(path),
        }
    )


def _samples(config: ProjectConfig, root: Path) -> str | None:
    if config.responses is None:
        return None
    path = root / config.responses.path
    return digest(
        {
            "path": config.responses.path,
            "content": file_digest(path),
            "records": _line_count(path),
        }
    )


def _outcome_protocol(config: ProjectConfig, root: Path) -> str | None:
    if config.outcome is None:
        return None
    return digest(
        {
            "kind": config.outcome.kind,
            "path": config.outcome.path,
            "tree": _tree(root / config.outcome.path),
        }
    )


def _training_semantics(config: ProjectConfig) -> str | None:
    reward = config.reward
    if reward.trainer is None:
        return None
    return digest(
        {
            "trainer": reward.trainer,
            "kind": reward.kind,
            "watch": sorted(reward.watch),
            "components": [{"name": c.name, "weight": c.weight} for c in reward.components],
        }
    )


def _build_version() -> str:
    try:
        from importlib.metadata import version as _version

        return _version("reward-lens")
    except Exception:  # pragma: no cover - an uninstalled tree still has a version
        from reward_lens import __version__

        return __version__


def instrument_method_digest(methods: Sequence[str] | Iterable[str] = ()) -> str:
    """The method set this build would run: the tool, its version, and the instruments named."""
    return digest({"tool": "reward-lens", "version": _build_version(), "methods": sorted(methods)})


def method_set(record: Mapping[str, Any]) -> tuple[str, ...]:
    """The instrument method set a record names: `<method id>@<version>`, deduplicated and sorted.

    One derivation, so that the `instrument_method` digest a record declares and the one the
    project computes when that record is checked are computed over the same set. A caller building
    a record passes this same set to `Project.version(methods=...)`; the store derives it back off
    the record's own entries rather than guessing that the set was empty.
    """
    found: set[str] = set()
    for entries in (record.get("measurement") or {}).values():
        for entry in entries or ():
            method = entry.get("method") or {}
            name = method.get("id")
            if not name:
                continue
            version = method.get("version")
            found.add(f"{name}@{version}" if version else name)
    return tuple(sorted(found))


def compute(config: ProjectConfig, root: Path, *, methods: Iterable[str] = ()) -> Digests:
    """The nine digests of a project as it stands on disk."""
    return Digests(
        source=_source(config, root),
        environment=_environment(root),
        scorer_config=_scorer_config(config),
        task_distribution=_task_distribution(config, root),
        samples=_samples(config, root),
        outcome_protocol=_outcome_protocol(config, root),
        policy=None,
        training_semantics=_training_semantics(config),
        instrument_method=instrument_method_digest(methods),
    )


