"""Install the shipped portable reward-lens skill into one declared agent layout.

The destination is deliberately a project root, never an arbitrary target path.  That leaves the
agent choice finite and auditable and prevents a command-line value from redirecting a skill write
through a user configuration symlink.  The installed source comes from package resources so a
wheel and a source checkout install the identical bytes.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import os
from dataclasses import dataclass
from pathlib import Path

SKILL_NAME = "reward-lens"
_AGENTS = {"claude": ".claude", "codex": ".agents", "copilot": ".github"}


class SkillInstallRefusal(ValueError):
    """An installation request whose target cannot be selected or safely written."""


@dataclass(frozen=True)
class SkillInstall:
    """The exact source and destination confirmed after an installation."""

    agent: str
    destination: Path
    digest: str


def install_skill(project: Path, *, agent: str | None = None) -> SkillInstall:
    """Write one fresh skill copy below ``project`` and read the exact bytes back.

    Detection considers only established agent marker directories.  An explicit agent selects one
    fixed marker and may create it, but does not turn an arbitrary path into a write destination.
    Existing installed content is a conflict: this command never replaces skill instructions.
    """
    root = _project_root(project)
    chosen, marker_name = _choose_agent(root, agent)
    marker = root / marker_name
    _safe_directory(marker, create=True)
    skill_parent = marker / "skills"
    _safe_directory(skill_parent, create=True)
    target = skill_parent / SKILL_NAME
    if target.exists() or target.is_symlink():
        raise SkillInstallRefusal(f"skill destination already exists: {target}")
    source = _skill_bytes()
    _write_and_read_back(target / "SKILL.md", source)
    return SkillInstall(
        agent=chosen,
        destination=target / "SKILL.md",
        digest=hashlib.sha256(source).hexdigest(),
    )


def _project_root(project: Path) -> Path:
    spelled = Path(os.path.abspath(project))
    resolved = spelled.resolve()
    if not spelled.exists() or not spelled.is_dir() or spelled.is_symlink() or spelled != resolved:
        raise SkillInstallRefusal(
            f"unsafe skill destination: {project} must be an existing, non-symlinked directory"
        )
    return resolved


def _choose_agent(root: Path, requested: str | None) -> tuple[str, str]:
    if requested is not None:
        marker = _AGENTS.get(requested)
        if marker is None:
            choices = ", ".join(_AGENTS)
            raise SkillInstallRefusal(f"unknown agent {requested!r}; choose one of {choices}")
        return requested, marker

    found: list[tuple[str, str]] = []
    for name, marker in _AGENTS.items():
        candidate = root / marker
        if candidate.exists() or candidate.is_symlink():
            _safe_directory(candidate, create=False)
            found.append((name, marker))
    if not found:
        raise SkillInstallRefusal(
            "cannot detect an agent at this project; pass --agent claude, codex, or copilot"
        )
    if len(found) > 1:
        names = ", ".join(name for name, _ in found)
        raise SkillInstallRefusal(f"conflicting agent markers ({names}); pass --agent explicitly")
    return found[0]


def _safe_directory(path: Path, *, create: bool) -> None:
    if path.is_symlink():
        raise SkillInstallRefusal(f"unsafe skill destination: {path} is a symlink")
    if path.exists():
        if not path.is_dir():
            raise SkillInstallRefusal(f"unsafe skill destination: {path} is not a directory")
        return
    if not create:
        raise SkillInstallRefusal(f"unsafe skill destination: {path} is missing")
    path.mkdir()


def _skill_bytes() -> bytes:
    """Read the wheel resource first; the source-tree fallback supports editable development."""
    packaged = importlib.resources.files("reward_lens").joinpath("skills", SKILL_NAME, "SKILL.md")
    if packaged.is_file():
        return packaged.read_bytes()
    source_tree = Path(__file__).resolve().parents[3] / "skills" / SKILL_NAME / "SKILL.md"
    if source_tree.is_file():
        return source_tree.read_bytes()
    raise SkillInstallRefusal("the reward-lens skill is not present in this installed artifact")


def _write_and_read_back(path: Path, source: bytes) -> None:
    """Create the target once, then prove its saved bytes are the shipped source bytes."""
    path.parent.mkdir()
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(source)
    finally:
        if descriptor is not None:
            os.close(descriptor)
    saved = path.read_bytes()
    if saved != source:
        raise SkillInstallRefusal(f"skill write readback did not match source: {path}")
