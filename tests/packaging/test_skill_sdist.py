"""The portable skill survives the ordinary source-distribution build route."""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SKILL = Path("skills/reward-lens/SKILL.md")
WHEEL_SKILL = Path("reward_lens/skills/reward-lens/SKILL.md")


def _env(**extra: str) -> dict[str, str]:
    env = {
        key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}
    }
    env.update({"UV_OFFLINE": "1", **extra})
    return env


def _run(
    argv: list[str], *, cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        argv,
        cwd=str(cwd),
        env=env or _env(),
        capture_output=True,
        text=True,
        timeout=180,
    )


def _ok(process: subprocess.CompletedProcess[str], label: str) -> subprocess.CompletedProcess[str]:
    if process.returncode != 0:
        pytest.fail(f"{label} failed ({process.returncode}):\n{process.stdout}\n{process.stderr}")
    return process


def test_skill_survives_sdist_to_wheel_and_installed_init(tmp_path: Path) -> None:
    """Build from a source distribution, not the checkout where force-include can mask absence."""
    dist = tmp_path / "dist"
    _ok(
        _run(["uv", "build", "--no-sources", "--sdist", "--out-dir", str(dist)], cwd=ROOT),
        "source distribution build",
    )
    sdists = sorted(dist.glob("*.tar.gz"))
    assert len(sdists) == 1, [path.name for path in sdists]
    with tarfile.open(sdists[0]) as archive:
        names = archive.getnames()
        assert any(name.endswith(str(SKILL)) for name in names), names
        archive.extractall(tmp_path / "unpacked", filter="data")
    roots = [path for path in (tmp_path / "unpacked").iterdir() if path.is_dir()]
    assert len(roots) == 1, [path.name for path in roots]

    wheel_dist = tmp_path / "wheel-dist"
    _ok(
        _run(
            ["uv", "build", "--no-sources", "--wheel", "--out-dir", str(wheel_dist)], cwd=roots[0]
        ),
        "wheel build from source distribution",
    )
    wheels = sorted(wheel_dist.glob("*.whl"))
    assert len(wheels) == 1, [path.name for path in wheels]

    venv = tmp_path / "venv"
    _ok(
        _run(["uv", "venv", "--python", sys.executable, "-q", str(venv)], cwd=ROOT),
        "fresh temporary venv",
    )
    python = venv / "bin" / "python"
    _ok(
        _run(["uv", "pip", "install", "-q", "--python", str(python), str(wheels[0])], cwd=ROOT),
        "fresh wheel install",
    )
    project = tmp_path / "project"
    project.mkdir()
    empty_home = tmp_path / "empty-home"
    empty_home.mkdir()
    installed = _ok(
        _run(
            [
                str(venv / "bin" / "reward-lens"),
                "init",
                str(project),
                "--skill",
                "--agent",
                "codex",
                "--format",
                "text",
            ],
            cwd=tmp_path,
            env=_env(HOME=str(empty_home), NO_COLOR="1", TERM="dumb"),
        ),
        "installed init --skill",
    )
    written = project / ".agents" / "skills" / "reward-lens" / "SKILL.md"
    assert "reward-lens skill installed for codex" in installed.stdout
    assert written.read_bytes() == (ROOT / SKILL).read_bytes()
    assert not (empty_home / ".agents").exists()
