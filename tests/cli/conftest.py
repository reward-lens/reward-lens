"""Shared machinery for the CLI tests.

Two ways in. `invoke` runs the click tree in process, which is what most of these tests want
because they are about wording, ordering and exit codes. `cli` runs the installed console script
in a subprocess, which is what the tests about streams, import sets, wall clock and re-invocation
need, because those are properties of a real process and not of a runner.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = REPO / "fleet" / "golden"
SCRIPT = pathlib.Path(sys.executable).parent / "reward-lens"


@dataclass
class Run:
    argv: list[str]
    exit_code: int
    stdout: str
    stderr: str

    def json(self) -> dict:
        return json.loads(self.stdout)


def _env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    for name in list(env):
        if name.startswith("REWARD_LENS_"):
            del env[name]
    env.update({"NO_COLOR": "1", "REWARD_LENS_NO_PROGRESS": "1", "COLUMNS": "100", "TERM": "dumb"})
    env.update(extra or {})
    return env


@pytest.fixture
def cli(tmp_path):
    """Run the installed console script, with no TTY on any stream."""

    def run(
        *argv: str, cwd: pathlib.Path | None = None, env: dict | None = None, stdin: str = ""
    ) -> Run:
        proc = subprocess.run(
            [str(SCRIPT), *argv],
            cwd=str(cwd or tmp_path),
            env=_env(env),
            input=stdin,
            capture_output=True,
            text=True,
            timeout=180,
        )
        return Run(list(argv), proc.returncode, proc.stdout, proc.stderr)

    return run


@pytest.fixture
def child_env():
    return _env


@pytest.fixture
def invoke(monkeypatch, tmp_path):
    """Run the click tree in process through click's own runner."""
    from click.testing import CliRunner

    from reward_lens.cli.main import cli as root

    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("REWARD_LENS_NO_PROGRESS", "1")
    for name in list(os.environ):
        if name.startswith("REWARD_LENS_") and name != "REWARD_LENS_NO_PROGRESS":
            monkeypatch.delenv(name, raising=False)

    def run(*argv: str, input: str | None = None) -> Run:
        # A fresh empty cwd per invocation, the way `CliRunner.isolated_filesystem` used to give
        # one. It is deprecated in click 8 and gone in click 9, and a deprecation warning in this
        # suite's output is indistinguishable from one the CLI itself provoked.
        runner = CliRunner()
        sandbox = tempfile.mkdtemp(dir=str(tmp_path))
        here = os.getcwd()
        os.chdir(sandbox)
        try:
            result = runner.invoke(root, list(argv), input=input, catch_exceptions=False)
        finally:
            os.chdir(here)
        return Run(list(argv), result.exit_code, result.stdout, result.stderr)

    return run


@pytest.fixture
def state_home(monkeypatch, tmp_path):
    """A private XDG state root, so `runs` never touches the developer's own runs."""
    root = tmp_path / "state"
    root.mkdir()
    monkeypatch.setenv("XDG_STATE_HOME", str(root))
    return root


def golden_body(name: str) -> str:
    """The expected output of a golden transcript: the file minus its `$ <command>` line.

    `fleet/golden/run_demo.py:parse_golden` splits these files on lines starting with `$ `, so the
    first line of a single-command transcript is the invocation and the rest is the output.
    """
    path = GOLDEN / name
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    assert lines[0].startswith("$ "), f"{path} does not open with an invocation line"
    return "\n".join(lines[1:])
