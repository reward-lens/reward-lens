"""What a write root grants, at every tier that has a filesystem ruleset at all.

P-AUDIT-1 measured that no tier on this machine let a staged grader run: at L1 and L2 the
interpreter could not open the supervisor script (`[Errno 13]`), at L3 the path was not in the
mount namespace (`[Errno 2]`), and making the path readable only moved the failure on to the
capture file. The cause was one hole in this seam, and these tests are the fence around it: a
write root is read as well as written, the working directory is always one, and the caller can
name others for the machinery it stages outside the directory the code under test runs in.

Every test here runs real processes. Nothing reaches the network.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

from reward_lens.execution import Limits, probe, tier_ladder
from reward_lens.execution.landlock import ACCESS_FS_READ, ACCESS_FS_WRITE
from reward_lens.execution.linux import LinuxSandbox

#: The tiers with a filesystem ruleset, in ladder order, that hold on this machine. T0 and L0 are
#: not here: they confine no path, and they refuse `network=False` before they would run anything
#: (RL0402), so there is no root for them to grant. `test_the_ladder_reaches_every_confining_tier`
#: is what makes a missing tier loud instead of quietly shrinking this list.
_HELD = tuple(t for t in tier_ladder("linux") if t in ("L1", "L2", "L3") and probe().tiers[t].held)

_LIMITS = Limits(wall_s=25.0, cpu_s=25.0)

#: Read a file staged in the working directory, create one, append to it, truncate it, and
#: overwrite it through a truncating `open(..., "w")`. Five distinct Landlock access rights.
WORK_SOURCE = """
import os
with open("payload.txt", encoding="utf-8") as fh:
    print("read", fh.read().strip())
with open("made.txt", "w", encoding="utf-8") as fh:
    fh.write("one")
with open("made.txt", "a", encoding="utf-8") as fh:
    fh.write("-two")
with open("made.txt", encoding="utf-8") as fh:
    print("appended", fh.read())
os.truncate("made.txt", 3)
with open("made.txt", encoding="utf-8") as fh:
    print("truncated", fh.read())
with open("made.txt", "w", encoding="utf-8") as fh:
    fh.write("final")
print("ok")
"""

#: The adapter's shape: the supervisor and the capture file sit in a scratch directory whose
#: child is the working directory, so neither is under `cwd`.
SUPERVISOR_SOURCE = """
import os, sys
with open(os.path.join(os.getcwd(), "grader.py"), encoding="utf-8") as fh:
    print("grader", fh.read().strip())
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    fh.write("captured")
print("ok")
"""

#: Read a path the roots do not name and report the exception by name and errno.
OUTSIDE_SOURCE = """
import errno, sys
try:
    with open(sys.argv[1], encoding="utf-8") as fh:
        fh.read()
except OSError as exc:
    print(type(exc).__name__, errno.errorcode[exc.errno])
else:
    print("opened", "NONE")
"""


def _stage(work: Path, name: str, source: str) -> Path:
    script = work / name
    script.write_text(source, encoding="utf-8")
    return script


def _argv(script: Path, *args: str) -> list[str]:
    return [sys.executable, "-B", "-s", str(script), *args]


# --- the working directory ------------------------------------------------------------------


def test_the_ladder_reaches_every_confining_tier() -> None:
    """The parametrisation below is only as wide as this. A tier that stopped holding is a fact."""
    assert _HELD == ("L1", "L2", "L3"), {
        t: probe().tiers[t].reason for t in ("L1", "L2", "L3") if not probe().tiers[t].held
    }


@pytest.mark.parametrize("tier", _HELD)
def test_the_working_directory_is_read_and_write_at_every_tier(tier: str, tmp_path: Path) -> None:
    """Read a staged file, create one, append, truncate, overwrite. Asserted from outside too."""
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    (work / "payload.txt").write_text("hello", encoding="utf-8")
    script = _stage(work, "runner.py", WORK_SOURCE)

    result = LinuxSandbox(tier=tier).run(
        _argv(script), limits=_LIMITS, cwd=work, env={}
    )

    assert result.exit_code == 0, result.stderr.decode()
    assert result.tier == tier
    lines = result.stdout.decode().split("\n")
    assert lines[0] == "read hello"
    assert lines[1] == "appended one-two"
    assert lines[2] == "truncated one"
    assert lines[3] == "ok"
    assert (work / "made.txt").read_text(encoding="utf-8") == "final"


@pytest.mark.parametrize("tier", _HELD)
def test_the_working_directorys_parent_is_not_readable(tier: str, tmp_path: Path) -> None:
    """The grant is the directory, not the directory above it, and not `/tmp`."""
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    outside = tmp_path / "outside.txt"
    outside.write_text("not for the grader", encoding="utf-8")
    script = _stage(work, "runner.py", OUTSIDE_SOURCE)

    result = LinuxSandbox(tier=tier).run(
        _argv(script, str(outside)), limits=_LIMITS, cwd=work, env={}
    )

    assert result.exit_code == 0, result.stderr.decode()
    error, code = result.stdout.decode().split()
    if tier == "L3":
        # Bubblewrap binds the write roots and the stated read roots and nothing else, so the
        # parent's contents are not in the run's mount namespace at all.
        assert (error, code) == ("FileNotFoundError", "ENOENT")
    else:
        assert (error, code) == ("PermissionError", "EACCES")


# --- a write root the caller names ------------------------------------------------------------


def _stage_adapter_shape(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """The geometry `graders/python_grader.py` builds: scratch holds the machinery, its `project`
    child is the working directory, and the copy is stripped of write before the run."""
    scratch = tmp_path / "rl-grader"
    scratch.mkdir(mode=0o700)
    supervisor = _stage(scratch, "_rl_supervisor.py", SUPERVISOR_SOURCE)
    capture = scratch / "grader_stdout.txt"
    project = scratch / "project"
    project.mkdir()
    (project / "grader.py").write_text("GRADER", encoding="utf-8")
    for path in (project / "grader.py", project):
        path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    return scratch, project, supervisor, capture


@pytest.mark.parametrize("tier", _HELD)
def test_machinery_outside_the_working_directory_is_unreachable_unnamed(
    tier: str, tmp_path: Path
) -> None:
    """P-AUDIT-1's tier table, reproduced: the mechanism differs, the absence does not."""
    _scratch, project, supervisor, capture = _stage_adapter_shape(tmp_path)

    result = LinuxSandbox(tier=tier).run(
        _argv(supervisor, str(capture)), limits=_LIMITS, cwd=project, env={}
    )

    assert result.exit_code != 0
    stderr = result.stderr.decode()
    assert "_rl_supervisor.py" in stderr
    assert ("Errno 2" if tier == "L3" else "Errno 13") in stderr
    assert not capture.exists()


@pytest.mark.parametrize("tier", _HELD)
def test_a_named_write_root_is_read_and_written_at_every_tier(tier: str, tmp_path: Path) -> None:
    """Naming the scratch root closes both walls at once: the script opens and the capture writes."""
    scratch, project, supervisor, capture = _stage_adapter_shape(tmp_path)

    result = LinuxSandbox(tier=tier).run(
        _argv(supervisor, str(capture)),
        limits=_LIMITS,
        cwd=project,
        env={},
        write_roots=[scratch],
    )

    assert result.exit_code == 0, result.stderr.decode()
    assert result.stdout.decode().split("\n")[:2] == ["grader GRADER", "ok"]
    assert capture.read_text(encoding="utf-8") == "captured"
    assert str(scratch) in result.detail["write_roots"]
    assert str(project) in result.detail["write_roots"]


# --- the rule itself -------------------------------------------------------------------------


def test_a_landlock_write_root_carries_every_read_right() -> None:
    """The implication `write_roots_for` documents, asserted on the bits rather than on a run."""
    assert ACCESS_FS_WRITE & ACCESS_FS_READ == ACCESS_FS_READ


def test_the_working_directory_is_always_a_write_root(tmp_path: Path) -> None:
    """`write_roots` adds to the working directory; it never replaces it, and it de-duplicates."""
    sandbox = LinuxSandbox(tier="L1")
    work = tmp_path / "work"
    other = tmp_path / "other"
    assert sandbox.write_roots_for(work) == (str(work),)
    assert sandbox.write_roots_for(work, [work]) == (str(work),)
    assert sandbox.write_roots_for(work, [other]) == tuple(sorted((str(work), str(other))))
