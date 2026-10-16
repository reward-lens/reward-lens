"""A terminal interrupt reaps active response graders before printing a resume command."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from reward_lens.examples.code_reward import write as write_example

SCRIPT = Path(sys.executable).parent / "reward-lens"


def _proc(pid: int) -> tuple[int, int, int, str] | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        tail = stat.rsplit(") ", 1)[1].split()
        return int(tail[1]), int(tail[19]), int(tail[2]), stat.split("(", 1)[1].rsplit(") ", 1)[0]
    except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
        return None


def _descendants(root: int) -> dict[int, tuple[int, int, int, str]]:
    table = {
        int(name): found
        for name in os.listdir("/proc")
        if name.isdigit() and (found := _proc(int(name))) is not None
    }
    selected = {root}
    while True:
        extended = selected | {pid for pid, row in table.items() if row[0] in selected}
        if extended == selected:
            break
        selected = extended
    return {pid: table[pid] for pid in selected if pid in table}


def _same_process(pid: int, original: tuple[int, int, int, str]) -> bool:
    current = _proc(pid)
    return current is not None and current[1] == original[1]


def _remaining(seen: dict[int, tuple[int, int, int, str]]) -> list[int]:
    """Allow the spawn resource tracker to observe CLI pipe closure before checking identities."""
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        live = [pid for pid, row in seen.items() if _same_process(pid, row)]
        if not live:
            return []
        time.sleep(0.02)
    return [pid for pid, row in seen.items() if _same_process(pid, row)]


def _wait_for_active_score(
    process: subprocess.Popen[str],
    state: Path,
    profile: Path,
    seen: dict[int, tuple[int, int, int, str]],
) -> Path:
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        partials = list((state / "reward-lens" / "runs").glob("run-*/partial/validity.static.json"))
        events = profile.read_text(encoding="utf-8") if profile.exists() else ""
        tree = _descendants(process.pid)
        # Preserve identities even when the gate fails after a worker has
        # left the CLI session; the caller's finally block must reap it.
        seen.update(tree)
        python_children = sum(
            row[3].startswith("python") for pid, row in tree.items() if pid != process.pid
        )
        bwraps = sum(row[3] == "bwrap" for row in tree.values())
        if (
            partials
            and "response_score_bank_start " in events
            and python_children >= 12
            and bwraps >= 2
        ):
            return partials[0].parents[1]
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(
                f"audit exited before scoring began: {process.returncode}\n{stdout}\n{stderr}"
            )
        time.sleep(0.02)
    pytest.fail("sealed static work and active contained response graders were not observed")


@pytest.mark.parametrize("signal_mode", ["pid", "group"])
def test_sigint_during_active_response_scoring_reaps_graders_and_resumes(
    tmp_path: Path, signal_mode: str
) -> None:
    """Both direct SIGINT and foreground Ctrl+C stop real, bounded grader subprocesses."""
    project = tmp_path / "example"
    write_example(project)
    # The original 120-row resume test remains intact. Twelve rows are enough
    # to fill every scorer slot here while keeping this separate control short.
    responses = project / "responses.jsonl"
    responses.write_text(
        "".join(responses.read_text(encoding="utf-8").splitlines(keepends=True)[:12]),
        encoding="utf-8",
    )
    grader = project / "grader.py"
    grader.write_text(
        grader.read_text(encoding="utf-8")
        + "\nfrom pathlib import Path\nimport time\n"
        + "_pause = Path(__file__).with_name('.pause-audit')\n"
        + "_score = score\n"
        + "def score(task, response):\n"
        + "    if _pause.exists():\n"
        + "        time.sleep(60)\n"
        + "    return _score(task, response)\n",
        encoding="utf-8",
    )
    pause = project / ".pause-audit"
    pause.write_text("block only the first invocation", encoding="utf-8")
    state = tmp_path / "state"
    profile = tmp_path / "score-profile.log"
    state.mkdir()
    env = {
        key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}
    }
    env.update(
        {
            "XDG_STATE_HOME": str(state),
            "REWARD_LENS_MCP_PROFILE_PATH": str(profile),
            "REWARD_LENS_NO_PROGRESS": "1",
            "NO_COLOR": "1",
            "TERM": "dumb",
        }
    )
    process = subprocess.Popen(
        [str(SCRIPT), "audit", str(project), "--name", "active-score-sigint", "--format", "text"],
        cwd=project,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    seen: dict[int, tuple[int, int, int, str]] = {}
    try:
        run_dir = _wait_for_active_score(process, state, profile, seen)
        seen.update(_descendants(process.pid))
        if signal_mode == "group":
            os.killpg(process.pid, signal.SIGINT)
        else:
            process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 130, f"stdout:\n{stdout}\nstderr:\n{stderr}"
        assert "RL0130" in stderr
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        resume_line = (
            f"reward-lens audit {manifest['project']} --resume "
            f"{manifest['name'] or manifest['run_id']}"
        )
        assert stderr.rstrip().splitlines()[-1] == resume_line
        assert (run_dir / "partial" / "validity.static.json").is_file()
        assert not _remaining(seen)

        pause.unlink()
        resumed = subprocess.run(
            [str(SCRIPT), *resume_line.split()[1:], "--format", "text"],
            cwd=project,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert resumed.returncode in (0, 1, 2), resumed.stderr
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["state"] == "finished"
        assert Path(manifest["record_path"]).is_file()
        log = (run_dir / "log.txt").read_text(encoding="utf-8")
        assert "instrument validity.static replayed" in log
        assert (
            "instrument validity.static ran"
            not in log.split("instrument validity.static replayed", 1)[1]
        )
    finally:
        if process.poll() is None:
            # A gate failure can occur after a worker has left the CLI's
            # session. Capture its identity before killing the CLI group.
            seen.update(_descendants(process.pid))
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)
        for pid, row in seen.items():
            if _same_process(pid, row):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
