"""Run directories (D-27): found by a caller that did not start them, resumable, forked when done."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

RUN_ID = re.compile(r"^run-[0-9a-f]{12}$")


@pytest.fixture()
def runs_root(state_home: Path) -> Path:
    return state_home / "reward-lens" / "runs"


def test_create_writes_the_four_things_d27_names(project_root, runs_root):
    from reward_lens.store import RunDir

    run = RunDir.create(project=project_root, name="nightly")
    assert RUN_ID.match(run.run_id)
    assert run.path == runs_root / run.run_id
    assert (run.path / "manifest.json").is_file()
    assert (run.path / "plan.json").is_file()
    assert (run.path / "partial").is_dir()
    assert (run.path / "log.txt").is_file()
    manifest = json.loads((run.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == run.run_id
    assert manifest["name"] == "nightly"
    assert manifest["project"] == str(project_root)
    assert manifest["state"] == "running"


def test_xdg_state_home_is_read_at_call_time(tmp_path: Path, monkeypatch):
    from reward_lens.store import RunDir

    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setenv("XDG_STATE_HOME", str(elsewhere))
    run = RunDir.create(project=None, name=None)
    assert run.path.parent == elsewhere / "reward-lens" / "runs"


def test_the_default_root_is_local_state(monkeypatch, tmp_path: Path):
    from reward_lens.store.runs import runs_root

    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert runs_root() == tmp_path / ".local" / "state" / "reward-lens" / "runs"


def test_a_run_is_found_by_id_and_by_name(project_root, state_home):
    from reward_lens.store import RunDir

    run = RunDir.create(project=project_root, name="nightly")
    assert RunDir.find(run.run_id).run_id == run.run_id
    assert RunDir.find("nightly").run_id == run.run_id


def test_a_run_that_is_not_there_is_refused(state_home):
    from reward_lens.store.errors import RunNotFound

    from reward_lens.store import RunDir

    with pytest.raises(RunNotFound) as excinfo:
        RunDir.find("run-000000000000")
    assert excinfo.value.code == "RL0621"
    assert excinfo.value.exit_code == 4
    assert "run-000000000000" in excinfo.value.message
    assert excinfo.value.remediation


def test_list_is_newest_first_and_most_recent_is_per_project(tmp_path: Path, state_home):
    from reward_lens.store import RunDir

    a, b = tmp_path / "a", tmp_path / "b"
    first = RunDir.create(project=a, name="one")
    second = RunDir.create(project=a, name="two")
    other = RunDir.create(project=b, name="three")

    listed = [run.run_id for run in RunDir.list()]
    assert listed[0] == other.run_id
    assert set(listed) == {first.run_id, second.run_id, other.run_id}
    assert RunDir.most_recent(a).run_id == second.run_id
    assert RunDir.most_recent(b).run_id == other.run_id
    assert RunDir.most_recent(tmp_path / "c") is None


def test_the_run_index_is_a_cache_the_directories_can_rebuild(tmp_path: Path, runs_root):
    from reward_lens.store import RunDir

    RunDir.create(project=tmp_path / "a", name="one")
    second = RunDir.create(project=tmp_path / "a", name="two")
    before = [run.run_id for run in RunDir.list()]

    index = runs_root / "index.json"
    assert index.is_file()
    index.unlink()
    assert [run.run_id for run in RunDir.list()] == before
    assert RunDir.find("two").run_id == second.run_id
    assert index.is_file()


def test_the_resume_command_is_the_one_a_caller_can_paste(project_root, state_home):
    from reward_lens.store import RunDir

    run = RunDir.create(project=project_root, name="nightly")
    command = run.resume_command()
    assert isinstance(command, list)
    assert all(isinstance(part, str) for part in command)
    assert command[0] == "reward-lens"
    assert "--resume" in command
    assert command[command.index("--resume") + 1] == "nightly"
    assert str(project_root) in command


def test_a_run_without_a_name_resumes_by_id(state_home):
    from reward_lens.store import RunDir

    run = RunDir.create(project=None, name=None)
    command = run.resume_command()
    assert command[command.index("--resume") + 1] == run.run_id


def test_partial_results_are_written_per_instrument(project_root, state_home):
    from reward_lens.store import RunDir

    run = RunDir.create(project=project_root, name="nightly")
    path = run.write_partial("validity", {"entries": ["validity.grader_source"]})
    assert path == run.path / "partial" / "validity.json"
    assert run.partial() == {"validity": {"entries": ["validity.grader_source"]}}


def test_resuming_an_unfinished_run_continues_it(project_root, state_home):
    from reward_lens.store import RunDir

    run = RunDir.create(project=project_root, name="nightly")
    resumed = run.resume()
    assert resumed.run_id == run.run_id
    assert resumed.manifest["forked_from"] is None


def test_resuming_a_finished_run_forks_it_and_leaves_the_record_alone(project_root, state_home):
    from reward_lens.store import RunDir

    run = RunDir.create(project=project_root, name="nightly")
    run.write_partial("validity", {"entries": ["validity.grader_source"]})
    run.finish()
    before = (run.path / "manifest.json").read_bytes()

    fork = run.resume()
    assert fork.run_id != run.run_id
    assert RUN_ID.match(fork.run_id)
    assert fork.manifest["forked_from"] == run.run_id
    assert fork.manifest["state"] == "running"
    assert fork.partial() == run.partial()  # the partial result is carried, never discarded
    assert (run.path / "manifest.json").read_bytes() == before  # the old record is not mutated
    assert RunDir.find(run.run_id).manifest["state"] == "finished"


def test_a_finished_run_says_so(project_root, state_home):
    from reward_lens.store import RunDir

    run = RunDir.create(project=project_root, name="nightly")
    assert run.finished is False
    run.finish()
    assert run.finished is True
    assert RunDir.find(run.run_id).finished is True


def test_the_log_is_appended_to_not_rewritten(project_root, state_home):
    from reward_lens.store import RunDir

    run = RunDir.create(project=project_root, name="nightly")
    run.log("started")
    run.log("stopped")
    text = (run.path / "log.txt").read_text(encoding="utf-8")
    assert text.splitlines()[-2:] == ["started", "stopped"]
