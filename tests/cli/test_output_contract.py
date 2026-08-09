"""D-21 streams, D-23 formats and their env twins, D-31's pending action, and `--fields`."""

from __future__ import annotations

import json
import pathlib

import pytest

from reward_lens.cli.verbs._shared import RECORD_SUFFIX


@pytest.fixture
def grader(tmp_path) -> pathlib.Path:
    path = tmp_path / "grader.py"
    path.write_text("def score(task, response):\n    return 1.0\n", encoding="utf-8")
    return path


def test_json_writes_only_the_envelope_to_stdout(cli, grader):
    run = cli("audit", str(grader), "--format", "json")
    payload = json.loads(run.stdout)
    assert payload["schema_version"] == "assay-result/1.0"
    assert payload["command"] == "audit"
    assert set(payload) >= {"execution", "subject", "decision", "findings", "holes", "artifacts", "error"}


def test_json_is_the_default_without_a_tty(cli, grader):
    assert json.loads(cli("audit", str(grader)).stdout)["schema_version"] == "assay-result/1.0"


def test_json_flag_is_an_alias_for_format_json(cli, grader):
    assert json.loads(cli("audit", str(grader), "--json").stdout)["command"] == "audit"


def test_format_env_twin_is_honoured(cli, grader):
    run = cli("audit", str(grader), env={"REWARD_LENS_FORMAT": "text"})
    assert not run.stdout.lstrip().startswith("{")
    assert "validity" in run.stdout


def test_the_flag_beats_the_env_twin(cli, grader):
    run = cli("audit", str(grader), "--format", "json", env={"REWARD_LENS_FORMAT": "text"})
    assert json.loads(run.stdout)["command"] == "audit"


def test_jsonl_streams_one_versioned_event_per_line(cli, grader):
    run = cli("audit", str(grader), "--format", "jsonl")
    lines = [line for line in run.stdout.splitlines() if line.strip()]
    assert len(lines) >= 2
    events = [json.loads(line) for line in lines]
    assert all(event["schema_version"] == "assay-event/1.0" for event in events)
    assert events[0]["event"] == "started"
    assert events[-1]["event"] == "result"


def test_progress_and_notes_never_reach_stdout(cli, grader):
    run = cli("audit", str(grader), "--format", "json", "--progress", "plain")
    json.loads(run.stdout)  # would raise if a note had been mixed into the data stream


def test_the_error_twin_goes_to_stderr_and_stdout_stays_parseable(cli, grader):
    run = cli("audit", str(grader), "--format", "json", "--fields", "no.such.field")
    assert run.exit_code == 4
    twin = json.loads(run.stderr.splitlines()[-1])
    assert twin["code"] and twin["message"] and twin["remediation"]


def test_fields_bounds_the_envelope(cli, grader):
    run = cli("audit", str(grader), "--format", "json", "--fields", "command,decision.state")
    payload = json.loads(run.stdout)
    assert payload == {"command": "audit", "decision": {"state": "unresolved"}}


def test_the_pending_action_carries_argv_and_command(cli, state_home):
    from reward_lens.store import RunDir

    run_id = RunDir.create(project=None, name="doomed").run_id
    refused = cli("runs", "rm", run_id, "--format", "json")
    assert refused.exit_code == 3
    pending = json.loads(refused.stdout)["pending"]
    assert pending["argv"][0] == "reward-lens"
    assert "--yes" in pending["argv"]
    assert isinstance(pending["command"], str) and "--yes" in pending["command"]
    assert pending["decision"]


def test_the_pending_argv_runs_to_completion(cli, state_home):
    from reward_lens.store import RunDir

    run_id = RunDir.create(project=None, name="doomed").run_id
    refused = cli("runs", "rm", run_id, "--format", "json")
    argv = json.loads(refused.stdout)["pending"]["argv"]
    assert argv[0] == "reward-lens"
    again = cli(*argv[1:], "--format", "json")
    assert again.exit_code == 0, again.stderr
    assert cli("runs", "show", run_id, "--format", "json").exit_code == 4


def test_an_unknown_format_names_the_value_it_refused(cli, grader):
    run = cli("audit", str(grader), "--format", "xml")
    assert run.exit_code == 4
    assert run.stderr.startswith("error: ")
    assert "help:" in run.stderr
    assert "reward-lens explain RL0004" in run.stderr


def test_no_colour_when_stdout_is_not_a_tty(cli, grader):
    run = cli("audit", str(grader), "--format", "text", "--color", "always")
    assert "\x1b[" in run.stdout
    plain = cli("audit", str(grader), "--format", "text")
    assert "\x1b[" not in plain.stdout


def test_the_envelope_names_the_files_the_audit_left_behind(cli, tmp_path):
    """Section 5.7's `artifacts`, which is the only way a caller finds the record it just made.

    The api hands back the record and not the paths it wrote, so the verb asks the store which
    name holds a record with this id. Both files have to be there: a path in the envelope that
    names nothing on disk is worse than no path at all. A-019: they are named from the project
    directory, which is what the golden envelope carries and what survives the envelope being kept.
    """
    assert cli("init", "--example", "code-reward", ".", cwd=tmp_path).exit_code == 0

    run = cli("audit", "--format", "json", cwd=tmp_path)

    assert run.exit_code == 2, (run.stdout, run.stderr)
    artifacts = json.loads(run.stdout)["artifacts"]
    assay = pathlib.Path(artifacts["assay"])
    report = pathlib.Path(artifacts["report"])
    assert not assay.is_absolute(), artifacts
    assert not report.is_absolute(), artifacts
    assert (tmp_path / assay).is_file(), artifacts
    assert (tmp_path / report).is_file(), artifacts
    assert assay.name.endswith(".assay.json"), artifacts
    assert report.name.endswith(".assay.html"), artifacts
    assert assay.parent == pathlib.Path("assays")
    assert report.parent == pathlib.Path("assays")


def test_the_envelope_names_the_project_and_the_text_names_the_working_directory(cli, tmp_path):
    """A-019, the two forms of the same file, side by side.

    The envelope is kept, moved and read from somewhere else, so it names the path from the project
    it belongs to. The `Report` line is typed back into this shell, so it names the path from here.
    The audit below is run from the parent of the project, which is the case where the two strings
    genuinely differ: get this wrong and one of the two names nothing.
    """
    assert cli("init", "--example", "code-reward", "demo", cwd=tmp_path).exit_code == 0

    envelope = cli("audit", "demo", "--format", "json", cwd=tmp_path)
    text = cli("audit", "demo", "--format", "text", cwd=tmp_path)

    assert envelope.exit_code == 2, (envelope.stdout, envelope.stderr)
    artifacts = json.loads(envelope.stdout)["artifacts"]
    name = pathlib.Path(artifacts["report"]).name
    assert artifacts["report"] == f"assays/{name}", artifacts
    assert artifacts["assay"] == "assays/" + name[: -len(".assay.html")] + ".assay.json", artifacts
    assert (tmp_path / "demo" / artifacts["report"]).is_file(), artifacts
    report_line = [line for line in text.stdout.split("\n") if line.startswith("  Report ")]
    assert report_line, text.stdout
    assert f"./demo/assays/{name}" in report_line[0], report_line


def test_the_envelope_names_the_record_this_run_wrote_and_not_merely_the_newest(cli, tmp_path):
    """The match is on the record's own id, so a directory holding other records still works."""
    assert cli("init", "--example", "code-reward", ".", cwd=tmp_path).exit_code == 0
    decoy = tmp_path / "assays" / ("zz-decoy" + RECORD_SUFFIX)
    decoy.parent.mkdir(parents=True, exist_ok=True)
    decoy.write_text(json.dumps({"assay_id": "sha256:" + "f" * 64}), encoding="utf-8")

    run = cli("audit", "--format", "json", cwd=tmp_path)

    assert run.exit_code == 2, (run.stdout, run.stderr)
    named = tmp_path / json.loads(run.stdout)["artifacts"]["assay"]
    assert named != decoy, "the newest name is not the record; the id is"
    assert json.loads(named.read_text(encoding="utf-8"))["assay_id"].startswith("sha256:")
