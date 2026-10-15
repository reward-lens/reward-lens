"""The CI policy is an exact consumer of owner-declared input-handling intent."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from reward_lens import api
from reward_lens.acceptance import decide, load_policy
from reward_lens.cli.main import cli

POLICY = Path(__file__).parents[2] / "src/reward_lens/acceptance/policies/default-ci-v1.json"


def _project(root: Path, *, planted: bool) -> Path:
    root.mkdir()
    malformed = "return None" if planted else "return 0.0"
    (root / "grader.py").write_text(
        "def score(task, response):\n"
        "    if response == '' or response.startswith('```') or len(response) > 100000:\n"
        f"        {malformed}\n"
        "    return 1.0\n",
        encoding="utf-8",
    )
    (root / "tasks.jsonl").write_text('{"id":"task-1","prompt":"answer"}\n', encoding="utf-8")
    (root / "rewardlens.yaml").write_text(
        "reward:\n  kind: plain\n  entry: grader.py\n"
        "tasks:\n  path: tasks.jsonl\n  prompt: prompt\n"
        "outcome: null\n",
        encoding="utf-8",
    )
    return root


def _input_check(record):
    return next(
        entry
        for entry in record.measurement.validity
        if entry.entry_id == "validity.input_handling"
    )


def test_ci_policy_rejects_the_actual_planted_owner_intent_violation_and_reopens_it(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path / "planted", planted=True)

    record = api.audit(api.AuditRequest(path=project, name="ci-planted", policy=POLICY))

    saved = project / "assays" / "ci-planted.assay.json"
    reopened = api.open_record(saved)
    check = _input_check(reopened)
    assert saved.is_file()
    assert check.check is not None and check.check.passed is False
    assert any(
        finding.code == "RL0210" and finding.entries == ["validity.input_handling"]
        for finding in reopened.findings
    )
    assert record.decision.state == reopened.decision.state == "rejected"
    evaluated = decide(reopened, load_policy(POLICY))
    assert evaluated.evaluated["validity.owner_declared_input_handling"].state == "rejected"


def test_ci_policy_cli_uses_persisted_decision_and_repaired_intent_is_not_blocked_by_unrelated_absences(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path / "repaired", planted=False)

    result = CliRunner().invoke(
        cli,
        [
            "audit",
            str(project),
            "--name",
            "ci-repaired",
            "--policy",
            str(POLICY),
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    delivered = json.loads(result.stdout)
    reopened = api.open_record(project / delivered["artifacts"]["assay"])
    check = _input_check(reopened)
    assert check.check is not None and check.check.passed is True
    assert any(entry.kind == "absence" for entry in reopened.measurement.signal)
    assert reopened.decision.state == "qualified"
    assert decide(reopened, load_policy(POLICY)).state == "qualified"


def test_ci_policy_cli_blocks_on_the_saved_planted_decision(tmp_path: Path) -> None:
    project = _project(tmp_path / "planted-cli", planted=True)

    result = CliRunner().invoke(
        cli,
        [
            "audit",
            str(project),
            "--name",
            "ci-planted-cli",
            "--policy",
            str(POLICY),
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1, result.output
    delivered = json.loads(result.stdout)
    reopened = api.open_record(project / delivered["artifacts"]["assay"])
    assert reopened.decision.state == "rejected"
    assert _input_check(reopened).check is not None and _input_check(reopened).check.passed is False
