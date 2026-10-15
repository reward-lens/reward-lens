"""Public project capability-floor assessment through Audit's supported SDK and CLI routes."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from reward_lens import api, contracts
from reward_lens.acceptance import decide, load_policy
from reward_lens.cli.main import cli
from reward_lens.graders.python_grader import PythonGrader
from reward_lens.product.audit import registry
from reward_lens.product.audit.instruments import build_corpus
from reward_lens.product.audit.run import _planned_project_method_ids
from reward_lens.store import Project

_POLICY = (
    Path(__file__).parents[2] / "src/reward_lens/acceptance/policies/default-selection-v1.json"
)
_INTERFACE = "reward_lens.graders.python_grader.PythonGrader.score/v1"


def _task() -> dict[str, str]:
    return {"id": "heldout-1", "prompt": "return the reference", "reference": "correct"}


def _entry(record):
    return next(
        item
        for item in record.measurement.soundness
        if item.entry_id == "soundness.capability_floor"
    )


def _project(root: Path, *, candidate_response: str | None) -> Path:
    """Write one source-bound YAML project without accepting any caller outcome."""
    controls = root / "outcome" / "controls"
    targets = root / "outcome" / "targets"
    controls.mkdir(parents=True)
    targets.mkdir()
    (controls / "good.case").write_text("good\n", encoding="utf-8")
    (controls / "wrong.case").write_text("wrong\n", encoding="utf-8")
    (targets / "heldout-1.case").write_text("heldout\n", encoding="utf-8")
    reward_source = (
        "def score(task, response):\n    return 1.0 if response == 'paid-but-wrong' else 0.0\n"
    )
    (root / "grader.py").write_text(reward_source, encoding="utf-8")
    (root / "tasks.jsonl").write_text(json.dumps(_task()) + "\n", encoding="utf-8")
    evaluator = root / "outcome" / "owner_evaluator.py"
    evaluator.write_text(
        "def score(task, response):\n    return 1.0 if response == task['reference'] else 0.0\n",
        encoding="utf-8",
    )
    task = _task()
    execution: dict[str, object] = {
        "version": 1,
        "evaluator": {"path": "outcome/owner_evaluator.py", "entrypoint": "score"},
        "check": {
            "owner": "assay-owner",
            "source_digest": PythonGrader.from_path(evaluator).describe().source_digest,
            "interface": _INTERFACE,
            "partition_id": "controls-v1",
        },
        "candidate_owner": "candidate",
        "audited_grader": {
            "owner": "candidate",
            "source_digest": contracts.digest({"source": reward_source}),
        },
        # This valid placeholder is replaced below by Audit’s exact planned project identity.
        "subject_ref": "sha256:" + "0" * 64,
        "controls": {
            "partition_path": "outcome/controls",
            "partition_id": "controls-v1",
            "known_good": [{"item_id": "good.case", "task": task, "response": "correct"}],
            "known_wrong": [{"item_id": "wrong.case", "task": task, "response": "wrong"}],
        },
        "target": {
            "partition_path": "outcome/targets",
            "partition_id": "heldout-v1",
            "tasks": [{"item_id": "heldout-1.case", "task": task}],
        },
    }
    if candidate_response is not None:
        execution["capability_floor"] = {
            "partition_path": "outcome/targets",
            "partition_id": "heldout-v1",
            "candidates": [
                {"item_id": "heldout-1.case", "task": task, "response": candidate_response}
            ],
        }
    execution_path = root / "outcome" / "execution.json"
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    (root / "rewardlens.yaml").write_text(
        """reward:
  kind: plain
  entry: grader.py
tasks:
  path: tasks.jsonl
  prompt: prompt
outcome:
  kind: protected_test_suite
  path: outcome/controls
  execution: outcome/execution.json
  held_out_capability:
    partition_id: heldout-v1
    path: outcome/targets
    floor: 1.0
""",
        encoding="utf-8",
    )
    project = Project.open(root)
    execution["subject_ref"] = project.version(
        methods=_planned_project_method_ids(
            registry.discover(),
            project=project,
            corpus=build_corpus(tasks_path=root / "tasks.jsonl", responses_path=None),
        )
    ).digest()
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    return root


def test_sdk_capability_floor_retains_true_false_and_unknown_policy_states(tmp_path: Path) -> None:
    cases = (
        ("held", "correct", "qualified"),
        ("not-held", "wrong", "rejected"),
        ("unknown", None, "missing"),
    )
    for name, response, policy_state in cases:
        project = _project(tmp_path / name, candidate_response=response)
        record = api.audit(api.AuditRequest(path=project, name=name, policy=_POLICY))
        saved = project / "assays" / f"{name}.assay.json"
        assert saved.is_file()
        reopened = api.open_record(saved)
        entry = _entry(reopened)
        assert reopened.assay_id == record.assay_id
        evaluated = decide(reopened, load_policy(_POLICY))
        assert evaluated.evaluated["capability_floor.held"].state == policy_state
        assert reopened.decision.state == (
            "rejected" if policy_state == "rejected" else "unresolved"
        )
        if response is None:
            assert entry.kind == "absence"
            assert entry.absence.state == "NOT_MEASURED"
            assert entry.check is None
        else:
            assert entry.check is not None and entry.check.passed is (response == "correct")
            assert entry.result["held"] is (response == "correct")


def test_cli_capability_floor_persists_and_reopens_the_real_held_out_check(tmp_path: Path) -> None:
    project = _project(tmp_path / "cli", candidate_response="correct")

    result = CliRunner().invoke(
        cli,
        [
            "audit",
            str(project),
            "--name",
            "floor-cli",
            "--policy",
            str(_POLICY),
            "--format",
            "json",
            "--exit-zero",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    delivered = json.loads(result.stdout)
    saved = api.open_record(project / delivered["artifacts"]["assay"])
    entry = _entry(saved)
    assert entry.check is not None and entry.check.passed is True
    assert entry.result["held"] is True
    assert (
        decide(saved, load_policy(_POLICY)).evaluated["capability_floor.held"].state == "qualified"
    )
    assert saved.decision.state == "unresolved"
