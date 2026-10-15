"""Validity entries only claim outcomes actually exercised."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from reward_lens import contracts
from reward_lens.instruments.base import RunContext
from reward_lens.outcome.resolution import OutcomeState
from reward_lens.product.audit import instruments, validity_checks
from reward_lens.verifier import Rollout, VerifierUnderTest


@dataclass(frozen=True)
class Observed:
    verdict: str
    score: float | None
    request_digest: str
    observations: dict


class ControlledGrader:
    """A process-like scorer whose history reveals cross-episode contamination."""

    def __init__(self, *, contaminate: bool = False):
        self.calls = []
        self.contaminate = contaminate

    def score(self, task, response, *, sandbox, limits, project_dir):
        self.calls.append(response)
        if limits.wall_s < 0.01:
            return Observed("timeout", None, "probe", {"sandbox": {"breach": "wall"}})
        if limits.memory_bytes < 64 << 20:
            return Observed("resource_exhausted", None, "probe", {"sandbox": {"breach": "memory"}})
        value = 1.0 if response == "A" else 2.0
        if self.contaminate and response == "A" and len(self.calls) > 2:
            value = 3.0
        return Observed("scored", value, str(response), {"sandbox": {"breach": None}})


def _run(
    monkeypatch, tmp_path: Path, grader: ControlledGrader, *, tasks: bool = False, execution=None
):
    path = tmp_path / "grader.py"
    path.write_text("def score(task, response):\n    return 1.0\n")
    monkeypatch.setattr(validity_checks.PythonGrader, "from_path", lambda *args: grader)
    task = {"id": "one", "task_id": "one", "prompt": "do it"}
    corpus = instruments.AuditCorpus(
        rollouts=(
            Rollout(id="a", inputs={"task": task, "response": "A"}),
            Rollout(id="b", inputs={"task": task, "response": "B"}),
        ),
        origin="two supplied responses",
        tasks=(task,),
        n_tasks=1,
    )
    task_path = tmp_path / "tasks.jsonl"
    if tasks:
        task_path.write_text('{"id":"one","task_id":"one","prompt":"do it"}\n')
    return validity_checks.run(
        ctx=RunContext(sandbox=object()),
        subject_ref=contracts.digest({"grader": "fixture"}),
        vut=VerifierUnderTest(path, "score"),
        corpus=corpus,
        project=None,
        sandbox=object(),
        outcome_execution=execution,
        has_task_set=tasks,
        has_response_bank=True,
        grader_failure=None,
        tasks_path=task_path if tasks else None,
    )


def test_dynamic_checks_observe_fresh_serial_concurrent_and_breaches(monkeypatch, tmp_path):
    entries = _run(monkeypatch, tmp_path, ControlledGrader())
    assert [entry.entry_id for entry in entries] == [
        "validity.fresh_state_sterility",
        "validity.episode_isolation",
        "validity.timeout_behaviour",
        "validity.resource_exhaustion",
        "validity.task_validity",
        "validity.pass_rate_floor",
        "validity.flakiness",
    ]
    assert all(entry.kind == "check" and entry.check.passed for entry in entries[:4])
    assert all(entry.kind == "absence" for entry in entries[4:])
    assert entries[2].result["observation"]["score"] is None
    assert entries[3].result["observation"]["score"] is None


def test_cross_episode_contamination_is_failed_check(monkeypatch, tmp_path):
    entries = _run(monkeypatch, tmp_path, ControlledGrader(contaminate=True))
    assert entries[0].check.passed
    assert entries[1].kind == "check"
    assert not entries[1].check.passed


def test_task_checks_do_not_infer_known_good_or_solution_from_grader_score(monkeypatch, tmp_path):
    entries = _run(monkeypatch, tmp_path, ControlledGrader(), tasks=True)
    assert all(entry.kind == "absence" for entry in entries[4:])
    assert "known-good" in entries[4].absence.missing_access
    assert "outcome" in entries[5].absence.missing_access


class QualifiedOutcome:
    qualified = True

    def __init__(self, *, pass_responses):
        self.pass_responses = set(pass_responses)
        self.task = {"id": "one", "task_id": "one", "prompt": "do it"}
        self.target_items_by_task_digest = {contracts.digest(self.task): "one"}
        self.capability_floor = SimpleNamespace(
            cases=(SimpleNamespace(task=self.task, response="A"),)
        )

    def resolve_capability_floor(self):
        return (SimpleNamespace(outcome=SimpleNamespace(state=OutcomeState.PASS)),)

    def resolve_for_actual(self, *, task, response, item_id):
        assert item_id == "one" and task == self.task
        state = OutcomeState.PASS if response in self.pass_responses else OutcomeState.FAIL
        return SimpleNamespace(outcome=SimpleNamespace(state=state))


def test_qualified_task_outcomes_and_flakiness_are_executed(monkeypatch, tmp_path):
    entries = _run(
        monkeypatch,
        tmp_path,
        ControlledGrader(),
        tasks=True,
        execution=QualifiedOutcome(pass_responses={"A"}),
    )
    assert [entry.check.passed for entry in entries[4:]] == [True, True, True]
    assert entries[4].result["accepted_task_count"] == 1
    assert entries[5].result["passed_by_task"] == {"one": 1}
    assert entries[6].result["scores"] == [1.0, 1.0, 1.0]


def test_zero_outcome_passes_fail_pass_rate_floor(monkeypatch, tmp_path):
    entries = _run(
        monkeypatch,
        tmp_path,
        ControlledGrader(),
        tasks=True,
        execution=QualifiedOutcome(pass_responses=set()),
    )
    assert entries[5].kind == "check"
    assert not entries[5].check.passed
    assert entries[5].result["zero_pass_tasks"] == ["one"]


def test_real_qualified_outcome_accepts_good_flags_unsolved_and_refuses_unknown(tmp_path):
    """The actual supervisor, not a stub, controls every task-set verdict."""
    import json

    from reward_lens.execution import Limits
    from reward_lens.execution.probe import default_sandbox
    from reward_lens.graders.python_grader import PythonGrader
    from reward_lens.outcome.project_execution import load_project_outcome_execution
    from reward_lens.store import Project

    root = tmp_path / "project"
    controls = root / "outcome" / "controls"
    targets = root / "outcome" / "targets"
    controls.mkdir(parents=True)
    targets.mkdir()
    for path in (controls / "good.case", controls / "wrong.case", targets / "target.case"):
        path.write_text(path.name, encoding="utf-8")
    grader_source = "def score(task, response):\n    return float(len(response))\n"
    grader_path = root / "grader.py"
    grader_path.write_text(grader_source, encoding="utf-8")
    evaluator_path = root / "outcome" / "evaluator.py"
    evaluator_path.write_text(
        "def score(task, response):\n"
        "    return None if response == 'unknown' else "
        "(1.0 if response == task['reference'] else 0.0)\n",
        encoding="utf-8",
    )
    raw_task = {"task_id": "target", "prompt": "return the reference", "reference": "correct"}
    task = instruments.normalised_task(raw_task, "target")
    tasks_path = root / "tasks.jsonl"
    tasks_path.write_text(json.dumps(raw_task) + "\n", encoding="utf-8")
    responses_path = root / "responses.jsonl"
    subject_ref = contracts.digest({"fixture": "actual-qualified-control"})
    declaration = {
        "version": 1,
        "evaluator": {"path": "outcome/evaluator.py", "entrypoint": "score"},
        "check": {
            "owner": "independent-owner",
            "source_digest": PythonGrader.from_path(evaluator_path).describe().source_digest,
            "interface": "reward_lens.graders.python_grader.PythonGrader.score/v1",
            "partition_id": "controls-v1",
        },
        "candidate_owner": "candidate-owner",
        "audited_grader": {
            "owner": "candidate-owner",
            "source_digest": contracts.digest({"source": grader_source}),
        },
        "subject_ref": subject_ref,
        "controls": {
            "partition_path": "outcome/controls",
            "partition_id": "controls-v1",
            "known_good": [{"item_id": "good.case", "task": task, "response": "correct"}],
            "known_wrong": [{"item_id": "wrong.case", "task": task, "response": "wrong"}],
        },
        "target": {
            "partition_path": "outcome/targets",
            "partition_id": "target-v1",
            "tasks": [{"item_id": "target.case", "task": task}],
        },
        "capability_floor": {
            "partition_path": "outcome/targets",
            "partition_id": "target-v1",
            "candidates": [{"item_id": "target.case", "task": task, "response": "correct"}],
        },
    }
    (root / "outcome" / "execution.json").write_text(json.dumps(declaration), encoding="utf-8")
    (root / "rewardlens.yaml").write_text(
        "reward:\n  entry: grader.py\n  kind: plain\n"
        "tasks:\n  path: tasks.jsonl\n  prompt: prompt\n"
        "responses:\n  path: responses.jsonl\n"
        "outcome:\n  kind: protected_test_suite\n  path: outcome/controls\n"
        "  execution: outcome/execution.json\n"
        "  held_out_capability:\n    partition_id: target-v1\n"
        "    path: outcome/targets\n    floor: 1.0\n",
        encoding="utf-8",
    )
    project = Project.open(root)
    sandbox = default_sandbox()
    execution = load_project_outcome_execution(
        project,
        expected_subject_ref=subject_ref,
        sandbox=sandbox,
        limits=Limits(network=False),
    )
    assert execution.qualified

    def measure(response):
        responses_path.write_text(json.dumps({"task_id": "target", "text": response}) + "\n")
        corpus = instruments.build_corpus(tasks_path=tasks_path, responses_path=responses_path)
        return validity_checks.run(
            ctx=RunContext(sandbox=sandbox),
            subject_ref=subject_ref,
            vut=VerifierUnderTest(grader_path, "score"),
            corpus=corpus,
            project=project,
            sandbox=sandbox,
            outcome_execution=execution,
            has_task_set=True,
            has_response_bank=True,
            grader_failure=None,
            tasks_path=tasks_path,
        )

    good = measure("correct")
    assert all(entry.kind == "check" and entry.check.passed for entry in good[4:])
    assert good[5].result["passed_by_task"] == {"target": 1}
    assert good[6].result["scores"] == [7.0, 7.0, 7.0]
    assert {"digest:task_distribution", "digest:outcome_protocol"} <= set(good[4].depends_on)
    assert {"digest:samples", "digest:outcome_protocol"} <= set(good[5].depends_on)

    wrong = measure("wrong")
    assert wrong[5].kind == "check" and not wrong[5].check.passed
    assert wrong[5].result["zero_pass_tasks"] == ["target"]

    unknown = measure("unknown")
    assert unknown[5].kind == "absence"
    assert "unqualified outcome" in unknown[5].absence.missing_access
