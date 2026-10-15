"""C-061 project-local producer for actual supervised outcome execution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import reward_lens.outcome.project_execution as project_execution
from reward_lens.contracts import ProjectConfig, UsageError
from reward_lens.execution import Limits, default_sandbox
from reward_lens.outcome import OutcomeState
from reward_lens.outcome.project_execution import (
    ProjectOutcomeExecution,
    ProjectOutcomeExecutionUnavailable,
    load_project_outcome_execution,
)
from reward_lens.store import Project

AUDITED_SOURCE = "sha256:" + "a" * 64
SUBJECT = "sha256:" + "b" * 64
INTERFACE = "reward_lens.graders.python_grader.PythonGrader.score/v1"


@pytest.fixture
def sandbox():
    return default_sandbox()


@pytest.fixture
def limits() -> Limits:
    return Limits(wall_s=15.0, cpu_s=15.0, network=False)


def _task(name: str) -> dict[str, str]:
    return {"id": name, "prompt": "return the reference", "reference": "correct"}


def _write_project(tmp_path: Path, *, execution: str | None = "outcome/execution.json") -> Project:
    (tmp_path / "outcome" / "controls").mkdir(parents=True)
    (tmp_path / "outcome" / "targets").mkdir()
    for item in ("good.case", "wrong.case"):
        (tmp_path / "outcome" / "controls" / item).write_text(f"{item}\n", encoding="utf-8")
    for item in ("pass.case", "fail.case"):
        (tmp_path / "outcome" / "targets" / item).write_text(f"{item}\n", encoding="utf-8")
    evaluator = tmp_path / "outcome" / "owner_evaluator.py"
    evaluator.write_text(
        "def score(task, response):\n    return 1.0 if response == task['reference'] else 0.0\n",
        encoding="utf-8",
    )
    from reward_lens.graders.python_grader import PythonGrader

    source_digest = PythonGrader.from_path(evaluator).describe().source_digest
    if execution is not None:
        declaration = {
            "version": 1,
            "evaluator": {"path": "outcome/owner_evaluator.py", "entrypoint": "score"},
            "check": {
                "owner": "assay-owner",
                "source_digest": source_digest,
                "interface": INTERFACE,
                "partition_id": "controls-v1",
            },
            "candidate_owner": "candidate",
            "audited_grader": {"owner": "candidate", "source_digest": AUDITED_SOURCE},
            "subject_ref": SUBJECT,
            "controls": {
                "partition_path": "outcome/controls",
                "partition_id": "controls-v1",
                "known_good": [
                    {"item_id": "good.case", "task": _task("good.case"), "response": "correct"}
                ],
                "known_wrong": [
                    {"item_id": "wrong.case", "task": _task("wrong.case"), "response": "wrong"}
                ],
            },
            "target": {
                "partition_path": "outcome/targets",
                "partition_id": "heldout-v1",
                "tasks": [
                    {"item_id": "pass.case", "task": _task("pass.case")},
                    {"item_id": "fail.case", "task": _task("fail.case")},
                ],
            },
        }
        path = tmp_path / execution
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(declaration), encoding="utf-8")
    config = ProjectConfig.model_validate(
        {
            "reward": {"entry": "outcome/owner_evaluator.py:score"},
            "tasks": {"path": "tasks.jsonl", "prompt": "prompt"},
            "outcome": None
            if execution is None
            else {
                "kind": "protected_test_suite",
                "path": "outcome/controls",
                "execution": execution,
            },
        }
    )
    return Project(tmp_path, config)


def _declaration(project: Project) -> tuple[Path, dict[str, object]]:
    assert project.config.outcome is not None
    path = project.root / project.config.outcome.execution
    return path, json.loads(path.read_text(encoding="utf-8"))


def _bind_declared_subject(project: Project, subject_ref: str | None = None) -> str:
    """Bind a test declaration to the public caller's independently computed project version."""
    expected = project.version().digest() if subject_ref is None else subject_ref
    path, declaration = _declaration(project)
    declaration["subject_ref"] = expected
    path.write_text(json.dumps(declaration), encoding="utf-8")
    return expected


def test_project_loader_runs_real_known_good_and_wrong_controls_once_then_resolves_declared_targets(
    tmp_path: Path, sandbox, limits
) -> None:
    project = _write_project(tmp_path)
    expected_subject_ref = _bind_declared_subject(project)

    execution = load_project_outcome_execution(
        project, expected_subject_ref=expected_subject_ref, sandbox=sandbox, limits=limits
    )

    assert isinstance(execution, ProjectOutcomeExecution)
    assert execution.qualified is True
    assert execution.config_source_digest.startswith("sha256:")
    assert execution.expected_subject_ref == expected_subject_ref
    assert execution.supervised.qualification.entry.subject_ref == expected_subject_ref
    assert execution.control_partition_id == "controls-v1"
    assert execution.target_partition_id == "heldout-v1"
    assert len(execution.supervised.control_receipts) == 2
    assert all(receipt.verdict == "scored" for receipt in execution.supervised.control_receipts)
    assert (
        len(
            (tmp_path / ".reward-lens" / "outcome-execution-controls.jsonl")
            .read_text()
            .splitlines()
        )
        == 2
    )

    passed = execution.resolve_for_actual(
        task=_task("pass.case"), response="correct", item_id="pass.case"
    )
    failed = execution.resolve_for_actual(
        task=_task("fail.case"), response="wrong", item_id="fail.case"
    )

    assert passed.outcome.state is OutcomeState.PASS
    assert failed.outcome.state is OutcomeState.FAIL
    assert passed.receipt.verdict == "scored"
    assert passed.target_item_digest
    assert passed.task_digest != passed.response_digest
    assert (
        len(
            (tmp_path / ".reward-lens" / "outcome-execution-targets.jsonl").read_text().splitlines()
        )
        == 2
    )


def test_missing_declaration_is_an_explicit_unqualified_state_and_never_needs_host_execution(
    tmp_path: Path,
) -> None:
    project = _write_project(tmp_path, execution=None)

    unavailable = load_project_outcome_execution(project)

    assert unavailable == ProjectOutcomeExecutionUnavailable(
        state="unqualified", reason="missing_outcome_declaration"
    )
    assert unavailable.qualified is False


def test_missing_execution_field_is_an_explicit_unqualified_state(tmp_path: Path) -> None:
    project = _write_project(tmp_path)
    expected_subject_ref = _bind_declared_subject(project)
    assert project.config.outcome is not None
    config = project.config.model_copy(
        update={"outcome": project.config.outcome.model_copy(update={"execution": None})}
    )

    unavailable = load_project_outcome_execution(
        Project(tmp_path, config), expected_subject_ref=expected_subject_ref
    )

    assert unavailable == ProjectOutcomeExecutionUnavailable(
        state="unqualified", reason="missing_outcome_execution_config"
    )


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda declaration: declaration.__setitem__("extra", "no"), "outcome.execution"),
        (
            lambda declaration: declaration["evaluator"].__setitem__("path", "../owner.py"),
            "outcome.execution.evaluator.path",
        ),
        (
            lambda declaration: declaration["target"].__setitem__(
                "partition_path", "outcome/controls"
            ),
            "outcome.execution.target.partition_path",
        ),
    ],
)
def test_loader_refuses_extra_keys_escaped_paths_and_nondisjoint_partitions(
    tmp_path: Path, sandbox, limits, mutate, field: str
) -> None:
    project = _write_project(tmp_path)
    expected_subject_ref = _bind_declared_subject(project)
    path, declaration = _declaration(project)
    mutate(declaration)
    path.write_text(json.dumps(declaration), encoding="utf-8")

    with pytest.raises(UsageError) as excinfo:
        load_project_outcome_execution(
            project, expected_subject_ref=expected_subject_ref, sandbox=sandbox, limits=limits
        )

    assert excinfo.value.code == "RL0003"
    assert excinfo.value.context["field"] == field


def test_target_task_item_mapping_is_exact_and_receipts_cannot_be_replaced_by_caller_values(
    tmp_path: Path, sandbox, limits
) -> None:
    project = _write_project(tmp_path)
    execution = load_project_outcome_execution(
        project,
        expected_subject_ref=_bind_declared_subject(project),
        sandbox=sandbox,
        limits=limits,
    )
    assert isinstance(execution, ProjectOutcomeExecution)

    with pytest.raises(ValueError, match="does not match"):
        execution.resolve_for_actual(
            task=_task("pass.case"), response="correct", item_id="fail.case"
        )
    with pytest.raises(ValueError, match="not a declared"):
        execution.resolve_for_actual(
            task=_task("invented"), response="correct", item_id="pass.case"
        )
    with pytest.raises(TypeError):
        execution.supervised.control_receipts[0].payload["verdict"] = "scored"  # type: ignore[index]


@pytest.mark.parametrize(
    ("declared_subject", "expected_subject", "field"),
    [
        ("bind", None, "expected_subject_ref"),
        ("bind", "not-a-digest", "expected_subject_ref"),
        (SUBJECT, "project", "outcome.execution.subject_ref"),
    ],
)
def test_declared_execution_requires_authoritative_subject_preflight_before_any_execution(
    tmp_path: Path,
    sandbox,
    limits,
    monkeypatch,
    declared_subject: str,
    expected_subject: str | None,
    field: str,
) -> None:
    project = _write_project(tmp_path)
    project_subject = project.version().digest()
    if declared_subject == "bind":
        _bind_declared_subject(project, project_subject)
    else:
        _bind_declared_subject(project, declared_subject)
    argument = project_subject if expected_subject == "project" else expected_subject

    def unexpected_execution(*args, **kwargs):
        raise AssertionError("subject preflight reached an execution constructor")

    monkeypatch.setattr(
        project_execution.PythonGrader, "from_path", staticmethod(unexpected_execution)
    )
    monkeypatch.setattr(
        project_execution.Partition,
        "from_dir",
        classmethod(lambda cls, *args, **kwargs: unexpected_execution()),
    )

    with pytest.raises(UsageError) as excinfo:
        load_project_outcome_execution(
            project,
            expected_subject_ref=argument,
            sandbox=sandbox,
            limits=limits,
        )

    assert excinfo.value.code == "RL0003"
    assert excinfo.value.context["field"] == field
    assert not (tmp_path / ".reward-lens").exists()
