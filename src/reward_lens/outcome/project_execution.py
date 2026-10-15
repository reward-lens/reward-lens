"""Load one sealed project-local supervised outcome execution declaration (C-061).

The project file names a local JSON declaration.  That declaration identifies an owner-selected
``PythonGrader``, actual protected controls, and the held-out task-to-item bindings.  It cannot
carry caller-supplied scores, counts, receipts, or callbacks: qualification and every target
resolution are delegated to :mod:`reward_lens.outcome.supervised`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from reward_lens import contracts
from reward_lens.execution import Limits, Sandbox
from reward_lens.graders.python_grader import PythonGrader
from reward_lens.outcome.qualify import OutcomeCheck
from reward_lens.outcome.supervised import (
    SupervisedCase,
    SupervisedQualification,
    SupervisedResolution,
    qualify_supervised,
    resolve_supervised,
)
from reward_lens.partitions import Partition, PartitionKind, file_digest
from reward_lens.store.project import Project

__all__ = [
    "ProjectOutcomeExecution",
    "ProjectOutcomeExecutionUnavailable",
    "load_project_outcome_execution",
]


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_INTERFACE = "reward_lens.graders.python_grader.PythonGrader.score/v1"
_CONFIG_KEYS = frozenset(
    {
        "version",
        "evaluator",
        "check",
        "candidate_owner",
        "audited_grader",
        "subject_ref",
        "controls",
        "target",
        "capability_floor",
    }
)
_CONFIG_REQUIRED_KEYS = _CONFIG_KEYS - {"capability_floor"}


def _refusal(detail: str, *, field: str = "outcome.execution") -> contracts.UsageError:
    return contracts.UsageError(
        code="RL0003",
        message=f"project outcome execution configuration is invalid: {detail}",
        remediation="supply the documented project-local supervised outcome JSON declaration",
        context={"field": field, "detail": detail},
    )


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _refusal(f"{field} must be an object", field=field)
    return dict(value)


def _exact_keys(value: Mapping[str, object], wanted: frozenset[str], field: str) -> None:
    actual = frozenset(value)
    if actual != wanted:
        raise _refusal(
            f"{field} must contain exactly {sorted(wanted)}; got {sorted(actual)}", field=field
        )


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _refusal(f"{field} must be a non-empty string", field=field)
    return value


def _digest(value: object, field: str) -> str:
    value = _text(value, field)
    if not _DIGEST.fullmatch(value):
        raise _refusal(f"{field} must be a sha256 digest", field=field)
    return value


def _project_path(root: Path, value: object, field: str, *, directory: bool) -> Path:
    declared = _text(value, field)
    relative = Path(declared)
    if relative.is_absolute() or ".." in relative.parts or "\\" in declared:
        raise _refusal(f"{field} must name a project-local relative path", field=field)
    resolved = (root / relative).resolve()
    if resolved == root or root not in resolved.parents:
        raise _refusal(f"{field} must stay beneath the project root", field=field)
    if directory:
        if not resolved.is_dir():
            raise _refusal(f"{field} names no directory", field=field)
    elif not resolved.is_file():
        raise _refusal(f"{field} names no regular file", field=field)
    return resolved


def _overlap(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left in right.parents or right in left.parents


def _case(value: object, field: str) -> SupervisedCase:
    row = _mapping(value, field)
    _exact_keys(row, frozenset({"item_id", "task", "response"}), field)
    task = _mapping(row["task"], f"{field}.task")
    response = row["response"]
    if not isinstance(response, (str, dict)):
        raise _refusal(f"{field}.response must be a string or object", field=f"{field}.response")
    return SupervisedCase(_text(row["item_id"], f"{field}.item_id"), task, response)


def _cases(value: object, field: str) -> tuple[SupervisedCase, ...]:
    if not isinstance(value, list) or not value:
        raise _refusal(f"{field} must be a non-empty array", field=field)
    return tuple(_case(case, f"{field}[{index}]") for index, case in enumerate(value))


@dataclass(frozen=True)
class ProjectOutcomeExecutionUnavailable:
    """A project declaration that cannot produce an outcome, stated as an absence."""

    state: Literal["unqualified"]
    reason: Literal["missing_outcome_declaration", "missing_outcome_execution_config"]

    @property
    def qualified(self) -> bool:
        return False


@dataclass(frozen=True)
class CapabilityFloorExecution:
    """Fixed source-declared target cases for the policy capability floor."""

    partition: Partition
    floor: Decimal
    cases: tuple[SupervisedCase, ...]


@dataclass(frozen=True)
class ProjectOutcomeExecution:
    """Actual supervisor-backed project outcome execution, qualified once at construction."""

    config_path: Path
    config_source_digest: str
    expected_subject_ref: str
    evaluator_source_digest: str
    evaluator_manifest_digest: str
    control_partition_id: str
    control_partition_digest: str
    target_partition_id: str
    target_partition_digest: str
    supervised: SupervisedQualification
    _target: Partition
    _target_items_by_task_digest: Mapping[str, str]
    capability_floor: CapabilityFloorExecution | None
    _sandbox: Sandbox
    _limits: Limits

    @property
    def qualified(self) -> bool:
        return self.supervised.protocol is not None

    @property
    def target_items_by_task_digest(self) -> Mapping[str, str]:
        """Frozen explicit declared held-out task-to-partition-item mapping."""
        return self._target_items_by_task_digest

    def resolve_for_actual(
        self,
        *,
        task: Mapping[str, Any],
        response: str | Mapping[str, Any],
        item_id: str,
    ) -> SupervisedResolution:
        """Resolve one exact declared target task through the qualified supervisor.

        A caller cannot redirect a response to a different held-out item: the exact canonical task
        payload and its configured item id must agree before the selected evaluator is invoked.
        """
        if not isinstance(task, Mapping):
            raise TypeError("task must be a mapping")
        if not isinstance(response, (str, Mapping)):
            raise TypeError("response must be a string or mapping")
        if not isinstance(item_id, str) or not item_id.strip():
            raise TypeError("item_id must be a non-empty string")
        task_payload = dict(task)
        expected = self._target_items_by_task_digest.get(contracts.digest(task_payload))
        if expected is None:
            raise ValueError("task is not a declared held-out outcome target")
        if item_id != expected:
            raise ValueError("item_id does not match the declared held-out task target")
        return resolve_supervised(
            self.supervised,
            target=self._target,
            case=SupervisedCase(item_id=item_id, task=task_payload, response=response),
            sandbox=self._sandbox,
            limits=self._limits,
        )

    def resolve_capability_floor(self) -> tuple[SupervisedResolution, ...]:
        """Execute only fixed source-declared floor candidates through the qualified supervisor."""
        declared = self.capability_floor
        if declared is None:
            raise ValueError("no declared capability floor candidates")
        if not self.qualified:
            raise ValueError("outcome controls are unqualified")
        return tuple(
            resolve_supervised(
                self.supervised,
                target=declared.partition,
                case=case,
                sandbox=self._sandbox,
                limits=self._limits,
            )
            for case in declared.cases
        )


def load_project_outcome_execution(
    project: Project,
    *,
    expected_subject_ref: str | None = None,
    sandbox: Sandbox | None = None,
    limits: Limits | None = None,
) -> ProjectOutcomeExecution | ProjectOutcomeExecutionUnavailable:
    """Load and actually qualify the optional local project outcome-execution declaration.

    ``expected_subject_ref`` comes from the consuming public record's authoritative subject
    version, never from this declaration. ``sandbox`` and ``limits`` are execution containment
    inputs, never an evaluator callback or outcome input. They are needed only when the project
    declares execution; a missing declaration returns a precise unqualified state without
    attempting any host execution.
    """
    if type(project) is not Project:
        raise TypeError("project must be the accepted Project opened from rewardlens.yaml")
    outcome = project.config.outcome
    if outcome is None:
        return ProjectOutcomeExecutionUnavailable("unqualified", "missing_outcome_declaration")
    if outcome.execution is None:
        return ProjectOutcomeExecutionUnavailable("unqualified", "missing_outcome_execution_config")
    root = project.root.resolve()
    if not root.is_dir():
        raise _refusal("project root must be an existing directory", field="project.root")
    config_path = _project_path(root, outcome.execution, "outcome.execution", directory=False)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise _refusal(f"outcome.execution is not valid JSON: {error}") from error
    config = _mapping(raw, "outcome.execution")
    if frozenset(config) not in {_CONFIG_REQUIRED_KEYS, _CONFIG_KEYS}:
        raise _refusal(
            f"outcome.execution must contain {_CONFIG_REQUIRED_KEYS} with optional capability_floor; got {sorted(config)}"
        )
    if type(config["version"]) is not int or config["version"] != 1:
        raise _refusal(
            "outcome.execution.version must be integer 1", field="outcome.execution.version"
        )
    declared_subject_ref = _digest(config["subject_ref"], "outcome.execution.subject_ref")
    if expected_subject_ref is None:
        raise _refusal(
            "expected_subject_ref is required for declared outcome execution",
            field="expected_subject_ref",
        )
    expected_subject_ref = _digest(expected_subject_ref, "expected_subject_ref")
    if declared_subject_ref != expected_subject_ref:
        raise _refusal(
            "outcome.execution.subject_ref must equal the authoritative expected_subject_ref",
            field="outcome.execution.subject_ref",
        )
    if sandbox is None or not isinstance(sandbox, Sandbox):
        raise TypeError("a concrete accepted Sandbox is required for declared outcome execution")
    if type(limits) is not Limits:
        raise TypeError("limits must be the accepted Limits value for declared outcome execution")
    if limits.network is not False:
        raise ValueError("project outcome execution requires limits.network=False")

    evaluator_row = _mapping(config["evaluator"], "outcome.execution.evaluator")
    _exact_keys(evaluator_row, frozenset({"path", "entrypoint"}), "outcome.execution.evaluator")
    evaluator = PythonGrader.from_path(
        _project_path(
            root, evaluator_row["path"], "outcome.execution.evaluator.path", directory=False
        ),
        _text(evaluator_row["entrypoint"], "outcome.execution.evaluator.entrypoint"),
    )

    check_row = _mapping(config["check"], "outcome.execution.check")
    _exact_keys(
        check_row,
        frozenset({"owner", "source_digest", "interface", "partition_id"}),
        "outcome.execution.check",
    )
    check = OutcomeCheck(
        owner=_text(check_row["owner"], "outcome.execution.check.owner"),
        source_digest=_digest(check_row["source_digest"], "outcome.execution.check.source_digest"),
        interface=_text(check_row["interface"], "outcome.execution.check.interface"),
        partition_id=_text(check_row["partition_id"], "outcome.execution.check.partition_id"),
    )
    if check.interface != _INTERFACE:
        raise _refusal(
            "outcome.execution.check.interface must name PythonGrader.score/v1",
            field="outcome.execution.check.interface",
        )

    controls_row = _mapping(config["controls"], "outcome.execution.controls")
    _exact_keys(
        controls_row,
        frozenset({"partition_path", "partition_id", "known_good", "known_wrong"}),
        "outcome.execution.controls",
    )
    control_id = _text(controls_row["partition_id"], "outcome.execution.controls.partition_id")
    if control_id != check.partition_id:
        raise _refusal(
            "outcome.execution controls partition_id must match check.partition_id",
            field="outcome.execution.controls.partition_id",
        )
    control_root = _project_path(
        root,
        controls_row["partition_path"],
        "outcome.execution.controls.partition_path",
        directory=True,
    )
    controls = Partition.from_dir(
        control_id,
        PartitionKind.PROTECTED_SUITE,
        control_root,
        log_path=root / ".reward-lens" / "outcome-execution-controls.jsonl",
    )
    known_good = _cases(controls_row["known_good"], "outcome.execution.controls.known_good")
    known_wrong = _cases(controls_row["known_wrong"], "outcome.execution.controls.known_wrong")

    target_row = _mapping(config["target"], "outcome.execution.target")
    _exact_keys(
        target_row,
        frozenset({"partition_path", "partition_id", "tasks"}),
        "outcome.execution.target",
    )
    target_root = _project_path(
        root,
        target_row["partition_path"],
        "outcome.execution.target.partition_path",
        directory=True,
    )
    if _overlap(control_root, target_root):
        raise _refusal(
            "outcome execution controls and target partitions must be disjoint",
            field="outcome.execution.target.partition_path",
        )
    target = Partition.from_dir(
        _text(target_row["partition_id"], "outcome.execution.target.partition_id"),
        PartitionKind.HELD_OUT_TASKS,
        target_root,
        log_path=root / ".reward-lens" / "outcome-execution-targets.jsonl",
    )
    if not isinstance(target_row["tasks"], list) or not target_row["tasks"]:
        raise _refusal(
            "outcome.execution.target.tasks must be a non-empty array",
            field="outcome.execution.target.tasks",
        )
    task_items: dict[str, str] = {}
    target_manifest = dict(target.manifest())
    for index, declared in enumerate(target_row["tasks"]):
        field = f"outcome.execution.target.tasks[{index}]"
        row = _mapping(declared, field)
        _exact_keys(row, frozenset({"item_id", "task"}), field)
        item_id = _text(row["item_id"], f"{field}.item_id")
        task = _mapping(row["task"], f"{field}.task")
        task_digest = contracts.digest(task)
        if item_id not in target_manifest:
            raise _refusal(
                f"{field}.item_id is not an item in the held-out target partition",
                field=f"{field}.item_id",
            )
        if task_digest in task_items:
            raise _refusal(
                f"{field}.task duplicates an earlier declared target task", field=f"{field}.task"
            )
        if item_id in task_items.values():
            raise _refusal(
                f"{field}.item_id is already bound to another target task", field=f"{field}.item_id"
            )
        task_items[task_digest] = item_id

    capability_floor = None
    floor_row = config.get("capability_floor")
    declaration = outcome.held_out_capability
    if floor_row is not None:
        if declaration is None or declaration.floor is None:
            raise _refusal(
                "outcome.execution.capability_floor requires outcome.held_out_capability.floor",
                field="outcome.execution.capability_floor",
            )
        floor_config = _mapping(floor_row, "outcome.execution.capability_floor")
        _exact_keys(
            floor_config,
            frozenset({"partition_path", "partition_id", "candidates"}),
            "outcome.execution.capability_floor",
        )
        floor_root = _project_path(
            root,
            floor_config["partition_path"],
            "outcome.execution.capability_floor.partition_path",
            directory=True,
        )
        floor_id = _text(
            floor_config["partition_id"], "outcome.execution.capability_floor.partition_id"
        )
        declared_root = _project_path(
            root, declaration.path, "outcome.held_out_capability.path", directory=True
        )
        if floor_id != declaration.partition_id or floor_root != declared_root:
            raise _refusal(
                "capability floor partition must exactly match outcome.held_out_capability id and path",
                field="outcome.execution.capability_floor",
            )
        if target.id != floor_id or target.root.resolve() != floor_root:
            raise _refusal(
                "outcome.execution.target must exactly reuse the declared capability-floor partition",
                field="outcome.execution.target",
            )
        candidates = _cases(
            floor_config["candidates"], "outcome.execution.capability_floor.candidates"
        )
        candidate_tasks: dict[str, SupervisedCase] = {}
        for index, candidate in enumerate(candidates):
            task_digest = contracts.digest(dict(candidate.task))
            expected = task_items.get(task_digest)
            if expected is None or expected != candidate.item_id:
                raise _refusal(
                    "capability floor candidate must name one exact declared target task and item",
                    field=f"outcome.execution.capability_floor.candidates[{index}]",
                )
            if task_digest in candidate_tasks:
                raise _refusal(
                    "capability floor candidates duplicate a declared target task",
                    field=f"outcome.execution.capability_floor.candidates[{index}]",
                )
            candidate_tasks[task_digest] = candidate
        if set(candidate_tasks) != set(task_items):
            raise _refusal(
                "capability floor candidates must cover every declared target task exactly once",
                field="outcome.execution.capability_floor.candidates",
            )
        capability_floor = CapabilityFloorExecution(
            partition=target,
            floor=Decimal(str(declaration.floor)),
            cases=tuple(candidate_tasks[key] for key in sorted(candidate_tasks)),
        )

    audited_row = _mapping(config["audited_grader"], "outcome.execution.audited_grader")
    _exact_keys(
        audited_row, frozenset({"owner", "source_digest"}), "outcome.execution.audited_grader"
    )
    supervised = qualify_supervised(
        evaluator=evaluator,
        check=check,
        owner=check.owner,
        candidate_owner=_text(config["candidate_owner"], "outcome.execution.candidate_owner"),
        controls=controls,
        known_good=known_good,
        known_wrong=known_wrong,
        audited_grader_owner=_text(audited_row["owner"], "outcome.execution.audited_grader.owner"),
        audited_grader_source_digest=_digest(
            audited_row["source_digest"], "outcome.execution.audited_grader.source_digest"
        ),
        subject_ref=expected_subject_ref,
        sandbox=sandbox,
        limits=limits,
    )
    if supervised.qualification.entry.subject_ref != expected_subject_ref:
        raise AssertionError(
            "supervised qualification subject_ref diverged from expected_subject_ref"
        )
    return ProjectOutcomeExecution(
        config_path=config_path,
        config_source_digest=file_digest(config_path),
        expected_subject_ref=expected_subject_ref,
        evaluator_source_digest=supervised.evaluator_source_digest,
        evaluator_manifest_digest=supervised.evaluator_manifest_digest,
        control_partition_id=controls.id,
        control_partition_digest=controls.digest,
        target_partition_id=target.id,
        target_partition_digest=target.digest,
        supervised=supervised,
        _target=target,
        _target_items_by_task_digest=MappingProxyType(dict(task_items)),
        capability_floor=capability_floor,
        _sandbox=sandbox,
        _limits=limits,
    )
