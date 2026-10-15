"""Finite-bank, reward-selected independent-outcome measurement for compare rung 2."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from itertools import combinations
from math import comb
from pathlib import Path

from reward_lens import contracts
from reward_lens.bank import Response, ResponseBank, TaskId, TieBreak, best_of_n
from reward_lens.bank.select import require_bank
from reward_lens.execution import Limits, Sandbox
from reward_lens.execution.linux import LinuxSandbox
from reward_lens.graders.python_grader import PythonGrader
from reward_lens.outcome import Outcome, OutcomeCheck, OutcomeState, QualifiedProtocol
from reward_lens.outcome.supervised import (
    SupervisedCase,
    SupervisedResolution,
    qualify_supervised,
    resolve_supervised,
)
from reward_lens.partitions import Partition, PartitionKind

from .registry import CompareResult, Rung, register
from .scorecard import ROWS, Absence, Estimate, Scorecard, ScorecardRow

__all__ = [
    "Calibration",
    "OutcomeProvider",
    "SelectionPoint",
    "SelectionStressExecution",
    "SelectionStressReading",
    "Susceptibility",
    "canonical_assay_entry",
    "expected_max_order_statistic",
    "load_selection_stress_execution",
    "pass_at_k",
    "selection_stress",
]
_SCOPE = "selection_stress"
_Q = Decimal("0.000001")
_RHO = Decimal("0.964286")
_POWER = Decimal("0.13")
_KILL = "K-meta fired"
_EVIDENCE = "ev:c978e3958769ecf93dcf079771a3aedb"
_CITE = "campaign-results/RESULTS.md:99"
_ABSENCE_EXTENSION = "https://reward-lens.github.io/assay/selection-stress-absence/v1"


def _round(value: Decimal) -> Decimal:
    return value.quantize(_Q, rounding=ROUND_HALF_UP)


def _refusal(code: str, message: str, context: dict[str, object]) -> contracts.UsageError:
    return contracts.UsageError(
        code=code,
        message=message,
        remediation="retain the required selection-stress conditions and use a finite bank budget",
        context=context,
    )


@dataclass(frozen=True)
class Calibration:
    rho: Decimal
    power: Decimal
    kill_criterion: str
    evidence_id: str
    cite: str

    def __post_init__(self) -> None:
        if (self.rho, self.power, self.kill_criterion, self.evidence_id, self.cite) != (
            _RHO,
            _POWER,
            _KILL,
            _EVIDENCE,
            _CITE,
        ):
            raise _refusal(
                "RL0133",
                "susceptibility calibration conditions are missing or altered",
                {"field": "calibration", "required_evidence": _EVIDENCE},
            )


@dataclass(frozen=True)
class Susceptibility:
    value: Decimal
    calibration: Calibration | None

    def __post_init__(self) -> None:
        if not isinstance(self.value, Decimal):
            raise TypeError("susceptibility value must be a Decimal")
        if not isinstance(self.calibration, Calibration):
            raise _refusal(
                "RL0133",
                "susceptibility reading requires its calibration conditions",
                {"field": "calibration", "reading": "susceptibility"},
            )


@dataclass(frozen=True)
class OutcomeProvider:
    """Actual declared check, verified protocol, and response-level outcome resolver."""

    check: OutcomeCheck
    protocol: QualifiedProtocol | None
    resolve: Callable[[TaskId, Response], Outcome]
    receipt_identities: dict[str, dict[str, object]] | None = None
    subject_ref: str | None = None

    def __post_init__(self) -> None:
        if type(self.check) is not OutcomeCheck:
            raise TypeError("outcome provider check must be the accepted OutcomeCheck")
        if self.protocol is not None and type(self.protocol) is not QualifiedProtocol:
            raise TypeError("outcome provider protocol must be a QualifiedProtocol or None")
        if not callable(self.resolve):
            raise TypeError("outcome provider resolve must be callable")
        if self.receipt_identities is not None and not isinstance(self.receipt_identities, dict):
            raise TypeError("outcome provider receipts must be a producer-owned dictionary")
        if self.subject_ref is not None and (
            not isinstance(self.subject_ref, str) or not _DIGEST.fullmatch(self.subject_ref)
        ):
            raise TypeError(
                "outcome provider subject_ref must be a canonical sha256 digest or None"
            )
        if self.subject_ref is not None and self.protocol is not None:
            qualification = self.protocol._qualification
            if qualification is None or qualification.entry.subject_ref != self.subject_ref:
                raise ValueError(
                    "outcome provider subject_ref must match the qualified producer receipt"
                )

    @property
    def qualified(self) -> bool:
        if self.protocol is None:
            return False
        method_digest = contracts.digest(
            {
                "interface": self.check.interface,
                "owner": self.check.owner,
                "partition_id": self.check.partition_id,
                "source_digest": self.check.source_digest,
                "suite_digest": self.protocol.qualified_on_partition_digest,
            }
        )
        return method_digest == self.protocol.protocol_digest


@dataclass(frozen=True)
class SelectionStressExecution:
    """Executable, sealed selection-stress inputs built from one local declaration.

    The JSON declaration supplies paths and identities only.  It never supplies an
    outcome label or a ``qualified`` flag: controls are executed locally and item
    resolution has to return a real Outcome built by the outcome package.
    """

    selector: object
    outcome_provider: OutcomeProvider | None
    sandbox: Sandbox
    limits: Limits
    ladder: tuple[int, ...]
    bootstrap_resamples: int
    validated_subject: str | None
    canonical_declaration: dict[str, object]

    @property
    def canonical_declaration_digest(self) -> str:
        return contracts.digest(self.canonical_declaration)


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CONFIG_KEYS = frozenset(
    {"version", "selector", "outcome", "ladder", "bootstrap_resamples", "sandbox_tier", "limits"}
)


def _config_refusal(detail: str, *, field: str = "selection_config") -> contracts.UsageError:
    return contracts.UsageError(
        code="RL0003",
        message=f"selection-stress configuration is invalid: {detail}",
        remediation="supply the documented project-local selection-stress JSON declaration",
        context={"field": field, "detail": detail},
    )


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _config_refusal(f"{field} must be an object", field=field)
    return dict(value)


def _exact_keys(value: dict[str, object], wanted: set[str], field: str) -> None:
    actual = set(value)
    if actual != wanted:
        raise _config_refusal(
            f"{field} must contain exactly {sorted(wanted)}; got {sorted(actual)}", field=field
        )


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _config_refusal(f"{field} must be a non-empty string", field=field)
    return value


def _digest_text(value: object, field: str) -> str:
    result = _text(value, field)
    if not _DIGEST.fullmatch(result):
        raise _config_refusal(f"{field} must be a sha256 digest", field=field)
    return result


def _project_file(root: Path, value: object, field: str) -> Path:
    declared = _text(value, field)
    relative = Path(declared)
    if relative.is_absolute() or ".." in relative.parts:
        raise _config_refusal(f"{field} must name a project-local relative path", field=field)
    resolved = (root / relative).resolve()
    if root != resolved and root not in resolved.parents:
        raise _config_refusal(f"{field} must stay beneath the project root", field=field)
    if not resolved.is_file():
        raise _config_refusal(f"{field} names no regular file", field=field)
    return resolved


def _project_directory(root: Path, value: object, field: str) -> Path:
    declared = _text(value, field)
    relative = Path(declared)
    if relative.is_absolute() or ".." in relative.parts:
        raise _config_refusal(f"{field} must name a project-local relative path", field=field)
    resolved = (root / relative).resolve()
    if root != resolved and root not in resolved.parents:
        raise _config_refusal(f"{field} must stay beneath the project root", field=field)
    if not resolved.is_dir():
        raise _config_refusal(f"{field} names no directory", field=field)
    return resolved


def _overlap(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left in right.parents or right in left.parents


def _limits(value: object) -> Limits:
    row = _mapping(value, "limits")
    wanted = {"wall_s", "cpu_s", "memory_bytes", "processes", "stdout_bytes", "artifact_bytes"}
    _exact_keys(row, wanted, "limits")
    float_keys = {"wall_s", "cpu_s"}
    converted: dict[str, object] = {}
    for key in wanted:
        item = row[key]
        if type(item) is bool or not isinstance(item, (int, float)) or item <= 0:
            raise _config_refusal(f"limits.{key} must be a positive number", field=f"limits.{key}")
        if key in float_keys:
            converted[key] = float(item)
        elif type(item) is not int:
            raise _config_refusal(f"limits.{key} must be a positive integer", field=f"limits.{key}")
        else:
            converted[key] = item
    return Limits(**converted, network=False)  # type: ignore[arg-type]


def _selection_ladder(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise _config_refusal("ladder must be a non-empty array", field="ladder")
    ladder = tuple(value)
    if any(type(item) is not int or item < 1 for item in ladder):
        raise _config_refusal("ladder must contain positive integers", field="ladder")
    if tuple(sorted(set(ladder))) != ladder:
        raise _config_refusal(
            "ladder must be strictly increasing without duplicates", field="ladder"
        )
    return ladder


def _case(value: object, field: str) -> SupervisedCase:
    row = _mapping(value, field)
    _exact_keys(row, {"item_id", "task", "response"}, field)
    task = _mapping(row["task"], f"{field}.task")
    return SupervisedCase(_text(row["item_id"], f"{field}.item_id"), task, row["response"])


def _receipt_identity(resolution: SupervisedResolution, *, subject_ref: str) -> dict[str, object]:
    return {
        "receipt_digest": resolution.receipt.receipt_digest,
        "run_id": resolution.receipt.run_id,
        "request_digest": resolution.receipt.request_digest,
        "target_partition_id": resolution.target_partition_id,
        "target_partition_digest": resolution.target_partition_digest,
        "target_item_digest": resolution.target_item_digest,
        "task_digest": resolution.task_digest,
        "response_digest": resolution.response_digest,
        "state": resolution.outcome.state.value,
        "subject_ref": subject_ref,
    }


def _outcome_provider(
    root: Path, value: object, *, sandbox: Sandbox, limits: Limits
) -> tuple[OutcomeProvider | None, object, str | None]:
    if value is None:
        return None, None, None
    row = _mapping(value, "outcome")
    _exact_keys(
        row,
        {
            "check",
            "evaluator",
            "controls",
            "target",
            "candidate_owner",
            "audited_grader",
            "subject_ref",
        },
        "outcome",
    )
    check_row = _mapping(row["check"], "outcome.check")
    _exact_keys(check_row, {"owner", "source_digest", "interface", "partition_id"}, "outcome.check")
    check = OutcomeCheck(
        _text(check_row["owner"], "outcome.check.owner"),
        _digest_text(check_row["source_digest"], "outcome.check.source_digest"),
        _text(check_row["interface"], "outcome.check.interface"),
        _text(check_row["partition_id"], "outcome.check.partition_id"),
    )
    evaluator_row = _mapping(row["evaluator"], "outcome.evaluator")
    _exact_keys(evaluator_row, {"path", "entrypoint"}, "outcome.evaluator")
    evaluator_path = _project_file(root, evaluator_row["path"], "outcome.evaluator.path")
    evaluator = PythonGrader.from_path(
        evaluator_path, _text(evaluator_row["entrypoint"], "outcome.evaluator.entrypoint")
    )
    controls = _mapping(row["controls"], "outcome.controls")
    _exact_keys(
        controls,
        {"partition_path", "partition_id", "known_good", "known_wrong"},
        "outcome.controls",
    )
    partition_id = _text(controls["partition_id"], "outcome.controls.partition_id")
    if partition_id != check.partition_id:
        raise _config_refusal(
            "outcome controls partition_id must match outcome check partition_id",
            field="outcome.controls.partition_id",
        )
    partition_root = _project_directory(
        root, controls["partition_path"], "outcome.controls.partition_path"
    )
    partition = Partition.from_dir(
        partition_id,
        PartitionKind.PROTECTED_SUITE,
        partition_root,
        log_path=root / ".reward-lens" / "selection-stress-controls.jsonl",
    )
    good = tuple(
        _case(case, f"outcome.controls.known_good[{index}]")
        for index, case in enumerate(
            controls["known_good"] if isinstance(controls["known_good"], list) else ()
        )
    )
    wrong = tuple(
        _case(case, f"outcome.controls.known_wrong[{index}]")
        for index, case in enumerate(
            controls["known_wrong"] if isinstance(controls["known_wrong"], list) else ()
        )
    )
    target_row = _mapping(row["target"], "outcome.target")
    _exact_keys(
        target_row, {"partition_path", "partition_id", "response_items", "tasks"}, "outcome.target"
    )
    target_root = _project_directory(
        root, target_row["partition_path"], "outcome.target.partition_path"
    )
    if _overlap(partition_root, target_root):
        raise _config_refusal(
            "outcome controls and target partitions must be disjoint",
            field="outcome.target.partition_path",
        )
    target = Partition.from_dir(
        _text(target_row["partition_id"], "outcome.target.partition_id"),
        PartitionKind.HELD_OUT_TASKS,
        target_root,
        log_path=root / ".reward-lens" / "selection-stress-targets.jsonl",
    )
    item_map = _mapping(target_row["response_items"], "outcome.target.response_items")
    if any(not isinstance(key, str) or not isinstance(item, str) for key, item in item_map.items()):
        raise _config_refusal(
            "outcome.target.response_items must map response digests to item ids",
            field="outcome.target.response_items",
        )
    target_tasks = _mapping(target_row["tasks"], "outcome.target.tasks")
    if any(
        not isinstance(key, str) or not isinstance(item, dict) for key, item in target_tasks.items()
    ):
        raise _config_refusal(
            "outcome.target.tasks must map task ids to task objects", field="outcome.target.tasks"
        )
    audited = _mapping(row["audited_grader"], "outcome.audited_grader")
    _exact_keys(audited, {"owner", "source_digest"}, "outcome.audited_grader")
    declared_subject = _digest_text(row["subject_ref"], "outcome.subject_ref")
    supervised = qualify_supervised(
        evaluator=evaluator,
        check=check,
        owner=check.owner,
        candidate_owner=_text(row["candidate_owner"], "outcome.candidate_owner"),
        controls=partition,
        known_good=good,
        known_wrong=wrong,
        audited_grader_owner=_text(audited["owner"], "outcome.audited_grader.owner"),
        audited_grader_source_digest=_digest_text(
            audited["source_digest"], "outcome.audited_grader.source_digest"
        ),
        subject_ref=declared_subject,
        sandbox=sandbox,
        limits=limits,
    )
    validated_subject = supervised.qualification.entry.subject_ref
    if validated_subject != declared_subject:
        raise ValueError("supervised qualification did not retain the declared outcome subject")
    receipts: dict[str, dict[str, object]] = {}

    def resolve(task: TaskId, response: Response) -> Outcome:
        item_id = item_map.get(response.digest)
        if not isinstance(item_id, str):
            raise ValueError(f"no declared held-out target item for response {response.digest}")
        task_payload = target_tasks.get(str(task))
        if not isinstance(task_payload, dict):
            raise ValueError(f"no declared held-out target task for {task}")
        resolution = resolve_supervised(
            supervised,
            target=target,
            case=SupervisedCase(item_id, task_payload, response.text),
            sandbox=sandbox,
            limits=limits,
        )
        receipts[response.digest] = _receipt_identity(resolution, subject_ref=validated_subject)
        return resolution.outcome

    provider = OutcomeProvider(check, supervised.protocol, resolve, receipts, validated_subject)
    control_receipts = [
        {"receipt_digest": receipt.receipt_digest, "request_digest": receipt.request_digest}
        for receipt in supervised.control_receipts
    ]
    return (
        provider,
        {
            "subject_ref": validated_subject,
            "check": check_row,
            "evaluator": {
                "path": str(evaluator_path.relative_to(root)),
                "entrypoint": evaluator.entrypoint,
                "source_digest": supervised.evaluator_source_digest,
                "manifest_digest": supervised.evaluator_manifest_digest,
            },
            "controls": {
                "partition_path": str(partition_root.relative_to(root)),
                "partition_id": partition.id,
                "partition_digest": partition.digest,
                "control_receipts": control_receipts,
            },
            "target": {
                "partition_path": str(target_root.relative_to(root)),
                "partition_id": target.id,
                "partition_digest": target.digest,
                "response_items": item_map,
                "tasks": target_tasks,
            },
            "qualification": "supervised_receipts",
            "qualification_entry": {
                "entry_id": supervised.qualification.entry.entry_id,
                "kind": supervised.qualification.entry.kind,
                "subject_ref": validated_subject,
            },
            "protocol_digest": None
            if supervised.protocol is None
            else supervised.protocol.protocol_digest,
        },
        validated_subject,
    )


def load_selection_stress_execution(
    project_root: Path | str,
    selection_config: Path | str,
    *,
    expected_subject_ref: str | None = None,
) -> SelectionStressExecution:
    """Load one sealed project-local Rung-2 declaration into real execution objects.

    A public Assay consumer supplies its already-computed ``expected_subject_ref``.
    This loader compares that identity to the declared independent-outcome subject
    before it constructs outcome execution or touches protected controls.
    """
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise _config_refusal("project root must be an existing directory", field="project_root")
    config_path = Path(selection_config).resolve()
    if (root != config_path and root not in config_path.parents) or not config_path.is_file():
        raise _config_refusal(
            "selection_config must be an existing project-local file", field="selection_config"
        )
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise _config_refusal(f"selection_config is not valid JSON: {error}") from error
    config = _mapping(raw, "selection_config")
    _exact_keys(config, set(_CONFIG_KEYS), "selection_config")
    if config["version"] != 1:
        raise _config_refusal("version must be integer 1", field="version")
    if expected_subject_ref is not None:
        expected_subject = _digest_text(expected_subject_ref, "expected_subject_ref")
        outcome = config["outcome"]
        if outcome is None:
            raise _config_refusal(
                "expected_subject_ref requires a declared independent outcome subject",
                field="outcome.subject_ref",
            )
        outcome_row = _mapping(outcome, "outcome")
        declared_subject = _digest_text(outcome_row.get("subject_ref"), "outcome.subject_ref")
        if declared_subject != expected_subject:
            raise _config_refusal(
                "outcome.subject_ref must match expected_subject_ref before outcome execution",
                field="outcome.subject_ref",
            )
    selector_row = _mapping(config["selector"], "selector")
    _exact_keys(selector_row, {"path", "entrypoint"}, "selector")
    selector_path = _project_file(root, selector_row["path"], "selector.path")
    selector_entrypoint = _text(selector_row["entrypoint"], "selector.entrypoint")
    selector = PythonGrader.from_path(selector_path, selector_entrypoint)
    tier = _text(config["sandbox_tier"], "sandbox_tier")
    if tier in {"T0", "L0"}:
        raise _config_refusal(
            "sandbox_tier must enforce network=False; T0 and L0 have no egress filter",
            field="sandbox_tier",
        )
    sandbox = LinuxSandbox(tier=tier)
    limits = _limits(config["limits"])
    ladder = _selection_ladder(config["ladder"])
    resamples = config["bootstrap_resamples"]
    if type(resamples) is not int or resamples < 1:
        raise _config_refusal(
            "bootstrap_resamples must be a positive integer", field="bootstrap_resamples"
        )
    provider, outcome_declaration, validated_subject = _outcome_provider(
        root, config["outcome"], sandbox=sandbox, limits=limits
    )
    declaration: dict[str, object] = {
        "version": 1,
        "selection_config_source_digest": "sha256:"
        + hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "selector": {
            "path": str(selector_path.relative_to(root)),
            "entrypoint": selector_entrypoint,
            "source_digest": "sha256:" + hashlib.sha256(selector_path.read_bytes()).hexdigest(),
        },
        "outcome": outcome_declaration,
        "validated_subject": validated_subject,
        "ladder": list(ladder),
        "bootstrap_resamples": resamples,
        "sandbox_tier": tier,
        "limits": {
            "wall_s": limits.wall_s,
            "cpu_s": limits.cpu_s,
            "memory_bytes": limits.memory_bytes,
            "processes": limits.processes,
            "stdout_bytes": limits.stdout_bytes,
            "artifact_bytes": limits.artifact_bytes,
            "network": False,
        },
    }
    return SelectionStressExecution(
        selector, provider, sandbox, limits, ladder, resamples, validated_subject, declaration
    )


@dataclass(frozen=True)
class SelectionPoint:
    budget: int
    selected_success: Estimate | Absence
    oracle_pass_at_k: Decimal
    tie_break: TieBreak
    unqualified_outcomes: int
    ties_observed: int


@dataclass(frozen=True)
class SelectionStressReading:
    points: tuple[SelectionPoint, ...]
    susceptibility: Susceptibility


def pass_at_k(n: int, c: int, k: int) -> Decimal:
    """Exact oracle availability from a finite without-replacement subset."""
    if any(type(x) is not int for x in (n, c, k)):
        raise TypeError("n, c, and k must be integers")
    if n < 1 or not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("pass@k requires 0 <= c <= n and 1 <= k <= n")
    return _round(Decimal(1) - Decimal(0 if k > n - c else comb(n - c, k)) / Decimal(comb(n, k)))


def expected_max_order_statistic(outcomes: Sequence[Outcome], k: int) -> Decimal:
    """Exact selected-winner success, with rows sorted low-to-high by reward rank."""
    rows = tuple(outcomes)
    n = len(rows)
    if n < 1 or type(k) is not int or not 1 <= k <= n:
        raise ValueError("order statistic requires a non-empty finite sequence and 1 <= k <= N")
    if any(type(row) is not Outcome for row in rows):
        raise TypeError("order statistic evidence requires accepted Outcome values")
    passed = resolved = Decimal(0)
    for rank, outcome in enumerate(rows, 1):
        weight = Decimal(comb(rank - 1, k - 1)) / Decimal(comb(n, k))
        state = outcome.state
        if state is OutcomeState.UNQUALIFIED:
            continue
        if state is True or state is OutcomeState.PASS:
            passed += weight
            resolved += weight
        elif state is False or state is OutcomeState.FAIL:
            resolved += weight
        else:
            raise TypeError("accepted Outcome has an unrecognised state")
    if not resolved:
        raise ValueError("no resolved selected outcomes support an order-statistic estimate")
    return _round(passed / resolved)


def _score(
    selector: object, task: dict[str, object], response: Response, sandbox: Sandbox, limits: Limits
) -> Decimal:
    result = selector.score(task, response.text, sandbox=sandbox, limits=limits)
    value = getattr(result, "score", result)
    if value is None:
        raise ValueError("the selector returned an unscored response")
    return Decimal(str(value))


def _winner(
    subset: Sequence[Response],
    selector: object,
    task: dict[str, object],
    sandbox: Sandbox,
    limits: Limits,
) -> Response:
    scores = [(row, _score(selector, task, row, sandbox, limits)) for row in subset]
    highest = max(value for _, value in scores)
    return min((row for row, value in scores if value == highest), key=lambda row: row.digest)


def _task_point(
    bank: ResponseBank,
    task: TaskId,
    selector: object,
    provider: OutcomeProvider,
    budget: int,
    sandbox: Sandbox,
    limits: Limits,
) -> tuple[Decimal | None, Decimal, int]:
    rows = bank.responses_for(task)
    passed = resolved = oracle = unknown = 0
    for subset in combinations(rows, budget):
        outcomes = tuple(provider.resolve(task, row) for row in subset)
        if any(type(item) is not Outcome for item in outcomes):
            raise TypeError("outcome provider must return accepted Outcome values")
        if any(item.item_id != row.digest for row, item in zip(subset, outcomes)):
            raise ValueError("outcome item_id must identify the selected response digest")
        if any(
            item.protocol_digest != provider.protocol.protocol_digest
            for item in outcomes
            if item.resolved
        ):
            raise ValueError(
                "resolved outcome protocol does not match the declared qualified protocol"
            )
        winner = _winner(subset, selector, dict(bank.task_for(task)), sandbox, limits)
        chosen = next(item for row, item in zip(subset, outcomes) if row is winner)
        if chosen.state is OutcomeState.UNQUALIFIED:
            unknown += 1
        else:
            resolved += 1
            passed += chosen.state is OutcomeState.PASS
        oracle += any(item.state is OutcomeState.PASS for item in outcomes)
    total = comb(len(rows), budget)
    return (
        None if not resolved else Decimal(passed) / Decimal(resolved),
        Decimal(oracle) / Decimal(total),
        unknown,
    )


def _bootstrap(values: Sequence[Decimal], resamples: int) -> tuple[Decimal, Decimal]:
    if type(resamples) is not int or resamples < 1:
        raise ValueError("bootstrap_resamples must be a positive integer")
    if not values:
        raise ValueError("task bootstrap requires at least one resolved task")
    rng = random.Random(0)
    samples = sorted(
        sum((values[rng.randrange(len(values))] for _ in values), Decimal(0)) / Decimal(len(values))
        for _ in range(resamples)
    )
    return _round(samples[int(Decimal("0.025") * (resamples - 1))]), _round(
        samples[int(Decimal("0.975") * (resamples - 1))]
    )


def _scorecard(value: Estimate | Absence) -> Scorecard:
    rows = []
    for row_id in ROWS:
        if row_id == "intended_task_success":
            row_value = value
        elif row_id == "legitimate_solutions_kept":
            row_value = Absence(
                "not measured: no held-out capability set declared",
                "declare outcome.held_out_capability separately from the qualification suite",
            )
        else:
            row_value = Absence(
                "not measured: rung 2 measures selected independent outcome only",
                "run the rung that measures this scorecard row",
            )
        rows.append(ScorecardRow(row_id, row_id.replace("_", " "), row_value, _SCOPE))
    return Scorecard.render(rows, rung=Rung.SELECTION_STRESS)


def _subject(bank: ResponseBank) -> str:
    return "sha256:" + hashlib.sha256(bank.digest.encode()).hexdigest()


def _entry_subject(bank: ResponseBank, provider: OutcomeProvider | None = None) -> str:
    return (
        _subject(bank) if provider is None or provider.subject_ref is None else provider.subject_ref
    )


def _selection_method(bank: ResponseBank, budget: int | None) -> contracts.Method:
    return contracts.Method(
        id="reward_statistics.selection_stress",
        version="1.0.0",
        params_digest=contracts.digest(
            {"bank": bank.digest, "budget": budget, "tie_break": TieBreak.FIRST_BY_DIGEST.value}
        ),
        procedure="enumerate finite without-replacement subsets, select with FIRST_BY_DIGEST, resolve independent outcomes, then bootstrap task means",
    )


def _absence(
    bank: ResponseBank,
    reason: str,
    remedy: str,
    susceptibility: Susceptibility,
    provenance: contracts.EntryProvenance,
    provider: OutcomeProvider | None = None,
) -> CompareResult:
    entry = contracts.absence(
        "reward_statistics",
        "reward_statistics.selection_stress",
        "selected-winner independent outcome under finite-bank selection",
        reason,
        remedy,
        ("selected-winner independent-outcome curve",),
        subject_ref=_entry_subject(bank, provider),
        provenance=provenance,
        method=_selection_method(bank, None),
        scope=_SCOPE,
    )
    return CompareResult(
        _SCOPE,
        _scorecard(Absence(reason, remedy)),
        [entry],
        "the independent-outcome reading is absent",
        "selected-winner quality or a target-specific capability claim",
        SelectionStressReading((), susceptibility),
    )


def _point_payload(point: SelectionPoint) -> dict[str, object]:
    selected = point.selected_success
    if isinstance(selected, Estimate):
        selected_payload: dict[str, object] = {
            "selected_success": str(selected.value),
            "selected_interval": [str(selected.interval[0]), str(selected.interval[1])],
            "resolved_tasks": selected.n,
        }
    else:
        selected_payload = {
            "selected_success_absence": {"reason": selected.reason, "remedy": selected.remedy}
        }
    return {
        "budget": point.budget,
        **selected_payload,
        "oracle_pass_at_k": str(point.oracle_pass_at_k),
        "unqualified_outcomes": point.unqualified_outcomes,
        "ties_observed": point.ties_observed,
    }


def _producer_payload(
    bank: ResponseBank,
    points: Sequence[SelectionPoint],
    provider: OutcomeProvider,
    susceptibility: Susceptibility,
) -> dict[str, object]:
    point = points[0]
    return {
        "curve": [_point_payload(item) for item in points],
        "calibration": {
            "rho": str(susceptibility.calibration.rho),
            "power": str(susceptibility.calibration.power),
            "kill_criterion": susceptibility.calibration.kill_criterion,
            "evidence_id": susceptibility.calibration.evidence_id,
            "cite": susceptibility.calibration.cite,
        },
        "qualification": {
            "protocol_digest": provider.protocol.protocol_digest,
            "qualified_on_partition_id": provider.protocol.qualified_on_partition_id,
            "qualified_on_partition_digest": provider.protocol.qualified_on_partition_digest,
        },
        "outcome_identities": {
            "owner": provider.check.owner,
            "source_digest": provider.check.source_digest,
            "interface": provider.check.interface,
            "partition_id": provider.check.partition_id,
        },
        "subject_ref": provider.subject_ref,
        "supervised_target_receipts": []
        if provider.receipt_identities is None
        else [provider.receipt_identities[key] for key in sorted(provider.receipt_identities)],
        "finite_bank_population": {
            "N": bank.N,
            "tasks": {str(task): len(bank.responses_for(task)) for task in bank.tasks},
        },
        "source_bank": {
            "id": bank.id,
            "digest": bank.digest,
            "source_path": bank.provenance.source_path,
            "revision": bank.provenance.revision,
        },
        "draw_law": "uniform finite subsets without replacement",
        "ties": {"policy": point.tie_break.value, "observed": point.ties_observed},
        "selection_seed": 0,
    }


def _entry(
    bank: ResponseBank,
    estimate: Estimate,
    points: Sequence[SelectionPoint],
    provider: OutcomeProvider,
    susceptibility: Susceptibility,
    bootstrap_resamples: int,
    provenance: contracts.EntryProvenance,
) -> contracts.Entry:
    point = points[0]
    uncertainty = contracts.Uncertainty(
        method="task_level_bootstrap",
        level=0.95,
        interval=[float(estimate.interval[0]), float(estimate.interval[1])],
        clusters=estimate.n,
        cluster_unit="task",
    )
    method = _selection_method(bank, point.budget)
    result = _producer_payload(bank, points, provider, susceptibility)
    result["interval"] = [str(estimate.interval[0]), str(estimate.interval[1])]
    result["task_bootstrap"] = {
        "method": "task_level_bootstrap",
        "resamples": bootstrap_resamples,
        "cluster_unit": "task",
        "resolved_tasks": estimate.n,
    }
    return contracts.estimate(
        "reward_statistics",
        "reward_statistics.selection_stress",
        "selected-winner independent outcome under finite-bank selection",
        float(estimate.value),
        "fraction",
        estimate.n,
        "task",
        uncertainty,
        subject_ref=_entry_subject(bank, provider),
        provenance=provenance,
        method=method,
        result=result,
        scope=_SCOPE,
        limitations=[
            "pass@k is oracle availability and is reported separately from reward-selected success"
        ],
    )


def _resolved_outcome_absence(
    bank: ResponseBank,
    points: Sequence[SelectionPoint],
    provider: OutcomeProvider,
    susceptibility: Susceptibility,
    bootstrap_resamples: int,
    provenance: contracts.EntryProvenance,
) -> contracts.Entry:
    reason = "not measured: no resolved selected independent outcomes"
    remedy = "resolve independent outcomes for selected responses under the qualified protocol"
    entry = contracts.absence(
        "reward_statistics",
        "reward_statistics.selection_stress",
        "selected-winner independent outcome under finite-bank selection",
        reason,
        remedy,
        ("selected-winner independent-outcome curve",),
        subject_ref=_entry_subject(bank, provider),
        provenance=provenance,
        method=_selection_method(bank, points[0].budget),
        scope=_SCOPE,
    )
    payload = _producer_payload(bank, points, provider, susceptibility)
    payload["reason"] = reason
    payload["remedy"] = remedy
    payload["interval"] = None
    payload["task_bootstrap"] = {
        "method": "task_level_bootstrap",
        "resamples": bootstrap_resamples,
        "cluster_unit": "task",
        "resolved_tasks": 0,
    }
    entry.extensions = {_ABSENCE_EXTENSION: payload}
    return entry


def canonical_assay_entry(result: CompareResult) -> contracts.Entry:
    """Reopen the actual engine-produced stable entry through canonical contract bytes."""
    if type(result) is not CompareResult or result.scope != _SCOPE:
        raise TypeError(
            "canonical selection-stress entry requires a selection_stress CompareResult"
        )
    entries = [
        entry for entry in result.entries if entry.entry_id == "reward_statistics.selection_stress"
    ]
    if len(entries) != 1:
        raise ValueError("selection-stress CompareResult requires exactly one canonical entry")
    return contracts.Entry.model_validate_json(contracts.canonical_bytes(entries[0]))


def selection_stress(
    bank: ResponseBank | None,
    selector: object,
    outcome_check: OutcomeProvider | None,
    *,
    ladder: Iterable[int],
    bootstrap_resamples: int,
    sandbox: Sandbox,
    limits: Limits,
) -> CompareResult:
    """Measure selected winner outcome over the allowed finite-bank selection budgets."""
    # Capture this measurement before invoking the selector or resolving outcomes.
    provenance = contracts.EntryProvenance(
        started=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        duration_s=0.0,
        sandbox_tier="T0",
        offline=True,
    )
    bank = require_bank(bank)
    susceptibility = Susceptibility(Decimal(0), Calibration(_RHO, _POWER, _KILL, _EVIDENCE, _CITE))
    if outcome_check is None:
        return _absence(
            bank,
            "not measured: no independent outcome check for this bank",
            "declare and qualify the bank's independent outcome protocol",
            susceptibility,
            provenance,
        )
    if not outcome_check.qualified:
        return _absence(
            bank,
            "not measured: independent outcome protocol is unqualified",
            "qualify the declared outcome check against its actual protected suite",
            susceptibility,
            provenance,
            outcome_check,
        )
    budgets = tuple(ladder)
    if not budgets:
        raise ValueError("selection ladder must contain at least one budget")
    if any(type(item) is not int or item < 1 for item in budgets):
        raise TypeError("selection ladder budgets must be positive integers")
    if any(item > bank.N // 2 for item in budgets):
        requested = max(item for item in budgets if item > bank.N // 2)
        raise _refusal(
            "RL0132",
            f"requested selection budget {requested} exceeds safe bank budget {bank.N // 2} for N={bank.N}",
            {"N": bank.N, "requested_budget": requested},
        )
    points = []
    first: Estimate | Absence | None = None
    for budget in budgets:
        # Materialise the declared P-BANK selection under the caller's real execution
        # context.  The curve below is the exact expectation over all such finite draws.
        materialized = best_of_n(
            bank,
            selector,
            n=budget,
            seed=0,
            tie_break=TieBreak.FIRST_BY_DIGEST,
            sandbox=sandbox,
            limits=limits,
        )
        values: list[Decimal] = []
        oracle: list[Decimal] = []
        unknown = 0
        for task in bank.tasks:
            selected, availability, count = _task_point(
                bank, task, selector, outcome_check, budget, sandbox, limits
            )
            oracle.append(availability)
            unknown += count
            if selected is not None:
                values.append(selected)
        estimate = (
            Estimate(
                _round(sum(values, Decimal(0)) / Decimal(len(values))),
                _bootstrap(values, bootstrap_resamples),
                n=len(values),
                unit="selected independent outcome",
            )
            if values
            else Absence(
                "not measured: no resolved selected independent outcomes",
                "resolve independent outcomes for selected responses under the qualified protocol",
            )
        )
        point = SelectionPoint(
            budget,
            estimate,
            _round(sum(oracle, Decimal(0)) / Decimal(len(oracle))),
            TieBreak.FIRST_BY_DIGEST,
            unknown,
            materialized.ties_observed,
        )
        points.append(point)
        if first is None:
            first = estimate
    assert first is not None
    reading = SelectionStressReading(tuple(points), susceptibility)
    entries = (
        [
            _resolved_outcome_absence(
                bank, points, outcome_check, susceptibility, bootstrap_resamples, provenance
            )
        ]
        if isinstance(first, Absence)
        else [
            _entry(
                bank, first, points, outcome_check, susceptibility, bootstrap_resamples, provenance
            )
        ]
    )
    return CompareResult(
        _SCOPE,
        _scorecard(first),
        entries,
        "reward-selected independent outcome over the supplied finite bank",
        "oracle availability, target-specific capability retention, or behavior after training",
        reading,
    )


def run(inputs) -> CompareResult:
    return selection_stress(
        inputs.bank,
        inputs.selector,
        inputs.outcome_check,
        ladder=inputs.ladder,
        bootstrap_resamples=inputs.bootstrap_resamples,
        sandbox=inputs.sandbox,
        limits=inputs.limits,
    )


register(Rung.SELECTION_STRESS, run)
