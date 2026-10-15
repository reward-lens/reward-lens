"""Rung 6: qualified held-out transfer across declared partitions."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from reward_lens import contracts
from reward_lens.execution import Limits, Sandbox, default_sandbox
from reward_lens.graders.python_grader import PythonGrader
from reward_lens.outcome import (
    Outcome,
    OutcomeCheck,
    OutcomeState,
    Qualification,
    qualified_protocol,
)
from reward_lens.outcome.supervised import (
    SupervisedCase,
    SupervisedQualification,
    SupervisedResolution,
    qualify_supervised,
    resolve_supervised,
)
from reward_lens.partitions import Partition, PartitionKind

from .registry import CompareInputs, CompareResult, Rung, register
from .scorecard import ROWS, Absence, Estimate, Scorecard, ScorecardRow

__all__ = [
    "R6_ENTRY_ID",
    "R6_SUPERVISED_SCHEMA_VERSION",
    "Rung6Input",
    "SupervisedRung6Config",
    "SupervisedRung6Original",
    "SupervisedRung6Result",
    "SupervisedRung6Target",
    "TransferPartition",
    "TransferPartitionClass",
    "TransferPartitionAbsent",
    "append_transfer_assay",
    "build_supervised_rung6",
    "evaluate_transfer",
    "load_supervised_rung6_execution",
    "run",
]


R6_ENTRY_ID = "reward_statistics.held_out_transfer"
R6_METHOD_ID = "reward_statistics.repair_transfer"
R6_SUPERVISED_SCHEMA_VERSION = 1
_SCOPE = "held_out_transfer"
_Q = Decimal("0.000001")
_SHA256_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_PARTITION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SUPERVISED_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "subject_ref",
        "owners",
        "grader",
        "limits",
        "controls",
        "original",
        "targets",
    }
)
_PYTHON_GRADER_INTERFACE = "reward_lens.graders.python_grader.PythonGrader.score/v1"


class TransferPartitionAbsent(ValueError):
    """TRANSFER_PARTITION_ABSENT: a required independent transfer class was not declared."""

    def __init__(self, partition_class: str) -> None:
        self.partition_class = partition_class
        super().__init__(
            f"TRANSFER_PARTITION_ABSENT: missing declared transfer partition class {partition_class!r}"
        )


class TransferPartitionClass(StrEnum):
    """The one axis a live held-out target is declared to vary."""

    FAMILY = "FAMILY"
    EXPLOIT = "EXPLOIT"
    CHECKPOINT = "CHECKPOINT"


def _supervised_config_error(field: str, detail: str) -> ValueError:
    return ValueError(f"RUNG6_SUPERVISED_CONFIG_INVALID: {field} {detail}")


def _strict_mapping(value: object, field: str, *, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _supervised_config_error(field, "must be an object")
    found = set(value)
    if found != keys:
        missing = sorted(keys - found)
        extra = sorted(found - keys)
        raise _supervised_config_error(field, f"has missing={missing!r} extra={extra!r}")
    if any(not isinstance(key, str) for key in value):
        raise _supervised_config_error(field, "must use string keys")
    return dict(value)


def _nonempty_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _supervised_config_error(field, "must be a non-empty string")
    return value


def _sha256_ref(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_REF.fullmatch(value):
        raise _supervised_config_error(field, "must be a sha256 reference")
    return value


def _finite(value: object, field: str, *, upper: float) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(float(value))
        or not 0 < float(value) <= upper
    ):
        raise _supervised_config_error(field, f"must be a positive finite number at most {upper}")
    return float(value)


def _bounded_integer(value: object, field: str, *, upper: int) -> int:
    if type(value) is not int or not 0 < value <= upper:
        raise _supervised_config_error(field, f"must be an integer from 1 through {upper}")
    return value


def _json_value(value: object, field: str) -> Any:
    """Copy only JSON values so a mapping loader cannot transport host objects."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _supervised_config_error(field, "must not contain a non-finite number")
        return value
    if isinstance(value, list):
        return [_json_value(item, f"{field}[]") for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise _supervised_config_error(field, "must use string object keys")
        return {key: _json_value(item, f"{field}.{key}") for key, item in value.items()}
    raise _supervised_config_error(field, f"must contain JSON values, not {type(value).__name__}")


def _relative_local_path(value: object, field: str, *, root: Path, file: bool) -> Path:
    text = _nonempty_text(value, field)
    candidate = Path(text)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise _supervised_config_error(field, "must be a relative local path below root")
    resolved = (root / candidate).resolve()
    if resolved == root or root not in resolved.parents:
        raise _supervised_config_error(field, "must resolve below root")
    if file and not resolved.is_file():
        raise _supervised_config_error(field, "must name an existing local file")
    if not file and not resolved.is_dir():
        raise _supervised_config_error(field, "must name an existing local directory")
    return resolved


def _case_from_mapping(value: object, field: str) -> SupervisedCase:
    data = _strict_mapping(value, field, keys={"item_id", "task", "response"})
    item_id = _nonempty_text(data["item_id"], f"{field}.item_id")
    task = _json_value(data["task"], f"{field}.task")
    response = _json_value(data["response"], f"{field}.response")
    if not isinstance(task, Mapping):
        raise _supervised_config_error(f"{field}.task", "must be an object")
    if not isinstance(response, (str, Mapping)):
        raise _supervised_config_error(f"{field}.response", "must be a string or object")
    return SupervisedCase(item_id=item_id, task=task, response=response)


def _cases_from_mapping(value: object, field: str, *, nonempty: bool) -> tuple[SupervisedCase, ...]:
    if not isinstance(value, list) or (nonempty and not value):
        requirement = "a non-empty array" if nonempty else "an array"
        raise _supervised_config_error(field, f"must be {requirement}")
    return tuple(_case_from_mapping(item, f"{field}[{index}]") for index, item in enumerate(value))


def _partition_from_mapping(
    value: object,
    field: str,
    *,
    root: Path,
    kind: PartitionKind,
    extra_keys: set[str],
) -> tuple[dict[str, Any], Partition]:
    data = _strict_mapping(
        value, field, keys={"partition_id", "path", "digest", "tree_version", *extra_keys}
    )
    partition_id = _nonempty_text(data["partition_id"], f"{field}.partition_id")
    if not _SAFE_PARTITION_ID.fullmatch(partition_id):
        raise _supervised_config_error(f"{field}.partition_id", "must be a safe local identifier")
    if data["tree_version"] != "tree-v1":
        raise _supervised_config_error(f"{field}.tree_version", "must equal 'tree-v1'")
    expected_digest = _sha256_ref(data["digest"], f"{field}.digest")
    path = _relative_local_path(data["path"], f"{field}.path", root=root, file=False)
    partition = Partition.from_dir(
        partition_id,
        kind,
        path,
        log_path=root / ".reward-lens-rung6-access" / f"{partition_id}.jsonl",
    )
    if partition.digest != expected_digest:
        raise _supervised_config_error(f"{field}.digest", "does not match the local partition tree")
    return data, partition


@dataclass(frozen=True)
class SupervisedRung6Original:
    """The distinct original example that the supervised evaluator must execute."""

    partition: Partition
    family: str
    exploit: str
    checkpoint: str
    case: SupervisedCase


@dataclass(frozen=True)
class SupervisedRung6Target:
    """One public declared transfer target, with requests but no caller-supplied outcomes."""

    partition_class: TransferPartitionClass
    partition: Partition
    family: str
    exploit: str
    checkpoint: str
    declared_at: str
    cases: Sequence[SupervisedCase]


@dataclass(frozen=True)
class SupervisedRung6Config:
    """The public producer configuration for a supervisor-backed Rung 6 measurement.

    It intentionally has no Outcome, receipt, SuiteResult, count, callback, or runner field.
    ``build_supervised_rung6`` obtains all qualification and target facts itself.
    """

    evaluator: PythonGrader
    check: OutcomeCheck
    owner: str
    candidate_owner: str
    protected_controls: Partition
    known_good: Sequence[SupervisedCase]
    known_wrong: Sequence[SupervisedCase]
    audited_grader_owner: str
    audited_grader_source_digest: str
    original: SupervisedRung6Original
    targets: Sequence[SupervisedRung6Target]
    subject_ref: str
    sandbox: Sandbox
    limits: Limits

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, root: Path | str) -> "SupervisedRung6Config":
        """Build typed local execution inputs from the R6 supervised JSON v1 contract.

        This is intentionally a loader for declared local files, not a transport
        format for outcomes, receipts, callbacks, partitions, or a sandbox object.
        The resulting typed config still has to pass ``build_supervised_rung6`` to
        obtain its qualified facts and fresh receipts.
        """
        base = Path(root).resolve()
        if not base.is_dir():
            raise _supervised_config_error("root", "must be an existing local directory")
        data = _strict_mapping(raw, "config", keys=set(_SUPERVISED_CONFIG_KEYS))
        if data["schema_version"] != R6_SUPERVISED_SCHEMA_VERSION:
            raise _supervised_config_error(
                "schema_version", f"must equal {R6_SUPERVISED_SCHEMA_VERSION}"
            )
        subject_ref = _sha256_ref(data["subject_ref"], "subject_ref")
        owners = _strict_mapping(
            data["owners"],
            "owners",
            keys={"evaluator", "candidate", "audited_grader", "audited_grader_source_digest"},
        )
        owner = _nonempty_text(owners["evaluator"], "owners.evaluator")
        candidate_owner = _nonempty_text(owners["candidate"], "owners.candidate")
        audited_grader_owner = _nonempty_text(owners["audited_grader"], "owners.audited_grader")
        audited_grader_source_digest = _sha256_ref(
            owners["audited_grader_source_digest"], "owners.audited_grader_source_digest"
        )
        grader = _strict_mapping(
            data["grader"], "grader", keys={"version", "path", "source_digest"}
        )
        if grader["version"] != "python-grader-v1":
            raise _supervised_config_error("grader.version", "must equal 'python-grader-v1'")
        grader_path = _relative_local_path(grader["path"], "grader.path", root=base, file=True)
        expected_source_digest = _sha256_ref(grader["source_digest"], "grader.source_digest")
        evaluator = PythonGrader.from_path(grader_path)
        if evaluator.describe().source_digest != expected_source_digest:
            raise _supervised_config_error(
                "grader.source_digest", "does not match the local grader source"
            )
        limits_raw = _strict_mapping(
            data["limits"],
            "limits",
            keys={
                "wall_s",
                "cpu_s",
                "memory_bytes",
                "processes",
                "network",
                "stdout_bytes",
                "artifact_bytes",
            },
        )
        if limits_raw["network"] is not False:
            raise _supervised_config_error(
                "limits.network", "must be false for local offline execution"
            )
        limits = Limits(
            wall_s=_finite(limits_raw["wall_s"], "limits.wall_s", upper=300.0),
            cpu_s=_finite(limits_raw["cpu_s"], "limits.cpu_s", upper=300.0),
            memory_bytes=_bounded_integer(
                limits_raw["memory_bytes"], "limits.memory_bytes", upper=8 * 2**30
            ),
            processes=_bounded_integer(limits_raw["processes"], "limits.processes", upper=256),
            network=False,
            stdout_bytes=_bounded_integer(
                limits_raw["stdout_bytes"], "limits.stdout_bytes", upper=16 << 20
            ),
            artifact_bytes=_bounded_integer(
                limits_raw["artifact_bytes"], "limits.artifact_bytes", upper=128 << 20
            ),
        )
        controls_raw, controls = _partition_from_mapping(
            data["controls"],
            "controls",
            root=base,
            kind=PartitionKind.PROTECTED_SUITE,
            extra_keys={"known_good", "known_wrong"},
        )
        known_good = _cases_from_mapping(
            controls_raw["known_good"], "controls.known_good", nonempty=True
        )
        known_wrong = _cases_from_mapping(
            controls_raw["known_wrong"], "controls.known_wrong", nonempty=True
        )
        original_raw, original_partition = _partition_from_mapping(
            data["original"],
            "original",
            root=base,
            kind=PartitionKind.HELD_OUT_TASKS,
            extra_keys={"family", "exploit", "checkpoint", "case"},
        )
        original = SupervisedRung6Original(
            partition=original_partition,
            family=_nonempty_text(original_raw["family"], "original.family"),
            exploit=_nonempty_text(original_raw["exploit"], "original.exploit"),
            checkpoint=_nonempty_text(original_raw["checkpoint"], "original.checkpoint"),
            case=_case_from_mapping(original_raw["case"], "original.case"),
        )
        if not isinstance(data["targets"], list):
            raise _supervised_config_error("targets", "must be an array")
        targets: list[SupervisedRung6Target] = []
        for index, value in enumerate(data["targets"]):
            field = f"targets[{index}]"
            target_raw, target_partition = _partition_from_mapping(
                value,
                field,
                root=base,
                kind=PartitionKind.HELD_OUT_TASKS,
                extra_keys={
                    "partition_class",
                    "family",
                    "exploit",
                    "checkpoint",
                    "declared_at",
                    "cases",
                },
            )
            try:
                partition_class = TransferPartitionClass(target_raw["partition_class"])
            except (TypeError, ValueError) as exc:
                raise _supervised_config_error(
                    f"{field}.partition_class", "must be FAMILY, EXPLOIT, or CHECKPOINT"
                ) from exc
            targets.append(
                SupervisedRung6Target(
                    partition_class=partition_class,
                    partition=target_partition,
                    family=_nonempty_text(target_raw["family"], f"{field}.family"),
                    exploit=_nonempty_text(target_raw["exploit"], f"{field}.exploit"),
                    checkpoint=_nonempty_text(target_raw["checkpoint"], f"{field}.checkpoint"),
                    declared_at=_nonempty_text(target_raw["declared_at"], f"{field}.declared_at"),
                    cases=_cases_from_mapping(target_raw["cases"], f"{field}.cases", nonempty=True),
                )
            )
        return cls(
            evaluator=evaluator,
            check=OutcomeCheck(
                owner, evaluator.describe().source_digest, _PYTHON_GRADER_INTERFACE, controls.id
            ),
            owner=owner,
            candidate_owner=candidate_owner,
            protected_controls=controls,
            known_good=known_good,
            known_wrong=known_wrong,
            audited_grader_owner=audited_grader_owner,
            audited_grader_source_digest=audited_grader_source_digest,
            original=original,
            targets=tuple(targets),
            subject_ref=subject_ref,
            sandbox=default_sandbox(),
            limits=limits,
        )


@dataclass(frozen=True)
class SupervisedRung6Result:
    """The typed engine input, canonical comparison, and retained actual supervisor receipts."""

    input: Rung6Input
    comparison: CompareResult
    qualification: SupervisedQualification
    original: SupervisedResolution
    targets: Mapping[str, tuple[SupervisedResolution, ...]]


@dataclass(frozen=True)
class TransferPartition:
    """A declared transfer identity, separate from the protected qualification controls."""

    partition_id: str
    partition_class: TransferPartitionClass
    family: str
    exploit: str
    checkpoint: str
    declared_at: str
    digest: str

    def __post_init__(self) -> None:
        if type(self.partition_class) is not TransferPartitionClass:
            raise TypeError("transfer partition_class must be a TransferPartitionClass")
        for field in ("partition_id", "family", "exploit", "checkpoint", "declared_at", "digest"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise TypeError(f"transfer partition {field} must be a non-empty string")
        if not self.digest.startswith("sha256:") or len(self.digest) != 71:
            raise ValueError("transfer partition digest must be a sha256 digest")


@dataclass(frozen=True)
class Rung6Input:
    """The strict producer seam for a transfer result.

    The future public Compare owner must materialize this object from its request,
    then pass it as ``CompareInputs.run_record``.  This rung accepts no names or
    success labels in place of factory-created item outcomes.
    """

    partitions: tuple[TransferPartition, ...]
    evaluation_partitions: Mapping[str, Partition]
    outcomes: Mapping[str, Sequence[Outcome]]
    qualification: Qualification
    protected_controls: Partition
    original_partition: Partition | None
    original_outcome: Outcome | None
    original_family: str | None
    original_exploit: str | None
    original_checkpoint: str | None
    subject_ref: str
    provenance: contracts.EntryProvenance

    def __post_init__(self) -> None:
        if any(type(value) is not TransferPartition for value in self.partitions):
            raise TypeError("rung 6 partitions must be TransferPartition values")
        if type(self.qualification) is not Qualification:
            raise TypeError("rung 6 qualification must be the accepted Outcome Qualification")
        if type(self.protected_controls) is not Partition:
            raise TypeError("rung 6 protected_controls must be a live Partition")
        if self.original_partition is not None and type(self.original_partition) is not Partition:
            raise TypeError("rung 6 original_partition must be a live Partition or None")
        if self.original_outcome is not None and type(self.original_outcome) is not Outcome:
            raise TypeError("rung 6 original_outcome must be an Outcome or None")
        for field in ("original_family", "original_exploit", "original_checkpoint"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or not value):
                raise TypeError(f"rung 6 {field} must be a non-empty string or None")
        if not isinstance(self.subject_ref, str) or not self.subject_ref.startswith("sha256:"):
            raise TypeError("rung 6 subject_ref must be a canonical sha256 digest")
        if not isinstance(self.provenance, contracts.EntryProvenance):
            raise TypeError("rung 6 provenance must be actual EntryProvenance")
        object.__setattr__(
            self, "evaluation_partitions", MappingProxyType(dict(self.evaluation_partitions))
        )
        object.__setattr__(
            self,
            "outcomes",
            MappingProxyType({key: tuple(value) for key, value in self.outcomes.items()}),
        )


def _absence(input: Rung6Input) -> CompareResult:
    qualification = input.qualification
    reason = f"qualification.state=OUTCOME_UNQUALIFIED: {qualification.reason}"
    entry = contracts.absence(
        "reward_statistics",
        R6_ENTRY_ID,
        "whether the repair transfers across the separately declared held-out partitions",
        reason,
        "supply a P-OUTCOME-qualified independent evaluator before resolving transfer items",
        (
            "a repair transfer claim",
            "a held-out family, exploit, or checkpoint result",
        ),
        subject_ref=input.subject_ref,
        provenance=input.provenance,
        depends_on=("digest:outcome_protocol",),
        method=contracts.Method(
            id=R6_METHOD_ID,
            version="3.1.0",
            params_digest=contracts.digest(
                {
                    "qualification_state": qualification.state,
                    "qualification_reason": qualification.reason,
                    "partitions": [
                        {
                            "id": partition.partition_id,
                            "partition_class": partition.partition_class.value,
                            "digest": partition.digest,
                        }
                        for partition in input.partitions
                    ],
                }
            ),
            procedure=(
                "attempt the declared supervised transfer measurement and retain its "
                "independent-outcome qualification state when the protocol is unqualified"
            ),
        ),
        scope=_SCOPE,
        limitations=(
            "OUTCOME_UNQUALIFIED is an absence of independent outcome evidence, not a transfer failure.",
        ),
    )
    rows = [
        ScorecardRow(
            row_id,
            row_id.replace("_", " "),
            Absence(
                "not measured: OUTCOME_UNQUALIFIED prevents held-out transfer measurement",
                "qualify the independent outcome protocol and resolve every declared transfer item",
            ),
            _SCOPE,
        )
        for row_id in ROWS
    ]
    return CompareResult(
        _SCOPE,
        Scorecard.render(rows, rung=Rung.TRANSFER),
        [entry],
        "no transfer result because the outcome protocol is unqualified",
        "a failed transfer, an original-example-only block, or any qualified outcome effect",
    )


def _require_partition_classes(input: Rung6Input) -> None:
    partitions = input.partitions
    if len({item.partition_id for item in partitions}) != len(partitions):
        raise ValueError("transfer partition declarations must use distinct partition_id values")
    expected = tuple(TransferPartitionClass)
    actual = {item.partition_class for item in partitions}
    for partition_class in expected:
        if partition_class not in actual:
            raise TransferPartitionAbsent(partition_class.value.lower())
    if len(partitions) != len(expected):
        raise ValueError(
            "rung 6 requires exactly one distinct live target for each transfer partition class"
        )
    if any(
        value is None
        for value in (input.original_family, input.original_exploit, input.original_checkpoint)
    ):
        raise ValueError("rung 6 requires original family, exploit, and checkpoint identities")
    baseline = {
        "family": input.original_family,
        "exploit": input.original_exploit,
        "checkpoint": input.original_checkpoint,
    }
    changed_by_class = {
        TransferPartitionClass.FAMILY: "family",
        TransferPartitionClass.EXPLOIT: "exploit",
        TransferPartitionClass.CHECKPOINT: "checkpoint",
    }
    for declaration in partitions:
        changed = changed_by_class[declaration.partition_class]
        for axis, original_value in baseline.items():
            value = getattr(declaration, axis)
            if axis == changed and value == original_value:
                raise ValueError(
                    f"{declaration.partition_class.value} transfer partition must change original {axis}"
                )
            if axis != changed and value != original_value:
                raise ValueError(
                    f"{declaration.partition_class.value} transfer partition must retain original {axis}"
                )


def _require_control_separation(input: Rung6Input) -> None:
    controls = input.protected_controls
    if controls.kind is not PartitionKind.PROTECTED_SUITE:
        raise TypeError("rung 6 protected_controls must be a PROTECTED_SUITE Partition")
    qualification = input.qualification
    if (
        controls.id != qualification.partition_id
        or controls.digest != qualification.partition_digest
    ):
        raise ValueError("protected_controls must be the live partition that P-OUTCOME qualified")
    for partition in input.evaluation_partitions.values():
        if type(partition) is not Partition:
            raise TypeError("rung 6 evaluation partitions must be live Partition values")
        if partition.id == controls.id or partition.digest == controls.digest:
            raise ValueError(
                f"protected_controls {controls.id!r} must be disjoint from transfer evaluation partition {partition.id!r}"
            )


def _require_live_partitions(input: Rung6Input) -> tuple[tuple[TransferPartition, Partition], ...]:
    pairs: list[tuple[TransferPartition, Partition]] = []
    declared_ids = {item.partition_id for item in input.partitions}
    if set(input.evaluation_partitions) != declared_ids:
        raise ValueError(
            "evaluation_partitions must contain exactly the declared transfer partition ids"
        )
    if set(input.outcomes) != declared_ids:
        raise ValueError("outcomes must contain exactly the declared transfer partition ids")
    for declaration in input.partitions:
        partition = input.evaluation_partitions[declaration.partition_id]
        if partition.kind is not PartitionKind.HELD_OUT_TASKS:
            raise TypeError(
                f"transfer evaluation partition {declaration.partition_id!r} must be HELD_OUT_TASKS"
            )
        if partition.id != declaration.partition_id or partition.digest != declaration.digest:
            raise ValueError(
                f"transfer declaration {declaration.partition_id!r} does not match its live partition identity"
            )
        pairs.append((declaration, partition))
    return tuple(pairs)


def _outcomes_for_partition(
    input: Rung6Input,
    declaration: TransferPartition,
    partition: Partition,
    protocol_digest: str,
) -> tuple[Outcome, ...]:
    rows = input.outcomes[declaration.partition_id]
    if any(type(row) is not Outcome for row in rows):
        raise TypeError("transfer outcomes must be P-OUTCOME Outcome values")
    manifest = dict(partition.manifest())
    ids = tuple(row.item_id for row in rows)
    if len(ids) != len(set(ids)):
        raise ValueError(f"transfer outcomes for {partition.id!r} contain duplicate item ids")
    if set(ids) != set(manifest):
        raise ValueError(
            f"transfer outcomes for {partition.id!r} do not cover its active manifest exactly"
        )
    for row in rows:
        if row.protocol_digest != protocol_digest:
            raise ValueError(
                f"transfer outcome {row.item_id!r} does not carry the qualified outcome protocol digest"
            )
        if row.state is OutcomeState.UNQUALIFIED:
            raise ValueError(
                f"OUTCOME_UNQUALIFIED: transfer outcome {row.item_id!r} is not a resolved independent outcome"
            )
        if (
            row.evaluated_on_partition_id != partition.id
            or row.evaluated_on_partition_digest != partition.digest
            or row.evaluated_item_digest != manifest[row.item_id]
        ):
            raise ValueError(
                f"transfer outcome {row.item_id!r} lacks the live evaluated-on witness for {partition.id!r}"
            )
    return rows


def _require_original_observation(
    input: Rung6Input,
    protocol_digest: str,
    pairs: tuple[tuple[TransferPartition, Partition], ...],
) -> Outcome:
    partition = input.original_partition
    outcome = input.original_outcome
    if partition is None or outcome is None:
        raise ValueError(
            "rung 6 requires the actual original_partition and original_outcome observation"
        )
    if partition.kind is not PartitionKind.HELD_OUT_TASKS:
        raise TypeError("rung 6 original_partition must be a HELD_OUT_TASKS Partition")
    if (
        partition.id == input.protected_controls.id
        or partition.digest == input.protected_controls.digest
    ):
        raise ValueError("original_partition must be disjoint from protected_controls")
    for _declaration, target in pairs:
        if partition.id == target.id or partition.digest == target.digest:
            raise ValueError(
                f"original_partition {partition.id!r} must be disjoint from transfer target {target.id!r}"
            )
    manifest = dict(partition.manifest())
    if outcome.item_id not in manifest:
        raise ValueError("original_outcome item_id is not active in original_partition")
    if outcome.protocol_digest != protocol_digest:
        raise ValueError("original_outcome does not carry the qualified outcome protocol digest")
    if outcome.state is OutcomeState.UNQUALIFIED:
        raise ValueError(
            "OUTCOME_UNQUALIFIED: original_outcome is not a resolved independent outcome"
        )
    if (
        outcome.evaluated_on_partition_id != partition.id
        or outcome.evaluated_on_partition_digest != partition.digest
        or outcome.evaluated_item_digest != manifest[outcome.item_id]
    ):
        raise ValueError(
            "original_outcome lacks the live evaluated-on witness for original_partition"
        )
    return outcome


def _wilson_interval(passed: int, total: int) -> tuple[Decimal, Decimal]:
    z = 1.959964
    proportion = passed / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return (
        Decimal(str(max(0.0, centre - half))).quantize(_Q, rounding=ROUND_HALF_UP),
        Decimal(str(min(1.0, centre + half))).quantize(_Q, rounding=ROUND_HALF_UP),
    )


def _scorecard(successes: int, total: int) -> Scorecard:
    value = (Decimal(successes) / Decimal(total)).quantize(_Q, rounding=ROUND_HALF_UP)
    measured = Estimate(value, _wilson_interval(successes, total), total)
    rows = []
    for row_id in ROWS:
        row_value = (
            measured
            if row_id == "intended_task_success"
            else Absence(
                "not measured: held-out transfer resolves repair success only",
                "run the registered measurement for this scorecard row",
            )
        )
        rows.append(ScorecardRow(row_id, row_id.replace("_", " "), row_value, _SCOPE))
    return Scorecard.render(rows, rung=Rung.TRANSFER)


def evaluate_transfer(input: Rung6Input) -> CompareResult:
    """Produce one transfer check from real partitions and qualified outcome rows."""
    if type(input) is not Rung6Input:
        raise TypeError("rung 6 canonical producer requires Rung6Input")
    if not input.qualification.qualified:
        return _absence(input)
    protocol = qualified_protocol(input.qualification)
    _require_partition_classes(input)
    _require_control_separation(input)
    pairs = _require_live_partitions(input)
    original_outcome = _require_original_observation(input, protocol.protocol_digest, pairs)

    evidence: dict[str, object] = {}
    all_rows: list[Outcome] = []
    for declaration, partition in pairs:
        rows = _outcomes_for_partition(input, declaration, partition, protocol.protocol_digest)
        all_rows.extend(rows)
        evidence[declaration.partition_id] = {
            "partition_class": declaration.partition_class.value,
            "family": declaration.family,
            "exploit": declaration.exploit,
            "checkpoint": declaration.checkpoint,
            "declared_at": declaration.declared_at,
            "digest": declaration.digest,
            "outcomes": [
                {
                    "item_id": row.item_id,
                    "state": row.state.value,
                    "evaluated_item_digest": row.evaluated_item_digest,
                }
                for row in rows
            ],
        }
    successes = sum(row.state is OutcomeState.PASS for row in all_rows)
    transferred = successes == len(all_rows)
    original_blocked = original_outcome.state is OutcomeState.FAIL
    original_only = original_blocked and not transferred
    passed = original_blocked and transferred
    entry = contracts.Entry(
        entry_id=R6_ENTRY_ID,
        section="reward_statistics",
        kind="check",
        measurand="whether an original observed block also holds across separately declared held-out transfer partitions",
        method=contracts.Method(
            id=R6_METHOD_ID,
            version="3.1.0",
            params_digest=contracts.digest(
                {
                    "partitions": [
                        {
                            "id": item.partition_id,
                            "partition_class": item.partition_class.value,
                            "family": item.family,
                            "exploit": item.exploit,
                            "checkpoint": item.checkpoint,
                            "declared_at": item.declared_at,
                            "digest": item.digest,
                        }
                        for item, _ in pairs
                    ],
                    "protocol_digest": protocol.protocol_digest,
                    "protected_controls": {
                        "id": input.protected_controls.id,
                        "digest": input.protected_controls.digest,
                    },
                    "original": {
                        "partition_id": input.original_partition.id,
                        "partition_digest": input.original_partition.digest,
                        "family": input.original_family,
                        "exploit": input.original_exploit,
                        "checkpoint": input.original_checkpoint,
                        "item_id": original_outcome.item_id,
                        "item_digest": original_outcome.evaluated_item_digest,
                    },
                }
            ),
            procedure="require exact live-partition coverage by factory-created qualified outcomes, then derive the transfer predicate from their resolved states",
        ),
        scope=_SCOPE,
        subject_ref=input.subject_ref,
        depends_on=["digest:outcome_protocol"],
        state="complete",
        provenance=input.provenance,
        limitations=[
            "This compares the declared held-out families, exploits, and checkpoints only; it does not establish transfer beyond them."
        ],
        result={
            "qualification": {
                "state": "qualified",
                "protocol_digest": protocol.protocol_digest,
                "protected_controls": {
                    "partition_id": input.protected_controls.id,
                    "partition_digest": input.protected_controls.digest,
                },
            },
            "partitions": evidence,
            "original_example": {
                "partition_id": input.original_partition.id,
                "partition_digest": input.original_partition.digest,
                "item_id": original_outcome.item_id,
                "item_digest": original_outcome.evaluated_item_digest,
                "state": original_outcome.state.value,
            },
            "original_example_only": original_only,
            "resolved_transfer_successes": successes,
            "resolved_transfer_total": len(all_rows),
        },
        check=contracts.Check(
            predicate="the disjoint original observation resolves FAIL and every item in one live declared FAMILY, EXPLOIT, and CHECKPOINT target resolves PASS under one qualified protocol",
            passed=passed,
            scope_tested="one distinct live target per held-out class: FAMILY, EXPLOIT, and CHECKPOINT",
        ),
    )
    return CompareResult(
        _SCOPE,
        _scorecard(successes, len(all_rows)),
        [entry],
        "a qualified transfer check over the three declared held-out partition axes"
        if passed
        else "the qualified transfer predicate did not hold",
        "a transfer claim beyond the declared partitions; an original example block alone does not establish transfer",
    )


def append_transfer_assay(assay: contracts.Assay, comparison: CompareResult) -> contracts.Assay:
    """Append the one canonical R6 entry before the shared store seals an Assay."""
    if not isinstance(assay, contracts.Assay) or type(comparison) is not CompareResult:
        raise TypeError("rung 6 assay producer requires an Assay and CompareResult")
    entries = [entry for entry in comparison.entries if entry.entry_id == R6_ENTRY_ID]
    if len(entries) != 1:
        raise ValueError("rung 6 comparison must carry exactly one canonical entry")
    data = assay.to_dict()
    existing = data["measurement"]["reward_statistics"]
    if any(entry["entry_id"] == R6_ENTRY_ID for entry in existing):
        raise ValueError("rung 6 canonical entry is already present in this assay")
    existing.append(entries[0].model_dump(mode="json", by_alias=True, exclude_unset=True))
    data["assay_id"] = contracts.digest(data)
    return contracts.Assay.model_validate(data)


def _supervised_receipt(resolution: SupervisedResolution) -> dict[str, object]:
    """Keep the supervisor's identity-bound receipt beside the Outcome it produced."""
    return {
        "receipt_digest": resolution.receipt.receipt_digest,
        "run_id": resolution.receipt.run_id,
        "request_digest": resolution.receipt.request_digest,
        "verdict": resolution.receipt.verdict,
        "target_partition_id": resolution.target_partition_id,
        "target_partition_digest": resolution.target_partition_digest,
        "target_item_digest": resolution.target_item_digest,
        "task_digest": resolution.task_digest,
        "response_digest": resolution.response_digest,
    }


def load_supervised_rung6_execution(
    path: Path | str, *, root: Path | str | None = None
) -> SupervisedRung6Config:
    """Load the strict R6 supervised JSON contract from a local file.

    Relative grader and partition paths are interpreted beneath ``root`` or, by
    default, the declaration's parent.  The declaration is input only: it does
    not carry outcomes, qualification counts, receipts, callbacks, or runners.
    """
    declaration = Path(path).resolve()
    if not declaration.is_file():
        raise _supervised_config_error("declaration", "must be an existing local JSON file")
    local_root = declaration.parent if root is None else Path(root).resolve()
    if not local_root.is_dir() or (
        declaration != local_root and local_root not in declaration.parents
    ):
        raise _supervised_config_error("declaration", "must be located below the local root")
    try:
        raw = json.loads(declaration.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _supervised_config_error("declaration", "must contain one valid JSON object") from exc
    if not isinstance(raw, Mapping):
        raise _supervised_config_error("declaration", "must contain one JSON object")
    return SupervisedRung6Config.from_mapping(raw, root=local_root)


def _same_or_nested(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _require_physical_partition_separation(named: Sequence[tuple[str, Partition]]) -> None:
    """Reject shared files, roots, and nested trees before the evaluator can read one item."""
    for index, (left_name, left) in enumerate(named):
        left_root = left.root.resolve()
        for right_name, right in named[index + 1 :]:
            if _same_or_nested(left_root, right.root.resolve()):
                raise ValueError(
                    f"supervised rung 6 physical partition separation failed: {left_name} and {right_name} share or nest a root"
                )
    members: dict[tuple[int, int], str] = {}
    for name, partition in named:
        for item_id, _digest in partition.manifest():
            try:
                stat = (partition.root / item_id).stat()
            except OSError as exc:
                raise ValueError(
                    f"supervised rung 6 physical partition separation could not stat {name}/{item_id}"
                ) from exc
            identity = (stat.st_dev, stat.st_ino)
            existing = members.get(identity)
            if existing is not None:
                raise ValueError(
                    f"supervised rung 6 physical partition separation failed: {existing} and {name}/{item_id} alias one file"
                )
            members[identity] = f"{name}/{item_id}"


def _preflight_supervised_rung6(
    config: SupervisedRung6Config, targets: tuple[SupervisedRung6Target, ...]
) -> None:
    """Prove all declared transfer memberships are distinct before a supervisor call."""
    if (
        type(config.original.partition) is not Partition
        or type(config.original.case) is not SupervisedCase
    ):
        raise TypeError("supervised rung 6 original must use a live Partition and SupervisedCase")
    if (
        config.original.partition.id == config.protected_controls.id
        or config.original.partition.digest == config.protected_controls.digest
        or _same_or_nested(
            config.original.partition.root.resolve(), config.protected_controls.root.resolve()
        )
    ):
        raise ValueError("supervised rung 6 original must be disjoint from protected_controls")
    if config.protected_controls.kind is not PartitionKind.PROTECTED_SUITE:
        raise TypeError("supervised rung 6 protected controls must be a PROTECTED_SUITE Partition")
    if config.original.partition.kind is not PartitionKind.HELD_OUT_TASKS:
        raise TypeError("supervised rung 6 original must use a HELD_OUT_TASKS Partition")
    expected = set(TransferPartitionClass)
    if len(targets) != len(expected) or {target.partition_class for target in targets} != expected:
        missing = next(
            (
                value
                for value in TransferPartitionClass
                if value not in {target.partition_class for target in targets}
            ),
            None,
        )
        if missing is not None:
            raise TransferPartitionAbsent(missing.value.lower())
        raise ValueError(
            "rung 6 requires exactly one distinct live target for each transfer partition class"
        )
    baseline = {
        "family": config.original.family,
        "exploit": config.original.exploit,
        "checkpoint": config.original.checkpoint,
    }
    changed_by_class = {
        TransferPartitionClass.FAMILY: "family",
        TransferPartitionClass.EXPLOIT: "exploit",
        TransferPartitionClass.CHECKPOINT: "checkpoint",
    }
    named: list[tuple[str, Partition]] = [
        ("protected_controls", config.protected_controls),
        ("original", config.original.partition),
    ]
    ids = {config.protected_controls.id, config.original.partition.id}
    if len(ids) != 2:
        raise ValueError("supervised rung 6 partitions must use distinct partition ids")
    for target in targets:
        if (
            type(target.partition) is not Partition
            or target.partition.kind is not PartitionKind.HELD_OUT_TASKS
        ):
            raise TypeError("supervised rung 6 targets must use HELD_OUT_TASKS Partition values")
        cases = tuple(target.cases)
        if not cases or any(type(case) is not SupervisedCase for case in cases):
            raise ValueError("every supervised rung 6 target needs declared SupervisedCase values")
        if target.partition.id in ids:
            raise ValueError("supervised rung 6 partitions must use distinct partition ids")
        ids.add(target.partition.id)
        changed = changed_by_class[target.partition_class]
        for axis, original_value in baseline.items():
            value = getattr(target, axis)
            if axis == changed and value == original_value:
                raise ValueError(
                    f"{target.partition_class.value} transfer partition must change original {axis}"
                )
            if axis != changed and value != original_value:
                raise ValueError(
                    f"{target.partition_class.value} transfer partition must retain original {axis}"
                )
        named.append((f"target {target.partition_class.value}", target.partition))
    _require_physical_partition_separation(named)


def build_supervised_rung6(config: SupervisedRung6Config) -> SupervisedRung6Result:
    """Execute the owner-selected evaluator and materialize a canonical Rung 6 input.

    The caller supplies declarative live partitions and cases only.  Qualification
    counts, outcome states, and every receipt arise from ``outcome.supervised``.
    """
    if type(config) is not SupervisedRung6Config:
        raise TypeError("supervised rung 6 factory requires SupervisedRung6Config")
    if type(config.evaluator) is not PythonGrader:
        raise TypeError("supervised rung 6 evaluator must be the accepted PythonGrader")
    if type(config.check) is not OutcomeCheck or type(config.protected_controls) is not Partition:
        raise TypeError("supervised rung 6 requires an OutcomeCheck and live protected controls")
    if type(config.original) is not SupervisedRung6Original:
        raise TypeError("supervised rung 6 requires SupervisedRung6Original")
    targets = tuple(config.targets)
    if any(type(target) is not SupervisedRung6Target for target in targets):
        raise TypeError("supervised rung 6 targets must be SupervisedRung6Target values")
    _preflight_supervised_rung6(config, targets)
    qualification = qualify_supervised(
        evaluator=config.evaluator,
        check=config.check,
        owner=config.owner,
        candidate_owner=config.candidate_owner,
        controls=config.protected_controls,
        known_good=tuple(config.known_good),
        known_wrong=tuple(config.known_wrong),
        audited_grader_owner=config.audited_grader_owner,
        audited_grader_source_digest=config.audited_grader_source_digest,
        subject_ref=config.subject_ref,
        sandbox=config.sandbox,
        limits=config.limits,
    )
    original = resolve_supervised(
        qualification,
        target=config.original.partition,
        case=config.original.case,
        sandbox=config.sandbox,
        limits=config.limits,
    )
    declarations: list[TransferPartition] = []
    partitions: dict[str, Partition] = {}
    outcomes: dict[str, tuple[Outcome, ...]] = {}
    resolutions: dict[str, tuple[SupervisedResolution, ...]] = {}
    for target in targets:
        if type(target.partition) is not Partition:
            raise TypeError("supervised rung 6 targets must use live Partition values")
        cases = tuple(target.cases)
        if not cases or any(type(case) is not SupervisedCase for case in cases):
            raise ValueError("every supervised rung 6 target needs declared SupervisedCase values")
        declaration = TransferPartition(
            partition_id=target.partition.id,
            partition_class=target.partition_class,
            family=target.family,
            exploit=target.exploit,
            checkpoint=target.checkpoint,
            declared_at=target.declared_at,
            digest=target.partition.digest,
        )
        resolved = tuple(
            resolve_supervised(
                qualification,
                target=target.partition,
                case=case,
                sandbox=config.sandbox,
                limits=config.limits,
            )
            for case in cases
        )
        declarations.append(declaration)
        partitions[target.partition.id] = target.partition
        outcomes[target.partition.id] = tuple(item.outcome for item in resolved)
        resolutions[target.partition.id] = resolved
    input = Rung6Input(
        partitions=tuple(declarations),
        evaluation_partitions=partitions,
        outcomes=outcomes,
        qualification=qualification.qualification,
        protected_controls=config.protected_controls,
        original_partition=config.original.partition,
        original_outcome=original.outcome,
        original_family=config.original.family,
        original_exploit=config.original.exploit,
        original_checkpoint=config.original.checkpoint,
        subject_ref=config.subject_ref,
        provenance=qualification.qualification.entry.provenance,
    )
    comparison = evaluate_transfer(input)
    entry = next(item for item in comparison.entries if item.entry_id == R6_ENTRY_ID)
    # Frozen records deliberately forbid ``result`` on an absence.  The public
    # dispatcher retains the actual qualification receipts in its execution
    # identity for that route; an absence must remain an absence in the
    # canonical measurement section.
    if entry.kind == "absence":
        return SupervisedRung6Result(
            input=input,
            comparison=comparison,
            qualification=qualification,
            original=original,
            targets=MappingProxyType(dict(resolutions)),
        )
    result = dict(entry.result or {})
    result["supervised_receipts"] = {
        "controls": [
            {
                "receipt_digest": receipt.receipt_digest,
                "run_id": receipt.run_id,
                "request_digest": receipt.request_digest,
                "verdict": receipt.verdict,
            }
            for receipt in qualification.control_receipts
        ],
        "original": _supervised_receipt(original),
        "targets": {
            partition_id: [_supervised_receipt(resolution) for resolution in rows]
            for partition_id, rows in resolutions.items()
        },
    }
    canonical_entry = entry.model_copy(update={"result": result})
    comparison = replace(
        comparison,
        entries=[
            canonical_entry if item.entry_id == R6_ENTRY_ID else item for item in comparison.entries
        ],
    )
    return SupervisedRung6Result(
        input=input,
        comparison=comparison,
        qualification=qualification,
        original=original,
        targets=MappingProxyType(resolutions),
    )


def run(inputs: CompareInputs) -> CompareResult:
    """Registry entrypoint for the strict executable Rung6Input seam."""
    if not isinstance(inputs, CompareInputs):
        raise TypeError("rung 6 requires CompareInputs.run_record to be a Rung6Input")
    if type(inputs.run_record) is SupervisedRung6Result:
        return inputs.run_record.comparison
    if type(inputs.run_record) is not Rung6Input:
        raise TypeError("rung 6 requires CompareInputs.run_record to be a Rung6Input")
    return evaluate_transfer(inputs.run_record)


register(Rung.TRANSFER, run)
