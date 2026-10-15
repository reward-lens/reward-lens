"""Explicit item-level outcome states for a protocol that has already qualified.

Protocol qualification and item resolution answer different questions.  The first asks whether an
independent evaluator may support a correctness claim at all.  This module records each item that
such an evaluator actually resolved.  It deliberately has no conversion from an unknown evaluator
result to a failed one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable, Mapping

from reward_lens.contracts.models import EntryProvenance
from reward_lens.partitions import Partition

from .qualify import Qualification, _is_verified_success

__all__ = [
    "Outcome",
    "OutcomeState",
    "OutcomeSummary",
    "QualifiedProtocol",
    "TaskValidity",
    "UnqualifiedReason",
    "qualified_protocol",
    "resolved_fail",
    "resolved_pass",
    "unqualified",
]


class OutcomeState(StrEnum):
    """The only semantic item-outcome states.

    `UNQUALIFIED` is an unknown item result, never a false result.  It remains in the record and
    outside every resolved denominator.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    UNQUALIFIED = "UNQUALIFIED"


class TaskValidity(StrEnum):
    """Whether the task itself was valid for an evaluator result."""

    VALID = "VALID"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class UnqualifiedReason(StrEnum):
    """Named ways an evaluator may fail to give an item outcome."""

    AMBIGUITY = "ambiguity"
    MISSING_CHECK = "missing_check"
    TIMEOUT = "timeout"
    EVALUATOR_ERROR = "evaluator_error"
    PANEL_TIE = "panel_tie"
    REFUSAL = "refusal"


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_UNQUALIFIED_VALUES = frozenset(reason.value for reason in UnqualifiedReason)
_PROTOCOL_BRIDGE = object()
_OUTCOME_FACTORY = object()


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _protocol_digest(value: object) -> str:
    value = _required_text(value, "protocol_digest")
    if not _DIGEST.fullmatch(value):
        raise ValueError("protocol_digest must be sha256:<64 lowercase hexadecimal characters>")
    return value


def _provenance(value: object) -> EntryProvenance:
    if not isinstance(value, EntryProvenance):
        raise TypeError("provenance must be an EntryProvenance from the evaluator attempt")
    return value


def _validity(value: object) -> TaskValidity:
    if type(value) is not TaskValidity:
        raise TypeError("task_validity must be an explicit TaskValidity, never a truthy value")
    return value


@dataclass(frozen=True)
class _PartitionWitness:
    """A factory-captured partition identity, optionally down to one active item."""

    partition_id: str
    partition_digest: str
    item_id: str | None = None
    item_digest: str | None = None


def _evaluation_witness(partition: Partition | None, item_id: str) -> _PartitionWitness | None:
    if partition is None:
        return None
    if type(partition) is not Partition:
        raise TypeError("evaluation_partition must be a Partition")
    items = dict(partition.manifest())
    if item_id not in items:
        raise ValueError(f"{item_id} is not an active item of evaluation partition {partition.id}")
    return _PartitionWitness(partition.id, partition.digest, item_id, items[item_id])


@dataclass(frozen=True)
class QualifiedProtocol:
    """The metadata of a protocol that qualified before any item was resolved.

    This does not itself label a task.  Its only use is to bind every per-item outcome to the
    protocol digest, evaluator provenance, and declared scope under which it was evaluated.
    """

    protocol_digest: str
    provenance: EntryProvenance
    scope: str
    _qualification: Qualification | None = field(default=None, repr=False, compare=False)
    _qualified_on: _PartitionWitness | None = field(default=None, repr=False, compare=False)
    _bridge: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _protocol_digest(self.protocol_digest)
        _provenance(self.provenance)
        _required_text(self.scope, "scope")
        witness = self._qualified_on
        if self._bridge is not _PROTOCOL_BRIDGE or not _is_verified_success(self._qualification):
            raise TypeError(
                "QualifiedProtocol must be constructed by qualified_protocol from a successful "
                "Qualification"
            )
        if (
            witness is None
            or self._qualification.partition_id != witness.partition_id
            or self._qualification.partition_digest != witness.partition_digest
            or self.protocol_digest != self._qualification.entry.method.params_digest
            or self.provenance is not self._qualification.entry.provenance
            or self.scope != self._qualification.entry.scope
        ):
            raise ValueError(
                "QualifiedProtocol metadata must match its qualification partition witness"
            )

    @property
    def qualified_on_partition_id(self) -> str:
        """The protected control partition that actually qualified this protocol."""
        assert self._qualified_on is not None
        return self._qualified_on.partition_id

    @property
    def qualified_on_partition_digest(self) -> str:
        """The digest of the protected control partition that qualified this protocol."""
        assert self._qualified_on is not None
        return self._qualified_on.partition_digest


def qualified_protocol(qualification: Qualification) -> QualifiedProtocol:
    """Bridge a verified successful protocol qualification into item-resolution metadata."""
    if type(qualification) is not Qualification:
        raise TypeError("qualification must be a Qualification returned by qualify")
    if not _is_verified_success(qualification):
        raise ValueError("only a successful Qualification returned by qualify can resolve items")
    return QualifiedProtocol(
        protocol_digest=qualification.entry.method.params_digest,
        provenance=qualification.entry.provenance,
        scope=qualification.entry.scope,
        _qualification=qualification,
        _qualified_on=_PartitionWitness(
            qualification.partition_id or "",
            qualification.partition_digest or "",
        ),
        _bridge=_PROTOCOL_BRIDGE,
    )


@dataclass(frozen=True)
class Outcome:
    """One evaluator result, retaining enough context to make its boundary inspectable."""

    item_id: str
    state: OutcomeState
    reason: str
    protocol_digest: str | None
    provenance: EntryProvenance
    task_validity: TaskValidity
    scope: str
    evaluated_on_partition_id: str | None = None
    evaluated_on_partition_digest: str | None = None
    evaluated_item_digest: str | None = None
    _protocol: QualifiedProtocol | None = field(default=None, repr=False, compare=False)
    _evaluation: _PartitionWitness | None = field(default=None, repr=False, compare=False)
    _factory: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _required_text(self.item_id, "item_id")
        if type(self.state) is not OutcomeState:
            raise TypeError("state must be an explicit OutcomeState, never a truthy value")
        _required_text(self.reason, "reason")
        if self.protocol_digest is not None:
            _protocol_digest(self.protocol_digest)
        _provenance(self.provenance)
        validity = _validity(self.task_validity)
        _required_text(self.scope, "scope")
        if self._evaluation is None:
            if any(
                value is not None
                for value in (
                    self.evaluated_on_partition_id,
                    self.evaluated_on_partition_digest,
                    self.evaluated_item_digest,
                )
            ):
                raise ValueError(
                    "evaluation partition metadata requires an evaluation partition witness"
                )
        elif (
            self.evaluated_on_partition_id != self._evaluation.partition_id
            or self.evaluated_on_partition_digest != self._evaluation.partition_digest
            or self.item_id != self._evaluation.item_id
            or self.evaluated_item_digest != self._evaluation.item_digest
        ):
            raise ValueError("Outcome metadata must match its evaluation partition witness")
        if self.state in (OutcomeState.PASS, OutcomeState.FAIL):
            if self._protocol is None or self._factory is not _OUTCOME_FACTORY:
                raise TypeError(
                    "resolved Outcome values must be constructed through a resolution factory"
                )
            if self.protocol_digest != self._protocol.protocol_digest:
                raise ValueError(
                    "resolved Outcome protocol digest must match its qualification witness"
                )
            if validity is not TaskValidity.VALID:
                raise ValueError("a resolved state requires TaskValidity.VALID")
            if self.reason in _UNQUALIFIED_VALUES:
                raise ValueError("an unknown evaluator result must be UNQUALIFIED, never FAIL")
        elif self.reason not in _UNQUALIFIED_VALUES:
            raise ValueError("an unqualified state requires a named UnqualifiedReason")
        elif self._factory is not _OUTCOME_FACTORY:
            raise TypeError("Outcome values must be constructed through a resolution factory")
        elif self._protocol is None and self.protocol_digest is not None:
            raise ValueError(
                "an unqualified outcome without a protocol cannot invent a protocol digest"
            )

    @property
    def resolved(self) -> bool:
        """Whether the outcome belongs to a rate denominator."""
        return self.state in (OutcomeState.PASS, OutcomeState.FAIL)

    def __bool__(self) -> bool:
        raise TypeError("Outcome has no truth value; compare state explicitly")


def _outcome(
    *,
    item_id: str,
    state: OutcomeState,
    reason: str,
    protocol: QualifiedProtocol | None,
    provenance: EntryProvenance | None = None,
    scope: str | None = None,
    task_validity: TaskValidity,
    evaluation_partition: Partition | None = None,
) -> Outcome:
    if protocol is not None and type(protocol) is not QualifiedProtocol:
        raise TypeError("protocol must be a QualifiedProtocol, separate from the item result")
    if state in (OutcomeState.PASS, OutcomeState.FAIL) and protocol is None:
        raise TypeError("a resolved outcome requires a QualifiedProtocol witness")
    if protocol is None:
        provenance = _provenance(provenance)
        scope = _required_text(scope, "scope")
        protocol_digest = None
    else:
        provenance = protocol.provenance
        scope = protocol.scope
        protocol_digest = protocol.protocol_digest
    evaluation = _evaluation_witness(evaluation_partition, item_id)
    return Outcome(
        item_id=item_id,
        state=state,
        reason=reason,
        protocol_digest=protocol_digest,
        provenance=provenance,
        task_validity=task_validity,
        scope=scope,
        evaluated_on_partition_id=None if evaluation is None else evaluation.partition_id,
        evaluated_on_partition_digest=None if evaluation is None else evaluation.partition_digest,
        evaluated_item_digest=None if evaluation is None else evaluation.item_digest,
        _protocol=protocol,
        _evaluation=evaluation,
        _factory=_OUTCOME_FACTORY,
    )


def resolved_pass(
    *,
    item_id: str,
    reason: str,
    protocol: QualifiedProtocol,
    task_validity: TaskValidity,
    evaluation_partition: Partition | None = None,
) -> Outcome:
    """Record a real evaluator acceptance under a qualified protocol."""
    return _outcome(
        item_id=item_id,
        state=OutcomeState.PASS,
        reason=reason,
        protocol=protocol,
        task_validity=task_validity,
        evaluation_partition=evaluation_partition,
    )


def resolved_fail(
    *,
    item_id: str,
    reason: str,
    protocol: QualifiedProtocol,
    task_validity: TaskValidity,
    evaluation_partition: Partition | None = None,
) -> Outcome:
    """Record a real evaluator rejection under a qualified protocol."""
    return _outcome(
        item_id=item_id,
        state=OutcomeState.FAIL,
        reason=reason,
        protocol=protocol,
        task_validity=task_validity,
        evaluation_partition=evaluation_partition,
    )


def unqualified(
    *,
    item_id: str,
    reason: UnqualifiedReason,
    protocol: QualifiedProtocol | None,
    provenance: EntryProvenance | None = None,
    scope: str | None = None,
    task_validity: TaskValidity,
    evaluation_partition: Partition | None = None,
) -> Outcome:
    """Record an unknown evaluator result without converting it to a failure."""
    if type(reason) is not UnqualifiedReason:
        raise TypeError("reason must be an explicit UnqualifiedReason, never a string or boolean")
    return _outcome(
        item_id=item_id,
        state=OutcomeState.UNQUALIFIED,
        reason=reason.value,
        protocol=protocol,
        provenance=provenance,
        scope=scope,
        task_validity=task_validity,
        evaluation_partition=evaluation_partition,
    )


@dataclass(frozen=True)
class OutcomeSummary:
    """A rate summary that retains, and excludes, unknown item outcomes."""

    total_count: int
    pass_count: int
    fail_count: int
    unqualified_count: int
    unqualified_by_reason: Mapping[UnqualifiedReason, int]

    @classmethod
    def from_outcomes(cls, outcomes: Iterable[Outcome]) -> OutcomeSummary:
        """Summarise explicit outcomes without mixing qualified protocols or scopes."""
        rows = tuple(outcomes)
        if any(type(row) is not Outcome for row in rows):
            raise TypeError("outcomes must contain Outcome instances")
        if rows:
            first = rows[0]
            if any(
                row.protocol_digest != first.protocol_digest or row.scope != first.scope
                for row in rows[1:]
            ):
                raise ValueError(
                    "one summary cannot combine different qualified protocols or scopes"
                )
        counts: dict[UnqualifiedReason, int] = {}
        for row in rows:
            if row.state is OutcomeState.UNQUALIFIED:
                reason = UnqualifiedReason(row.reason)
                counts[reason] = counts.get(reason, 0) + 1
        return cls(
            total_count=len(rows),
            pass_count=sum(row.state is OutcomeState.PASS for row in rows),
            fail_count=sum(row.state is OutcomeState.FAIL for row in rows),
            unqualified_count=sum(row.state is OutcomeState.UNQUALIFIED for row in rows),
            unqualified_by_reason=counts,
        )

    def __post_init__(self) -> None:
        counts = (self.total_count, self.pass_count, self.fail_count, self.unqualified_count)
        if any(type(count) is not int or count < 0 for count in counts):
            raise TypeError("outcome summary counts must be non-negative integers, never booleans")
        if self.total_count != self.pass_count + self.fail_count + self.unqualified_count:
            raise ValueError("outcome summary counts must account for every outcome exactly once")
        if any(type(reason) is not UnqualifiedReason for reason in self.unqualified_by_reason):
            raise TypeError("unqualified reasons must be UnqualifiedReason values")
        if any(
            type(count) is not int or count < 0 for count in self.unqualified_by_reason.values()
        ):
            raise TypeError("unqualified reason counts must be non-negative integers")
        if sum(self.unqualified_by_reason.values()) != self.unqualified_count:
            raise ValueError("unqualified reason counts must equal unqualified_count")

    @property
    def resolved_count(self) -> int:
        return self.pass_count + self.fail_count

    @property
    def resolved_rate(self) -> float | None:
        if self.resolved_count == 0:
            return None
        return self.pass_count / self.resolved_count

    @property
    def measurement_state(self) -> str:
        return "NOT_MEASURED" if self.resolved_count == 0 else "MEASURED"

    def __bool__(self) -> bool:
        raise TypeError("OutcomeSummary has no truth value; inspect its explicit counts and state")

    ("qualified_protocol",)
