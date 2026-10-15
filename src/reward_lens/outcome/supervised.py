"""Actual supervisor-backed qualification and held-out outcome resolution (C-054).

The public outcome constructors deliberately accept already-resolved semantic values.  This module
is the producer that obtains those values: it invokes the owner-selected ``PythonGrader`` itself,
keeps a frozen record of the resulting supervisor envelopes, and is the only path here from an
evaluator score to a resolved ``Outcome``.  There is no callable-runner input and no envelope,
suite-count, PASS, or FAIL input supplied by a caller.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from reward_lens.contracts import EntryProvenance, canonical_bytes, digest
from reward_lens.execution import Limits, Sandbox
from reward_lens.graders.base import EvidenceEnvelope
from reward_lens.graders.python_grader import PythonGrader
from reward_lens.partitions import ACCEPTANCE_EVALUATOR, Partition

from .qualify import OutcomeCheck, Qualification, SuiteResult, qualify
from .resolution import (
    Outcome,
    QualifiedProtocol,
    TaskValidity,
    UnqualifiedReason,
    qualified_protocol,
    resolved_fail,
    resolved_pass,
    unqualified,
)

__all__ = [
    "SupervisorReceipt",
    "SupervisedCase",
    "SupervisedQualification",
    "SupervisedResolution",
    "qualify_supervised",
    "resolve_supervised",
]


_INTERFACE = "reward_lens.graders.python_grader.PythonGrader.score/v1"
_QUALIFICATION_BRIDGE = object()
_QUALIFICATIONS: dict[int, weakref.ReferenceType[SupervisedQualification]] = {}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _same_or_nested(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left in right.parents or right in left.parents


@dataclass(frozen=True)
class SupervisedCase:
    """One declared control or target request, bound to a partition item."""

    item_id: str
    task: Mapping[str, Any]
    response: str | Mapping[str, Any]

    def __post_init__(self) -> None:
        _text(self.item_id, "item_id")
        if not isinstance(self.task, Mapping):
            raise TypeError("task must be a mapping")
        if not isinstance(self.response, (str, Mapping)):
            raise TypeError("response must be a string or mapping")


@dataclass(frozen=True)
class SupervisorReceipt:
    """An immutable public projection of one envelope the selected evaluator actually returned.

    ``payload`` is deliberately a mapping proxy, not a user-provided dict.  A deserialised copy is
    useful for evidence transport but is not trusted as a producer receipt: only a registered
    ``SupervisedQualification`` created below can be resolved.
    """

    receipt_digest: str
    run_id: str
    subject_digest: str
    manifest_digest: str
    request_digest: str
    verdict: str
    score: float | None
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze(_thaw(self.payload)))

    def canonical_bytes(self) -> bytes:
        """The explicit canonical transport form for a retained supervisor receipt."""
        return canonical_bytes(_thaw(self.payload))

    @property
    def provenance(self) -> Mapping[str, Any]:
        """Frozen supervisor-stamped provenance, retained without exposing mutable envelope data."""
        value = self.payload.get("provenance")
        assert isinstance(value, Mapping)
        return value

    @property
    def observations(self) -> Mapping[str, Any]:
        """Frozen adapter observations, including quarantined candidate stdout."""
        value = self.payload.get("observations")
        assert isinstance(value, Mapping)
        return value

    def to_record(self) -> dict[str, Any]:
        return _thaw(self.payload)


@dataclass(frozen=True)
class SupervisedQualification:
    """A source-owned actual control run and its accepted outcome-protocol bridge."""

    qualification: Qualification
    protocol: QualifiedProtocol | None
    control_receipts: tuple[SupervisorReceipt, ...]
    control_partition_id: str
    control_partition_digest: str
    evaluator_source_digest: str
    evaluator_manifest_digest: str
    check: OutcomeCheck
    _evaluator: PythonGrader | None = field(default=None, repr=False, compare=False)
    _control_root: Path | None = field(default=None, repr=False, compare=False)
    _bridge: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class SupervisedResolution:
    """One held-out actual execution, its immutable receipt, and the accepted Outcome."""

    outcome: Outcome
    receipt: SupervisorReceipt
    target_partition_id: str
    target_partition_digest: str
    target_item_digest: str
    task_digest: str
    response_digest: str


def _remember(value: SupervisedQualification) -> SupervisedQualification:
    object_id = id(value)

    def _drop(reference: weakref.ReferenceType[SupervisedQualification]) -> None:
        if _QUALIFICATIONS.get(object_id) is reference:
            _QUALIFICATIONS.pop(object_id, None)

    _QUALIFICATIONS[object_id] = weakref.ref(value, _drop)
    return value


def _trusted(value: object) -> SupervisedQualification:
    if type(value) is not SupervisedQualification:
        raise TypeError("supervised qualification must be created by qualify_supervised")
    reference = _QUALIFICATIONS.get(id(value))
    if reference is None or reference() is not value or value._bridge is not _QUALIFICATION_BRIDGE:
        raise ValueError(
            "resolve_supervised requires an actual qualification created by qualify_supervised"
        )
    if value._evaluator is None:
        raise ValueError("actual qualification has no selected evaluator")
    return value


def _evaluator_identity(evaluator: PythonGrader) -> tuple[str, str]:
    if type(evaluator) is not PythonGrader:
        raise TypeError("evaluator must be the accepted PythonGrader, never an arbitrary runner")
    manifest = evaluator.describe()
    source_digest = digest({"source": evaluator.source(), "entrypoint": evaluator.entrypoint})
    if source_digest != manifest.source_digest:
        raise ValueError("evaluator source or manifest was tampered with")
    return source_digest, manifest.digest()


def _assert_receipt(
    envelope: EvidenceEnvelope,
    *,
    evaluator_source_digest: str,
    evaluator_manifest_digest: str,
    case: SupervisedCase,
    entrypoint: str,
    seed: int,
) -> SupervisorReceipt:
    """Accept only the fresh envelope returned by PythonGrader.score in this call path."""
    if type(envelope) is not EvidenceEnvelope:
        raise TypeError("PythonGrader.score did not return an EvidenceEnvelope")
    expected_request = digest(
        {"task": case.task, "response": case.response, "seed": seed, "entrypoint": entrypoint}
    )
    payload = envelope.to_dict()
    if (
        envelope.subject_digest != evaluator_source_digest
        or envelope.manifest_digest != evaluator_manifest_digest
        or envelope.request_digest != expected_request
        or payload.get("observations", {}).get("request")
        != {"task": case.task, "response": case.response, "entrypoint": entrypoint, "seed": seed}
    ):
        raise ValueError("supervisor receipt does not bind the selected evaluator and request")
    try:
        EntryProvenance.model_validate(envelope.provenance)
    except Exception as exc:  # the adapter is the producer; malformed receipts are not evidence
        raise ValueError("supervisor receipt has malformed provenance") from exc
    return SupervisorReceipt(
        receipt_digest=digest(payload),
        run_id=envelope.run_id,
        subject_digest=envelope.subject_digest,
        manifest_digest=envelope.manifest_digest,
        request_digest=envelope.request_digest,
        verdict=envelope.verdict,
        score=envelope.score,
        payload=payload,
    )


def _run_case(
    evaluator: PythonGrader,
    partition: Partition,
    case: SupervisedCase,
    *,
    sandbox: Sandbox,
    limits: Limits,
    evaluator_source_digest: str,
    evaluator_manifest_digest: str,
    purpose: str,
) -> tuple[EvidenceEnvelope, SupervisorReceipt]:
    if case.item_id not in dict(partition.manifest()):
        raise ValueError(f"{case.item_id} is not an active item of partition {partition.id}")
    # This read both proves the control/target is the declared partition item and leaves its audit log.
    partition.read(case.item_id, capability=ACCEPTANCE_EVALUATOR, purpose=purpose)
    envelope = evaluator.score(case.task, case.response, sandbox=sandbox, limits=limits)
    return envelope, _assert_receipt(
        envelope,
        evaluator_source_digest=evaluator_source_digest,
        evaluator_manifest_digest=evaluator_manifest_digest,
        case=case,
        entrypoint=evaluator.entrypoint,
        seed=evaluator.seed,
    )


def _strict_score(envelope: EvidenceEnvelope) -> float | None:
    """Interpret only an actual finite numeric binary score, never Python truthiness."""
    if envelope.verdict != "scored" or type(envelope.score) is not float:
        return None
    if any(finding.rule == "validity.bool_as_score" for finding in envelope.findings):
        return None
    if not math.isfinite(envelope.score) or envelope.score not in (0.0, 1.0):
        return None
    return envelope.score


def _provenance(envelope: EvidenceEnvelope) -> EntryProvenance:
    return EntryProvenance.model_validate(envelope.provenance)


def qualify_supervised(
    *,
    evaluator: PythonGrader,
    check: OutcomeCheck,
    owner: str,
    candidate_owner: str,
    controls: Partition,
    known_good: Sequence[SupervisedCase],
    known_wrong: Sequence[SupervisedCase],
    audited_grader_owner: str,
    audited_grader_source_digest: str,
    subject_ref: str,
    sandbox: Sandbox,
    limits: Limits,
) -> SupervisedQualification:
    """Run known-good and known-wrong controls before exposing a qualified protocol.

    This function accepts cases, never counts or envelopes.  Every number handed to ``qualify`` is
    derived below from freshly returned, identity-bound supervisor receipts.
    """
    if type(check) is not OutcomeCheck or type(controls) is not Partition:
        raise TypeError("check and controls must be accepted OutcomeCheck and Partition values")
    owner, candidate_owner = _text(owner, "owner"), _text(candidate_owner, "candidate_owner")
    if owner == candidate_owner:
        raise ValueError("the selected qualified evaluator belongs to the owner, not the candidate")
    if check.owner != owner:
        raise ValueError("OutcomeCheck owner must select the evaluator")
    if check.partition_id != controls.id:
        raise ValueError("OutcomeCheck must name the actual control partition")
    if check.interface != _INTERFACE:
        raise ValueError("OutcomeCheck interface does not name PythonGrader.score")
    if limits.network is not False:
        raise ValueError("supervised outcome execution requires network=False")
    evaluator_source_digest, evaluator_manifest_digest = _evaluator_identity(evaluator)
    if check.source_digest != evaluator_source_digest:
        raise ValueError("OutcomeCheck source does not match the selected evaluator")
    good, wrong = tuple(known_good), tuple(known_wrong)
    if not good or not wrong or not all(type(case) is SupervisedCase for case in (*good, *wrong)):
        raise ValueError(
            "qualification needs declared known-good and known-wrong SupervisedCase controls"
        )
    item_ids = [case.item_id for case in (*good, *wrong)]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("each qualification control must name one distinct partition item")

    raw: list[EvidenceEnvelope] = []
    receipts: list[SupervisorReceipt] = []
    for case in good:
        envelope, receipt = _run_case(
            evaluator,
            controls,
            case,
            sandbox=sandbox,
            limits=limits,
            evaluator_source_digest=evaluator_source_digest,
            evaluator_manifest_digest=evaluator_manifest_digest,
            purpose="supervised known-good qualification control",
        )
        raw.append(envelope)
        receipts.append(receipt)
    for case in wrong:
        envelope, receipt = _run_case(
            evaluator,
            controls,
            case,
            sandbox=sandbox,
            limits=limits,
            evaluator_source_digest=evaluator_source_digest,
            evaluator_manifest_digest=evaluator_manifest_digest,
            purpose="supervised known-wrong qualification control",
        )
        raw.append(envelope)
        receipts.append(receipt)

    qualification = qualify(
        check=check,
        suite=controls,
        grader_owner=_text(audited_grader_owner, "audited_grader_owner"),
        grader_source_digest=_text(audited_grader_source_digest, "audited_grader_source_digest"),
        subject_ref=_text(subject_ref, "subject_ref"),
        provenance=_provenance(raw[0]),
        result=SuiteResult(
            known_good_passed=sum(_strict_score(envelope) == 1.0 for envelope in raw[: len(good)]),
            known_good_total=len(good),
            known_wrong_failed=sum(_strict_score(envelope) == 0.0 for envelope in raw[len(good) :]),
            known_wrong_total=len(wrong),
        ),
    )
    protocol = qualified_protocol(qualification) if qualification.qualified else None
    return _remember(
        SupervisedQualification(
            qualification=qualification,
            protocol=protocol,
            control_receipts=tuple(receipts),
            control_partition_id=controls.id,
            control_partition_digest=controls.digest,
            evaluator_source_digest=evaluator_source_digest,
            evaluator_manifest_digest=evaluator_manifest_digest,
            check=check,
            _evaluator=evaluator,
            _control_root=controls.root.resolve(),
            _bridge=_QUALIFICATION_BRIDGE,
        )
    )


def resolve_supervised(
    supervised: SupervisedQualification,
    *,
    target: Partition,
    case: SupervisedCase,
    sandbox: Sandbox,
    limits: Limits,
) -> SupervisedResolution:
    """Execute one held-out item and derive PASS, FAIL, or UNQUALIFIED from its receipt."""
    supervised = _trusted(supervised)
    if type(target) is not Partition or type(case) is not SupervisedCase:
        raise TypeError("target and case must be accepted Partition and SupervisedCase values")
    if limits.network is not False:
        raise ValueError("supervised outcome execution requires network=False")
    if (
        target.id == supervised.control_partition_id
        or target.digest == supervised.control_partition_digest
        or supervised._control_root is None
        or _same_or_nested(target.root, supervised._control_root)
    ):
        raise ValueError("qualification controls and held-out targets must be disjoint")
    source_digest, manifest_digest = _evaluator_identity(supervised._evaluator)
    if (
        source_digest != supervised.evaluator_source_digest
        or manifest_digest != supervised.evaluator_manifest_digest
    ):
        raise ValueError("selected evaluator source or manifest changed after qualification")
    envelope, receipt = _run_case(
        supervised._evaluator,
        target,
        case,
        sandbox=sandbox,
        limits=limits,
        evaluator_source_digest=source_digest,
        evaluator_manifest_digest=manifest_digest,
        purpose="supervised held-out target resolution",
    )
    score = _strict_score(envelope)
    if supervised.protocol is None:
        outcome = unqualified(
            item_id=case.item_id,
            reason=UnqualifiedReason.MISSING_CHECK,
            protocol=None,
            provenance=_provenance(envelope),
            scope="validity",
            task_validity=TaskValidity.UNKNOWN,
            evaluation_partition=target,
        )
    elif envelope.verdict == "timeout":
        outcome = unqualified(
            item_id=case.item_id,
            reason=UnqualifiedReason.TIMEOUT,
            protocol=supervised.protocol,
            task_validity=TaskValidity.UNKNOWN,
            evaluation_partition=target,
        )
    elif envelope.verdict != "scored":
        outcome = unqualified(
            item_id=case.item_id,
            reason=UnqualifiedReason.EVALUATOR_ERROR,
            protocol=supervised.protocol,
            task_validity=TaskValidity.UNKNOWN,
            evaluation_partition=target,
        )
    elif score == 1.0:
        outcome = resolved_pass(
            item_id=case.item_id,
            reason="owner-selected evaluator returned strict score 1.0",
            protocol=supervised.protocol,
            task_validity=TaskValidity.VALID,
            evaluation_partition=target,
        )
    elif score == 0.0:
        outcome = resolved_fail(
            item_id=case.item_id,
            reason="owner-selected evaluator returned strict score 0.0",
            protocol=supervised.protocol,
            task_validity=TaskValidity.VALID,
            evaluation_partition=target,
        )
    else:
        outcome = unqualified(
            item_id=case.item_id,
            reason=UnqualifiedReason.AMBIGUITY,
            protocol=supervised.protocol,
            task_validity=TaskValidity.UNKNOWN,
            evaluation_partition=target,
        )
    return SupervisedResolution(
        outcome=outcome,
        receipt=receipt,
        target_partition_id=target.id,
        target_partition_digest=target.digest,
        target_item_digest=dict(target.manifest())[case.item_id],
        task_digest=digest(case.task),
        response_digest=digest({"response": case.response}),
    )
