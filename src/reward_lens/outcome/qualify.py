"""D-39: the owner supplies the independent outcome check, or the record says it is unqualified.

The state this returns is P-AUDIT-1's RL0301. It is imported with the catalogue's own message and
the finding id P-AUDIT-1 records it under; nothing here writes a second wording for it, and nothing
here allocates it. The unqualified case is an absence entry, because a missing outcome check is a
hole in the record and never a zero.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass

from reward_lens.contracts.canonical import digest as canonical_digest
from reward_lens.contracts.models import (
    NO_INTERVAL,
    Entry,
    EntryProvenance,
    Method,
    Uncertainty,
    absence,
    estimate,
)
from reward_lens.errors import make
from reward_lens.partitions import Partition
from reward_lens.product.audit.run import FINDING_IDS

__all__ = ["OutcomeCheck", "Qualification", "SuiteResult", "qualify"]

#: The state a record carries when nothing independent stood behind the scores. P-AUDIT-1's.
_STATE = "RL0301"
_STATE_NAME = "OUTCOME_UNQUALIFIED"

_ENTRY_ID = "validity.outcome_qualification"
_MEASURAND = "whether an independent outcome check qualified the correctness claim"
_METHOD_VERSION = "1.0.0"

#: Below this many sampling units the record states that there is no interval rather than
#: inventing endpoints (D-75, rule 9).
_MIN_CLUSTERS = 15
_SUCCESSFUL_QUALIFICATIONS: dict[int, weakref.ReferenceType[Qualification]] = {}


@dataclass(frozen=True)
class OutcomeCheck:
    """The check the owner supplied: who owns it, what it is, and how it is called."""

    owner: str
    source_digest: str
    interface: str
    partition_id: str


@dataclass(frozen=True)
class SuiteResult:
    """What the protected suite did: known-good programs pass, known-wrong variants fail."""

    known_good_passed: int
    known_good_total: int
    known_wrong_failed: int
    known_wrong_total: int


@dataclass(frozen=True)
class Qualification:
    """The answer, with the entry that goes in the record."""

    qualified: bool
    reason: str
    entry: Entry
    code: str | None = None
    state: str | None = None
    message: str = ""
    finding_id: str | None = None
    label: str | None = None
    partition_digest: str | None = None
    partition_id: str | None = None


def _remember_success(qualification: Qualification) -> Qualification:
    """Register one exact successful object without retaining it beyond its caller.

    A field marker is copied by ``dataclasses.replace``. The registry instead accepts only the
    identity returned by this module's successful ``qualify`` call. A weak reference avoids turning
    the verification registry into a lifetime owner of assay records.
    """
    object_id = id(qualification)

    def _remove(reference: weakref.ReferenceType[Qualification]) -> None:
        if _SUCCESSFUL_QUALIFICATIONS.get(object_id) is reference:
            _SUCCESSFUL_QUALIFICATIONS.pop(object_id, None)

    _SUCCESSFUL_QUALIFICATIONS[object_id] = weakref.ref(qualification, _remove)
    return qualification


def _is_verified_success(qualification: object) -> bool:
    """Whether this exact object came from a successful call to :func:`qualify`.

    The bridge to an item-level resolved outcome consumes this identity. A public dataclass with
    `qualified=True`, including a copy of a real result, is not evidence that the outcome protocol
    actually passed its controls.
    """
    return (
        type(qualification) is Qualification
        and qualification.qualified is True
        and (reference := _SUCCESSFUL_QUALIFICATIONS.get(id(qualification))) is not None
        and reference() is qualification
        and qualification.entry.kind == "estimate"
        and qualification.entry.method.id == "outcome.protected_suite"
    )


def _wilson(passed: int, total: int) -> list[float]:
    z = 1.959964
    p = passed / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [round(max(0.0, centre - half), 6), round(min(1.0, centre + half), 6)]


def _unqualified(
    reason: str, *, subject_ref: str, provenance: EntryProvenance, suite: Partition | None
) -> Qualification:
    error = make(_STATE)
    entry = absence(
        "validity",
        _ENTRY_ID,
        _MEASURAND,
        reason,
        "supply an outcome check the owner stands behind, whose source is not the audited grader",
        ("any claim that a high score on this reward system means correct work",),
        subject_ref=subject_ref,
        provenance=provenance,
        limitations=(
            "the scores in this record are the grader's own; nothing independent confirmed them",
        ),
    )
    return Qualification(
        qualified=False,
        reason=reason,
        entry=entry,
        code=error.code,
        state=_STATE_NAME,
        message=error.message,
        finding_id=FINDING_IDS[_STATE],
        partition_id=None if suite is None else suite.id,
        partition_digest=None if suite is None else suite.digest,
    )


def _suite_result_problem(result: SuiteResult) -> str | None:
    """Name a malformed or vacuous evaluator-control result without throwing it away.

    A zero/zero control result made the old equality checks pass and then divided by zero. It is
    neither proof of a qualified evaluator nor a failed item result, so qualification returns its
    normal absence state instead.
    """
    counts = {
        "known_good_passed": result.known_good_passed,
        "known_good_total": result.known_good_total,
        "known_wrong_failed": result.known_wrong_failed,
        "known_wrong_total": result.known_wrong_total,
    }
    for name, count in counts.items():
        if type(count) is not int:
            return f"the outcome check returned a non-integer {name}"
        if count < 0:
            return f"the outcome check returned a negative {name}"
    if result.known_good_total == 0 or result.known_wrong_total == 0:
        return "the outcome check supplied no non-vacuous known-good and known-wrong controls"
    if result.known_good_passed > result.known_good_total:
        return "the outcome check reported more known-good passes than known-good controls"
    if result.known_wrong_failed > result.known_wrong_total:
        return "the outcome check reported more known-wrong failures than known-wrong controls"
    return None


def qualify(
    *,
    check: OutcomeCheck | None,
    suite: Partition | None,
    grader_owner: str,
    grader_source_digest: str,
    subject_ref: str,
    provenance: EntryProvenance,
    result: SuiteResult | None = None,
) -> Qualification:
    """Decide whether the correctness claim is qualified, and by what.

    The order of the questions is the order of D-39: was a check supplied at all, is it something
    other than the grader being audited, does the protected suite hold anything, was it run, and
    did it behave the way a usable check behaves.
    """
    if check is None:
        return _unqualified(
            "no outcome check was supplied",
            subject_ref=subject_ref,
            provenance=provenance,
            suite=suite,
        )
    if suite is not None and check.partition_id != suite.id:
        return _unqualified(
            f"the outcome check names partition {check.partition_id}, which does not name {suite.id}",
            subject_ref=subject_ref,
            provenance=provenance,
            suite=suite,
        )
    if check.source_digest == grader_source_digest:
        return _unqualified(
            "the outcome check is the audited grader under another name",
            subject_ref=subject_ref,
            provenance=provenance,
            suite=suite,
        )
    if suite is None or suite.is_empty:
        return _unqualified(
            "the protected suite holds no item",
            subject_ref=subject_ref,
            provenance=provenance,
            suite=suite,
        )
    if result is None:
        return _unqualified(
            "the outcome check was not run against the protected suite",
            subject_ref=subject_ref,
            provenance=provenance,
            suite=suite,
        )
    problem = _suite_result_problem(result)
    if problem is not None:
        return _unqualified(
            problem,
            subject_ref=subject_ref,
            provenance=provenance,
            suite=suite,
        )
    if result.known_wrong_failed != result.known_wrong_total:
        return _unqualified(
            "a known-wrong variant passed the outcome check",
            subject_ref=subject_ref,
            provenance=provenance,
            suite=suite,
        )
    if result.known_good_passed != result.known_good_total:
        return _unqualified(
            "a known-good program failed the outcome check, which is an evaluator defect",
            subject_ref=subject_ref,
            provenance=provenance,
            suite=suite,
        )

    total = result.known_good_total
    value = round(result.known_good_passed / total, 6)
    if total < _MIN_CLUSTERS:
        uncertainty = Uncertainty(
            method=NO_INTERVAL,
            level=0.95,
            reason=(
                f"{total} tasks is below the fifteen this project requires before it states an "
                "interval, so the record states the count and no interval"
            ),
            clusters=total,
        )
    else:
        uncertainty = Uncertainty(
            interval=_wilson(result.known_good_passed, total),
            method="wilson",
            level=0.95,
            clusters=total,
        )
    method = Method(
        id="outcome.protected_suite",
        version=_METHOD_VERSION,
        params_digest=canonical_digest(
            {
                "interface": check.interface,
                "owner": check.owner,
                "partition_id": check.partition_id,
                "source_digest": check.source_digest,
                "suite_digest": suite.digest,
            }
        ),
        procedure=(
            "the owner's outcome check was run over the protected partition: known-good programs "
            "had to pass and known-wrong variants had to fail for the expected reason"
        ),
    )
    entry = estimate(
        "validity",
        _ENTRY_ID,
        "the share of known-good programs the outcome check passed on the protected partition",
        value,
        "fraction",
        total,
        "task",
        uncertainty,
        subject_ref=subject_ref,
        provenance=provenance,
        method=method,
        result={
            "known_good_passed": result.known_good_passed,
            "known_good_total": result.known_good_total,
            "known_wrong_failed": result.known_wrong_failed,
            "known_wrong_total": result.known_wrong_total,
            "partition_digest": suite.digest,
        },
        denominator="known-good programs in the protected partition",
    )
    return _remember_success(
        Qualification(
            qualified=True,
            reason="the owner's outcome check qualified the correctness claim on this partition",
            entry=entry,
            label=f"outcome-qualified for protected partition {suite.id}",
            partition_id=suite.id,
            partition_digest=suite.digest,
        )
    )
