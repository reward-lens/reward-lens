"""The revision-3 per-item outcome state is explicit and cannot collapse unknown into failure."""

from __future__ import annotations

import pytest

from reward_lens.contracts import EntryProvenance
from reward_lens.outcome import (
    Outcome,
    OutcomeCheck,
    OutcomeState,
    OutcomeSummary,
    QualifiedProtocol,
    SuiteResult,
    TaskValidity,
    UnqualifiedReason,
    qualified_protocol,
    qualify,
    resolved_fail,
    resolved_pass,
    unqualified,
)

CHECK_SOURCE = "sha256:" + "c" * 64
GRADER = "sha256:" + "b" * 64
SUBJECT = "sha256:" + "a" * 64
SCOPE = "evaluator_comparison"


@pytest.fixture
def provenance() -> EntryProvenance:
    return EntryProvenance(
        started="2026-09-26T00:00:00Z", duration_s=0.25, sandbox_tier="L3", offline=True
    )


@pytest.fixture
def protocol(protected, provenance: EntryProvenance) -> QualifiedProtocol:
    qualification = qualify(
        check=OutcomeCheck(
            owner="independent evaluator",
            source_digest=CHECK_SOURCE,
            interface="pytest",
            partition_id=protected.id,
        ),
        suite=protected,
        grader_owner="audited grader",
        grader_source_digest=GRADER,
        subject_ref=SUBJECT,
        provenance=provenance,
        result=SuiteResult(
            known_good_passed=1,
            known_good_total=1,
            known_wrong_failed=1,
            known_wrong_total=1,
        ),
    )
    return qualified_protocol(qualification)


def test_outcome_state_has_exactly_the_three_semantic_values() -> None:
    assert tuple(OutcomeState) == (
        OutcomeState.PASS,
        OutcomeState.FAIL,
        OutcomeState.UNQUALIFIED,
    )


def test_resolved_outcome_carries_all_required_metadata(
    protocol: QualifiedProtocol, provenance: EntryProvenance
) -> None:
    outcome = resolved_pass(
        item_id="task-1",
        reason="the independent evaluator accepted the candidate output",
        protocol=protocol,
        task_validity=TaskValidity.VALID,
    )
    assert outcome.state is OutcomeState.PASS
    assert outcome.reason.startswith("the independent")
    assert outcome.protocol_digest == protocol.protocol_digest
    assert outcome.provenance is provenance
    assert outcome.task_validity is TaskValidity.VALID
    assert outcome.scope == SCOPE


def test_protocol_qualification_is_distinct_from_an_item_resolution(
    protocol: QualifiedProtocol,
) -> None:
    outcome = unqualified(
        item_id="task-timeout",
        reason=UnqualifiedReason.TIMEOUT,
        protocol=protocol,
        task_validity=TaskValidity.VALID,
    )
    assert protocol.protocol_digest == outcome.protocol_digest
    assert outcome.state is OutcomeState.UNQUALIFIED
    assert outcome.reason == UnqualifiedReason.TIMEOUT.value


@pytest.mark.parametrize(
    "reason",
    (
        UnqualifiedReason.AMBIGUITY,
        UnqualifiedReason.MISSING_CHECK,
        UnqualifiedReason.TIMEOUT,
        UnqualifiedReason.EVALUATOR_ERROR,
        UnqualifiedReason.PANEL_TIE,
        UnqualifiedReason.REFUSAL,
    ),
)
def test_named_unknown_sources_are_unqualified_not_fail(
    protocol: QualifiedProtocol, reason: UnqualifiedReason
) -> None:
    outcome = unqualified(
        item_id=f"task-{reason.value}",
        reason=reason,
        protocol=protocol,
        task_validity=TaskValidity.UNKNOWN,
    )
    assert outcome.state is OutcomeState.UNQUALIFIED
    assert outcome.state is not OutcomeState.FAIL


@pytest.mark.parametrize("state", ("PASS", True, False, not OutcomeState.PASS, 0, 1))
def test_state_constructor_rejects_strings_truthiness_negation_and_default_zero(
    protocol: QualifiedProtocol, state: object
) -> None:
    with pytest.raises(TypeError):
        Outcome(
            item_id="task-1",
            state=state,  # type: ignore[arg-type]
            reason="forged coercion",
            protocol_digest=protocol.protocol_digest,
            provenance=protocol.provenance,
            task_validity=TaskValidity.VALID,
            scope=protocol.scope,
        )


@pytest.mark.parametrize("validity", (True, False, "VALID", 0, 1))
def test_resolved_factories_reject_truthiness_for_task_validity(
    protocol: QualifiedProtocol, validity: object
) -> None:
    with pytest.raises(TypeError):
        resolved_fail(
            item_id="task-1",
            reason="a real evaluator rejected the candidate output",
            protocol=protocol,
            task_validity=validity,  # type: ignore[arg-type]
        )


def test_outcome_and_summary_refuse_boolean_coercion(protocol: QualifiedProtocol) -> None:
    outcome = resolved_fail(
        item_id="task-1",
        reason="a real evaluator rejected the candidate output",
        protocol=protocol,
        task_validity=TaskValidity.VALID,
    )
    summary = OutcomeSummary.from_outcomes((outcome,))
    with pytest.raises(TypeError):
        bool(outcome)
    with pytest.raises(TypeError):
        bool(summary)


def test_zero_resolved_outcomes_are_not_measured_and_reasons_remain_counted(
    protocol: QualifiedProtocol,
) -> None:
    summary = OutcomeSummary.from_outcomes(
        (
            unqualified(
                item_id="task-timeout",
                reason=UnqualifiedReason.TIMEOUT,
                protocol=protocol,
                task_validity=TaskValidity.VALID,
            ),
            unqualified(
                item_id="task-refusal",
                reason=UnqualifiedReason.REFUSAL,
                protocol=protocol,
                task_validity=TaskValidity.UNKNOWN,
            ),
        )
    )
    assert summary.total_count == 2
    assert summary.resolved_count == 0
    assert summary.unqualified_count == 2
    assert summary.measurement_state == "NOT_MEASURED"
    assert summary.resolved_rate is None
    assert summary.unqualified_by_reason == {
        UnqualifiedReason.TIMEOUT: 1,
        UnqualifiedReason.REFUSAL: 1,
    }


def test_resolved_rate_excludes_unqualified_outcomes_but_keeps_their_count(
    protocol: QualifiedProtocol,
) -> None:
    summary = OutcomeSummary.from_outcomes(
        (
            resolved_pass(
                item_id="task-pass",
                reason="accepted",
                protocol=protocol,
                task_validity=TaskValidity.VALID,
            ),
            resolved_fail(
                item_id="task-fail",
                reason="rejected",
                protocol=protocol,
                task_validity=TaskValidity.VALID,
            ),
            unqualified(
                item_id="task-tie",
                reason=UnqualifiedReason.PANEL_TIE,
                protocol=protocol,
                task_validity=TaskValidity.UNKNOWN,
            ),
        )
    )
    assert summary.resolved_count == 2
    assert summary.unqualified_count == 1
    assert summary.resolved_rate == 0.5
    assert summary.measurement_state == "MEASURED"


def test_unqualified_reason_cannot_be_supplied_as_a_string(protocol: QualifiedProtocol) -> None:
    with pytest.raises(TypeError):
        unqualified(
            item_id="task-1",
            reason="timeout",  # type: ignore[arg-type]
            protocol=protocol,
            task_validity=TaskValidity.UNKNOWN,
        )


def test_raw_unqualified_state_cannot_hide_an_unnamed_reason(
    protocol: QualifiedProtocol,
) -> None:
    with pytest.raises(ValueError, match="named UnqualifiedReason"):
        Outcome(
            item_id="task-1",
            state=OutcomeState.UNQUALIFIED,
            reason="some unrecorded evaluator problem",
            protocol_digest=protocol.protocol_digest,
            provenance=protocol.provenance,
            task_validity=TaskValidity.UNKNOWN,
            scope=protocol.scope,
        )


def test_missing_required_metadata_is_rejected(provenance: EntryProvenance) -> None:
    with pytest.raises(TypeError, match="provenance"):
        unqualified(
            item_id="task-1",
            reason=UnqualifiedReason.MISSING_CHECK,
            protocol=None,
            provenance=None,
            scope=SCOPE,
            task_validity=TaskValidity.UNKNOWN,
        )
    with pytest.raises(ValueError, match="scope"):
        unqualified(
            item_id="task-1",
            reason=UnqualifiedReason.MISSING_CHECK,
            protocol=None,
            provenance=provenance,
            scope="",
            task_validity=TaskValidity.UNKNOWN,
        )
    with pytest.raises(TypeError, match="provenance"):
        Outcome(
            item_id="task-1",
            state=OutcomeState.UNQUALIFIED,
            reason=UnqualifiedReason.MISSING_CHECK.value,
            protocol_digest=None,
            provenance=None,  # type: ignore[arg-type]
            task_validity=TaskValidity.UNKNOWN,
            scope=SCOPE,
        )
