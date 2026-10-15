"""A resolved item outcome must be bound to a successful, non-vacuous qualification."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from reward_lens.contracts import EntryProvenance
from reward_lens.outcome import (
    Outcome,
    OutcomeCheck,
    OutcomeState,
    Qualification,
    SuiteResult,
    TaskValidity,
    UnqualifiedReason,
    qualified_protocol,
    qualify,
    resolved_pass,
    unqualified,
)
from reward_lens.partitions import Partition, PartitionKind

SUBJECT = "sha256:" + "a" * 64
GRADER = "sha256:" + "b" * 64
CHECK_SOURCE = "sha256:" + "c" * 64
SCOPE = "evaluator_comparison"


@pytest.fixture
def provenance() -> EntryProvenance:
    return EntryProvenance(
        started="2026-09-26T00:00:00Z", duration_s=0.25, sandbox_tier="L3", offline=True
    )


@pytest.fixture
def suite(tmp_path: Path) -> Partition:
    root = tmp_path / "acceptance"
    root.mkdir()
    (root / "case.py").write_text("assert output == 1", encoding="utf-8")
    return Partition.from_dir(
        "acceptance-suite-1",
        PartitionKind.PROTECTED_SUITE,
        root,
        log_path=tmp_path / "access.jsonl",
    )


def check() -> OutcomeCheck:
    return OutcomeCheck(
        owner="independent evaluator",
        source_digest=CHECK_SOURCE,
        interface="pytest",
        partition_id="acceptance-suite-1",
    )


def successful_qualification(suite: Partition, provenance: EntryProvenance):
    return qualify(
        check=check(),
        suite=suite,
        grader_owner="audited grader",
        grader_source_digest=GRADER,
        subject_ref=SUBJECT,
        provenance=provenance,
        result=SuiteResult(
            known_good_passed=1, known_good_total=1, known_wrong_failed=1, known_wrong_total=1
        ),
    )


def test_only_a_verified_successful_qualification_can_bridge_to_resolved_outcomes(
    suite: Partition, provenance: EntryProvenance
) -> None:
    protocol = qualified_protocol(successful_qualification(suite, provenance))
    outcome = resolved_pass(
        item_id="task-1",
        reason="the evaluator accepted the output",
        protocol=protocol,
        task_validity=TaskValidity.VALID,
    )
    assert outcome.state is OutcomeState.PASS
    assert outcome.protocol_digest == protocol.protocol_digest


def test_missing_or_failed_qualification_cannot_bridge_to_resolved_outcomes(
    suite: Partition, provenance: EntryProvenance
) -> None:
    missing = qualify(
        check=None,
        suite=suite,
        grader_owner="audited grader",
        grader_source_digest=GRADER,
        subject_ref=SUBJECT,
        provenance=provenance,
    )
    assert missing.qualified is False
    with pytest.raises(ValueError, match="successful Qualification"):
        qualified_protocol(missing)
    with pytest.raises(TypeError, match="Qualification"):
        qualified_protocol(None)  # type: ignore[arg-type]


def test_copying_a_failed_qualification_cannot_elevate_it_to_a_successful_witness(
    suite: Partition, provenance: EntryProvenance
) -> None:
    failed = qualify(
        check=None,
        suite=suite,
        grader_owner="audited grader",
        grader_source_digest=GRADER,
        subject_ref=SUBJECT,
        provenance=provenance,
    )
    successful = successful_qualification(suite, provenance)
    forged = replace(failed, qualified=True, entry=successful.entry)
    with pytest.raises(ValueError, match="successful Qualification"):
        qualified_protocol(forged)


def test_copying_a_successful_qualification_does_not_copy_its_trusted_identity(
    suite: Partition, provenance: EntryProvenance
) -> None:
    successful = successful_qualification(suite, provenance)
    copied = replace(successful)
    with pytest.raises(ValueError, match="successful Qualification"):
        qualified_protocol(copied)


def test_qualification_flag_must_be_exactly_true_even_on_a_real_registered_object(
    suite: Partition, provenance: EntryProvenance
) -> None:
    successful = successful_qualification(suite, provenance)
    forged_flag = replace(successful, qualified=1)
    with pytest.raises(ValueError, match="successful Qualification"):
        qualified_protocol(forged_flag)


def test_legacy_ninth_positional_qualification_argument_stays_partition_digest(
    suite: Partition, provenance: EntryProvenance
) -> None:
    successful = successful_qualification(suite, provenance)
    legacy_digest = "sha256:" + "d" * 64
    legacy = Qualification(
        True,
        "legacy positional construction",
        successful.entry,
        None,
        None,
        "",
        None,
        None,
        legacy_digest,
    )
    assert legacy.partition_digest == legacy_digest
    assert legacy.partition_id is None


def test_raw_public_outcome_constructor_cannot_bypass_the_qualification_witness(
    suite: Partition, provenance: EntryProvenance
) -> None:
    protocol = qualified_protocol(successful_qualification(suite, provenance))
    with pytest.raises(TypeError, match="factory"):
        Outcome(
            item_id="task-1",
            state=OutcomeState.PASS,
            reason="forged success",
            protocol_digest=protocol.protocol_digest,
            provenance=protocol.provenance,
            task_validity=TaskValidity.VALID,
            scope=protocol.scope,
        )


def test_unqualified_missing_check_needs_no_invented_qualified_protocol(
    provenance: EntryProvenance,
) -> None:
    outcome = unqualified(
        item_id="task-missing-check",
        reason=UnqualifiedReason.MISSING_CHECK,
        protocol=None,
        provenance=provenance,
        scope=SCOPE,
        task_validity=TaskValidity.UNKNOWN,
    )
    assert outcome.state is OutcomeState.UNQUALIFIED
    assert outcome.protocol_digest is None
    assert outcome.provenance is provenance


@pytest.mark.parametrize(
    "result",
    (
        SuiteResult(
            known_good_passed=0, known_good_total=0, known_wrong_failed=0, known_wrong_total=0
        ),
        SuiteResult(
            known_good_passed=True, known_good_total=1, known_wrong_failed=1, known_wrong_total=1
        ),
        SuiteResult(
            known_good_passed=1, known_good_total=1, known_wrong_failed=-1, known_wrong_total=1
        ),
        SuiteResult(
            known_good_passed=2, known_good_total=1, known_wrong_failed=1, known_wrong_total=1
        ),
    ),
)
def test_vacuous_or_invalid_control_counts_return_an_unqualified_absence(
    suite: Partition, provenance: EntryProvenance, result: SuiteResult
) -> None:
    qualification = qualify(
        check=check(),
        suite=suite,
        grader_owner="audited grader",
        grader_source_digest=GRADER,
        subject_ref=SUBJECT,
        provenance=provenance,
        result=result,
    )
    assert qualification.qualified is False
    assert qualification.entry.kind == "absence"
    assert qualification.entry.absence.state == "NOT_MEASURED"
