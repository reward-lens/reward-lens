"""D-39: the owner supplies the outcome check, or the record says OUTCOME_UNQUALIFIED.

RL0301 is P-AUDIT-1's. It is imported here with its finding id and the catalogue's own text, and
this packet neither allocates it nor writes a second message for it.
"""

from __future__ import annotations

import pytest

from reward_lens.contracts import EntryProvenance, Uncertainty
from reward_lens.errors import make
from reward_lens.outcome import OutcomeCheck, SuiteResult, qualify
from reward_lens.partitions import Partition
from reward_lens.product.audit.run import FINDING_IDS

SUBJECT = "sha256:" + "a" * 64
GRADER = "sha256:" + "b" * 64
CHECK_SOURCE = "sha256:" + "c" * 64


@pytest.fixture
def provenance() -> EntryProvenance:
    return EntryProvenance(
        started="2026-09-14T00:00:00Z", duration_s=0.5, sandbox_tier="L3", offline=True
    )


def owner_check(source: str = CHECK_SOURCE) -> OutcomeCheck:
    return OutcomeCheck(
        owner="the reward system's owner",
        source_digest=source,
        interface="pytest",
        partition_id="acceptance-suite-1",
    )


def test_no_outcome_check_leaves_the_record_unqualified(protected: Partition, provenance: EntryProvenance) -> None:
    result = qualify(
        check=None,
        suite=protected,
        grader_owner="the reward system's owner",
        grader_source_digest=GRADER,
        subject_ref=SUBJECT,
        provenance=provenance,
    )
    assert result.qualified is False
    assert result.code == "RL0301"
    assert result.state == "OUTCOME_UNQUALIFIED"
    assert result.reason == "no outcome check was supplied"


def test_the_message_and_finding_id_are_imported_not_rewritten(protected: Partition, provenance: EntryProvenance) -> None:
    result = qualify(
        check=None,
        suite=protected,
        grader_owner="the reward system's owner",
        grader_source_digest=GRADER,
        subject_ref=SUBJECT,
        provenance=provenance,
    )
    assert result.message == make("RL0301").message
    assert result.finding_id == FINDING_IDS["RL0301"]


def test_the_unqualified_record_is_an_absence_entry_not_a_zero(protected: Partition, provenance: EntryProvenance) -> None:
    entry = qualify(
        check=None,
        suite=protected,
        grader_owner="the reward system's owner",
        grader_source_digest=GRADER,
        subject_ref=SUBJECT,
        provenance=provenance,
    ).entry
    assert entry.kind == "absence"
    assert entry.section == "validity"
    assert entry.state == "absent"
    assert entry.absence.state == "NOT_MEASURED"
    assert entry.absence.missing_access == "no outcome check was supplied"
    assert entry.value is None
    assert entry.subject_ref == SUBJECT
    assert entry.provenance.sandbox_tier == "L3"


def test_a_renamed_copy_of_the_audited_grader_is_never_independent(protected: Partition, provenance: EntryProvenance) -> None:
    result = qualify(
        check=owner_check(GRADER),
        suite=protected,
        grader_owner="the reward system's owner",
        grader_source_digest=GRADER,
        subject_ref=SUBJECT,
        provenance=provenance,
        result=SuiteResult(known_good_passed=3, known_good_total=3, known_wrong_failed=2, known_wrong_total=2),
    )
    assert result.qualified is False
    assert result.code == "RL0301"
    assert result.reason == "the outcome check is the audited grader under another name"


def test_an_empty_protected_suite_fails_qualification(protected: Partition, provenance: EntryProvenance) -> None:
    for name in ("test_case_0.py", "test_case_1.py", "test_case_2.py"):
        protected.retire(name, disclosed_to="repair-agent", reason="shown in a diff")
    result = qualify(
        check=owner_check(),
        suite=protected,
        grader_owner="the reward system's owner",
        grader_source_digest=GRADER,
        subject_ref=SUBJECT,
        provenance=provenance,
        result=SuiteResult(known_good_passed=0, known_good_total=0, known_wrong_failed=0, known_wrong_total=0),
    )
    assert result.qualified is False
    assert result.reason == "the protected suite holds no item"


def test_a_known_wrong_variant_that_passes_fails_qualification(protected: Partition, provenance: EntryProvenance) -> None:
    result = qualify(
        check=owner_check(),
        suite=protected,
        grader_owner="the reward system's owner",
        grader_source_digest=GRADER,
        subject_ref=SUBJECT,
        provenance=provenance,
        result=SuiteResult(known_good_passed=3, known_good_total=3, known_wrong_failed=1, known_wrong_total=2),
    )
    assert result.qualified is False
    assert result.reason == "a known-wrong variant passed the outcome check"


def test_a_qualified_check_carries_an_estimate_entry(protected: Partition, provenance: EntryProvenance) -> None:
    result = qualify(
        check=owner_check(),
        suite=protected,
        grader_owner="the reward system's owner",
        grader_source_digest=GRADER,
        subject_ref=SUBJECT,
        provenance=provenance,
        result=SuiteResult(known_good_passed=3, known_good_total=3, known_wrong_failed=2, known_wrong_total=2),
    )
    assert result.qualified is True
    assert result.code is None
    entry = result.entry
    assert entry.kind == "estimate"
    assert entry.value == 1.0
    assert entry.n == 3
    assert entry.sampling_unit == "task"
    assert isinstance(entry.uncertainty, Uncertainty)
    assert entry.method.id == "outcome.protected_suite"


def test_the_label_is_scoped_to_the_partition_it_was_measured_on(protected: Partition, provenance: EntryProvenance) -> None:
    result = qualify(
        check=owner_check(),
        suite=protected,
        grader_owner="the reward system's owner",
        grader_source_digest=GRADER,
        subject_ref=SUBJECT,
        provenance=provenance,
        result=SuiteResult(known_good_passed=3, known_good_total=3, known_wrong_failed=2, known_wrong_total=2),
    )
    assert result.label == "outcome-qualified for protected partition acceptance-suite-1"
    assert result.partition_digest == protected.digest
