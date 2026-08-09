"""The error type and the four refusal cases this packet owns."""

from __future__ import annotations

import copy
import json

import pytest

from reward_lens.contracts import (
    Assay,
    ProjectConfig,
    RewardLensError,
    canonical_bytes,
    validate_record,
)
from reward_lens.contracts.errors import (
    BudgetExceeded,
    CapabilityUnavailable,
    EntryKindEvidenceMissing,
    InternalFailure,
    NumberNotQuantised,
    PendingDecision,
    RecordInvalid,
    SchemaUnknownMajor,
    UsageError,
)


def test_the_base_error_carries_the_five_fields():
    err = RewardLensError(
        code="RL0604",
        message="a record was refused",
        remediation="run reward-lens audit again",
        context={"where": "here"},
    )
    assert err.code == "RL0604"
    assert err.exit_code == 7
    assert json.loads(err.to_json()) == {
        "code": "RL0604",
        "message": "a record was refused",
        "remediation": "run reward-lens audit again",
    }


@pytest.mark.parametrize(
    ("cls", "exit_code"),
    [
        (UsageError, 4),
        (CapabilityUnavailable, 5),
        (BudgetExceeded, 6),
        (InternalFailure, 7),
    ],
)
def test_the_family_is_keyed_by_exit_code(cls, exit_code):
    err = cls(code="RL0001", message="m", remediation="r")
    assert err.exit_code == exit_code
    assert isinstance(err, RewardLensError)


def test_pending_decision_carries_the_command_it_is_waiting_on():
    err = PendingDecision(
        code="RL0801",
        message="a decision is pending",
        remediation="answer it",
        argv=["reward-lens", "audit", "."],
        command="audit",
        decision="qualify",
    )
    assert err.exit_code == 3
    assert err.argv == ["reward-lens", "audit", "."]
    assert err.command == "audit"
    assert err.decision == "qualify"


# --- RL0601 ------------------------------------------------------------------------------------


def test_schema_unknown_major_preserves_the_original_bytes(reference):
    record = copy.deepcopy(reference)
    record["$schema"] = "https://reward-lens.github.io/schema/assay/2.0/assay.schema.json"
    raw = json.dumps(record).encode("utf-8")
    with pytest.raises(SchemaUnknownMajor) as excinfo:
        validate_record(record, raw=raw)
    err = excinfo.value
    assert err.code == "RL0601"
    assert err.exit_code == 4
    assert err.original_bytes == raw
    assert err.context["original_bytes"] == raw
    assert "2" in err.message
    assert err.remediation


def test_a_minor_bump_inside_the_major_is_not_refused_by_the_major_check(reference):
    record = copy.deepcopy(reference)
    record["$schema"] = "https://reward-lens.github.io/schema/assay/1.7/assay.schema.json"
    with pytest.raises(RecordInvalid):
        validate_record(record)


# --- RL0602 ------------------------------------------------------------------------------------


def test_entry_kind_evidence_missing_names_the_kind_and_the_field(reference):
    record = copy.deepcopy(reference)
    witness = next(
        e for e in record["measurement"]["validity"] if e["kind"] == "witness"
    )
    del witness["witness"]["observed"]
    with pytest.raises(EntryKindEvidenceMissing) as excinfo:
        validate_record(record)
    err = excinfo.value
    assert err.code == "RL0602"
    assert err.context["kind"] == "witness"
    assert err.context["missing"] == "observed"
    assert "witness" in err.message and "observed" in err.message


def test_an_estimate_without_its_uncertainty_names_uncertainty(reference):
    record = copy.deepcopy(reference)
    estimate = next(
        e for e in record["measurement"]["soundness"] if e["kind"] == "estimate"
    )
    del estimate["uncertainty"]
    with pytest.raises(EntryKindEvidenceMissing) as excinfo:
        validate_record(record)
    assert excinfo.value.context["kind"] == "estimate"
    assert excinfo.value.context["missing"] == "uncertainty"


# --- RL0603 ------------------------------------------------------------------------------------


def test_number_not_quantised_names_the_field(reference):
    record = copy.deepcopy(reference)
    record["calibration_links"][0]["transfer_gap"] = 0.4189832345678
    with pytest.raises(NumberNotQuantised) as excinfo:
        validate_record(record)
    err = excinfo.value
    assert err.code == "RL0603"
    assert err.context["field"] == "calibration_links[0].transfer_gap"
    assert err.context["scale"] == 6
    assert "calibration_links[0].transfer_gap" in err.message


# --- RL0604 ------------------------------------------------------------------------------------


def test_record_invalid_carries_the_schema_path(reference):
    record = copy.deepcopy(reference)
    record["measurement"]["validity"][0]["scope"] = "vibes"
    with pytest.raises(RecordInvalid) as excinfo:
        validate_record(record)
    err = excinfo.value
    assert err.code == "RL0604"
    assert err.exit_code == 4
    assert err.context["schema_path"]
    assert err.context["instance_path"] == [
        "measurement",
        "validity",
        0,
        "scope",
    ]
    assert "scope" in err.message


# --- malformed calls answer in a reserved code, not in a traceback ------------------------------
#
# These are the calls the errors packet drives through the handler a caller meets. A bare
# exception there is a failed case: the handler renders two lines and an exit code from a
# `RewardLensError`, and anything else reaches the caller as a traceback with no remediation.

MALFORMED = [
    ("validate an empty record", lambda: validate_record({}), "RL0604"),
    ("validate something that is not a record", lambda: validate_record("not a record"), "RL0604"),
    ("build a record from nothing", lambda: Assay.model_validate({}), "RL0604"),
    ("build a record from a list", lambda: Assay.model_validate([]), "RL0604"),
    ("canonicalise a value with no JSON form", lambda: canonical_bytes(object()), "RL0604"),
    (
        "read a project file with a bad reward kind",
        lambda: ProjectConfig.model_validate({"reward": {"kind": "nope"}}),
        "RL0003",
    ),
]


@pytest.mark.parametrize("case,call,code", MALFORMED, ids=[row[0] for row in MALFORMED])
def test_a_malformed_call_raises_the_reserved_class_and_names_the_place(case, call, code):
    with pytest.raises(RewardLensError) as excinfo:
        call()
    error = excinfo.value
    assert error.code == code, case
    assert error.exit_code == 4, case
    named = error.context.get("instance_path", error.context.get("field"))
    assert named is not None, f"{case}: refused, but named no place: {error.context}"
    assert error.remediation, case


def test_a_record_from_a_later_major_is_refused_as_unreadable_not_as_invalid(reference):
    """The sixth case the driver runs: a major this build cannot read is RL0601, not RL0604.

    The distinction is the point of D-64. RL0604 says the record is wrong; RL0601 says the record
    may be perfectly good and this build cannot interpret it, which is why the original bytes are
    handed back unaltered.
    """
    later = reference["$schema"].replace("/1.0/", "/9.0/")
    with pytest.raises(RewardLensError) as excinfo:
        validate_record({"$schema": later})
    assert excinfo.value.code == "RL0601"
    assert excinfo.value.exit_code == 4
