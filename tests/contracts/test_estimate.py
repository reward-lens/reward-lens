"""`estimate()` carries a measured number, and carries nothing it was not given.

An estimate entry is the record's answer to "this was measured, and here is how well". The
frozen schema makes six fields conditional on `kind: estimate` beyond the eleven every entry
carries: `value`, `unit`, `n`, `sampling_unit`, `uncertainty` and `result`. Each is a fact only
the instrument holds, so each is a parameter with no default here, and omitting one is a
`TypeError` at the call rather than a record the schema refuses three layers later.

Two things separate this constructor from `absence()`. An absence derives its own method, because
the thing that was done is the absence procedure itself; an estimate's method is the instrument's,
so `method` is required and is stored exactly as it arrives, `params_digest` included. And the
uncertainty arrives built: under D-75 a measurement below about fifteen clusters reports no
interval and says why, so the unbounded block carries `reason` and `clusters` and no interval at
all, and nothing here invents endpoints to fill the field.
"""

from __future__ import annotations

import inspect

import pytest

from reward_lens.contracts import (
    Assay,
    Uncertainty,
    canonical_bytes,
    estimate,
    validate_record,
)
from reward_lens.contracts.models import KIND_EVIDENCE, Entry, EntryProvenance, Method

SUBJECT = "sha256:" + "ab" * 32
PROVENANCE = EntryProvenance(
    started="2026-09-13T09:41:07Z", duration_s=1.25, sandbox_tier="L1", offline=True
)
METHOD = Method(
    id="rl.soundness.known_good",
    version="3.1.0",
    params_digest="sha256:" + "cd" * 32,
    procedure="score the known-good solutions; Clopper-Pearson interval",
)
BOUNDED = Uncertainty(interval=[0.7354, 1.0], method="clopper_pearson", level=0.95)
UNBOUNDED = Uncertainty(
    method="no_interval_below_15_clusters",
    level=0.95,
    reason="9 task clusters, below the 15 at which rule 9 stops reporting an interval",
    clusters=9,
    cluster_unit="task",
)
RESULT = {"accepted": 12, "scored": 12}
#: The eight the sketch takes positionally, in order, so a call written against the interface
#: keeps working when a keyword is added after it.
POSITIONAL = (
    "reach",
    "reach.known_good_accept_rate",
    "fraction of known-good solutions the grader accepts",
    1.0,
    "fraction",
    12,
    "task",
)


def an_estimate(**kwargs) -> Entry:
    return estimate(
        *POSITIONAL,
        kwargs.pop("uncertainty", BOUNDED),
        subject_ref=SUBJECT,
        provenance=PROVENANCE,
        method=kwargs.pop("method", METHOD),
        result=kwargs.pop("result", RESULT),
        depends_on=kwargs.pop("depends_on", ("digest:source", "digest:outcome_protocol")),
        **kwargs,
    )


def record_with(entry: Entry, reference: dict) -> dict:
    record = dict(reference)
    record["measurement"] = {**reference["measurement"], "reach": [entry.to_dict()]}
    return record


# --- the two estimates the schema admits ---------------------------------------------------------


@pytest.mark.parametrize("uncertainty", [BOUNDED, UNBOUNDED], ids=["bounded", "unbounded"])
def test_the_estimate_validates_inside_the_record_and_round_trips(uncertainty, reference):
    """The schema is the judge, and the bytes are a function of the values, not of the path.

    `validate_record` is the gate every record passes on its way in, so the entry is put where an
    entry lives and the whole record is offered to it. Then the record is loaded back through the
    models and re-serialised: `canonical_bytes` has to give the same answer for the record built
    here and the record parsed from it, or an SDK-built assay and the same assay read from disk
    would hash differently (D-10).
    """
    entry = an_estimate(uncertainty=uncertainty)
    record = record_with(entry, reference)

    validate_record(record)

    assert canonical_bytes(Assay.model_validate(record).to_dict()) == canonical_bytes(record)
    data = entry.to_dict()
    assert Entry.model_validate(data).to_dict() == data


def test_the_bounded_estimate_carries_its_interval_and_the_unbounded_one_carries_none():
    """D-75: below about fifteen clusters the record states that there is no interval and why."""
    bounded = an_estimate().to_dict()
    assert bounded["uncertainty"]["interval"] == [0.7354, 1.0]
    assert an_estimate().uncertainty.bounded is True

    unbounded = an_estimate(uncertainty=UNBOUNDED)
    assert unbounded.uncertainty.bounded is False
    block = unbounded.to_dict()["uncertainty"]
    assert "interval" not in block, "an unbounded estimate never carries an interval"
    assert block["reason"] == UNBOUNDED.reason
    assert block["clusters"] == 9
    #: The value itself is untouched by the absence of an interval: the measurement happened.
    assert unbounded.value == 1.0


def test_an_unbounded_interval_cannot_be_smuggled_in_beside_the_stated_absence():
    """The one way to get an interval is to name the method that produced it."""
    with pytest.raises(ValueError):
        Uncertainty(
            method="no_interval_below_15_clusters",
            level=0.95,
            reason="9 task clusters",
            clusters=9,
            interval=[0.0, 1.0],
        )


# --- the required set is the schema's, and it is required at the call ----------------------------


def test_every_field_the_kind_requires_is_a_parameter_with_no_default():
    """`KIND_EVIDENCE["estimate"]` is read out of the schema's conditional; this follows it."""
    parameters = inspect.signature(estimate).parameters
    for required in (*KIND_EVIDENCE["estimate"], "subject_ref", "provenance", "method"):
        assert required in parameters, required
        assert parameters[required].default is inspect.Parameter.empty, required
    for keyword_only in ("subject_ref", "provenance", "method", "result"):
        assert parameters[keyword_only].kind is inspect.Parameter.KEYWORD_ONLY, keyword_only
    positional = [
        name
        for name, p in parameters.items()
        if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert positional == [
        "section",
        "entry_id",
        "measurand",
        "value",
        "unit",
        "n",
        "sampling_unit",
        "uncertainty",
    ]


@pytest.mark.parametrize("omitted", ["subject_ref", "provenance", "method", "result"])
def test_a_missing_required_keyword_is_a_typeerror_at_the_call(omitted):
    """Not a record refusal later: the call that cannot be completed is the one that fails."""
    supplied = {
        "subject_ref": SUBJECT,
        "provenance": PROVENANCE,
        "method": METHOD,
        "result": RESULT,
    }
    del supplied[omitted]
    with pytest.raises(TypeError):
        estimate(*POSITIONAL, BOUNDED, **supplied)


@pytest.mark.parametrize("keep", [3, 4, 5, 6, 7])
def test_a_missing_positional_measurement_field_is_a_typeerror_at_the_call(keep):
    """`value`, `unit`, `n`, `sampling_unit` and `uncertainty` are the instrument's to supply."""
    head = (*POSITIONAL, BOUNDED)[:keep]
    with pytest.raises(TypeError):
        estimate(*head, subject_ref=SUBJECT, provenance=PROVENANCE, method=METHOD, result=RESULT)


def test_estimate_derives_no_method_of_its_own():
    """Unlike `absence()`, there is no default method and none is computed."""
    assert inspect.signature(estimate).parameters["method"].default is inspect.Parameter.empty


# --- the caller's method survives ----------------------------------------------------------------


def test_the_callers_method_is_stored_untouched_and_its_digest_is_not_recomputed():
    """An estimate's method is the instrument's. Recomputing the digest here would fingerprint
    this constructor's arguments instead of the parameters the instrument actually ran under, so
    two estimates from one instrument would stop agreeing at the one field that identifies it."""
    entry = an_estimate()
    assert entry.method == METHOD
    assert entry.method.params_digest == "sha256:" + "cd" * 32
    assert entry.method.id == "rl.soundness.known_good"
    assert entry.method.version == "3.1.0"
    assert entry.method.procedure == METHOD.procedure

    other = estimate(
        "reach",
        "reach.other_rate",
        "a different measurand entirely",
        0.5,
        "fraction",
        40,
        "episode",
        BOUNDED,
        subject_ref=SUBJECT,
        provenance=PROVENANCE,
        method=METHOD,
        result={"accepted": 20, "scored": 40},
    )
    assert other.method.params_digest == entry.method.params_digest
    assert entry.to_dict()["method"] == METHOD.to_dict()


# --- what the caller did not supply is absent, not empty -----------------------------------------


def test_the_optional_fields_the_caller_left_alone_are_absent_from_the_record():
    """A default that serialises as absent, not as null and not as an empty analysis."""
    data = an_estimate().to_dict()
    for optional in (
        "denominator",
        "power",
        "exclusions",
        "assumptions",
        "verdicts",
        "rate_definition",
        "extensions",
    ):
        assert optional not in data, optional
    assert data["depends_on"] == ["digest:source", "digest:outcome_protocol"]
    assert data["limitations"] == []
    assert data["scope"] == "evaluator_comparison"
    assert data["state"] == "complete"
    assert data["kind"] == "estimate"


def test_the_optional_fields_the_caller_supplies_reach_the_record(reference):
    """And the record with them in it is still one the schema admits."""
    entry = an_estimate(
        denominator="known-good items scored",
        rate_definition="other",
        verdicts={"scored": 12},
        exclusions=("two items with malformed prompts",),
        assumptions=("the twelve known-good solutions are independent",),
        limitations=("12 of 12 gives a 95% lower bound of 0.76 and nothing stronger",),
        state="partial",
    )
    validate_record(record_with(entry, reference))
    data = entry.to_dict()
    assert data["denominator"] == "known-good items scored"
    assert data["rate_definition"] == "other"
    assert data["verdicts"] == {"scored": 12}
    assert data["exclusions"] == ["two items with malformed prompts"]
    assert data["assumptions"] == ["the twelve known-good solutions are independent"]
    assert data["state"] == "partial"
