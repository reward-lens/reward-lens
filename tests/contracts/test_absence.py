"""`absence()` records a hole; it never fabricates the evidence around one (A-004).

An absence entry is the record's answer to "this was not measured". It still has to say which
subject the attempt was against and when the attempt happened, and those are facts only the
caller holds. The frozen signature supplied four of the eleven fields an `Entry` requires, and
the first attempt filled the rest in here: a zero digest for the subject, the epoch for the start
time, T0 for the sandbox tier. An entry written that way claims a start time and a tier that never
happened, which is the one thing an honest absence exists to avoid. A-004 amends the interface
instead: the frozen six parameters stay, `subject_ref` and `provenance` become required keyword
arguments, and there is no default for either, nor for `started` or `sandbox_tier` inside the
provenance the caller builds.
"""

from __future__ import annotations

import inspect

import pytest

from reward_lens.contracts import Assay, absence, digest
from reward_lens.contracts.models import (
    ABSENCE_METHOD_VERSION,
    ZERO_DIGEST,
    EntryProvenance,
    Method,
)

SUBJECT = "sha256:" + "ab" * 32
PROVENANCE = EntryProvenance(
    started="2026-09-13T09:41:07Z", duration_s=1.25, sandbox_tier="L1", offline=True
)


def an_entry(**kwargs):
    return absence(
        "reach",
        "reach.held_out_transfer",
        "the scorer's agreement on the held-out split",
        "the held-out split was not released with the task set",
        "request the split, or re-run once it is public",
        ("reach.transfer",),
        subject_ref=SUBJECT,
        provenance=PROVENANCE,
        **kwargs,
    )


def test_absence_has_no_default_for_subject_or_provenance():
    """Omitting either is a TypeError at the call, not a zero digest written into a record."""
    frozen_six = (
        "reach",
        "reach.held_out_transfer",
        "the scorer's agreement on the held-out split",
        "the held-out split was not released with the task set",
        "request the split, or re-run once it is public",
        ("reach.transfer",),
    )
    with pytest.raises(TypeError):
        absence(*frozen_six)
    with pytest.raises(TypeError):
        absence(*frozen_six, subject_ref=SUBJECT)
    with pytest.raises(TypeError):
        absence(*frozen_six, provenance=PROVENANCE)

    parameters = inspect.signature(absence).parameters
    for required in ("subject_ref", "provenance"):
        assert parameters[required].default is inspect.Parameter.empty
        assert parameters[required].kind is inspect.Parameter.KEYWORD_ONLY
    for gone in ("started", "sandbox_tier", "offline"):
        assert gone not in parameters, f"{gone} is the caller's fact, not this function's default"
    fields = EntryProvenance.model_fields
    for required in ("started", "duration_s", "sandbox_tier", "offline"):
        assert fields[required].is_required(), required


def test_the_frozen_six_parameters_are_still_positional():
    """A-004 amends the signature; it does not rewrite the call every caller already writes."""
    positional = [
        name
        for name, p in inspect.signature(absence).parameters.items()
        if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert positional == [
        "section",
        "entry_id",
        "measurand",
        "missing_access",
        "remedy",
        "affected_claims",
    ]


def test_the_entry_carries_what_the_caller_supplied():
    entry = an_entry()
    assert entry.kind == "absence"
    assert entry.state == "absent"
    assert entry.subject_ref == SUBJECT
    assert entry.provenance.started == "2026-09-13T09:41:07Z"
    assert entry.provenance.sandbox_tier == "L1"
    assert entry.provenance.duration_s == 1.25
    assert entry.absence.state == "NOT_MEASURED"
    assert entry.absence.affected_claims == ["reach.transfer"]


def test_the_entry_validates_as_an_entry_of_the_record(reference):
    """The schema is the judge: the entry the helper builds is one the record admits, and the
    hole index rebuilt from it names the same id."""
    record = dict(reference)
    record["measurement"] = {**reference["measurement"], "reach": [an_entry().to_dict()]}
    assay = Assay.model_validate(record)
    holes = assay.holes_from_entries()
    assert [hole.entry_id for hole in holes if hole.section == "reach"] == [
        "reach.held_out_transfer"
    ]


def test_the_absence_method_is_the_absence_procedure_and_its_digest_is_computed():
    """An entry's method says what was done, so an absence's method is the absence procedure.

    The first implementation wrote `ZERO_DIGEST` there. A constant says that every absence was
    reached the same way, which makes the digest evidence of nothing: two absences with different
    missing access, different remedy and different affected claims would hash identically, and a
    reader comparing two records could not tell them apart at the one field that is supposed to
    fingerprint the procedure's parameters.
    """
    entry = an_entry()
    assert entry.method.id == "reach.absence"
    assert entry.method.version == ABSENCE_METHOD_VERSION
    assert entry.method.params_digest != ZERO_DIGEST
    assert entry.method.params_digest == digest(
        {
            "missing_access": "the held-out split was not released with the task set",
            "remedy": "request the split, or re-run once it is public",
            "affected_claims": ["reach.transfer"],
        }
    )


@pytest.mark.parametrize("position", [3, 4, 5])
def test_two_absences_that_differ_in_any_parameter_differ_in_their_digest(position):
    """One digest per set of parameters: `missing_access`, `remedy`, `affected_claims`."""
    base = [
        "reach",
        "reach.held_out_transfer",
        "the scorer's agreement on the held-out split",
        "the held-out split was not released with the task set",
        "request the split, or re-run once it is public",
        ("reach.transfer",),
    ]
    other = list(base)
    other[position] = ("reach.other_claim",) if position == 5 else base[position] + " (revised)"
    first = absence(*base, subject_ref=SUBJECT, provenance=PROVENANCE)
    second = absence(*other, subject_ref=SUBJECT, provenance=PROVENANCE)
    assert first.method.params_digest != second.method.params_digest
    assert absence(*base, subject_ref=SUBJECT, provenance=PROVENANCE).method.params_digest == (
        first.method.params_digest
    )


def test_the_caller_may_still_supply_a_method_and_it_is_not_overwritten():
    """`absence()` computes the absence procedure only where the caller named none."""
    supplied = Method(
        id="reach.replay",
        version="2.1.0",
        params_digest="sha256:" + "cd" * 32,
        procedure="the run was replayed against the recorded seed",
    )
    assert an_entry(method=supplied).method == supplied
