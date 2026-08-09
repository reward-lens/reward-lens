"""The honest record: what a verb returns when its instrument is not in this build.

A hole is a first-class state. The record validates, it says what is missing and what would fix
it, and it never fabricates a digest or a zero in place of a measurement.

The subject here is the fallback, not the build's inventory of engines: every verb below has its
engine removed through the seam first, so each rule holds for a verb whose packet has landed and
for one whose packet has not. `test_dry_run_...` is the exception and says why.
"""

from __future__ import annotations

import pytest

from reward_lens import api, contracts
from reward_lens.api import _dispatch

from .conftest import RECORD_VERBS, SECTIONS, engines_absent

MISSING = "instrument not in this build"
ZERO_DIGEST = "sha256:" + "0" * 64


@pytest.fixture
def records(
    requests_by_verb: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> dict[str, contracts.Assay]:
    """Every record verb answering without its engine, which is what this file is about."""
    engines_absent(monkeypatch, *RECORD_VERBS)
    for verb in RECORD_VERBS:
        assert _dispatch.engine(verb) is None
    return {verb: getattr(api, verb)(req) for verb, req in requests_by_verb.items()}


@pytest.mark.parametrize("verb", RECORD_VERBS)
def test_the_verb_returns_an_assay_and_never_raises(verb, records) -> None:
    assert isinstance(records[verb], contracts.Assay)


@pytest.mark.parametrize("verb", RECORD_VERBS)
def test_the_record_validates_under_the_frozen_schema(verb, records) -> None:
    contracts.validate_record(records[verb].to_dict())


@pytest.mark.parametrize("verb", RECORD_VERBS)
def test_the_holes_index_carries_one_entry_per_section(verb, records) -> None:
    holes = records[verb].holes
    assert [h.section for h in holes] == list(SECTIONS)
    assert len(holes) == 10


@pytest.mark.parametrize("verb", RECORD_VERBS)
def test_every_hole_names_the_missing_access_and_a_remedy(verb, records) -> None:
    for hole in records[verb].holes:
        assert hole.missing_access == MISSING
        assert hole.remedy.strip()
        assert hole.state == "NOT_MEASURED"
        assert hole.affected_claims


@pytest.mark.parametrize("verb", RECORD_VERBS)
def test_every_entry_is_an_absence_and_the_index_is_rebuilt_from_them(verb, records) -> None:
    record = records[verb]
    entries = record.entries()
    assert len(entries) == 10
    assert {e.kind for e in entries} == {"absence"}
    before = [h.model_dump() for h in record.holes]
    assert [h.model_dump() for h in record.holes_from_entries()] == before


@pytest.mark.parametrize("verb", RECORD_VERBS)
def test_the_assay_id_is_the_digest_of_the_record(verb, records) -> None:
    record = records[verb]
    assert record.assay_id == contracts.digest(record)


@pytest.mark.parametrize("verb", RECORD_VERBS)
def test_nothing_in_the_record_is_a_fabricated_zero_digest(verb, records) -> None:
    record = records[verb]
    assert record.subject.version.digest != ZERO_DIGEST
    assert record.subject.digests.instrument_method != ZERO_DIGEST
    for entry in record.entries():
        assert entry.subject_ref != ZERO_DIGEST


@pytest.mark.parametrize("verb", RECORD_VERBS)
def test_the_record_says_no_subject_was_read(verb, records) -> None:
    record = records[verb]
    assert record.subject.version.id == "unread"
    assert record.subject.digests.source is None
    assert any("no subject was read" in lim for lim in record.entries()[0].limitations)


@pytest.mark.parametrize("verb", RECORD_VERBS)
def test_the_decision_is_unresolved_and_names_typed_reasons(verb, records) -> None:
    decision = records[verb].decision
    assert decision.state == "unresolved"
    assert decision.action is None
    assert decision.policy is None
    assert "required_missing:subject" in decision.reasons
    for section in SECTIONS:
        assert f"required_missing:{section}" in decision.reasons


@pytest.mark.parametrize("verb", RECORD_VERBS)
def test_the_provenance_is_the_provenance_of_this_run(verb, records) -> None:
    provenance = records[verb].provenance
    assert provenance.offline is True
    assert provenance.sandbox_tier == "T0"
    assert provenance.reproduce and all(r.strip() for r in provenance.reproduce)
    assert records[verb].cost.usd == "0.00"
    assert records[verb].cost.api_calls == 0


@pytest.mark.parametrize("verb", RECORD_VERBS)
def test_the_entry_provenance_is_not_the_epoch(verb, records) -> None:
    for entry in records[verb].entries():
        assert not entry.provenance.started.startswith("1970-")


def test_two_calls_of_the_same_request_agree_outside_the_volatile_fields(
    requests_by_verb, monkeypatch
) -> None:
    """The honest record is a function of the request, not of the clock.

    `assay_id` is volatile along with the timestamps, because it is the digest of a record that
    carries them: two calls a second apart hash differently and are right to. What must not move
    is everything else, and each id must still be the digest of its own record.
    """
    engines_absent(monkeypatch, "audit")
    first, second = api.audit(requests_by_verb["audit"]), api.audit(requests_by_verb["audit"])
    assert first.assay_id == contracts.digest(first)
    assert second.assay_id == contracts.digest(second)
    one, two = first.to_dict(), second.to_dict()
    for blob in (one, two):
        blob.pop("created")
        blob.pop("cost")
        blob.pop("assay_id")
        blob.pop("environment_excluded_from_digest", None)
        for entry in (e for section in blob["measurement"].values() for e in section):
            entry["provenance"].pop("started")
    assert one == two


def test_dry_run_promises_what_the_run_fills_and_nothing_it_cannot_measure(
    requests_by_verb, monkeypatch
) -> None:
    """Both halves of the promise, against the build rather than against a constant.

    With no engine the plan promises no panel at all. With the engine in the build it promises
    exactly the panels the run then fills with a measurement: every promised panel comes back
    carrying a real entry, and no section the plan passed over does. Either way the dry run
    promises no spend, because a plan that costs money is not a plan.
    """
    request = requests_by_verb["audit"]

    with monkeypatch.context() as absent:
        engines_absent(absent, "dry_run")
        empty = api.dry_run(request)
    assert isinstance(empty, api.Plan)
    assert empty.panels == ()
    assert empty.paid_calls == 0
    assert empty.estimate_usd == "0.00"
    assert empty.notes

    plan = api.dry_run(request)
    assert isinstance(plan, api.Plan)
    assert plan.paid_calls == 0
    assert plan.estimate_usd == "0.00"
    assert plan.notes
    assert set(plan.panels) <= set(SECTIONS)

    measured = api.audit(request).to_dict()["measurement"]
    filled = {
        section
        for section, entries in measured.items()
        if any(entry["kind"] != "absence" for entry in entries)
    }
    assert filled == set(plan.panels)


def test_doctor_reports_rather_than_raising(monkeypatch) -> None:
    """Either way round: with the access engine in the build, and with the SDK answering alone.

    P-DOCTOR's engine is now in the build, so `doctor()` reports what it measured. The absence
    shape below is the SDK's own fallback, reached by pointing the verb at a module nobody ships.
    """
    assert isinstance(api.doctor(), api.Capabilities)

    monkeypatch.setitem(_dispatch.ENGINES, "doctor", "reward_lens.product.nothing_here:doctor")
    capabilities = api.doctor()
    assert isinstance(capabilities, api.Capabilities)
    assert len(capabilities) == len(list(capabilities))
    assert all(c.status == "unavailable" for c in capabilities)
    assert all(c.reason for c in capabilities)
    assert capabilities.sandbox is None
    assert capabilities.install.version
