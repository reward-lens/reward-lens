"""P-RGX contract: locally seeded identifiers are records, not CVEs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from importlib import resources

import pytest

from reward_lens.commons.rgx import (
    Affected,
    Assessment,
    Catalogue,
    Dispute,
    EvidenceOrigin,
    Lifecycle,
    RGXRecord,
    allocate_id,
    seeded_records,
)

_INCIDENT_KEYS = {
    *(f"A{number}" for number in range(1, 12)),
    *(f"B{number}" for number in range(1, 12)),
    *(f"C{number}" for number in range(1, 6)),
    *(f"D{number}" for number in range(1, 6)),
    "E1",
    "E2",
    "E4",
    "F4",
}


def _record(**changes: object) -> RGXRecord:
    values: dict[str, object] = {
        "id": "RGX-2026-0042",
        "definition": {"incident": "test", "summary": "a test incident"},
        "published": "2026-09-26T00:00:00Z",
        "affected": (Affected(package="example/grader", ranges=None),),
        "evidence_origin": EvidenceOrigin.PROJECT_CATALOGUE,
        "measured_here": False,
        "evidence_source": "product-strategy-notes/02-incident-catalog.md",
        "evidence_digest": "sha256:" + "a" * 64,
    }
    values.update(changes)
    return RGXRecord(**values)  # type: ignore[arg-type]


def test_content_id_uses_canonical_definition_bytes_not_mapping_order():
    first = _record(definition={"incident": "test", "summary": "same"})
    second = _record(definition={"summary": "same", "incident": "test"})
    assert first.content_id == second.content_id
    assert first.content_id.startswith("sha256:")


def test_content_id_changes_when_the_canonical_definition_changes():
    record = _record()
    changed = _record(definition={"incident": "test", "summary": "changed"})
    assert changed.content_id != record.content_id


def test_record_definition_is_immutable_after_its_digest_is_assigned():
    record = _record()
    with pytest.raises(TypeError):
        record.definition["summary"] = "rewritten"  # type: ignore[index]


def test_all_caller_owned_collections_are_frozen_before_content_is_projected():
    aliases = ["NOT-CVE"]
    related = ["RGX-2026-0001"]
    affected = [Affected(package="example/grader", ranges=None)]
    severity = [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N"}]
    references = [{"type": "REPORT", "url": "source-a"}]
    credits = [{"name": "reporter"}]
    database_specific = {"source": {"row": "A1"}}
    dispute = Dispute(
        Assessment("original", "report-a", "2026-09-26T00:00:00Z"),
        Assessment("response", "response-b", "2026-09-27T00:00:00Z"),
    )
    record = _record(
        aliases=aliases,
        related=related,
        affected=affected,
        severity=severity,
        references=references,
        credits=credits,
        database_specific=database_specific,
    )
    disputed = record.transition(Lifecycle.PUBLISHED).dispute(dispute)
    digest = record.content_id
    aliases.append("CVE-2026-9999")
    related.append("RGX-2026-0002")
    affected.append(Affected(package="example/other", ranges=None))
    severity[0]["score"] = "rewritten"
    references[0]["url"] = "rewritten"
    credits[0]["name"] = "rewritten"
    database_specific["source"]["row"] = "rewritten"
    assert record.content_id == digest
    assert record.to_osv()["aliases"] == ["NOT-CVE"]
    assert len(record.to_osv()["affected"]) == 1
    assert record.to_osv()["severity"][0]["score"] == "CVSS:3.1/AV:N"
    assert record.to_osv()["references"][0]["url"] == "source-a"
    assert record.to_osv()["credits"][0]["name"] == "reporter"
    assert record.to_osv()["database_specific"]["source"]["row"] == "A1"
    assert len(disputed.to_osv()["database_specific"]["rgx"]["disputes"]) == 1
    projection = record.to_osv()
    projection["aliases"].append("CVE-2026-9999")
    assert "CVE-2026-9999" not in record.to_osv()["aliases"]


def test_ids_are_strict_and_allocation_never_reuses_an_existing_id():
    assert allocate_id(2026, {"RGX-2026-0001", "RGX-2026-0003"}) == "RGX-2026-0002"
    with pytest.raises(ValueError, match="RGX-YYYY-NNNN"):
        _record(id="RGX-local-0001")
    with pytest.raises(ValueError, match="collision"):
        Catalogue((_record(), _record()))


def test_unknown_affected_range_is_preserved_and_known_ranges_cannot_be_erased():
    unknown = _record()
    assert unknown.affected[0].ranges is None
    known = _record(affected=(Affected(package="example/grader", ranges=({"introduced": "1.0"},)),))
    with pytest.raises(ValueError, match="unknown"):
        known.with_affected((Affected(package="example/grader", ranges=None),))
    with pytest.raises(ValueError, match="repeat"):
        _record(
            affected=(
                Affected(package="example/grader", ranges=({"introduced": "1.0"},)),
                Affected(package="example/grader", ranges=None),
            )
        )
    two_targets = _record(
        affected=(
            Affected(package="example/grader", ranges=({"introduced": "1.0"},)),
            Affected(package="example/other", ranges=None),
        )
    )
    with pytest.raises(ValueError, match="existing"):
        two_targets.with_affected(
            (Affected(package="example/grader", ranges=({"introduced": "1.0"},)),)
        )


def test_lifecycle_refuses_illegal_edges_and_returns_a_new_immutable_record():
    draft = _record()
    private = draft.transition(Lifecycle.PRIVATE_DISCLOSURE)
    published = private.transition(Lifecycle.PUBLISHED)
    assert draft.lifecycle is Lifecycle.DRAFT
    assert published.lifecycle is Lifecycle.PUBLISHED
    fixed = published.transition(Lifecycle.FIXED)
    assert fixed.transition(Lifecycle.RE_AUDITED).lifecycle is Lifecycle.RE_AUDITED
    with pytest.raises(ValueError, match="illegal lifecycle"):
        published.transition(Lifecycle.PRIVATE_DISCLOSURE)
    with pytest.raises(TypeError, match="lifecycle"):
        replace(draft, lifecycle=Lifecycle.FIXED)
    with pytest.raises(ValueError, match="withdraw"):
        published.transition(Lifecycle.WITHDRAWN)
    withdrawn = published.withdraw(Assessment("withdrawn", "report-c", "2026-09-28T00:00:00Z"))
    assert withdrawn.lifecycle is Lifecycle.WITHDRAWN
    assert withdrawn.to_osv()["withdrawn"] == "2026-09-28T00:00:00Z"


def test_dispute_keeps_original_and_response_as_distinct_assessments():
    original = Assessment(
        text="original report", source="report-a", assessed="2026-09-26T00:00:00Z"
    )
    response = Assessment(
        text="author response", source="response-b", assessed="2026-09-27T00:00:00Z"
    )
    disputed = (
        _record()
        .transition(Lifecycle.PUBLISHED)
        .dispute(Dispute(original=original, response=response))
    )
    assert disputed.lifecycle is Lifecycle.DISPUTED
    assert disputed.disputes[0].original.text == "original report"
    assert disputed.disputes[0].response.text == "author response"
    with pytest.raises(ValueError, match="separate"):
        Dispute(original=original, response=original)


def test_malformed_records_and_unmarked_measurement_claims_are_refused():
    with pytest.raises(ValueError, match="evidence source"):
        _record(evidence_source="")
    with pytest.raises(ValueError, match="measured_here"):
        _record(measured_here=True, evidence_origin=EvidenceOrigin.PROJECT_CATALOGUE)
    with pytest.raises(ValueError, match="evidence digest"):
        _record(evidence_digest="sha256:upper-case-is-not-a-digest")
    with pytest.raises(ValueError, match="CVE"):
        _record(aliases=("CVE-2026-9999",))
    with pytest.raises(ValueError, match="dispute"):
        _record(
            disputes=[
                Dispute(
                    Assessment("original", "report-a", "2026-09-26T00:00:00Z"),
                    Assessment("response", "response-b", "2026-09-27T00:00:00Z"),
                )
            ]
        )


def test_seeded_catalogue_has_every_distinct_incident_and_its_actual_provenance():
    source = resources.files("reward_lens.commons.rgx").joinpath("incident_catalog.md").read_bytes()
    digest = "sha256:" + hashlib.sha256(source).hexdigest()
    headings = dict(re.findall(r"^([A-F][0-9]+)\.\s+(.+)$", source.decode("utf-8"), re.MULTILINE))
    aliases = dict(re.findall(r"\b([A-F][0-9]+)\s*=\s*([A-F][0-9]+)\b", source.decode("utf-8")))
    assert digest == "sha256:ad5847d1531cf2845a62b3af2d1fc7535357feed42788c8d4baaffc9007d66bf"
    assert aliases == {"F1": "A4", "F2": "A10", "E3": "A11", "F3": "A11"}
    assert len(headings) == 40
    records = seeded_records()
    assert len(records) == 36
    assert len({record.id for record in records}) == 36
    assert {record.definition["incident_key"] for record in records} == _INCIDENT_KEYS
    for record in records:
        assert record.lifecycle is Lifecycle.PUBLISHED
        assert record.affected[0].ranges is None
        assert record.evidence_origin is EvidenceOrigin.PROJECT_CATALOGUE
        assert record.measured_here is False
        assert record.evidence_source == "src/reward_lens/commons/rgx/incident_catalog.md"
        assert (
            record.evidence_digest
            == "sha256:ad5847d1531cf2845a62b3af2d1fc7535357feed42788c8d4baaffc9007d66bf"
        )
        assert record.definition["source_heading"] == headings[record.definition["incident_key"]]
        assert (
            record.database_specific["source_origin"]
            == "product-strategy-notes/02-incident-catalog.md"
        )
        assert record.references == (
            {
                "type": "REPORT",
                "url": f"{record.evidence_source}#{record.definition['incident_key']}",
            },
        )
        osv = record.to_osv()
        assert set(
            (
                "id",
                "aliases",
                "related",
                "published",
                "modified",
                "severity",
                "references",
                "credits",
                "affected",
                "database_specific",
            )
        ) <= set(osv)
        assert "ranges" not in osv["affected"][0]
        assert osv["database_specific"]["rgx"]["content_id"] == record.content_id
