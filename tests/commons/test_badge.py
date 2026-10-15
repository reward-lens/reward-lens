"""P-BADGE: record-linked D-52 endpoint payloads use canonical Assay fields."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]
REFERENCE = ROOT / "schema" / "assay" / "1.0" / "fixtures" / "reference.json"


def _assay(*, reaches: int = 1, exploit_findings: int = 0):
    from reward_lens import contracts

    assay = contracts.Assay.model_validate(json.loads(REFERENCE.read_text(encoding="utf-8")))
    provenance = contracts.EntryProvenance(
        started="2026-09-12T09:14:02Z",
        duration_s=0,
        sandbox_tier="L0",
        offline=True,
        arm="black_box",
    )
    for index in range(reaches):
        entry = contracts.estimate(
            "reach",
            f"reach.surfaces.surface_{index}",
            "an executed surface probe",
            0.82 if index == 0 else 0.15,
            "fraction",
            4,
            "probe attempt",
            contracts.Uncertainty(method="wilson", level=0.95, interval=[0.1, 0.9]),
            subject_ref=assay.subject.version.digest,
            provenance=provenance,
            method=contracts.Method(
                id="reach.surfaces",
                version="1.0.0",
                params_digest=contracts.digest({"test": "badge"}),
                procedure="recorded probe",
            ),
            result={"surface": f"surface_{index}", "reached": bool(index == 0)},
        )
        assay.measurement.reach.append(entry)
    if exploit_findings:
        entry = contracts.Entry(
            entry_id="exploits.seeker.local.witness_0",
            section="exploits",
            kind="witness",
            measurand="reproduced seeker exploit",
            method=contracts.Method(
                id="exploits.seeker",
                version="1.0.0",
                params_digest=contracts.digest({"test": "seeker"}),
                procedure="reproduced pair",
            ),
            scope="evaluator_comparison",
            subject_ref=assay.subject.version.digest,
            depends_on=[],
            state="complete",
            provenance=provenance,
            limitations=[],
            result={"outcome_passed": False, "outcome_evidence": {"state": "FAIL"}},
            witness=contracts.Witness(
                inputs={"task_id": "fixture"}, procedure="reproduce", observed="outcome failed"
            ),
        )
        assay.measurement.exploits.append(entry)
        for _ in range(exploit_findings):
            assay.findings.append(
                contracts.Finding(
                    id="RGX-local-0071",
                    rule=entry.entry_id,
                    level="error",
                    kind="fail",
                    severity_rationale="typed reproduced seeker witness",
                    scope="evaluator_comparison",
                    entries=[entry.entry_id],
                    partial_fingerprint="badge-seeker-finding",
                )
            )
    return assay


def test_badge_projects_one_selected_canonical_reach_and_reopens_its_payload(
    tmp_path: Path,
) -> None:
    from reward_lens.commons.badge import (
        FORBIDDEN_WORDS,
        badge_from_assay,
        read_payload,
        write_payload,
    )

    assay = _assay(exploit_findings=2)
    payload = badge_from_assay(
        assay, record_url="file:///records/assay.json", now="2026-09-12T12:00:00Z"
    )
    written = write_payload(payload, tmp_path / "badge.json")

    assert read_payload(written) == payload
    assert payload["schemaVersion"] == 1
    assert payload["label"] == "reward assay"
    assert payload["message"] == "reach 0.82 | exploits 1 | 2026-09-12"
    assert payload["color"] == "blue"
    assert payload["labelColor"] == "grey"
    assert payload["isError"] is False
    assert payload["cacheSeconds"] == 3600
    assert payload["link"] == "file:///records/assay.json"
    assert payload["record"]["finding_ids"] == ["RGX-local-0071"]
    assert payload["provenance"] == {
        "reach_entry_id": "reach.surfaces.surface_0",
        "reach_surface": "surface_0",
        "reach_pointer": "/measurement/reach/0",
    }
    words = {
        word
        for value in (payload["label"], payload["message"])
        for word in value.lower().replace("|", " ").split()
    }
    assert not words & FORBIDDEN_WORDS


@pytest.mark.parametrize(
    ("state", "color", "is_error"),
    [
        ("DISPUTED", "orange", True),
        ("WITHDRAWN", "red", True),
    ],
)
def test_badge_reopens_a_typed_linked_record_lifecycle(
    state, color, is_error, tmp_path: Path
) -> None:
    import rfc8785

    from reward_lens import contracts
    from reward_lens.commons.badge import LINKED_RECORD_EXTENSION, badge_from_path

    assay = _assay()
    assay.extensions[LINKED_RECORD_EXTENSION] = {
        "url": "file:///records/rgx.json",
        "record": {"id": "RGX-2026-0001", "lifecycle": state, "disputes": [{"both": "retained"}]},
    }
    data = assay.to_dict()
    data["assay_id"] = contracts.digest(data)
    saved = tmp_path / f"{state.lower()}.assay.json"
    saved.write_bytes(rfc8785.dumps(data))
    payload = badge_from_path(saved, now="2026-09-12T12:00:00Z")
    assert payload["record"]["id"] == "RGX-2026-0001"
    assert payload["record"]["state"] == state.lower()
    assert payload["color"] == color
    assert payload["isError"] is is_error


def test_badge_refuses_absent_or_ambiguous_reach_instead_of_averaging() -> None:
    from reward_lens.commons.badge import BadgeInputError, badge_from_assay

    with pytest.raises(BadgeInputError, match="unavailable"):
        badge_from_assay(
            _assay(reaches=0), record_url="file:///record.json", now="2026-09-12T12:00:00Z"
        )
    two = _assay(reaches=2)
    with pytest.raises(BadgeInputError, match="multiple"):
        badge_from_assay(two, record_url="file:///record.json", now="2026-09-12T12:00:00Z")
    selected = badge_from_assay(
        two,
        record_url="file:///record.json",
        badge_reach_entry_id="reach.surfaces.surface_1",
        now="2026-09-20T00:00:00Z",
        stale_after=timedelta(days=7),
    )
    assert selected["record"]["state"] == "stale"
    assert selected["provenance"]["reach_entry_id"] == "reach.surfaces.surface_1"


def test_badge_refuses_a_lookalike_object_with_borrowed_assay_fields() -> None:
    from reward_lens.commons.badge import BadgeInputError, badge_from_assay

    genuine = _assay()
    impostor = SimpleNamespace(
        assay_id=genuine.assay_id,
        created=genuine.created,
        measurement=genuine.measurement,
        findings=genuine.findings,
        extensions=genuine.extensions,
    )
    with pytest.raises(BadgeInputError, match="validated canonical Assay"):
        badge_from_assay(impostor, record_url="file:///record.json", now="2026-09-12T12:00:00Z")
