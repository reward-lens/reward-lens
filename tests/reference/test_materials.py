"""Acceptance coverage for the frozen reference-material seam."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from reward_lens.reference import (
    CertificateOfAnalysis,
    CharacterisationEvidence,
    CharacterisedValue,
    LicenceEvidence,
    MaterialGrade,
    ReferenceMaterial,
    SourceEvidence,
    adopt,
    load,
    reverify,
)
from reward_lens.reference.load import load_diagnostic_v3_material


def test_diagnostic_v3_is_actually_acquired_loaded_and_reverified() -> None:
    material = load("diagnostic-v3")

    assert material.grade is MaterialGrade.REPORTED
    assert material.source.commit == "607774d19f79140234260da712a18487eca5b45c"
    assert material.licence.spdx == "MIT"
    assert material.retrieval_evidence
    receipt = reverify(material)
    assert receipt.verified_digest == material.source_digest
    assert receipt.verified_at


def test_adopted_material_needs_source_licence_and_reverification_evidence(tmp_path: Path) -> None:
    payload = tmp_path / "material.txt"
    payload.write_text("known answer", encoding="utf-8")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="source"):
        adopt(
            identifier="missing-source",
            payload=payload,
            source=None,
            licence="MIT",
            source_digest=digest,
            retrieval_evidence="local receipt",
            reverification_evidence="local verification",
        )

    with pytest.raises(ValueError, match="licence"):
        adopt(
            identifier="missing-licence",
            payload=payload,
            source="local test source",
            licence=None,
            source_digest=digest,
            retrieval_evidence="local receipt",
            reverification_evidence="local verification",
        )

    with pytest.raises(ValueError, match="re-verification"):
        adopt(
            identifier="missing-verification",
            payload=payload,
            source="local test source",
            licence="MIT",
            source_digest=digest,
            retrieval_evidence="local receipt",
            reverification_evidence=None,
        )


def test_certified_requires_project_characterisation() -> None:
    with pytest.raises(TypeError, match="CharacterisationEvidence"):
        ReferenceMaterial(
            identifier="constructed-without-characterisation",
            grade=MaterialGrade.CERTIFIED,
            payload_path="missing",
            source_digest="0" * 64,
            source=None,
            licence=None,
            retrieval_evidence=None,
            reverification_evidence=None,
            characterisation_evidence=None,
        )


def test_reverify_refuses_tampered_material(tmp_path: Path) -> None:
    material = load("diagnostic-v3")
    copied = tmp_path / "diagnostic_v3.py"
    copied.write_bytes(Path(material.payload_path).read_bytes())
    tampered = replace(material, payload_path=str(copied))
    copied.write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        reverify(tampered)


def test_certificate_has_every_section_7_9_field() -> None:
    certificate = CertificateOfAnalysis.for_material(load("diagnostic-v3"))
    assert certificate.complete
    assert certificate.value_table
    assert certificate.expanded_uncertainty_note
    assert certificate.nullification_conditions


def test_public_constructor_refuses_string_grade_and_unverified_reported_payload() -> None:
    """Runtime callers cannot bypass the grade or payload invariants with nominal values."""

    with pytest.raises(TypeError, match="MaterialGrade"):
        ReferenceMaterial(
            identifier="string-grade",
            grade="certified",  # type: ignore[arg-type]
            payload_path="missing",
            source_digest="0" * 64,
            source=None,
            licence=None,
            retrieval_evidence=None,
            reverification_evidence=None,
            characterisation_evidence="sure",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="payload"):
        ReferenceMaterial(
            identifier="reported-missing-payload",
            grade=MaterialGrade.REPORTED,
            payload_path="missing",
            source_digest="0" * 64,
            source=SourceEvidence(location="local source"),
            licence=LicenceEvidence(spdx="MIT"),
            retrieval_evidence="local receipt",
            reverification_evidence="local verification",
            characterisation_evidence=None,
        )


def test_certified_refuses_truthy_string_characterisation() -> None:
    """A label is not a project characterisation receipt or a certified value."""

    with pytest.raises(TypeError, match="CharacterisationEvidence"):
        ReferenceMaterial(
            identifier="forged-certified",
            grade=MaterialGrade.CERTIFIED,
            payload_path="missing",
            source_digest="0" * 64,
            source=None,
            licence=None,
            retrieval_evidence=None,
            reverification_evidence=None,
            characterisation_evidence="sure",  # type: ignore[arg-type]
        )


def test_certified_needs_digest_bound_project_characterisation(tmp_path: Path) -> None:
    """A complete certified certificate requires measured values and an intact project record."""

    payload = tmp_path / "constructed-material.txt"
    payload.write_text("known constructed material", encoding="utf-8")
    payload_digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    record = tmp_path / "project-characterisation.json"
    record.write_text('{"method":"controlled uncertainty study"}', encoding="utf-8")
    evidence = CharacterisationEvidence(
        material_identifier="constructed-material",
        material_digest=payload_digest,
        project_record_path=str(record),
        project_record_digest=hashlib.sha256(record.read_bytes()).hexdigest(),
        values=(
            CharacterisedValue(
                measurand="known-answer agreement",
                value="0.94",
                expanded_uncertainty="0.03",
                definition="agreement under the declared controlled procedure",
            ),
        ),
        conditions=("controlled test material",),
        uncertainty_method="expanded uncertainty from the declared study",
        verifier="reward-lens project-characterisation verifier",
    )
    material = ReferenceMaterial(
        identifier="constructed-material",
        grade=MaterialGrade.CERTIFIED,
        payload_path=str(payload),
        source_digest=payload_digest,
        source=None,
        licence=None,
        retrieval_evidence=None,
        reverification_evidence=None,
        characterisation_evidence=evidence,
    )
    certificate = CertificateOfAnalysis.for_material(material)
    assert certificate.complete
    assert certificate.certified_values == ("known-answer agreement: 0.94 ± 0.03",)

    record.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="project record digest"):
        CertificateOfAnalysis.for_material(material)


def test_diagnostic_v3_builder_constructs_a_deterministic_consumer_material() -> None:
    """The acquired builder is actually executed to build its usable local DataView."""

    first = load_diagnostic_v3_material(load("diagnostic-v3"))
    second = load_diagnostic_v3_material(load("diagnostic-v3"))

    # The current exact builder constructs 84 pairs; the nearby 85-line JSONL fixture includes
    # one metadata line and is not this builder's pair count.
    assert first.pair_count == 84
    assert len(first.dimensions) == 14
    assert first.view.name == "diagnostic_v3"
    assert first.content_checksum == second.content_checksum
    assert first.builder_digest == load("diagnostic-v3").source_digest


def _composition_arguments():
    from decimal import Decimal

    from reward_lens.reference.calibration import (
        CalibrationConfig,
        CalibrationLevel,
        TransferStatement,
    )

    return dict(
        unit="detection-probability",
        config=CalibrationConfig(
            alpha=Decimal(".05"),
            beta=Decimal(".05"),
            minimum_blank_replicates=4,
            minimum_level_replicates=3,
            maximum_relative_standard_uncertainty=Decimal(".10"),
        ),
        blank_measurements=(Decimal("0"), Decimal(".01"), Decimal("-.01"), Decimal("0")),
        levels=(
            CalibrationLevel(Decimal(".20"), (Decimal(".19"), Decimal(".20"), Decimal(".21"))),
            CalibrationLevel(Decimal(".60"), (Decimal(".59"), Decimal(".60"), Decimal(".61"))),
            CalibrationLevel(Decimal("1.00"), (Decimal(".99"), Decimal("1.00"), Decimal("1.00"))),
        ),
        synthetic=True,
        transfer=TransferStatement(synthetic=True),
    )


def _source_assay(tmp_path):
    from reward_lens.api._record import absent_record

    # A genuine explicit absence record, never a fixture pretending to have measured correctness.
    path = tmp_path / "subject.py"
    path.write_text("def score(task, response):\n    return float(response == task['reference'])\n")
    return absent_record(command="audit", reproduce=["reward-lens", "audit", str(path)], path=path)


def test_public_compose_derives_valid_new_id_and_retains_primary_absences(tmp_path):
    import json

    from reward_lens import api, contracts
    from reward_lens.reference import compose

    source = _source_assay(tmp_path)
    before = source.to_dict()
    result = compose(source, **_composition_arguments())
    assert result.assay_id != source.assay_id and source.to_dict() == before
    assert result.subject == source.subject and result.measurement == source.measurement
    assert result.holes == source.holes and result.decision.state == source.decision.state
    evidence = result.extensions["https://reward-lens.dev/extensions/reference-composition/1"]
    assert evidence["parent_assay_id"] == source.assay_id
    assert evidence["calibration"]["qualified"]
    assert evidence["calibration"]["transfer"] == {"gap": "0.419", "synthetic": True}
    assert evidence["reference_material"]["grade"] == "reported"
    assert evidence["proficiency"]["state"] == "NOT_MEASURED"
    assert contracts.digest(result) == result.assay_id
    path = tmp_path / "composed.json"
    path.write_text(json.dumps(result.to_dict()))
    assert api.open_record(path).assay_id == result.assay_id


def test_private_material_consent_refuses_before_builder_then_exact_restore(tmp_path, monkeypatch):
    import importlib
    import json

    from reward_lens.reference import compose

    loader = importlib.import_module("reward_lens.reference.load")
    original = loader.load_diagnostic_v3_material
    calls = []

    def observed(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(loader, "load_diagnostic_v3_material", observed)
    source = _source_assay(tmp_path)
    material = load()
    path = tmp_path / "consent.json"
    consent = {
        "schema": "reward-lens/reference-consent/1",
        "explicit": True,
        "assay_id": source.assay_id,
        "material_identifier": material.identifier,
        "material_digest": material.source_digest,
        "scope": "private-reference-composition",
    }
    raw = json.dumps(consent).encode()
    path.write_bytes(raw)
    args = dict(
        _composition_arguments(),
        adopted_material=material,
        material_private=True,
        reference_consent=path,
    )
    assert compose(source, **args).calibration_links
    calls.clear()
    consent["explicit"] = False
    path.write_text(json.dumps(consent))
    with pytest.raises(ValueError, match="does not authorize"):
        compose(source, **args)
    assert calls == []
    path.write_bytes(raw)
    assert path.read_bytes() == raw and compose(source, **args).calibration_links


def test_compose_real_calibration_refusal_preserves_no_limits(tmp_path):
    from decimal import Decimal

    from reward_lens.reference import compose

    args = _composition_arguments()
    args["blank_measurements"] = (Decimal("0"),)
    result = compose(_source_assay(tmp_path), **args)
    panel = result.extensions["https://reward-lens.dev/extensions/reference-composition/1"][
        "calibration"
    ]
    assert (
        panel["qualified"] is False and panel["limits"] is None and "replicated" in panel["refusal"]
    )


def test_public_compose_consent_binds_material_charter_and_source_before_participation(
    tmp_path, monkeypatch
):
    import json

    import reward_lens.reference.proficiency as prof
    from reward_lens.reference import compose

    source = _source_assay(tmp_path)
    material = load()
    sealed = prof.publish_local_charter(
        tmp_path / "charter.json", participants=("D-REC", "D-BBX", "D-CUSUM", "D-BOCPD")
    )
    consent = {
        "schema": "reward-lens/proficiency-consent/1",
        "explicit": True,
        "assay_id": source.assay_id,
        "charter_digest": sealed.digest,
        "material_identifier": material.identifier,
        "material_digest": material.source_digest,
        "scope": "local-native-round",
    }
    path = tmp_path / "consent.json"
    raw = json.dumps(consent).encode()
    path.write_bytes(raw)
    args = dict(
        _composition_arguments(),
        proficiency_charter=sealed.path,
        proficiency_consent=path,
        public_report=tmp_path / "public.json",
        private_report=tmp_path / "private.json",
    )
    calls = []
    original = prof.run_local_round

    def observed(*a):
        calls.append(1)
        return original(*a)

    monkeypatch.setattr(prof, "run_local_round", observed)
    for key, value in [
        ("explicit", False),
        ("assay_id", "sha256:" + "0" * 64),
        ("material_digest", "0" * 64),
        ("charter_digest", "0" * 64),
    ]:
        changed = dict(consent)
        changed[key] = value
        path.write_text(json.dumps(changed))
        with pytest.raises(prof.ProficiencyRefusal, match="consent"):
            compose(source, **args)
        assert calls == [] and not (tmp_path / "public.json").exists()
    path.write_bytes(raw)
    result = compose(source, **args)
    attached = result.extensions["https://reward-lens.dev/extensions/reference-composition/1"][
        "proficiency"
    ]
    assert attached["actual_participants"] == 4 and attached["state"] == "EXECUTED_LOCAL"
    assert calls == [1] and (tmp_path / "private.json").stat().st_mode & 0o777 == 0o600
    assert path.read_bytes() == raw


def test_public_compose_refuses_changed_parent_content_id_and_private_payload_then_restores(
    tmp_path,
):
    from reward_lens import contracts
    from reward_lens.reference import compose

    source = _source_assay(tmp_path)
    changed = source.model_copy(deep=True)
    changed.assay_id = "sha256:" + "0" * 64
    with pytest.raises(contracts.RecordInvalid):
        compose(changed, **_composition_arguments())
    material = load()
    payload = tmp_path / "private.py"
    raw = Path(material.payload_path).read_bytes()
    payload.write_bytes(raw)
    private = adopt(
        identifier=material.identifier,
        payload=payload,
        source=material.source,
        licence=material.licence,
        source_digest=material.source_digest,
        retrieval_evidence="actual controlled acquisition",
        reverification_evidence="actual digest check",
    )
    payload.write_bytes(raw + b"\n# deliberate source corruption\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        compose(source, adopted_material=private, **_composition_arguments())
    payload.write_bytes(raw)
    assert (
        payload.read_bytes() == raw
        and compose(source, adopted_material=private, **_composition_arguments()).calibration_links
    )
