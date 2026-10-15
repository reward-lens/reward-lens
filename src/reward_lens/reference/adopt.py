"""Adopt a locally acquired material while preserving its provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .types import LicenceEvidence, MaterialGrade, ReferenceMaterial, SourceEvidence


def _source(value: SourceEvidence | str | None) -> SourceEvidence | None:
    if value is None or isinstance(value, SourceEvidence):
        return value
    return SourceEvidence(location=value)


def _licence(value: LicenceEvidence | str | None) -> LicenceEvidence | None:
    if value is None or isinstance(value, LicenceEvidence):
        return value
    return LicenceEvidence(spdx=value)


def adopt(
    *,
    identifier: str,
    payload: Path,
    source: SourceEvidence | str | None,
    licence: LicenceEvidence | str | None,
    source_digest: str,
    retrieval_evidence: str | None,
    reverification_evidence: str | None,
) -> ReferenceMaterial:
    """Bind real local bytes to adopted-material evidence.

    Adoption deliberately always produces ``REPORTED``.  A caller cannot turn an external source
    into certified material by choosing a different enum value.
    """

    local_payload = Path(payload)
    if not local_payload.is_file():
        raise ValueError("adopted material payload must be an existing file")
    actual_digest = hashlib.sha256(local_payload.read_bytes()).hexdigest()
    if actual_digest != source_digest:
        raise ValueError(
            f"adopted material source digest mismatch: expected {source_digest}, got {actual_digest}"
        )
    return ReferenceMaterial(
        identifier=identifier,
        grade=MaterialGrade.REPORTED,
        payload_path=str(local_payload),
        source_digest=source_digest,
        source=_source(source),
        licence=_licence(licence),
        retrieval_evidence=retrieval_evidence,
        reverification_evidence=reverification_evidence,
        characterisation_evidence=None,
    )


def compose(
    record,
    *,
    material_identifier: str = "diagnostic-v3",
    adopted_material: ReferenceMaterial | None = None,
    material_private: bool = False,
    reference_consent: Path | None = None,
    unit: str,
    config,
    blank_measurements,
    levels,
    synthetic: bool,
    transfer,
    detection_trace=None,
    detector_spec=None,
    proficiency_charter: Path | None = None,
    proficiency_consent: Path | None = None,
    public_report: Path | None = None,
    private_report: Path | None = None,
):
    """Attach actual reference evidence to a fresh Assay without relabelling its measurements.

    The material's scalar controls, detector trace partitions and local proficiency round retain
    their own scopes and units. A synthetic reference result does not qualify the source Assay's
    target-domain measurements, so its Subject, measurements and scientific absences are retained.
    """
    import copy
    import json
    import time
    from dataclasses import asdict
    from datetime import UTC, datetime
    from decimal import Decimal
    from enum import Enum

    from reward_lens import contracts
    from reward_lens.reference.calibration import calibrate
    from reward_lens.reference.load import load, load_diagnostic_v3_material, reverify

    extension = "https://reward-lens.dev/extensions/reference-composition/1"
    if not isinstance(record, contracts.Assay):
        raise TypeError("reference composition requires an actual Assay")
    source = record.to_dict()
    contracts.validate_record(source)
    if contracts.digest(source) != record.assay_id:
        raise contracts.RecordInvalid(
            message="reference composition source Assay content ID does not match its body",
            remediation="supply the original validated source Assay without changing its content ID",
            context={"field": "assay_id"},
        )
    if extension in (source.get("extensions") or {}):
        raise ValueError("this Assay already carries reference composition evidence")
    if (detection_trace is None) != (detector_spec is None):
        raise ValueError(
            "detection requires both its actual TraceRecord and registered DetectorSpec"
        )
    requested_proficiency = any(
        value is not None
        for value in (proficiency_charter, proficiency_consent, public_report, private_report)
    )
    if requested_proficiency and any(
        value is None
        for value in (proficiency_charter, proficiency_consent, public_report, private_report)
    ):
        raise ValueError(
            "proficiency requires a sealed charter, explicit consent and both report paths"
        )

    def plain(value):
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(item) for item in value]
        if hasattr(value, "__dataclass_fields__"):
            return plain(asdict(value))
        return value

    started = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    wall_started, cpu_started = time.perf_counter(), time.process_time()
    blanks, known_levels = tuple(blank_measurements), tuple(levels)
    if type(material_private) is not bool:
        raise TypeError("material_private must be a bool")
    material = load(material_identifier) if adopted_material is None else adopted_material
    if not isinstance(material, ReferenceMaterial) or material.identifier != material_identifier:
        raise ValueError("adopted material must match the declared material identifier")
    reference_permission = None
    if material_private:
        try:
            permission_raw = Path(reference_consent).read_bytes()
            permission = json.loads(permission_raw)
        except (OSError, ValueError, TypeError) as error:
            raise ValueError(
                "explicit private reference consent is unavailable or invalid"
            ) from error
        if not isinstance(permission, dict) or set(permission) != {
            "schema",
            "explicit",
            "assay_id",
            "material_identifier",
            "material_digest",
            "scope",
        }:
            raise ValueError("explicit private reference consent has an unsupported declaration")
        if (
            permission["schema"] != "reward-lens/reference-consent/1"
            or permission["explicit"] is not True
            or permission["assay_id"] != record.assay_id
            or permission["material_identifier"] != material.identifier
            or permission["material_digest"] != material.source_digest
            or permission["scope"] != "private-reference-composition"
        ):
            raise ValueError(
                "explicit private reference consent does not authorize this material and Assay"
            )
        reference_permission = hashlib.sha256(permission_raw).hexdigest()
    elif reference_consent is not None:
        raise ValueError("reference consent requires an explicitly private material declaration")
    receipt = reverify(material)
    reference = load_diagnostic_v3_material(material)
    panel = calibrate(
        material=material,
        reference_view=reference.view,
        unit=unit,
        config=config,
        blank_measurements=blanks,
        levels=known_levels,
        synthetic=synthetic,
        transfer=transfer,
    )
    panel_body = {
        "qualified": panel.qualified,
        "refusal": panel.refusal,
        "material_identifier": panel.material_identifier,
        "material_digest": panel.material_digest,
        "reference_view_checksum": panel.reference_view_checksum,
        "reference_view_effective_n": plain(panel.reference_view_effective_n),
        "unit": panel.unit,
        "method": panel.method,
        "config": plain(panel.config),
        "synthetic": panel.synthetic,
        "transfer": plain(panel.transfer),
        "limits": plain(panel.limits),
        "validity": plain(panel.validity),
        "assessments": plain(panel.assessments),
        "registered_inputs": {"blank_measurements": plain(blanks), "levels": plain(known_levels)},
        "scope": "declared reference scalar procedure; not target-domain calibration",
        "sampling_unit": "independent scalar replicate",
    }
    panel_id = contracts.digest(panel_body)
    detection = {"state": "NOT_MEASURED", "reason": "no detector trace and specification supplied"}
    if detection_trace is not None:
        from reward_lens.instruments.base import RunContext
        from reward_lens.instruments.detection import DetectorSpec, detect
        from reward_lens.product.trace import TraceRecord

        if not isinstance(detection_trace, TraceRecord) or not isinstance(
            detector_spec, DetectorSpec
        ):
            raise TypeError("detection requires actual TraceRecord and DetectorSpec objects")
        if detection_trace.subject_ref != record.subject.version.digest:
            raise ValueError("detection trace belongs to a different Assay subject")
        if detector_spec.reference_material is not None:
            reference_traces = detector_spec.reference_material.all_traces()
            if detector_spec.reference_material.identifier != material.identifier:
                raise ValueError(
                    "detector reference identifier does not match the adopted material"
                )
            for trace in reference_traces:
                identity = trace.series[detector_spec.series].provenance
                if (
                    identity.get("material_identifier") != material.identifier
                    or identity.get("material_digest") != material.source_digest
                ):
                    raise ValueError(
                        "detector reference trace does not bind the adopted material digest"
                    )
        context = RunContext(
            sandbox=None,
            project=None,
            clock=time.monotonic,
            offline=True,
            started=started,
        )
        entries = detect(detection_trace, detector_spec, context=context)
        detection = {
            "state": (
                "QUALIFIED"
                if all(
                    item.kind == "detection" and not item.detection.unqualified for item in entries
                )
                else "UNQUALIFIED"
            ),
            "spec_digest": detector_spec.digest(),
            "series": detector_spec.series,
            "source_subject_ref": detection_trace.subject_ref,
            "target_run_id": detection_trace.run_id,
            "entries": [
                item.model_dump(mode="json", by_alias=True, exclude_none=True) for item in entries
            ],
            "scope": "registered trace detector procedure; distinct from scalar reference limits",
            "sampling_unit": "disjoint reference/assessment/target trace",
        }
    proficiency = {"state": "NOT_MEASURED", "reason": "no consented sealed local round requested"}
    if requested_proficiency:
        from reward_lens.reference.proficiency import (
            CalibrationBinding,
            ProficiencyRefusal,
            load_sealed_charter,
            persist_round,
            run_local_round,
        )

        # Validate consent before any participation or identified report is produced.
        try:
            consent_raw = Path(proficiency_consent).read_bytes()
            consent = json.loads(consent_raw)
        except (OSError, ValueError) as error:
            raise ProficiencyRefusal(
                "explicit proficiency consent is unavailable or invalid"
            ) from error
        if not isinstance(consent, dict) or set(consent) != {
            "schema",
            "explicit",
            "assay_id",
            "charter_digest",
            "material_identifier",
            "material_digest",
            "scope",
        }:
            raise ProficiencyRefusal("explicit proficiency consent has an unsupported declaration")
        if (
            consent["schema"] != "reward-lens/proficiency-consent/1"
            or consent["explicit"] is not True
            or consent["assay_id"] != record.assay_id
            or consent["material_identifier"] != material.identifier
            or consent["material_digest"] != material.source_digest
            or consent["scope"] != "local-native-round"
        ):
            raise ProficiencyRefusal(
                "explicit proficiency consent does not authorize this Assay local round"
            )
        sealed = load_sealed_charter(proficiency_charter)
        if consent["charter_digest"] != sealed.digest:
            raise ProficiencyRefusal(
                "explicit proficiency consent names a different sealed charter"
            )
        if CalibrationBinding.from_panel(panel) != sealed.charter.calibration:
            raise ProficiencyRefusal(
                "composed calibration does not match the sealed local proficiency unit/binding"
            )
        if Path(public_report).resolve() == Path(private_report).resolve():
            raise ProficiencyRefusal(
                "public and identified private reports require different paths"
            )
        if any(
            Path(path).exists() or Path(path).is_symlink()
            for path in (public_report, private_report)
        ):
            raise ProficiencyRefusal("proficiency report destinations must be fresh")
        actual = run_local_round(sealed)
        persisted = persist_round(
            actual, public_destination=public_report, private_destination=private_report
        )
        proficiency = {
            "state": "EXECUTED_LOCAL",
            "charter_digest": sealed.digest,
            "calibration_panel_digest": sealed.charter.calibration.panel_digest,
            "public_digest": persisted.public_digest,
            "public_file_sha256": hashlib.sha256(persisted.public_path.read_bytes()).hexdigest(),
            "consent_sha256": hashlib.sha256(consent_raw).hexdigest(),
            "public_projection": actual.public_projection(),
            "actual_participants": len(actual.actual_results),
            "scope": "actual charter-declared local procedures; no external participation claim",
            "identified_report_mode": oct(Path(private_report).stat().st_mode & 0o777),
        }
    evidence = {
        "schema": "reward-lens/reference-composition/1",
        "parent_assay_id": record.assay_id,
        "reference_material": plain(material),
        "reverification": plain(receipt),
        "material_private": material_private,
        "reference_consent_sha256": reference_permission,
        "calibration": panel_body,
        "calibration_evidence_id": panel_id,
        "detection": detection,
        "proficiency": proficiency,
        "provenance": {
            "started": started,
            "wall_s": time.perf_counter() - wall_started,
            "cpu_s": time.process_time() - cpu_started,
            "offline": True,
            "api_calls": 0,
            "producer": "reward_lens.reference.compose",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "primary_measurements_preserved": True,
    }
    body = copy.deepcopy(source)
    body["extensions"] = dict(body.get("extensions") or {})
    body["extensions"][extension] = evidence
    link = contracts.CalibrationLink(
        material=material.identifier,
        grade=material.grade.value,
        transfer_gap=None if panel.transfer is None else panel.transfer.gap,
        evidence_id=panel_id,
    )
    body["calibration_links"] = list(body.get("calibration_links") or []) + [
        link.model_dump(mode="json", by_alias=True, exclude_none=True)
    ]
    # An attestation for the parent body cannot attest this new content.
    body.pop("attestation", None)
    body["decision"]["signature"] = None
    body["created"] = started
    body["assay_id"] = contracts.digest(body)
    contracts.validate_record(body)
    result = contracts.Assay.model_validate(body)
    if record.to_dict() != source:
        raise RuntimeError("reference composition mutated the original Assay")
    return result
