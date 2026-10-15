from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from reward_lens.instruments.base import RunContext
from reward_lens.product.trace import Series, TraceRecord, TraceRow


def _trace(run_id: str, values: tuple[float, ...]) -> TraceRecord:
    return TraceRecord(
        reader="fixture",
        source=f"fixture://{run_id}",
        run_id=run_id,
        rows=tuple(
            TraceRow(index=i, step=i, group_id=None, row_id=f"{run_id}-{i}")
            for i in range(len(values))
        ),
        series={
            "hack_rate": Series(
                name="hack_rate",
                values=values,
                definition="fixture",
                provenance={"unit": "hack-rate"},
            )
        },
        components={},
        access_level="L0",
        access_statement="fixture",
        policy="strict",
        complete=True,
        completeness_statement="fixture",
        subject_ref="sha256:" + "a" * 64,
        depends_on=("digest:training_semantics",),
    )


def _context() -> RunContext:
    return RunContext(offline=True)


@pytest.fixture
def reference_material():
    from reward_lens.instruments.detection import EffectReference, ReferenceMaterial

    low = (
        (0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02),
        (0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01),
        (0.015, 0.025, 0.015, 0.025, 0.015, 0.025, 0.015, 0.025),
    )
    high = (0.01, 0.02, 0.01, 0.02, 0.50, 0.55, 0.60, 0.60)
    return ReferenceMaterial(
        "fixture-reference-v2",
        calibration_controls=tuple(
            _trace(f"cal-control-{i}", values) for i, values in enumerate(low)
        ),
        assessment_controls=tuple(
            _trace(f"assess-control-{i}", values) for i, values in enumerate(low)
        ),
        calibration_effects=tuple(
            EffectReference(_trace(f"cal-effect-{i}", high), 4, 0.5) for i in range(2)
        ),
        assessment_effects=tuple(
            EffectReference(_trace(f"assess-effect-{i}", high), 4, 0.5) for i in range(2)
        ),
    )


def _spec(material, *, arl: float = 8.0, effect: float = 0.5):
    from reward_lens.instruments.detection import DetectorSpec

    return DetectorSpec("hack_rate", arl, effect, 0.10, 0.20, material)


def test_public_route_emits_disjoint_calibrated_cusum_and_bocpd_evidence(reference_material):
    from reward_lens.instruments.detection import detect

    entries = detect(
        _trace("target", (0.01, 0.02, 0.01, 0.02, 0.55, 0.60, 0.60, 0.60)),
        _spec(reference_material),
        context=_context(),
    )
    assert {entry.result["algorithm"] for entry in entries} == {
        "one_sided_cusum",
        "bocpd_run_length",
    }
    assert all(
        entry.kind == "detection" and entry.detection.unqualified is False for entry in entries
    )
    assert all(
        entry.detection.LoB is not None and entry.detection.LoD is not None for entry in entries
    )
    assert all(entry.result["series_unit"] == "hack-rate" for entry in entries)
    assert all(entry.result["reference_provenance"]["digest"] for entry in entries)


def test_onset_detection_is_evidence_only_without_mutating_trace(reference_material):
    from reward_lens.instruments.detection import detect

    observed = _trace("target", (0.01, 0.02, 0.01, 0.02, 0.55, 0.60, 0.60, 0.60))
    before = (observed.rows, observed.series, observed.components, observed.complete)
    entries = detect(observed, _spec(reference_material), context=_context())
    assert any(entry.result["alarm_index"] is not None for entry in entries)
    assert (observed.rows, observed.series, observed.components, observed.complete) == before
    assert all(entry.result["steering"] == "none" for entry in entries)


def test_missing_reference_and_run_control_are_precise(reference_material):
    from reward_lens.instruments.detection import DetectionSteeringForbidden, DetectorSpec, detect

    target = _trace("target", (0.01, 0.02, 0.01, 0.02))
    entries = detect(
        target, DetectorSpec("hack_rate", 8.0, 0.5, 0.1, 0.2, None), context=_context()
    )
    assert (
        entries[0].kind == "absence"
        and "DETECTION_REFERENCE_ABSENT" in entries[0].absence.missing_access
    )
    with pytest.raises(
        DetectionSteeringForbidden, match=r"DETECTION_STEERING_FORBIDDEN.*runs/current"
    ):
        detect(target, _spec(reference_material), context=_context(), run_control="runs/current")


def test_nonfinite_observation_is_a_named_absence(reference_material):
    from reward_lens.instruments.detection import detect

    entries = detect(
        _trace("target", (0.01, float("nan"), 0.02)), _spec(reference_material), context=_context()
    )
    assert entries[0].kind == "absence"
    assert "DETECTION_NONFINITE_OBSERVATION" in entries[0].absence.missing_access


def test_lod_is_unqualified_when_no_effect_level_meets_registered_beta(reference_material):
    from reward_lens.instruments.detection import EffectReference, detect

    weak = replace(
        reference_material,
        assessment_effects=(
            EffectReference(
                _trace("weak-a", (0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02)), 4, 0.5
            ),
            EffectReference(
                _trace("weak-b", (0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01)), 4, 0.5
            ),
        ),
    )
    entries = detect(_trace("target", (0.01,) * 8), _spec(weak), context=_context())
    assert all(entry.detection.unqualified is True for entry in entries)
    assert all("LoD" in entry.detection.unqualified_reason for entry in entries)


def test_short_right_censored_control_cannot_claim_registered_arl_100():
    from reward_lens.instruments.detection import EffectReference, ReferenceMaterial, detect

    controls = tuple(_trace(f"zero-{i}", (0.0,) * 8) for i in range(4))
    effects = tuple(
        EffectReference(_trace(f"onset-{i}", (0.0,) * 4 + (1.0,) * 4), 4, 1.0) for i in range(4)
    )
    material = ReferenceMaterial(
        "too-short",
        calibration_controls=controls[:2],
        assessment_controls=controls[2:],
        calibration_effects=effects[:2],
        assessment_effects=effects[2:],
    )
    entries = detect(
        _trace("target", (0.0,) * 4 + (1.0,) * 4),
        _spec(material, arl=100.0, effect=1.0),
        context=_context(),
    )
    assert all(entry.detection.unqualified is True for entry in entries)
    assert all("control exposure" in entry.detection.unqualified_reason for entry in entries)


def test_contract_rejects_cross_role_overlap():
    from reward_lens.instruments.detection import EffectReference, ReferenceMaterial

    same = _trace("same", (0.0,) * 4 + (1.0,) * 4)
    with pytest.raises(ValueError, match="disjoint"):
        ReferenceMaterial(
            "overlap",
            calibration_controls=(same,),
            assessment_controls=(same,),
            calibration_effects=(
                EffectReference(_trace("cal", same.series["hack_rate"].values), 4, 1.0),
            ),
            assessment_effects=(
                EffectReference(_trace("ass", same.series["hack_rate"].values), 4, 1.0),
            ),
        )


def test_target_cannot_be_its_own_reference(reference_material):
    from reward_lens.instruments.detection import detect

    with pytest.raises(ValueError, match="target trace must be disjoint"):
        detect(
            reference_material.calibration_controls[0],
            _spec(reference_material),
            context=_context(),
        )


def test_context_declared_onset_and_numeric_run_length_posterior():
    from reward_lens.instruments.detection import (
        EffectReference,
        ReferenceMaterial,
        bocpd_run_length_posterior,
        detect,
    )

    controls = tuple(_trace(f"c{i}", (0.0,) * 32) for i in range(24))
    effects = tuple(
        EffectReference(_trace(f"e{i}", (0.0,) * 16 + (1.0,) * 16), 16, 1.0) for i in range(4)
    )
    material = ReferenceMaterial(
        "split",
        calibration_controls=controls[:12],
        assessment_controls=controls[12:],
        calibration_effects=effects[:2],
        assessment_effects=effects[2:],
    )
    entries = detect(
        _trace("target", (0.0,) * 16 + (1.0,) * 16),
        _spec(material, effect=1.0),
        context=RunContext(offline=False, budget="12.34"),
    )
    assert all(entry.provenance.offline is False for entry in entries)
    assert all(
        entry.result["calibration"]["delay_origin"] == "declared onset index" for entry in entries
    )
    posterior = bocpd_run_length_posterior(
        (0.0, 1.0), baseline_mean=0.0, baseline_sd=1.0, hazard=0.1
    )
    assert posterior[0] == pytest.approx((0.1, 0.9))
    # Direct two-step Normal-Normal message calculation: new run, one-step growth, two-step growth.
    assert posterior[-1] == pytest.approx((0.1, 0.09, 0.81), abs=1e-6)


def test_discovery_exposes_the_trace_detection_panel():
    """The public Audit registry, rather than a private detector call, finds this panel."""
    from reward_lens.instruments.base import Panel
    from reward_lens.product.audit.registry import discover

    panel = next(
        item
        for item in discover()
        if item.section == "trace"
        and [instrument.id for instrument in item.instruments] == ["trace.detection"]
    )
    assert isinstance(panel, Panel)
    assert [instrument.id for instrument in panel.instruments] == ["trace.detection"]


def test_discovered_panel_uses_the_frozen_context_reference_binding(reference_material):
    """The panel consumes the released carrier; it does not open or infer a reference itself."""
    from reward_lens.instruments.base import DetectionReferenceBinding
    from reward_lens.instruments.detection import PANEL

    target = _trace("target", (0.01, 0.02, 0.01, 0.02, 0.55, 0.60, 0.60, 0.60))
    binding = DetectionReferenceBinding(
        spec=_spec(reference_material),
        source="fixture://registered-reference-bundle",
        digest="sha256:" + "b" * 64,
    )
    entries = PANEL.instruments[0].run(
        SimpleNamespace(), SimpleNamespace(trace=target), RunContext(detection_reference=binding)
    )

    assert {entry.result["algorithm"] for entry in entries} == {
        "one_sided_cusum",
        "bocpd_run_length",
    }
    assert all(entry.result["steering"] == "none" for entry in entries)
    assert all(
        entry.result["reference_provenance"]["audit_binding"]
        == {
            "source": binding.source,
            "digest": binding.digest,
        }
        for entry in entries
    )


def test_discovered_panel_reports_precise_absence_without_a_context_binding():
    from reward_lens.instruments.detection import PANEL

    target = _trace("target", (0.01, 0.02, 0.01, 0.02))
    entries = PANEL.instruments[0].run(
        SimpleNamespace(), SimpleNamespace(trace=target), RunContext(detection_reference=None)
    )

    assert len(entries) == 1
    assert entries[0].kind == "absence"
    assert entries[0].entry_id == "trace.detection.reference"
    assert "DETECTION_REFERENCE_ABSENT" in entries[0].absence.missing_access
    assert "AuditRequest.detection_reference" in entries[0].absence.missing_access
