"""Acceptance coverage for separate, evidence-bound calibration limits."""

from __future__ import annotations

from decimal import Decimal

import pytest

from reward_lens.reference import load
from reward_lens.reference.calibration import (
    AdjustmentRecord,
    CalibrationConfig,
    CalibrationLevel,
    CalibrationPanel,
    LimitSet,
    TransferStatement,
    ValidityEnvelope,
    calibrate,
)
from reward_lens.reference.load import load_diagnostic_v3_material


def _reference_input():
    material = load("diagnostic-v3")
    return material, load_diagnostic_v3_material(material).view


def _qualified_panel(*, synthetic: bool = True, transfer: TransferStatement | None = None):
    material, view = _reference_input()
    return calibrate(
        material=material,
        reference_view=view,
        unit="score-delta",
        config=CalibrationConfig(
            alpha=Decimal("0.05"),
            beta=Decimal("0.05"),
            minimum_blank_replicates=4,
            minimum_level_replicates=3,
            maximum_relative_standard_uncertainty=Decimal("0.10"),
        ),
        blank_measurements=(Decimal("0.00"), Decimal("0.01"), Decimal("-0.01"), Decimal("0.00")),
        levels=(
            CalibrationLevel(Decimal("0.05"), (Decimal("0.04"), Decimal("0.05"), Decimal("0.06"))),
            CalibrationLevel(Decimal("0.20"), (Decimal("0.19"), Decimal("0.20"), Decimal("0.21"))),
            CalibrationLevel(Decimal("0.40"), (Decimal("0.39"), Decimal("0.40"), Decimal("0.41"))),
        ),
        synthetic=synthetic,
        transfer=transfer or TransferStatement(synthetic=True),
    )


def test_computes_distinct_lob_lod_loq_and_validity_envelope_from_replicated_controls() -> None:
    panel = _qualified_panel()

    assert panel.qualified
    assert isinstance(panel.limits, LimitSet)
    assert panel.limits.blank < panel.limits.detection <= panel.limits.quantification
    assert panel.validity is not None
    assert panel.validity.lower_bound == panel.limits.quantification
    assert panel.validity.upper_bound == Decimal("0.40")
    assert panel.config.alpha == Decimal("0.05")
    assert panel.config.beta == Decimal("0.05")
    assert panel.reference_view_checksum.startswith("ds:")
    assert panel.transfer == TransferStatement(synthetic=True)


def test_synthetic_panel_requires_complete_fixed_transfer_statement() -> None:
    material, view = _reference_input()
    kwargs = dict(
        material=material,
        reference_view=view,
        unit="score-delta",
        config=CalibrationConfig(alpha=Decimal("0.05"), beta=Decimal("0.05")),
        blank_measurements=(Decimal("0"), Decimal("0.01")),
        levels=(CalibrationLevel(Decimal("0.20"), (Decimal("0.20"), Decimal("0.21"))),),
        synthetic=True,
    )
    with pytest.raises(ValueError, match="0.419 transfer"):
        calibrate(**kwargs, transfer=None)
    with pytest.raises(ValueError, match="0.419"):
        calibrate(**kwargs, transfer=TransferStatement(gap=Decimal("0.418"), synthetic=True))


@pytest.mark.parametrize("alpha,beta", [(None, Decimal("0.05")), (Decimal("0.05"), None)])
def test_alpha_and_beta_are_required(alpha: Decimal | None, beta: Decimal | None) -> None:
    with pytest.raises((TypeError, ValueError), match="alpha|beta"):
        CalibrationConfig(alpha=alpha, beta=beta)  # type: ignore[arg-type]


def test_insufficient_controls_publish_no_fabricated_limits() -> None:
    material, view = _reference_input()
    panel = calibrate(
        material=material,
        reference_view=view,
        unit="score-delta",
        config=CalibrationConfig(alpha=Decimal("0.05"), beta=Decimal("0.05")),
        blank_measurements=(Decimal("0.00"),),
        levels=(CalibrationLevel(Decimal("0.20"), (Decimal("0.20"),)),),
        synthetic=False,
        transfer=None,
    )

    assert not panel.qualified
    assert panel.limits is None
    assert panel.validity is None
    assert "replicated" in panel.refusal


def test_a_single_numerical_level_cannot_carry_both_lod_and_loq_claims() -> None:
    material, view = _reference_input()
    panel = calibrate(
        material=material,
        reference_view=view,
        unit="score-delta",
        config=CalibrationConfig(alpha=Decimal("0.05"), beta=Decimal("0.05")),
        blank_measurements=(Decimal("0.00"), Decimal("0.01")),
        levels=(CalibrationLevel(Decimal("0.20"), (Decimal("0.20"), Decimal("0.21"))),),
        synthetic=False,
        transfer=None,
    )

    assert not panel.qualified
    assert panel.limits is None
    assert "separately established" in panel.refusal


def test_adjustment_is_a_separately_labelled_operation() -> None:
    record = AdjustmentRecord(
        raw_value=Decimal("1.00"),
        adjusted_value=Decimal("0.90"),
        operation="centering against assessment distribution",
        fitted_on="assessment data",
    )

    assert record.kind == "adjustment"
    assert record.delta == Decimal("-0.10")
    assert "calibration" not in record.kind


def test_interface_value_objects_keep_the_frozen_short_constructors() -> None:
    """The section-1 value-object constructors do not require a hidden unit argument."""

    limits = LimitSet(Decimal("0.01"), Decimal("0.05"), Decimal("0.20"))
    from reward_lens.reference.calibration import ValidityEnvelope

    envelope = ValidityEnvelope(Decimal("0.20"), Decimal("0.40"))
    assert limits.unit == envelope.unit == "dimensionless"


def test_replicated_zero_mean_level_refuses_without_dividing_by_zero() -> None:
    material, view = _reference_input()
    panel = calibrate(
        material=material,
        reference_view=view,
        unit="score-delta",
        config=CalibrationConfig(alpha=Decimal("0.05"), beta=Decimal("0.05")),
        blank_measurements=(Decimal("0.00"), Decimal("0.01")),
        levels=(
            CalibrationLevel(Decimal("0.20"), (Decimal("-0.01"), Decimal("0.01"))),
            CalibrationLevel(Decimal("0.40"), (Decimal("0.39"), Decimal("0.41"))),
        ),
        synthetic=False,
        transfer=None,
    )

    assert not panel.qualified
    assert panel.limits is None
    assert panel.validity is None
    assert "quantification" in panel.refusal


@pytest.mark.parametrize(
    ("panel_synthetic", "transfer_synthetic"),
    ((False, True), (True, False)),
)
def test_panel_and_transfer_synthetic_flags_cannot_contradict(
    panel_synthetic: bool, transfer_synthetic: bool
) -> None:
    with pytest.raises(ValueError, match="synthetic flag"):
        _qualified_panel(
            synthetic=panel_synthetic,
            transfer=TransferStatement(synthetic=transfer_synthetic),
        )


def test_public_panel_constructor_rejects_counterfeit_binding_and_cross_unit_limits() -> None:
    material, view = _reference_input()
    config = CalibrationConfig(alpha=Decimal("0.05"), beta=Decimal("0.05"))
    limits = LimitSet(Decimal("0.01"), Decimal("0.05"), Decimal("0.20"), unit="other-unit")
    validity = ValidityEnvelope(Decimal("0.20"), Decimal("0.40"), unit="another-unit")

    with pytest.raises(ValueError, match="material digest|unit"):
        CalibrationPanel(
            material_identifier=material.identifier,
            material_digest="0" * 64,
            reference_view_checksum="ds:" + "0" * 32,
            reference_view_effective_n=Decimal(str(view.effective_n())),
            unit="score-delta",
            config=config,
            synthetic=False,
            transfer=None,
            limits=limits,
            validity=validity,
            assessments=(),
            reference_material=material,
            reference_view=view,
        )

    with pytest.raises(ValueError, match="unit"):
        CalibrationPanel(
            material_identifier=material.identifier,
            material_digest=material.source_digest,
            reference_view_checksum=str(view.checksum()),
            reference_view_effective_n=Decimal(str(view.effective_n())),
            unit="score-delta",
            config=config,
            synthetic=False,
            transfer=None,
            limits=limits,
            validity=validity,
            assessments=(),
            reference_material=material,
            reference_view=view,
        )
