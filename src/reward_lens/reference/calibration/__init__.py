"""Evidence-bound LoB, LoD, and LoQ calibration for reference episodes.

This module accepts scalar results from a declared, replicated control procedure.  The
``diagnostic-v3`` DataView binds a panel to a real reference episode, but its pairs are not scalar
characterisation measurements, so it cannot manufacture numerical limits.

The normal-model calculation follows Armbruster and Pry, *Limit of Blank, Limit of Detection and
Limit of Quantitation*, Clin Biochem Rev. 2008;29(Suppl 1):S49--S52 (PMCID: PMC2556583):
``LoB = mean(blank) + z_(1-alpha) s_blank``.  A tested known level is LoD only when its one-sided
beta lower bound clears LoB.  LoQ additionally requires declared relative bias and standard-
uncertainty goals.  These calculations assume independent, approximately Gaussian, same-procedure
replicates in the declared unit; no transfer beyond the validity envelope follows from them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from math import sqrt
from statistics import NormalDist
from typing import Sequence

from reward_lens.data import DataView
from reward_lens.reference import ReferenceMaterial

__all__ = [
    "AdjustmentRecord",
    "CalibrationConfig",
    "CalibrationLevel",
    "CalibrationPanel",
    "LimitSet",
    "TransferStatement",
    "ValidityEnvelope",
    "calibrate",
]

_SYNTHETIC_TRANSFER_GAP = Decimal("0.419")
_METHOD = "Armbruster and Pry (2008), Gaussian one-sided replicated-control procedure"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DATASET_ID = re.compile(r"ds:[0-9a-f]{32}\Z")


def _finite_decimal(value: Decimal, *, name: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")
    return value


def _positive_decimal(value: Decimal, *, name: str) -> Decimal:
    value = _finite_decimal(value, name=name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _mean(values: Sequence[Decimal]) -> Decimal:
    return sum(values, Decimal("0")) / Decimal(len(values))


def _sample_sd(values: Sequence[Decimal]) -> Decimal:
    if len(values) < 2:
        raise ValueError("sample standard deviation requires at least two replicates")
    centre = _mean(values)
    variance = sum(((value - centre) ** 2 for value in values), Decimal("0")) / Decimal(
        len(values) - 1
    )
    return Decimal(str(sqrt(float(variance))))


def _z(upper_probability: Decimal) -> Decimal:
    return Decimal(str(NormalDist().inv_cdf(float(upper_probability))))


@dataclass(frozen=True)
class CalibrationConfig:
    """Registered error targets and acceptance requirements for one procedure."""

    alpha: Decimal
    beta: Decimal
    minimum_blank_replicates: int = 2
    minimum_level_replicates: int = 2
    maximum_relative_standard_uncertainty: Decimal = Decimal("0.20")
    maximum_relative_bias: Decimal = Decimal("0.20")

    def __post_init__(self) -> None:
        for name in ("alpha", "beta"):
            value = _finite_decimal(getattr(self, name), name=name)
            if not Decimal("0") < value < Decimal("1"):
                raise ValueError(f"{name} must be strictly between zero and one")
        for name in ("minimum_blank_replicates", "minimum_level_replicates"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 2:
                raise ValueError(f"{name} must be an integer of at least two")
        for name in ("maximum_relative_standard_uncertainty", "maximum_relative_bias"):
            value = _positive_decimal(getattr(self, name), name=name)
            if value > Decimal("1"):
                raise ValueError(f"{name} must be at most one")


@dataclass(frozen=True)
class CalibrationLevel:
    """Known non-blank level and its independent scalar procedure replicates."""

    level: Decimal
    measurements: tuple[Decimal, ...]

    def __init__(self, level: Decimal, measurements: Sequence[Decimal]) -> None:
        object.__setattr__(self, "level", _positive_decimal(level, name="calibration level"))
        values = tuple(
            _finite_decimal(value, name="calibration measurement") for value in measurements
        )
        if not values:
            raise ValueError("calibration level requires at least one measurement")
        object.__setattr__(self, "measurements", values)


@dataclass(frozen=True)
class LimitSet:
    """Three separately defined limits, all expressed in the panel's declared unit."""

    blank: Decimal
    detection: Decimal
    quantification: Decimal
    unit: str = "dimensionless"

    def __post_init__(self) -> None:
        for name in ("blank", "detection", "quantification"):
            _finite_decimal(getattr(self, name), name=name)
        if not self.blank < self.detection < self.quantification:
            raise ValueError("LoB, LoD, and LoQ must be distinct ordered limits")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("limits require a declared unit")


@dataclass(frozen=True)
class ValidityEnvelope:
    """The characterised interval on which this panel's calibration is valid."""

    lower_bound: Decimal
    upper_bound: Decimal
    unit: str = "dimensionless"

    def __post_init__(self) -> None:
        _positive_decimal(self.lower_bound, name="validity lower bound")
        _positive_decimal(self.upper_bound, name="validity upper bound")
        if self.lower_bound > self.upper_bound:
            raise ValueError("validity lower bound must not exceed upper bound")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("validity envelope requires a declared unit")


@dataclass(frozen=True)
class TransferStatement:
    """Structured synthetic-to-target transfer metadata, never a footnote-only claim."""

    gap: Decimal = _SYNTHETIC_TRANSFER_GAP
    synthetic: bool = False

    def __post_init__(self) -> None:
        _finite_decimal(self.gap, name="transfer gap")
        if not isinstance(self.synthetic, bool):
            raise TypeError("synthetic must be a bool")
        if self.synthetic and self.gap != _SYNTHETIC_TRANSFER_GAP:
            raise ValueError("synthetic calibration requires the fixed 0.419 transfer gap")


@dataclass(frozen=True)
class AdjustmentRecord:
    """A separately labelled transformation fitted on scored data, never calibration evidence."""

    raw_value: Decimal
    adjusted_value: Decimal
    operation: str
    fitted_on: str
    kind: str = field(default="adjustment", init=False)

    def __post_init__(self) -> None:
        _finite_decimal(self.raw_value, name="raw value")
        _finite_decimal(self.adjusted_value, name="adjusted value")
        for name in ("operation", "fitted_on"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"adjustment {name} must be non-empty")

    @property
    def delta(self) -> Decimal:
        return self.adjusted_value - self.raw_value


@dataclass(frozen=True)
class _LevelAssessment:
    level: Decimal
    mean: Decimal
    sample_sd: Decimal
    relative_bias: Decimal
    relative_standard_uncertainty: Decimal
    clears_detection: bool
    quantifiable: bool


@dataclass(frozen=True)
class CalibrationPanel:
    """A calibration result or an explicit no-limits refusal for one material and procedure."""

    material_identifier: str
    material_digest: str
    reference_view_checksum: str
    reference_view_effective_n: Decimal
    unit: str
    config: CalibrationConfig
    synthetic: bool
    transfer: TransferStatement | None
    limits: LimitSet | None
    validity: ValidityEnvelope | None
    assessments: tuple[_LevelAssessment, ...]
    method: str = _METHOD
    refusal: str | None = None
    reference_material: ReferenceMaterial | None = None
    reference_view: DataView | None = None

    def __post_init__(self) -> None:
        if not self.material_identifier or not _SHA256.fullmatch(self.material_digest):
            raise ValueError("calibration panel requires material identity and digest")
        if not _DATASET_ID.fullmatch(self.reference_view_checksum):
            raise ValueError("calibration panel requires a canonical DataView checksum")
        _positive_decimal(self.reference_view_effective_n, name="reference view effective n")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("calibration panel requires a declared unit")
        if not isinstance(self.reference_material, ReferenceMaterial):
            raise TypeError("calibration panel requires its authentic ReferenceMaterial")
        self.reference_material.__post_init__()
        if self.material_identifier != self.reference_material.identifier:
            raise ValueError(
                "calibration panel material identifier must bind its ReferenceMaterial"
            )
        if self.material_digest != self.reference_material.source_digest:
            raise ValueError("calibration panel material digest must bind its ReferenceMaterial")
        if not isinstance(self.reference_view, DataView) or not len(self.reference_view):
            raise ValueError("calibration panel requires its authentic non-empty DataView")
        if self.reference_view_checksum != str(self.reference_view.checksum()):
            raise ValueError("calibration panel checksum must bind its ReferenceMaterial DataView")
        if self.reference_view_effective_n != Decimal(str(self.reference_view.effective_n())):
            raise ValueError(
                "calibration panel effective n must bind its ReferenceMaterial DataView"
            )
        if self.transfer is not None and self.transfer.synthetic != self.synthetic:
            raise ValueError("calibration panel and transfer synthetic flags must agree")
        if self.synthetic and (
            self.transfer is None
            or not self.transfer.synthetic
            or self.transfer.gap != _SYNTHETIC_TRANSFER_GAP
        ):
            raise ValueError("synthetic calibration requires the complete 0.419 transfer statement")
        if self.limits is None:
            if self.validity is not None or not self.refusal:
                raise ValueError(
                    "unqualified calibration must publish no envelope and name its refusal"
                )
        else:
            if self.validity is None:
                raise ValueError("published limits require a validity envelope")
            if self.validity.lower_bound != self.limits.quantification:
                raise ValueError("validity-envelope lower bound must equal LoQ")
            if self.limits.unit != self.unit or self.validity.unit != self.unit:
                raise ValueError(
                    "calibration panel, limits, and validity envelope must share one unit"
                )
            if self.refusal is not None:
                raise ValueError("qualified calibration cannot carry a refusal")

    @property
    def qualified(self) -> bool:
        return self.limits is not None


def calibrate(
    *,
    material: ReferenceMaterial,
    reference_view: DataView,
    unit: str,
    config: CalibrationConfig,
    blank_measurements: Sequence[Decimal],
    levels: Sequence[CalibrationLevel],
    synthetic: bool,
    transfer: TransferStatement | None,
) -> CalibrationPanel:
    """Compute a calibration panel, or refuse without publishing numerical limits.

    The DataView names the actual consumed reference episode.  Its pairs are never coerced into
    scalar measurements: numerical limits need the separate, same-unit replicate controls.
    """

    if not isinstance(material, ReferenceMaterial):
        raise TypeError("material must be an accepted ReferenceMaterial")
    material.__post_init__()
    if not isinstance(reference_view, DataView) or not len(reference_view):
        raise ValueError("calibration requires a non-empty usable DataView")
    if not isinstance(config, CalibrationConfig):
        raise TypeError("config must be CalibrationConfig")
    if not isinstance(synthetic, bool):
        raise TypeError("synthetic must be a bool")
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("calibration requires a declared unit")
    if synthetic and (transfer is None or not isinstance(transfer, TransferStatement)):
        raise ValueError("synthetic calibration requires the complete 0.419 transfer statement")
    if transfer is not None and not isinstance(transfer, TransferStatement):
        raise TypeError("transfer must be TransferStatement or None")
    if transfer is not None and transfer.synthetic != synthetic:
        raise ValueError("calibration panel and transfer synthetic flags must agree")
    if synthetic and (not transfer.synthetic or transfer.gap != _SYNTHETIC_TRANSFER_GAP):
        raise ValueError("synthetic calibration requires the complete 0.419 transfer statement")

    blanks = tuple(_finite_decimal(value, name="blank measurement") for value in blank_measurements)
    known_levels = tuple(levels)
    if not all(isinstance(level, CalibrationLevel) for level in known_levels):
        raise TypeError("levels must contain CalibrationLevel records")
    if len({level.level for level in known_levels}) != len(known_levels):
        raise ValueError("each known calibration level must occur once")
    identity = _panel_identity(material, reference_view, unit)
    missing = _replication_refusal(blanks, known_levels, config)
    if missing is not None:
        return CalibrationPanel(
            **identity,
            config=config,
            synthetic=synthetic,
            transfer=transfer,
            limits=None,
            validity=None,
            assessments=(),
            refusal=missing,
        )

    lob = _mean(blanks) + _z(Decimal("1") - config.alpha) * _sample_sd(blanks)
    assessments = tuple(
        _assess_level(level, lob, config)
        for level in sorted(known_levels, key=lambda item: item.level)
    )
    detected = tuple(item for item in assessments if item.clears_detection)
    quantifiable = tuple(item for item in assessments if item.quantifiable)
    if not detected or not quantifiable:
        reason = "no replicated known level meets the registered "
        reason += (
            "detection requirement"
            if not detected
            else "quantification bias and uncertainty requirements"
        )
        return CalibrationPanel(
            **identity,
            config=config,
            synthetic=synthetic,
            transfer=transfer,
            limits=None,
            validity=None,
            assessments=assessments,
            refusal=reason,
        )
    lod = detected[0].level
    loq_candidates = tuple(item for item in quantifiable if item.level > lod)
    if not loq_candidates:
        return CalibrationPanel(
            **identity,
            config=config,
            synthetic=synthetic,
            transfer=transfer,
            limits=None,
            validity=None,
            assessments=assessments,
            refusal="no separately established quantification level lies above LoD",
        )
    loq = loq_candidates[0].level
    limits = LimitSet(blank=lob, detection=lod, quantification=loq, unit=unit)
    validity = ValidityEnvelope(lower_bound=loq, upper_bound=loq_candidates[-1].level, unit=unit)
    return CalibrationPanel(
        **identity,
        config=config,
        synthetic=synthetic,
        transfer=transfer,
        limits=limits,
        validity=validity,
        assessments=assessments,
    )


def _panel_identity(
    material: ReferenceMaterial, reference_view: DataView, unit: str
) -> dict[str, object]:
    return {
        "material_identifier": material.identifier,
        "material_digest": material.source_digest,
        "reference_view_checksum": str(reference_view.checksum()),
        "reference_view_effective_n": Decimal(str(reference_view.effective_n())),
        "unit": unit,
        "reference_material": material,
        "reference_view": reference_view,
    }


def _replication_refusal(
    blanks: tuple[Decimal, ...], levels: tuple[CalibrationLevel, ...], config: CalibrationConfig
) -> str | None:
    if len(blanks) < config.minimum_blank_replicates:
        return f"blank controls are not replicated: received {len(blanks)}, require {config.minimum_blank_replicates}"
    if not levels:
        return "no known-level replicated controls were supplied"
    under_replicated = [
        str(level.level)
        for level in levels
        if len(level.measurements) < config.minimum_level_replicates
    ]
    if under_replicated:
        return (
            "known-level controls are not replicated at "
            + ", ".join(under_replicated)
            + f"; require {config.minimum_level_replicates} replicates per level"
        )
    return None


def _assess_level(
    level: CalibrationLevel, lob: Decimal, config: CalibrationConfig
) -> _LevelAssessment:
    mean = _mean(level.measurements)
    sd = _sample_sd(level.measurements)
    lower_beta_bound = mean - _z(Decimal("1") - config.beta) * sd
    relative_bias = abs(mean - level.level) / level.level
    relative_standard_uncertainty = (
        Decimal("Infinity")
        if mean == 0
        else (sd / Decimal(str(sqrt(len(level.measurements))))) / abs(mean)
    )
    clears_detection = lower_beta_bound > lob
    quantifiable = (
        clears_detection
        and relative_bias <= config.maximum_relative_bias
        and relative_standard_uncertainty <= config.maximum_relative_standard_uncertainty
    )
    return _LevelAssessment(
        level=level.level,
        mean=mean,
        sample_sd=sd,
        relative_bias=relative_bias,
        relative_standard_uncertainty=relative_standard_uncertainty,
        clears_detection=clears_detection,
        quantifiable=quantifiable,
    )
