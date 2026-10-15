"""Calibrated, evidence-only trace onset detection.

The detector receives immutable trace records and an invocation ``RunContext``.  It cannot
receive a trainer or a control action: an alarm is evidence, not an instruction to a live run.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log, sqrt
from statistics import mean
from typing import Any, Sequence

from reward_lens import contracts
from reward_lens.instruments.base import Corpus, Panel, RunContext, Subject
from reward_lens.product.trace import Series, TraceRecord

__all__ = [
    "DetectionSteeringForbidden",
    "DetectorSpec",
    "EffectReference",
    "OnsetDetection",
    "PANEL",
    "ReferenceMaterial",
    "bocpd_run_length_posterior",
    "detect",
]

_VERSION = "1.1.0"
_SOURCES = (
    "Page (1954) Continuous Inspection Schemes; Adams and MacKay (2007) Bayesian Online "
    "Changepoint Detection, arXiv:0710.3742; Armbruster and Pry (2008) LoB/LoD."
)


class DetectionSteeringForbidden(ValueError):
    def __init__(self, changed_run_path: str) -> None:
        super().__init__(
            "DETECTION_STEERING_FORBIDDEN: detection emits evidence only and cannot change "
            f"run path {changed_run_path}"
        )


@dataclass(frozen=True)
class OnsetDetection:
    """Audit-discovered wrapper around the registered, evidence-only detector.

    Audit owns reference opening and puts its immutable binding on ``RunContext``.  The panel
    reads only the canonical trace already attached to the corpus; it never opens a path or
    derives controls from its target.
    """

    id: str = "trace.detection"
    section: contracts.Section = "trace"

    def run(self, subject: Subject, corpus: Corpus, ctx: RunContext) -> list[contracts.Entry]:
        del subject
        trace = getattr(corpus, "trace", None)
        if trace is None:
            return []
        binding = getattr(ctx, "detection_reference", None)
        if binding is None:
            return [
                ctx.absence(
                    "trace",
                    "trace.detection.reference",
                    "calibrated onset detection over the canonical trace",
                    "DETECTION_REFERENCE_ABSENT: AuditRequest.detection_reference was not supplied",
                    "supply one registered, disjoint detector reference bundle",
                    ["calibrated ARL", "false-alarm rate", "detection delay", "LoB", "LoD"],
                    subject_ref=trace.subject_ref,
                    depends_on=list(trace.depends_on),
                )
            ]
        entries = list(detect(trace, binding.spec, context=ctx))
        for entry in entries:
            if entry.kind == "detection":
                entry.result["reference_provenance"]["audit_binding"] = {
                    "source": binding.source,
                    "digest": binding.digest,
                }
        return entries


PANEL = Panel(section="trace", instruments=(OnsetDetection(),))


@dataclass(frozen=True)
class EffectReference:
    """One labelled effect trace, with its declared transition and physical series level."""

    trace: TraceRecord
    onset_index: int
    level: float

    def __post_init__(self) -> None:
        if self.onset_index < 0:
            raise ValueError("effect onset index must be non-negative")
        if not isfinite(self.level):
            raise ValueError("effect level must be finite")


@dataclass(frozen=True)
class ReferenceMaterial:
    """Disjoint calibration and assessment trace partitions.

    Performance estimates are computed only on the assessment partition.  Calibration selects a
    threshold and establishes whether the requested in-control operating point is supportable.
    """

    identifier: str
    calibration_controls: tuple[TraceRecord, ...]
    calibration_effects: tuple[EffectReference, ...]
    assessment_controls: tuple[TraceRecord, ...]
    assessment_effects: tuple[EffectReference, ...]

    def __init__(
        self,
        identifier: str,
        *,
        calibration_controls: Sequence[TraceRecord],
        calibration_effects: Sequence[EffectReference],
        assessment_controls: Sequence[TraceRecord],
        assessment_effects: Sequence[EffectReference],
    ) -> None:
        if not identifier:
            raise ValueError("reference material requires a non-empty identifier")
        values = {
            "calibration_controls": tuple(calibration_controls),
            "calibration_effects": tuple(calibration_effects),
            "assessment_controls": tuple(assessment_controls),
            "assessment_effects": tuple(assessment_effects),
        }
        if not all(values.values()):
            raise ValueError(
                "reference material requires calibration and assessment control/effect runs"
            )
        identities: list[str] = []
        for runs in (values["calibration_controls"], values["assessment_controls"]):
            identities.extend(_identity(run) for run in runs)
        for runs in (values["calibration_effects"], values["assessment_effects"]):
            identities.extend(_identity(item.trace) for item in runs)
        if len(identities) != len(set(identities)):
            raise ValueError("reference calibration and assessment partitions must be disjoint")
        run_ids = [
            run.run_id for run in (*values["calibration_controls"], *values["assessment_controls"])
        ]
        run_ids.extend(
            item.trace.run_id
            for item in (*values["calibration_effects"], *values["assessment_effects"])
        )
        if len(run_ids) != len(set(run_ids)):
            raise ValueError(
                "reference run IDs must be unique across calibration and assessment partitions"
            )
        for item in (*values["calibration_effects"], *values["assessment_effects"]):
            if item.onset_index >= len(item.trace.rows):
                raise ValueError("effect onset index must lie within its trace")
        object.__setattr__(self, "identifier", identifier)
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def digest(self, series: str) -> str:
        return contracts.digest(
            {
                "identifier": self.identifier,
                "series": series,
                "calibration_controls": [
                    _trace_identity(run, series) for run in self.calibration_controls
                ],
                "calibration_effects": [
                    _effect_identity(item, series) for item in self.calibration_effects
                ],
                "assessment_controls": [
                    _trace_identity(run, series) for run in self.assessment_controls
                ],
                "assessment_effects": [
                    _effect_identity(item, series) for item in self.assessment_effects
                ],
            }
        )

    def all_traces(self) -> tuple[TraceRecord, ...]:
        return (
            *self.calibration_controls,
            *(item.trace for item in self.calibration_effects),
            *self.assessment_controls,
            *(item.trace for item in self.assessment_effects),
        )


@dataclass(frozen=True)
class DetectorSpec:
    """Registered operating point and registered definitions of LoB/LoD."""

    series: str
    registered_arl: float
    effect_size: float
    alpha: float
    beta: float
    reference_material: ReferenceMaterial | None
    lob_definition: str = "upper alpha quantile of calibration blank observations in series units"
    lod_definition: str = (
        "lowest replicated effect level with post-onset detection probability at least 1-beta"
    )
    min_effect_replicates: int = 2

    def __post_init__(self) -> None:
        if not self.series:
            raise ValueError("detector series must be named")
        if not isfinite(self.registered_arl) or self.registered_arl <= 0:
            raise ValueError("registered ARL must be positive and finite")
        if not isfinite(self.effect_size) or self.effect_size <= 0:
            raise ValueError("registered effect size must be positive and finite in series units")
        if not 0 < self.alpha < 1 or not 0 < self.beta < 1:
            raise ValueError("alpha and beta must each be strictly between zero and one")
        if self.min_effect_replicates < 2:
            raise ValueError(
                "LoD requires at least two independently labelled effect traces per level"
            )

    def digest(self) -> str:
        return contracts.digest(
            {
                "series": self.series,
                "registered_arl": self.registered_arl,
                "effect_size": self.effect_size,
                "alpha": self.alpha,
                "beta": self.beta,
                "lob_definition": self.lob_definition,
                "lod_definition": self.lod_definition,
                "min_effect_replicates": self.min_effect_replicates,
                "reference": self.reference_material.digest(self.series)
                if self.reference_material
                else None,
            }
        )


@dataclass(frozen=True)
class _Calibration:
    algorithm: str
    threshold: float
    baseline_mean: float
    baseline_sd: float
    calibration_exposure: int
    assessment_exposure: int
    assessment_false_alarms: int
    assessment_false_alarm_rate: float
    arl_lower_bound: float | None
    lob: float
    lod: float | None
    effect_metrics: tuple[dict[str, Any], ...]
    qualified: bool
    reason: str | None
    reference_digest: str


def detect(
    trace: TraceRecord,
    spec: DetectorSpec,
    *,
    context: RunContext,
    run_control: str | None = None,
) -> tuple[contracts.Entry, ...]:
    """Return evidence from a pre-registered detector; this function has no control capability."""
    if run_control is not None:
        raise DetectionSteeringForbidden(run_control)
    reference = spec.reference_material
    if reference is None:
        return (_absence(trace, spec, context, "control/effect reference material"),)
    try:
        observed = _values(trace, spec.series)
    except _NonFiniteObservation as error:
        return (
            _absence(
                trace,
                spec,
                context,
                f"non-finite observation in {error.run_id} at index {error.index}",
            ),
        )
    if _identity(trace) in {_identity(item) for item in reference.all_traces()}:
        raise ValueError(
            "target trace must be disjoint from every calibration and assessment reference"
        )
    if any(item.subject_ref != trace.subject_ref for item in reference.all_traces()):
        raise ValueError("target and reference traces must have matching subject provenance")
    try:
        controls_cal = tuple(_values(item, spec.series) for item in reference.calibration_controls)
        controls_assess = tuple(
            _values(item, spec.series) for item in reference.assessment_controls
        )
        _effects_cal = tuple(
            (item, _values(item.trace, spec.series)) for item in reference.calibration_effects
        )
        effects_assess = tuple(
            (item, _values(item.trace, spec.series)) for item in reference.assessment_effects
        )
    except _NonFiniteObservation as error:
        return (
            _absence(
                trace,
                spec,
                context,
                f"non-finite observation in {error.run_id} at index {error.index}",
            ),
        )
    if not observed or any(not values for values in (*controls_cal, *controls_assess)):
        return (_absence(trace, spec, context, "named control series values"),)

    baseline = tuple(value for values in controls_cal for value in values)
    baseline_mean = mean(baseline)
    baseline_sd = _sample_sd(baseline) or spec.effect_size
    reference_digest = reference.digest(spec.series)
    calibrations = tuple(
        _calibrate(
            algorithm,
            controls_cal,
            controls_assess,
            effects_assess,
            baseline_mean,
            baseline_sd,
            spec,
            reference_digest,
        )
        for algorithm in ("one_sided_cusum", "bocpd_run_length")
    )
    return tuple(_entry(trace, observed, spec, context, item) for item in calibrations)


def bocpd_run_length_posterior(
    values: Sequence[float], *, baseline_mean: float, baseline_sd: float, hazard: float
) -> tuple[tuple[float, ...], ...]:
    """Adams--MacKay online posterior ``P(r_t | x_1:t)`` for a Gaussian mean model.

    ``r_t`` is the current run length.  Each run length carries its own Normal posterior over the
    segment mean, the constant hazard proposes a new run, and all growth and changepoint messages
    are normalized at every observed prefix.  The returned tuple preserves the complete posterior
    at every time, including possible later changepoints.
    """
    if not 0 < hazard < 1:
        raise ValueError("BOCPD hazard must be strictly between zero and one")
    if not isfinite(baseline_mean) or not isfinite(baseline_sd) or baseline_sd <= 0:
        raise ValueError("BOCPD baseline mean and scale must be finite, with positive scale")
    log_posterior = [0.0]
    counts = [0]
    sums = [0.0]
    out: list[tuple[float, ...]] = []
    variance = baseline_sd * baseline_sd
    prior_precision = 1.0 / variance
    for value in values:
        if not isfinite(float(value)):
            raise ValueError("BOCPD observations must be finite")
        log_predictive = []
        for count, total in zip(counts, sums):
            precision = prior_precision + count / variance
            posterior_mean = (prior_precision * baseline_mean + total / variance) / precision
            log_predictive.append(
                _normal_log_density(float(value), posterior_mean, sqrt(variance + 1.0 / precision))
            )
        new_log = [
            log(hazard)
            + _logsumexp(
                [weight + density for weight, density in zip(log_posterior, log_predictive)]
            )
        ]
        new_log.extend(
            log(1.0 - hazard) + weight + density
            for weight, density in zip(log_posterior, log_predictive)
        )
        normalizer = _logsumexp(new_log)
        log_posterior = [weight - normalizer for weight in new_log]
        posterior = [exp(weight) for weight in log_posterior]
        counts = [1, *(count + 1 for count in counts)]
        sums = [float(value), *(total + float(value) for total in sums)]
        out.append(tuple(posterior))
    return tuple(out)


def _calibrate(
    algorithm: str,
    calibration_controls: tuple[tuple[float, ...], ...],
    assessment_controls: tuple[tuple[float, ...], ...],
    assessment_effects: tuple[tuple[EffectReference, tuple[float, ...]], ...],
    baseline_mean: float,
    baseline_sd: float,
    spec: DetectorSpec,
    reference_digest: str,
) -> _Calibration:
    calibration_scores = tuple(
        _scores(algorithm, values, baseline_mean, baseline_sd, spec)
        for values in calibration_controls
    )
    threshold = max(score for run in calibration_scores for score in run) + 1e-12
    calibration_exposure = sum(len(values) for values in calibration_controls)
    assessment_scores = tuple(
        _scores(algorithm, values, baseline_mean, baseline_sd, spec)
        for values in assessment_controls
    )
    assessment_alarm = tuple(_alarm_index(values, threshold) for values in assessment_scores)
    assessment_exposure = sum(len(values) for values in assessment_controls)
    false_alarms = sum(index is not None for index in assessment_alarm)
    false_alarm_rate = false_alarms / assessment_exposure
    required_exposure = -log(spec.alpha) * spec.registered_arl
    calibration_established = calibration_exposure >= required_exposure
    assessment_established = false_alarms == 0 and assessment_exposure >= required_exposure
    arl_lower_bound = assessment_exposure / -log(spec.alpha) if assessment_established else None
    effect_metrics = _effect_metrics(
        algorithm, assessment_effects, threshold, baseline_mean, baseline_sd, spec
    )
    valid_levels = [
        item
        for item in effect_metrics
        if item["n"] >= spec.min_effect_replicates and item["power"] >= 1.0 - spec.beta
    ]
    lod = min((item["level"] for item in valid_levels), default=None)
    registered_level = [item for item in effect_metrics if item["level"] == spec.effect_size]
    registered_power = registered_level[0]["power"] if registered_level else None
    qualified = (
        calibration_established
        and assessment_established
        and lod is not None
        and registered_power is not None
        and registered_power >= 1.0 - spec.beta
    )
    reasons = []
    if not calibration_established:
        reasons.append(
            f"calibration control exposure {calibration_exposure} is below required {required_exposure:.3f} for ARL {spec.registered_arl:g} at alpha {spec.alpha:g}"
        )
    if not assessment_established:
        reasons.append(
            f"assessment control exposure {assessment_exposure} with {false_alarms} alarms does not establish the registered ARL"
        )
    if registered_power is None:
        reasons.append("no assessment effect level equals the registered effect size")
    elif registered_power < 1.0 - spec.beta:
        reasons.append("registered effect level does not meet the registered beta power target")
    if lod is None:
        reasons.append(
            "no supplied effect level has replicated post-onset power sufficient for LoD"
        )
    return _Calibration(
        algorithm,
        threshold,
        baseline_mean,
        baseline_sd,
        calibration_exposure,
        assessment_exposure,
        false_alarms,
        false_alarm_rate,
        arl_lower_bound,
        _upper_quantile(
            tuple(value for values in calibration_controls for value in values), 1.0 - spec.alpha
        ),
        lod,
        effect_metrics,
        qualified,
        "; ".join(reasons) if reasons else None,
        reference_digest,
    )


def _effect_metrics(
    algorithm: str,
    effects: tuple[tuple[EffectReference, tuple[float, ...]], ...],
    threshold: float,
    mu: float,
    sd: float,
    spec: DetectorSpec,
) -> tuple[dict[str, Any], ...]:
    grouped: dict[float, list[tuple[int | None, int]]] = {}
    for item, values in effects:
        alarm = _alarm_index(_scores(algorithm, values, mu, sd, spec), threshold)
        detected = alarm is not None and alarm >= item.onset_index
        delay = alarm - item.onset_index + 1 if detected else None
        grouped.setdefault(item.level, []).append((delay, item.onset_index))
    out = []
    for level in sorted(grouped):
        runs = grouped[level]
        detected_delays = [delay for delay, _ in runs if delay is not None]
        out.append(
            {
                "level": level,
                "n": len(runs),
                "detected": len(detected_delays),
                "power": len(detected_delays) / len(runs),
                "mean_post_onset_delay": mean(detected_delays) if detected_delays else None,
                "declared_onset_indices": [onset for _, onset in runs],
            }
        )
    return tuple(out)


def _entry(
    trace: TraceRecord,
    values: tuple[float, ...],
    spec: DetectorSpec,
    context: RunContext,
    calibration: _Calibration,
) -> contracts.Entry:
    scores = _scores(
        calibration.algorithm, values, calibration.baseline_mean, calibration.baseline_sd, spec
    )
    alarm = _alarm_index(scores, calibration.threshold)
    fields: dict[str, Any] = {
        "target": spec.series,
        "material": spec.reference_material.identifier,
        "threshold": calibration.threshold,
        "LoB": calibration.lob,
        "alpha": spec.alpha,
        "beta": spec.beta,
        "false_alarm_rate": calibration.assessment_false_alarm_rate,
        "unqualified": not calibration.qualified,
    }
    if calibration.qualified:
        registered = next(
            item for item in calibration.effect_metrics if item["level"] == spec.effect_size
        )
        fields.update(
            {
                "LoD": calibration.lod,
                "sensitivity": registered["power"],
                "arl_in_control": calibration.arl_lower_bound,
                "arl_out_of_control": registered["mean_post_onset_delay"],
            }
        )
    else:
        fields["unqualified_reason"] = calibration.reason
    return contracts.Entry(
        entry_id=f"trace.detection.{calibration.algorithm}",
        section="trace",
        kind="detection",
        measurand=f"onset in trace {spec.series} ({_series_unit(trace, spec.series)}) under {calibration.algorithm}",
        method=contracts.Method(
            id=f"detection.{calibration.algorithm}",
            version=_VERSION,
            params_digest=contracts.digest(
                {
                    "spec": spec.digest(),
                    "algorithm": calibration.algorithm,
                    "reference": calibration.reference_digest,
                }
            ),
            procedure="Disjoint calibration and assessment; finite-horizon alpha-bound operating rule. "
            + _SOURCES,
            credited_to="Page; Adams and MacKay; Armbruster and Pry",
        ),
        scope="reconstructed_training_pressure",
        subject_ref=trace.subject_ref,
        depends_on=list(trace.depends_on),
        state="complete" if trace.complete else "partial",
        provenance=context.provenance(),
        limitations=[
            "Detection is evidence only: it cannot pause, throttle, forecast, or control a run.",
            "ARL is a lower bound only when the alpha-bounded assessment control exposure establishes it.",
        ],
        result={
            "algorithm": calibration.algorithm,
            "alarm_index": alarm,
            "statistic": scores[alarm] if alarm is not None else scores[-1],
            "series_unit": _series_unit(trace, spec.series),
            "steering": "none",
            "calibration": {
                "registered_arl": spec.registered_arl,
                "registered_effect_size": spec.effect_size,
                "delay_origin": "declared onset index",
                "calibration_control_exposure": calibration.calibration_exposure,
                "assessment_control_exposure": calibration.assessment_exposure,
                "assessment_false_alarms": calibration.assessment_false_alarms,
                "arl_lower_bound": calibration.arl_lower_bound,
                "right_censored_controls": calibration.assessment_false_alarms == 0,
            },
            "effect_power": list(calibration.effect_metrics),
            "reference_material": spec.reference_material.identifier,
            "reference_provenance": {
                "digest": calibration.reference_digest,
                "trace_interface_source": "ca8395b",
                "audit_instrument_protocol_source": "e01fc4",
            },
        },
        detection=contracts.Detection(**fields),
    )


def _absence(
    trace: TraceRecord, spec: DetectorSpec, context: RunContext, missing: str
) -> contracts.Entry:
    code = (
        "DETECTION_REFERENCE_ABSENT"
        if "non-finite" not in missing
        else "DETECTION_NONFINITE_OBSERVATION"
    )
    return context.absence(
        "trace",
        "trace.detection.reference",
        f"calibrated onset detection for {spec.series}",
        f"{code}: {missing}",
        "supply finite, disjoint held-out calibration and assessment traces",
        ["calibrated ARL", "false-alarm rate", "detection delay", "LoB", "LoD"],
        subject_ref=trace.subject_ref,
        state="NOT_MEASURED",
        depends_on=list(trace.depends_on),
    )


class _NonFiniteObservation(Exception):
    def __init__(self, run_id: str, index: int) -> None:
        self.run_id, self.index = run_id, index


def _values(trace: TraceRecord, name: str) -> tuple[float, ...]:
    series: Series | None = trace.series.get(name) or trace.components.get(name)
    if series is None or not series.present:
        return ()
    values = []
    for index, value in enumerate(series.values):
        if value is None or not isfinite(float(value)):
            raise _NonFiniteObservation(trace.run_id, index)
        values.append(float(value))
    return tuple(values)


def _series_unit(trace: TraceRecord, name: str) -> str:
    return str(
        (trace.series.get(name) or trace.components.get(name)).provenance.get(
            "unit", "declared trace series unit"
        )
    )


def _identity(trace: TraceRecord) -> str:
    return contracts.digest(
        {
            "run_id": trace.run_id,
            "source": trace.source,
            "subject_ref": trace.subject_ref,
            "rows": len(trace.rows),
            "series": {
                name: _identity_values(series.values) for name, series in trace.series.items()
            },
            "components": {
                name: _identity_values(series.values) for name, series in trace.components.items()
            },
        }
    )


def _identity_values(values: Sequence[float | None]) -> list[float | None | str]:
    return [
        value if value is None or isfinite(float(value)) else f"nonfinite:{value}"
        for value in values
    ]


def _trace_identity(trace: TraceRecord, series: str) -> dict[str, Any]:
    return {
        "identity": _identity(trace),
        "run_id": trace.run_id,
        "source": trace.source,
        "subject_ref": trace.subject_ref,
        "values": list(_values(trace, series)),
    }


def _effect_identity(item: EffectReference, series: str) -> dict[str, Any]:
    return {
        **_trace_identity(item.trace, series),
        "onset_index": item.onset_index,
        "level": item.level,
    }


def _sample_sd(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return 0.0
    centre = mean(values)
    return sqrt(sum((value - centre) ** 2 for value in values) / (len(values) - 1))


def _scores(
    algorithm: str, values: tuple[float, ...], mu: float, sd: float, spec: DetectorSpec
) -> tuple[float, ...]:
    if algorithm == "one_sided_cusum":
        state, out, k = 0.0, [], spec.effect_size / (2.0 * sd)
        for value in values:
            state = max(0.0, state + (value - mu) / sd - k)
            out.append(state)
        return tuple(out)
    # ``1 - P(r_t=t)`` is the causal posterior mass that at least one changepoint has happened
    # in the observed prefix.  Unlike ``P(r_t=0)``, which can equal a constant hazard under this
    # model, it uses the full run-length posterior to distinguish a new regime from no change.
    return tuple(
        1.0 - posterior[-1]
        for posterior in bocpd_run_length_posterior(
            values, baseline_mean=mu, baseline_sd=sd, hazard=1.0 / spec.registered_arl
        )
    )


def _normal_log_density(value: float, centre: float, scale: float) -> float:
    z = (value - centre) / scale
    return -0.5 * z * z - log(scale) - 0.5 * log(2.0 * 3.141592653589793)


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + log(sum(exp(value - maximum) for value in values))


def _alarm_index(scores: tuple[float, ...], threshold: float) -> int | None:
    return next((index for index, value in enumerate(scores) if value >= threshold), None)


def _upper_quantile(values: tuple[float, ...], probability: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(probability * len(ordered) + 0.999999999)))]
