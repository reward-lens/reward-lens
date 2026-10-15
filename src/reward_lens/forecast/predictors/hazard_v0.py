"""Frozen-reference hazard-v0 forecast predictor.

This module is intentionally small and dependency-free.  It fits a penalised
logistic discrete-time onset hazard and L2-penalised linear quantile regressions
against a named frozen reference set.  The resulting object only predicts: it
cannot update coefficients, scalers, or reference material after ``fit``.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from statistics import median
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from reward_lens.forecast.evidence import BASELINES, ForecastSpec

_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
_SERIES = (
    "mean_reward",
    "contrast",
    "within_group_variance",
    "kl",
    "length",
    "hack_rate",
)
_PENALTIES = (0.01, 0.1, 1.0)


class PredictorRefusal(ValueError):
    """A fail-closed predictor refusal with its violated information boundary."""


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _freeze(value: Any) -> Any:
    """Recursively freeze every value retained after fitting.

    ``MappingProxyType`` is intentionally a real non-``dict`` mapping: base
    ``dict.__setitem__`` cannot bypass it. Tuples make all retained sequences
    equally immutable. The predictor only exposes ordinary dict/list values at
    the explicit serialization boundary below.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Create ordinary JSON-shaped values only for a caller-facing wire material view."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def _float(value: object, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PredictorRefusal(f"PREDICTOR_REFERENCE_ABSENT field={field} is not numeric") from exc
    if not math.isfinite(result):
        raise PredictorRefusal(f"PREDICTOR_REFERENCE_ABSENT field={field} is not finite")
    return result


def _normalise_run(raw: Mapping[str, Any]) -> dict[str, Any]:
    name = str(raw.get("name", "")).strip()
    observations = raw.get("observations")
    if not name:
        raise PredictorRefusal("PREDICTOR_REFERENCE_ABSENT reference run has no name")
    if not isinstance(observations, Sequence) or len(observations) < 4:
        raise PredictorRefusal(
            f"PREDICTOR_REFERENCE_ABSENT run={name} needs at least four observations"
        )
    result: list[dict[str, float]] = []
    for expected_step, row in enumerate(observations):
        if not isinstance(row, Mapping):
            raise PredictorRefusal(
                f"PREDICTOR_REFERENCE_ABSENT run={name} observation is not an object"
            )
        if int(row.get("step", -1)) != expected_step:
            raise PredictorRefusal(
                f"PREDICTOR_REFERENCE_ABSENT run={name} requires consecutive zero-based steps"
            )
        result.append({key: _float(row.get(key), field=f"{name}.{key}") for key in _SERIES})
    onset = raw.get("onset_step")
    if onset is not None and (not isinstance(onset, int) or onset < 0 or onset >= len(result)):
        raise PredictorRefusal(f"PREDICTOR_REFERENCE_ABSENT run={name} has invalid onset_step")
    return {"name": name, "observations": result, "onset_step": onset}


def _reference_step_stats(
    runs: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], tuple[float, float]]:
    stats: dict[tuple[str, int], tuple[float, float]] = {}
    for key in _SERIES:
        max_steps = min(len(run["observations"]) for run in runs)
        for step in range(max_steps):
            values = [float(run["observations"][step][key]) for run in runs]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            stats[key, step] = (mean, math.sqrt(variance) or 1.0)
    return stats


def _feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for key in _SERIES:
        names.extend(
            (
                f"{key}_level",
                f"{key}_slope",
                f"{key}_second_difference",
                f"{key}_dispersion",
                f"{key}_z",
            )
        )
    return tuple(names + ["step", "log_step", "hack_rate_level"])


def _features(
    run: Mapping[str, Any], cutoff: int, stats: Mapping[tuple[str, int], tuple[float, float]]
) -> list[float]:
    observations = run["observations"]
    if cutoff < 2 or cutoff >= len(observations):
        raise PredictorRefusal(
            f"PREDICTOR_FUTURE_FEATURE run={run['name']} cutoff={cutoff}; features require observations through cutoff only"
        )
    values: list[float] = []
    for key in _SERIES:
        current = float(observations[cutoff][key])
        previous = float(observations[cutoff - 1][key])
        before = float(observations[cutoff - 2][key])
        window = [float(row[key]) for row in observations[max(0, cutoff - 2) : cutoff + 1]]
        mean = sum(window) / len(window)
        dispersion = math.sqrt(sum((value - mean) ** 2 for value in window) / len(window))
        ref_mean, ref_sd = stats.get((key, cutoff), (current, 1.0))
        values.extend(
            (
                current,
                current - previous,
                current - 2 * previous + before,
                dispersion,
                (current - ref_mean) / ref_sd,
            )
        )
    values.extend((float(cutoff), math.log1p(cutoff), float(observations[cutoff]["hack_rate"])))
    return values


def _scale(rows: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    width = len(rows[0])
    means = [sum(row[index] for row in rows) / len(rows) for index in range(width)]
    scales = []
    for index, mean in enumerate(means):
        variance = sum((row[index] - mean) ** 2 for row in rows) / len(rows)
        scales.append(math.sqrt(variance) or 1.0)
    return means, scales


def _apply_scale(
    row: Sequence[float], means: Sequence[float], scales: Sequence[float]
) -> list[float]:
    return [(value - mean) / scale for value, mean, scale in zip(row, means, scales, strict=True)]


def _fit_logistic(
    rows: Sequence[Sequence[float]], labels: Sequence[int], penalty: float
) -> list[float]:
    coefficients = [0.0] * (len(rows[0]) + 1)
    n = float(len(rows))
    for _ in range(60):
        gradient = [0.0] * len(coefficients)
        for row, label in zip(rows, labels, strict=True):
            probability = _sigmoid(
                coefficients[0]
                + sum(weight * value for weight, value in zip(coefficients[1:], row, strict=True))
            )
            error = probability - label
            gradient[0] += error
            for index, value in enumerate(row, start=1):
                gradient[index] += error * value
        rate = 0.08
        coefficients[0] -= rate * gradient[0] / n
        for index in range(1, len(coefficients)):
            coefficients[index] -= rate * (gradient[index] / n + penalty * coefficients[index])
    return coefficients


def _fit_quantile(
    rows: Sequence[Sequence[float]], values: Sequence[float], quantile: float, penalty: float
) -> list[float]:
    coefficients = [0.0] * (len(rows[0]) + 1)
    n = float(len(rows))
    for _ in range(80):
        gradient = [0.0] * len(coefficients)
        for row, value in zip(rows, values, strict=True):
            estimate = coefficients[0] + sum(
                weight * item for weight, item in zip(coefficients[1:], row, strict=True)
            )
            residual_gradient = -(quantile - (1.0 if value < estimate else 0.0))
            gradient[0] += residual_gradient
            for index, item in enumerate(row, start=1):
                gradient[index] += residual_gradient * item
        rate = 0.035
        coefficients[0] -= rate * gradient[0] / n
        for index in range(1, len(coefficients)):
            coefficients[index] -= rate * (gradient[index] / n + penalty * coefficients[index])
    return coefficients


def _linear(coefficients: Sequence[float], row: Sequence[float]) -> float:
    return coefficients[0] + sum(
        weight * value for weight, value in zip(coefficients[1:], row, strict=True)
    )


def _hazard_rows(
    runs: Sequence[Mapping[str, Any]],
    spec: ForecastSpec,
    stats: Mapping[tuple[str, int], tuple[float, float]],
) -> tuple[list[list[float]], list[int], list[str]]:
    rows: list[list[float]] = []
    labels: list[int] = []
    owners: list[str] = []
    horizon = spec.horizon
    for run in runs:
        onset = run["onset_step"]
        final_cutoff = len(run["observations"]) - horizon - 1
        for cutoff in range(2, max(2, final_cutoff + 1)):
            if onset is not None and cutoff >= onset:
                break
            future_end = min(len(run["observations"]) - 1, cutoff + horizon)
            event = onset is not None and cutoff < onset <= future_end
            rows.append(_features(run, cutoff, stats))
            labels.append(int(event))
            owners.append(str(run["name"]))
    if not rows:
        raise PredictorRefusal(
            "PREDICTOR_REFERENCE_ABSENT no at-risk checkpoints under registered horizon"
        )
    return rows, labels, owners


def _fraction_rows(
    runs: Sequence[Mapping[str, Any]],
    spec: ForecastSpec,
    stats: Mapping[tuple[str, int], tuple[float, float]],
    key: str,
) -> tuple[list[list[float]], list[float]]:
    rows: list[list[float]] = []
    targets: list[float] = []
    for run in runs:
        final_cutoff = len(run["observations"]) - spec.horizon - 1
        for cutoff in range(2, max(2, final_cutoff + 1)):
            future = run["observations"][cutoff + 1 : cutoff + spec.horizon + 1]
            if len(future) != spec.horizon:
                continue
            rows.append(_features(run, cutoff, stats))
            targets.append(sum(float(row[key]) for row in future) / len(future))
    if not rows:
        raise PredictorRefusal(
            "PREDICTOR_REFERENCE_ABSENT no complete fraction windows under registered horizon"
        )
    return rows, targets


def _target_rows(
    runs: Sequence[Mapping[str, Any]],
    spec: ForecastSpec,
    stats: Mapping[tuple[str, int], tuple[float, float]],
    target: str,
) -> tuple[list[list[float]], list[float], list[str]]:
    if target == "T1":
        rows, labels, owners = _hazard_rows(runs, spec, stats)
        return rows, [float(label) for label in labels], owners
    key = "hack_rate" if target == "T2" else "contrast"
    rows, values = _fraction_rows(runs, spec, stats, key)
    owners: list[str] = []
    for run in runs:
        final_cutoff = len(run["observations"]) - spec.horizon - 1
        owners.extend([str(run["name"])] * max(0, final_cutoff - 1))
    return rows, values, owners


def _fit_target(
    rows: Sequence[Sequence[float]],
    values: Sequence[float],
    target: str,
    quantile: float | None,
    penalty: float,
) -> list[float]:
    if target == "T1":
        return _fit_logistic(rows, [int(value) for value in values], penalty)
    assert quantile is not None
    return _fit_quantile(rows, values, quantile, penalty)


def _prediction(coefficients: Sequence[float], row: Sequence[float], target: str) -> float:
    value = _linear(coefficients, row)
    return _sigmoid(value) if target == "T1" else min(1.0, max(0.0, value))


def _loss(target: str, quantile: float | None, prediction: float, value: float) -> float:
    if target == "T1":
        return (prediction - value) ** 2
    assert quantile is not None
    return (
        quantile * (value - prediction)
        if value >= prediction
        else (1 - quantile) * (prediction - value)
    )


def _stats_digest(stats: Mapping[tuple[str, int], tuple[float, float]]) -> str:
    return _digest(
        [
            {"series": key, "step": step, "mean": mean, "sd": sd}
            for (key, step), (mean, sd) in sorted(stats.items())
        ]
    )


def _choose_penalty(
    runs: Sequence[Mapping[str, Any]], spec: ForecastSpec, target: str, quantile: float | None
) -> tuple[float, tuple[dict[str, Any], ...]]:
    """Nested LORO with feature statistics and transforms fitted only on each training fold."""
    reports: list[dict[str, Any]] = []
    names = [str(run["name"]) for run in runs]
    for held_out in names:
        train = [run for run in runs if run["name"] != held_out]
        validation = [run for run in runs if run["name"] == held_out]
        inner_scores: dict[float, float] = {}
        for penalty in _PENALTIES:
            fold_scores: list[float] = []
            for inner_out in [str(run["name"]) for run in train]:
                inner_train = [run for run in train if run["name"] != inner_out]
                inner_valid = [run for run in train if run["name"] == inner_out]
                inner_stats = _reference_step_stats(inner_train)
                x_train, y_train, _ = _target_rows(inner_train, spec, inner_stats, target)
                means, scales = _scale(x_train)
                coefficients = _fit_target(
                    [_apply_scale(row, means, scales) for row in x_train],
                    y_train,
                    target,
                    quantile,
                    penalty,
                )
                x_valid, y_valid, _ = _target_rows(inner_valid, spec, inner_stats, target)
                predictions = [
                    _prediction(coefficients, _apply_scale(row, means, scales), target)
                    for row in x_valid
                ]
                fold_scores.append(
                    sum(
                        _loss(target, quantile, prediction, value)
                        for prediction, value in zip(predictions, y_valid, strict=True)
                    )
                    / len(y_valid)
                )
            inner_scores[penalty] = sum(fold_scores) / len(fold_scores)
        selected = min(inner_scores, key=inner_scores.get)
        outer_stats = _reference_step_stats(train)
        x_train, y_train, _ = _target_rows(train, spec, outer_stats, target)
        means, scales = _scale(x_train)
        coefficients = _fit_target(
            [_apply_scale(row, means, scales) for row in x_train],
            y_train,
            target,
            quantile,
            selected,
        )
        x_valid, y_valid, _ = _target_rows(validation, spec, outer_stats, target)
        valid = [
            _prediction(coefficients, _apply_scale(row, means, scales), target) for row in x_valid
        ]
        report: dict[str, Any] = {
            "held_out": held_out,
            "selection_scope": "inner-loro",
            "selected_penalty": selected,
            "outer_reference_names": [str(run["name"]) for run in train],
            "outer_statistics_digest": _stats_digest(outer_stats),
            "outer_transform_digest": _digest({"means": means, "scales": scales}),
        }
        score = sum(
            _loss(target, quantile, prediction, value)
            for prediction, value in zip(valid, y_valid, strict=True)
        ) / len(y_valid)
        if target == "T1":
            report["outer_brier"] = score
        else:
            report["quantile"] = quantile
            report["outer_pinball"] = score
        reports.append(report)
    return float(median([float(report["selected_penalty"]) for report in reports])), tuple(reports)


def _canonical_reference_material(runs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": str(run["name"]),
            "onset_step": run["onset_step"],
            "observations": [
                {key: float(row[key]) for key in _SERIES} | {"step": step}
                for step, row in enumerate(run["observations"])
            ],
        }
        for run in runs
    ]


def _canonical_statistics(
    stats: Mapping[tuple[str, int], tuple[float, float]],
) -> list[dict[str, float | int | str]]:
    return [
        {"series": series, "step": step, "mean": mean, "sd": sd}
        for (series, step), (mean, sd) in sorted(stats.items())
    ]


def _onset_target(run: Mapping[str, Any], cutoff: int, spec: ForecastSpec) -> float:
    onset = run["onset_step"]
    end = min(len(run["observations"]) - 1, cutoff + spec.horizon)
    return float(onset is not None and cutoff < onset <= end)


def _event_history(run: Mapping[str, Any], cutoff: int, spec: ForecastSpec) -> list[float]:
    onset = run["onset_step"]
    return [float(onset is not None and step >= onset) for step in range(cutoff + 1)]


def _fraction_target(run: Mapping[str, Any], cutoff: int, spec: ForecastSpec, key: str) -> float:
    future = run["observations"][cutoff + 1 : cutoff + spec.horizon + 1]
    if len(future) != spec.horizon:
        raise PredictorRefusal(
            f"PREDICTOR_REFERENCE_ABSENT run={run['name']} lacks a frozen {key} target window"
        )
    return sum(float(row[key]) for row in future) / len(future)


def _baseline_vector(run: Mapping[str, Any], cutoff: int) -> list[float]:
    return [float(run["observations"][cutoff][key]) for key in _SERIES]


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    width = len(vector)
    augmented = [matrix[index][:] + [vector[index]] for index in range(width)]
    for col in range(width):
        pivot = max(range(col, width), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            return [0.0] * width
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        divisor = augmented[col][col]
        augmented[col] = [item / divisor for item in augmented[col]]
        for row in range(width):
            if row == col:
                continue
            factor = augmented[row][col]
            augmented[row] = [
                item - factor * base
                for item, base in zip(augmented[row], augmented[col], strict=True)
            ]
    return [augmented[index][-1] for index in range(width)]


def _reference_covariance_predict(
    runs: Sequence[Mapping[str, Any]],
    query: Sequence[float],
    cutoff: int,
    spec: ForecastSpec,
    target: str,
) -> tuple[float, dict[str, Any]]:
    rows: list[list[float]] = []
    values: list[float] = []
    key = "hack_rate" if target == "T2" else "contrast"
    for run in runs:
        last = min(cutoff, len(run["observations"]) - spec.horizon - 1)
        for step in range(2, last + 1):
            rows.append(_baseline_vector(run, step))
            values.append(
                _onset_target(run, step, spec)
                if target == "T1"
                else _fraction_target(run, step, spec, key)
            )
    if len(rows) < 2:
        raise PredictorRefusal(
            "PREDICTOR_REFERENCE_ABSENT covariance baseline has fewer than two frozen rows"
        )
    width = len(rows[0])
    means = [sum(row[index] for row in rows) / len(rows) for index in range(width)]
    mean_y = sum(values) / len(values)
    denom = max(1, len(rows) - 1)
    covariance = [
        [
            sum((row[i] - means[i]) * (row[j] - means[j]) for row in rows) / denom
            for j in range(width)
        ]
        for i in range(width)
    ]
    cross = [
        sum((value - mean_y) * (row[i] - means[i]) for row, value in zip(rows, values, strict=True))
        / denom
        for i in range(width)
    ]
    for index in range(width):
        covariance[index][index] += 1e-6
    coefficients = _solve(covariance, cross)
    estimate = min(
        1.0,
        max(
            0.0,
            mean_y
            + sum(
                weight * (value - mean)
                for weight, value, mean in zip(coefficients, query, means, strict=True)
            ),
        ),
    )
    provenance = {
        "target": target,
        "cutoff": cutoff,
        "n_reference_rows": len(rows),
        "means": means,
        "coefficients": coefficients,
    }
    return estimate, provenance


@dataclass(frozen=True)
class ForecastDistribution:
    onset_probability: float
    hack_rate_quantiles: tuple[float, ...]
    contrast_quantiles: tuple[float, ...]
    survival: tuple[float, ...]
    feature_cutoff: int


@dataclass(frozen=True)
class FittedPredictor:
    spec: ForecastSpec
    reference_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    scaler_means: tuple[float, ...]
    scaler_scales: tuple[float, ...]
    hazard_coefficients: tuple[float, ...]
    hack_quantile_coefficients: tuple[tuple[float, ...], ...]
    contrast_quantile_coefficients: tuple[tuple[float, ...], ...]
    penalty: float
    fold_reports: tuple[dict[str, Any], ...]
    fraction_fold_reports: Mapping[str, tuple[dict[str, Any], ...]]
    hack_quantile_penalties: tuple[float, ...]
    contrast_quantile_penalties: tuple[float, ...]
    hack_scaler_means: tuple[float, ...]
    hack_scaler_scales: tuple[float, ...]
    contrast_scaler_means: tuple[float, ...]
    contrast_scaler_scales: tuple[float, ...]
    censored_runs: tuple[str, ...]
    reference_stats: Mapping[tuple[str, int], tuple[float, float]]
    frozen_reference_material: tuple[Mapping[str, Any], ...]
    reference_digest: str
    digest: str

    @property
    def outer_folds(self) -> int:
        return len(self.fold_reports)

    def material(self) -> dict[str, Any]:
        return {
            "reference_names": list(self.reference_names),
            "feature_names": list(self.feature_names),
            "scaler": {"means": list(self.scaler_means), "scales": list(self.scaler_scales)},
            "hazard_coefficients": list(self.hazard_coefficients),
            "hack_quantile_coefficients": [list(row) for row in self.hack_quantile_coefficients],
            "contrast_quantile_coefficients": [
                list(row) for row in self.contrast_quantile_coefficients
            ],
            "penalty": self.penalty,
            "hack_quantile_penalties": list(self.hack_quantile_penalties),
            "contrast_quantile_penalties": list(self.contrast_quantile_penalties),
            "hack_scaler": {
                "means": list(self.hack_scaler_means),
                "scales": list(self.hack_scaler_scales),
            },
            "contrast_scaler": {
                "means": list(self.contrast_scaler_means),
                "scales": list(self.contrast_scaler_scales),
            },
            "fold_reports": [_thaw(report) for report in self.fold_reports],
            "fraction_fold_reports": {
                key: [_thaw(report) for report in value]
                for key, value in self.fraction_fold_reports.items()
            },
            "censored_runs": list(self.censored_runs),
            "frozen_reference_material": _canonical_reference_material(
                self.frozen_reference_material
            ),
            "reference_statistics": _canonical_statistics(self.reference_stats),
            "reference_digest": self.reference_digest,
            "digest": self.digest,
        }

    def predict(
        self, run: Mapping[str, Any], cutoff: int, *, future: bool = False
    ) -> ForecastDistribution:
        if future:
            raise PredictorRefusal(
                f"PREDICTOR_FUTURE_FEATURE run={run.get('name', 'unnamed')} cutoff={cutoff}; prediction may not request future observations"
            )
        normalised = _normalise_run(run)
        raw_features = _features(normalised, cutoff, self.reference_stats)
        features = _apply_scale(raw_features, self.scaler_means, self.scaler_scales)
        hazard = _sigmoid(_linear(self.hazard_coefficients, features))
        survival = [1.0]
        remaining = 1.0
        for _ in range(self.spec.horizon):
            remaining *= 1.0 - hazard
            survival.append(remaining)
        onset = 1.0 - survival[-1]
        hack = tuple(
            min(
                1.0,
                max(
                    0.0,
                    _linear(
                        coefficients,
                        _apply_scale(raw_features, self.hack_scaler_means, self.hack_scaler_scales),
                    ),
                ),
            )
            for coefficients in self.hack_quantile_coefficients
        )
        contrast = tuple(
            min(
                1.0,
                max(
                    0.0,
                    _linear(
                        coefficients,
                        _apply_scale(
                            raw_features, self.contrast_scaler_means, self.contrast_scaler_scales
                        ),
                    ),
                ),
            )
            for coefficients in self.contrast_quantile_coefficients
        )
        return ForecastDistribution(
            onset, tuple(sorted(hack)), tuple(sorted(contrast)), tuple(survival), cutoff
        )

    def issue_row(
        self, run: Mapping[str, Any], *, cutoff: int, target: str, source_name: str
    ) -> dict[str, Any]:
        from reward_lens.forecast.baselines import (
            d70_climatology,
            d70_drift,
            d70_drift_quantiles,
            d70_persistence,
            d70_persistence_quantiles,
            d70_quantile_distribution,
        )

        prediction = self.predict(run, cutoff)
        normalised = _normalise_run(run)
        details: dict[str, Any]
        if target == "T1":
            forecast: Any = prediction.onset_probability
            history = _event_history(normalised, cutoff, self.spec)
            persistence = d70_persistence(history, checkpoint=cutoff)
            targets = [
                _onset_target(reference, cutoff, self.spec)
                for reference in self.frozen_reference_material
            ]
            climatology = d70_climatology(targets, reference_cutoff=len(targets))
            drift = d70_drift(history, checkpoint=cutoff, horizon=self.spec.horizon)
            covariance, covariance_detail = _reference_covariance_predict(
                self.frozen_reference_material,
                _baseline_vector(normalised, cutoff),
                cutoff,
                self.spec,
                "T1",
            )
            baselines: Any = {
                "persistence": persistence,
                "climatology": climatology,
                "drift": drift,
                "covariance": covariance,
            }
            details = {
                "persistence": {"target": "T1", "cutoff": cutoff, "history": history},
                "climatology": {"target": "T1", "cutoff": cutoff, "reference_targets": targets},
                "drift": {
                    "target": "T1",
                    "cutoff": cutoff,
                    "history": history,
                    "horizon": self.spec.horizon,
                },
                "covariance": covariance_detail,
            }
        elif target == "T2":
            forecast = {"values": list(prediction.hack_rate_quantiles)}
            baselines, details = self._fraction_baselines(
                normalised,
                cutoff,
                "T2",
                "hack_rate",
                d70_persistence_quantiles,
                d70_drift_quantiles,
                d70_quantile_distribution,
            )
        elif target == "T3":
            forecast = {"values": list(prediction.contrast_quantiles)}
            baselines, details = self._fraction_baselines(
                normalised,
                cutoff,
                "T3",
                "contrast",
                d70_persistence_quantiles,
                d70_drift_quantiles,
                d70_quantile_distribution,
            )
        else:
            raise ValueError(f"target {target!r} is not registered")
        provenance = {
            name: {
                "cutoff": cutoff,
                "fit_digest": _digest(
                    {
                        "reference_digest": self.reference_digest,
                        "baseline": name,
                        "material": details[name],
                    }
                ),
            }
            for name in BASELINES
        }
        return {
            "target": target,
            "checkpoint": cutoff,
            "source_name": source_name,
            "source_cutoff": cutoff,
            "feature_cutoff": cutoff,
            "model_selection_cutoff": cutoff,
            "forecast": forecast,
            "baselines": baselines,
            "baseline_provenance": provenance,
            "model_digest": self.digest,
        }

    def _fraction_baselines(
        self,
        run: Mapping[str, Any],
        cutoff: int,
        target: str,
        key: str,
        persistence_fn: Any,
        drift_fn: Any,
        quantile_fn: Any,
    ) -> tuple[dict[str, dict[str, list[float]]], dict[str, Any]]:
        history = [float(row[key]) for row in run["observations"]]
        persistence = persistence_fn(history, checkpoint=cutoff)
        drift = drift_fn(history, checkpoint=cutoff, horizon=self.spec.horizon)
        targets = [
            _fraction_target(reference, cutoff, self.spec, key)
            for reference in self.frozen_reference_material
        ]
        climatology = quantile_fn(targets)
        query = _baseline_vector(run, cutoff)
        covariance_values: list[float] = []
        covariance_details: list[dict[str, Any]] = []
        for reference in self.frozen_reference_material:
            value, detail = _reference_covariance_predict(
                self.frozen_reference_material,
                _baseline_vector(reference, cutoff),
                cutoff,
                self.spec,
                target,
            )
            covariance_values.append(value)
            covariance_details.append(detail)
        query_covariance, query_detail = _reference_covariance_predict(
            self.frozen_reference_material, query, cutoff, self.spec, target
        )
        covariance = quantile_fn(covariance_values + [query_covariance])
        baselines = {
            "persistence": {"values": list(persistence["values"])},
            "climatology": {"values": list(climatology["values"])},
            "drift": {"values": list(drift["values"])},
            "covariance": {"values": list(covariance["values"])},
        }
        details = {
            "persistence": {"target": target, "cutoff": cutoff, "history": history[: cutoff + 1]},
            "climatology": {"target": target, "cutoff": cutoff, "reference_targets": targets},
            "drift": {
                "target": target,
                "cutoff": cutoff,
                "history": history[: cutoff + 1],
                "horizon": self.spec.horizon,
            },
            "covariance": {
                "target": target,
                "cutoff": cutoff,
                "query": query,
                "query_fit": query_detail,
                "reference_fits": covariance_details,
            },
        }
        return baselines, details


class HazardV0:
    """Factory for a frozen model.  There is deliberately no incremental-fit method."""

    @staticmethod
    def fit(reference_runs: Iterable[Mapping[str, Any]], spec: ForecastSpec) -> FittedPredictor:
        raw = tuple(reference_runs)
        if not raw:
            raise PredictorRefusal("PREDICTOR_REFERENCE_ABSENT reference_runs is empty")
        runs = tuple(_normalise_run(run) for run in raw)
        names = tuple(str(run["name"]) for run in runs)
        if len(set(names)) != len(names):
            raise PredictorRefusal("PREDICTOR_REFERENCE_ABSENT reference run names must be unique")
        stats = _reference_step_stats(runs)
        penalty, reports = _choose_penalty(runs, spec, "T1", None)
        x_hazard, y_hazard, _ = _hazard_rows(runs, spec, stats)
        means, scales = _scale(x_hazard)
        scaled_hazard = [_apply_scale(row, means, scales) for row in x_hazard]
        hazard = _fit_logistic(scaled_hazard, y_hazard, penalty)
        x_hack, y_hack = _fraction_rows(runs, spec, stats, "hack_rate")
        x_contrast, y_contrast = _fraction_rows(runs, spec, stats, "contrast")
        hack_means, hack_scales = _scale(x_hack)
        contrast_means, contrast_scales = _scale(x_contrast)
        scaled_hack = [_apply_scale(row, hack_means, hack_scales) for row in x_hack]
        scaled_contrast = [_apply_scale(row, contrast_means, contrast_scales) for row in x_contrast]
        hack_selection = [_choose_penalty(runs, spec, "T2", quantile) for quantile in _QUANTILES]
        contrast_selection = [
            _choose_penalty(runs, spec, "T3", quantile) for quantile in _QUANTILES
        ]
        hack_penalties = tuple(selection[0] for selection in hack_selection)
        contrast_penalties = tuple(selection[0] for selection in contrast_selection)
        hack = tuple(
            tuple(_fit_quantile(scaled_hack, y_hack, quantile, selected))
            for quantile, selected in zip(_QUANTILES, hack_penalties, strict=True)
        )
        contrast = tuple(
            tuple(_fit_quantile(scaled_contrast, y_contrast, quantile, selected))
            for quantile, selected in zip(_QUANTILES, contrast_penalties, strict=True)
        )
        censored = tuple(str(run["name"]) for run in runs if run["onset_step"] is None)
        reference_material = _canonical_reference_material(runs)
        reference_digest = _digest(reference_material)
        material = {
            "reference_names": list(names),
            "feature_names": list(_feature_names()),
            "scaler": {"means": means, "scales": scales},
            "hazard_coefficients": hazard,
            "hack_quantile_coefficients": hack,
            "contrast_quantile_coefficients": contrast,
            "penalty": penalty,
            "hack_quantile_penalties": hack_penalties,
            "contrast_quantile_penalties": contrast_penalties,
            "hack_scaler": {"means": hack_means, "scales": hack_scales},
            "contrast_scaler": {"means": contrast_means, "scales": contrast_scales},
            "fold_reports": reports,
            "fraction_fold_reports": {"T2": hack_selection, "T3": contrast_selection},
            "censored_runs": censored,
            "frozen_reference_material": reference_material,
            "reference_statistics": _canonical_statistics(stats),
            "reference_digest": reference_digest,
        }
        frozen_fold_reports = _freeze(reports)
        frozen_fraction_reports = _freeze(
            {
                "T2": tuple(
                    report
                    for _, selection_reports in hack_selection
                    for report in selection_reports
                ),
                "T3": tuple(
                    report
                    for _, selection_reports in contrast_selection
                    for report in selection_reports
                ),
            }
        )
        return FittedPredictor(
            spec,
            names,
            _feature_names(),
            tuple(means),
            tuple(scales),
            tuple(hazard),
            hack,
            contrast,
            penalty,
            frozen_fold_reports,
            frozen_fraction_reports,
            hack_penalties,
            contrast_penalties,
            tuple(hack_means),
            tuple(hack_scales),
            tuple(contrast_means),
            tuple(contrast_scales),
            censored,
            _freeze(stats),
            _freeze(reference_material),
            reference_digest,
            _digest(material),
        )


def rowwise_inflation_control(
    reference_runs: Iterable[Mapping[str, Any]], spec: ForecastSpec, *, seed: int = 0
) -> dict[str, float | str]:
    """Demonstrate the invalid row-wise leakage control beside the valid runwise score.

    The row-wise branch intentionally lets a held-out row borrow its closest same-run
    training row.  That is precisely the adjacency leak the predictor refuses to use;
    it is retained only as a falsifiable negative demonstration, never as selection.
    """
    runs = tuple(_normalise_run(run) for run in reference_runs)
    stats = _reference_step_stats(runs)
    rows, labels, owners = _hazard_rows(runs, spec, stats)
    rng = random.Random(seed)
    test_indices = {index for index in range(len(rows)) if rng.random() < 0.4}
    if not test_indices:
        test_indices = {0}
    train_indices = set(range(len(rows))) - test_indices
    rowwise_correct = 0
    for index in test_indices:
        candidates = [
            candidate for candidate in train_indices if owners[candidate] == owners[index]
        ]
        if candidates:
            closest = min(
                candidates,
                key=lambda candidate: sum(
                    (a - b) ** 2 for a, b in zip(rows[candidate], rows[index], strict=True)
                ),
            )
            prediction = labels[closest]
        else:
            prediction = 0
        rowwise_correct += int(prediction == labels[index])
    loro_correct = 0
    loro_total = 0
    for name in sorted(set(owners)):
        train = [index for index, owner in enumerate(owners) if owner != name]
        valid = [index for index, owner in enumerate(owners) if owner == name]
        means, scales = _scale([rows[index] for index in train])
        coefficients = _fit_logistic(
            [_apply_scale(rows[index], means, scales) for index in train],
            [labels[index] for index in train],
            0.1,
        )
        for index in valid:
            probability = _sigmoid(_linear(coefficients, _apply_scale(rows[index], means, scales)))
            loro_correct += int((probability >= 0.5) == bool(labels[index]))
            loro_total += 1
    rowwise = rowwise_correct / len(test_indices)
    loro = loro_correct / loro_total
    return {
        "split": "row-wise-negative-control",
        "rowwise_accuracy": rowwise,
        "leave_one_run_out_accuracy": loro,
        "inflation": rowwise - loro,
    }


__all__ = [
    "ForecastDistribution",
    "FittedPredictor",
    "HazardV0",
    "PredictorRefusal",
    "rowwise_inflation_control",
]
