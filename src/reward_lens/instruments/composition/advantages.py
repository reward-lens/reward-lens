"""Reconstruct recorded advantages without reinterpreting loss or update evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from reward_lens.product.trace import (
    CompositionTelemetry,
    ScalarLossStep,
    ScalarLossToken,
    TraceRecord,
)
from reward_lens.record.schema import EstimatorSpec

__all__ = [
    "Agreement",
    "CompositionRecord",
    "LossReconstruction",
    "CompositionTelemetry",
    "ScalarLossStep",
    "ScalarLossToken",
    "ReconstructedStep",
    "price_differential",
    "recompute_advantages",
    "reconstruct_scalar_loss",
]


def _price_extra_missing(*_args: object, **_kwargs: object) -> object:
    """A stable patch point until the optional numerical ledger is installed."""
    raise ModuleNotFoundError("composition price differential requires the record extra")


# The base install stays numpy-free. `price_differential` imports the canonical ledger only when a
# caller has supplied enough data to ask for that optional measurement; tests may replace this
# name to prove this module delegates rather than copying the ledger arithmetic.
selection_differential = _price_extra_missing
_LEDGER_LOADED = False


@dataclass(frozen=True)
class ReconstructedStep:
    step: int
    available: bool
    reconstructed_loss: float | None
    observed_loss: float | None
    agreement: bool | None
    aggregation_denominator: float | None
    share_denominator: float | None
    row_loss_contributions: Mapping[str, float] = field(default_factory=dict)
    row_signed_negative_loss_contributions: Mapping[str, float] = field(default_factory=dict)
    row_shares: Mapping[str, float | None] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class LossReconstruction:
    steps: tuple[ReconstructedStep, ...]

    @property
    def available(self) -> bool:
        return bool(self.steps) and all(step.available for step in self.steps)

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(step.reason for step in self.steps if step.reason)


@dataclass(frozen=True)
class CompositionRecord:
    """The explicit join Trace intentionally does not infer from a reward column.

    Trace holds declared groups and the observed reward/advantage series.  Trainer loss telemetry
    is a different input: an applied weight is a signed negative-loss contribution, never an alias
    for `TraceRow.advantage`. Numerical scalar-loss inputs are the immutable telemetry attached to
    ``TraceRecord.metadata.composition`` by the trace reader.
    """

    trace: TraceRecord
    estimators: Mapping[str, EstimatorSpec] = field(default_factory=dict)
    applied_weights: Mapping[str, float] | None = None
    losses: Mapping[int, float] | None = None
    composition_telemetry: CompositionTelemetry | None = None

    @property
    def telemetry(self) -> CompositionTelemetry | None:
        """Use the canonical trace payload unless an explicit immutable override was supplied."""
        return self.composition_telemetry or self.trace.metadata.composition

    @property
    def recorded_weights(self) -> Mapping[str, float] | None:
        """Use the source-recorded canonical weights unless this join supplies them explicitly."""
        return (
            self.applied_weights
            if self.applied_weights is not None
            else self.trace.metadata.applied_weights
        )


@dataclass(frozen=True)
class Agreement:
    """A replay comparison, where agreement is a numeric fraction when observations exist."""

    observed: bool
    agreement: float | None
    reconstructed: tuple[float | None, ...]
    recorded: tuple[float | None, ...]
    missing: tuple[str, ...] = ()


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


_SUPPORTED_PROFILE = "grpo_ppo_token_mean_kl_v1"


def reconstruct_scalar_loss(record: CompositionRecord) -> LossReconstruction:
    """Replay the declared token-mean clipped GRPO/PPO loss without implying an update.

    For each token the supported profile records ``-min(r * A, clip(r) * A) + beta * KL + aux``.
    Token masks are applied before the declared aggregation denominator.  The result reports both
    positive loss contributions and their signed negative-loss counterparts; a share is only given
    when the reconstructed scalar loss itself is a finite, sufficiently large denominator.
    """
    telemetry = record.telemetry
    if telemetry is None or not telemetry.steps:
        return LossReconstruction(
            (
                ReconstructedStep(
                    step=-1,
                    available=False,
                    reconstructed_loss=None,
                    observed_loss=None,
                    agreement=None,
                    aggregation_denominator=None,
                    share_denominator=None,
                    reason="no numerical scalar-loss telemetry was recorded",
                ),
            )
        )
    rows = {row.row_id: row for row in record.trace.rows}
    rebuilt: list[ReconstructedStep] = []
    for step, step_telemetry in sorted(telemetry.steps.items()):
        rebuilt.append(_reconstruct_step(int(step), step_telemetry, rows))
    return LossReconstruction(tuple(rebuilt))


def _reconstruct_step(
    step: int, telemetry: ScalarLossStep, rows: Mapping[str, object]
) -> ReconstructedStep:
    if telemetry.trainer_profile != _SUPPORTED_PROFILE:
        return _missing_step(
            step, telemetry, f"unsupported trainer profile {telemetry.trainer_profile!r}"
        )
    numbers = (
        telemetry.aggregation_denominator,
        telemetry.minimum_denominator,
        telemetry.clip_low,
        telemetry.clip_high,
        telemetry.kl_coefficient,
    )
    if not all(_finite(value) for value in numbers):
        return _missing_step(step, telemetry, "scalar-loss step has a non-finite declaration")
    if (
        telemetry.minimum_denominator < 0
        or telemetry.aggregation_denominator <= telemetry.minimum_denominator
    ):
        return _missing_step(
            step,
            telemetry,
            "aggregation denominator is non-positive or unstable under its declared floor",
        )
    if telemetry.clip_low <= 0 or telemetry.clip_high < telemetry.clip_low:
        return _missing_step(step, telemetry, "clipping bounds are invalid")
    if not telemetry.tokens:
        return _missing_step(step, telemetry, "no token contributions were recorded")
    by_row: dict[str, float] = {}
    for token in telemetry.tokens:
        row = rows.get(token.row_id)
        if row is None:
            return _missing_step(step, telemetry, f"token names unknown trace row {token.row_id!r}")
        if getattr(row, "step", None) != step:
            return _missing_step(
                step, telemetry, f"token row {token.row_id!r} belongs to another optimizer step"
            )
        recorded_advantage = getattr(row, "advantage", None)
        if not _finite(recorded_advantage):
            return _missing_step(
                step, telemetry, f"trace row {token.row_id!r} has no finite recorded advantage"
            )
        values = (
            token.mask,
            token.log_probability,
            token.old_log_probability,
            token.importance_ratio,
            token.advantage,
            token.kl_term,
            token.auxiliary_term,
        )
        if not all(_finite(value) for value in values):
            return _missing_step(
                step,
                telemetry,
                f"token for row {token.row_id!r} has non-finite numerical telemetry",
            )
        if token.mask < 0 or token.mask > 1:
            return _missing_step(
                step, telemetry, f"token mask for row {token.row_id!r} is outside [0, 1]"
            )
        if not math.isclose(
            token.advantage, float(recorded_advantage), rel_tol=1e-9, abs_tol=1e-12
        ):
            return _missing_step(
                step,
                telemetry,
                f"token advantage for row {token.row_id!r} disagrees with the recorded trace advantage",
            )
        expected_ratio = math.exp(token.log_probability - token.old_log_probability)
        if not math.isclose(token.importance_ratio, expected_ratio, rel_tol=1e-9, abs_tol=1e-12):
            return _missing_step(
                step,
                telemetry,
                f"importance ratio for row {token.row_id!r} disagrees with recorded log probabilities",
            )
        clipped_ratio = min(max(token.importance_ratio, telemetry.clip_low), telemetry.clip_high)
        policy_objective = min(
            token.importance_ratio * token.advantage, clipped_ratio * token.advantage
        )
        contribution = (
            token.mask
            * (-policy_objective + telemetry.kl_coefficient * token.kl_term + token.auxiliary_term)
            / telemetry.aggregation_denominator
        )
        by_row[token.row_id] = by_row.get(token.row_id, 0.0) + contribution
    reconstructed = math.fsum(by_row.values())
    observed = telemetry.observed_loss
    if not _finite(observed):
        return _missing_step(
            step,
            telemetry,
            "the trainer did not record the scalar loss to compare with this reconstruction",
        )
    share_denominator = (
        reconstructed if abs(reconstructed) > telemetry.minimum_denominator else None
    )
    shares = {
        row: (value / share_denominator if share_denominator is not None else None)
        for row, value in by_row.items()
    }
    return ReconstructedStep(
        step,
        True,
        reconstructed,
        float(observed),
        math.isclose(reconstructed, float(observed), rel_tol=1e-9, abs_tol=1e-9),
        telemetry.aggregation_denominator,
        share_denominator,
        row_loss_contributions=dict(by_row),
        row_signed_negative_loss_contributions={row: -value for row, value in by_row.items()},
        row_shares=shares,
        reason=(
            "shares are absent because reconstructed loss is zero or unstable under the declared floor"
            if share_denominator is None
            else ""
        ),
    )


def _missing_step(step: int, telemetry: ScalarLossStep, reason: str) -> ReconstructedStep:
    return ReconstructedStep(
        step,
        False,
        None,
        telemetry.observed_loss,
        None,
        telemetry.aggregation_denominator,
        None,
        reason=reason,
    )


def recompute_advantages(record: CompositionRecord) -> Agreement:
    """Replay each declared group using its recorded estimator and compare, never tune, values."""
    rows = record.trace.rows
    recorded = tuple(row.advantage for row in rows)
    observed = any(_finite(value) for value in recorded)
    rebuilt: list[float | None] = [None] * len(rows)
    missing: list[str] = []
    groups: dict[str, list[tuple[int, object]]] = {}
    for index, row in enumerate(rows):
        if row.group_id is None:
            missing.append(f"row {row.row_id} has no declared group")
            continue
        groups.setdefault(row.group_id, []).append((index, row))
    for group_id, members in groups.items():
        estimator = record.estimators.get(group_id)
        if estimator is None:
            missing.append(f"group {group_id} has no recorded estimator")
            continue
        scores = [getattr(row, "reward_applied") for _, row in members]
        if not all(_finite(value) for value in scores):
            missing.append(f"group {group_id} lacks a finite applied reward for every row")
            continue
        replayed, reason = _replay_group([float(value) for value in scores], estimator, group_id)
        if reason is not None:
            missing.append(reason)
            continue
        for (index, _), value in zip(members, replayed):
            rebuilt[index] = None if not math.isfinite(value) else float(value)
    comparable = [
        (expected, got)
        for expected, got in zip(rebuilt, recorded)
        if _finite(expected) and _finite(got)
    ]
    agreement = (
        sum(
            math.isclose(float(expected), float(got), rel_tol=1e-9, abs_tol=1e-9)
            for expected, got in comparable
        )
        / len(comparable)
        if observed and comparable
        else None
    )
    if observed and not comparable and not missing:
        missing.append(
            "recorded advantages exist but none can be paired with a reconstructed value"
        )
    return Agreement(observed, agreement, tuple(rebuilt), recorded, tuple(missing))


def _replay_group(
    scores: list[float], spec: EstimatorSpec, group_id: str
) -> tuple[tuple[float, ...], str | None]:
    """Use the record-layer replay so grouping, normalisation and missingness have one meaning.

    Policy-ratio clipping deliberately remains outside this replay: it is a loss operation, not an
    advantage transform.  The canonical routine carries that distinction and refuses an
    unrecorded baseline or denominator rather than approximating either one here.
    """
    try:
        from reward_lens.record.scores import replay_advantages

        replayed = replay_advantages(scores, spec, where=f"declared group {group_id}")
    except ModuleNotFoundError:
        # `record` is optional in the base install. Keep an equivalent, deliberately narrow
        # fallback here so absence of that extra does not turn score-level evidence into an error.
        return _replay_without_record_extra(scores, spec, group_id)
    if not isinstance(replayed, tuple):
        return (), replayed.detail
    return tuple(float(value) for value in replayed), None


def _replay_without_record_extra(
    scores: list[float], spec: EstimatorSpec, group_id: str
) -> tuple[tuple[float, ...], str | None]:
    if not spec.group_centred:
        return (
            (),
            f"group {group_id} is not group-centred, so its advantage needs an unrecorded baseline",
        )
    if spec.std_normalised and (spec.std_epsilon is None or spec.std_ddof is None):
        return (), f"group {group_id} omits the standard-deviation denominator semantics"
    live = [value for value in scores if math.isfinite(value)]
    if not live:
        return tuple(float("nan") for _ in scores), None
    mean = math.fsum(live) / len(live)
    ddof = int(spec.std_ddof or 0) if len(live) > 1 else 0
    variance = math.fsum((value - mean) ** 2 for value in live) / max(1, len(live) - ddof)
    std = math.sqrt(variance)
    epsilon = float(spec.std_epsilon or 0.0)
    degenerate = std <= epsilon
    if degenerate and spec.degenerate_policy == "unknown":
        return (), f"group {group_id} is degenerate and its recorded policy is unknown"
    replayed = [value - mean if math.isfinite(value) else float("nan") for value in scores]
    if spec.family.lower() in {"rloo", "loo"}:
        replayed = (
            [float("nan")] * len(scores)
            if len(live) < 2
            else [
                value * len(live) / (len(live) - 1) if math.isfinite(value) else value
                for value in replayed
            ]
        )
    if spec.std_normalised:
        replayed = [
            value / (std + epsilon) if math.isfinite(value) else value for value in replayed
        ]
    if degenerate and spec.degenerate_policy == "zero":
        replayed = [0.0 if math.isfinite(value) else value for value in replayed]
    elif degenerate and spec.degenerate_policy == "skip":
        replayed = [float("nan")] * len(scores)
    return tuple(replayed), None


def price_differential(record: CompositionRecord):
    """Use the ledger's checked within-group Price arithmetic when every needed input exists.

    This function does not use a differential to declare agreement.  It is a separate scalar-loss
    summary over the documented applied weights, and absent inputs remain absent rather than being
    filled from advantages or a convenience aggregate.
    """
    series = record.trace.series.get("reward_applied")
    names = tuple(series.components) if series is not None else ()
    if not names:
        return None
    recorded_weights = record.recorded_weights
    if recorded_weights is None:
        return None
    feature_rows: list[list[float]] = []
    weights: list[float] = []
    groups: list[str] = []
    for row in record.trace.rows:
        weight = recorded_weights.get(row.row_id)
        values = [row.components.get(name) for name in names]
        if (
            row.group_id is None
            or not _finite(weight)
            or not all(_finite(value) for value in values)
        ):
            return None
        feature_rows.append([float(value) for value in values])
        weights.append(float(weight))
        groups.append(row.group_id)
    if not feature_rows:
        return None
    global selection_differential, _LEDGER_LOADED
    if selection_differential is _price_extra_missing:
        try:
            from reward_lens.measure.ledger.price import selection_differential as ledger_price
        except ModuleNotFoundError:
            return None
        selection_differential = ledger_price
        _LEDGER_LOADED = True
    if not _LEDGER_LOADED:
        # An injected delegate is used by the no-numpy base-install regression to prove that this
        # packet calls the ledger seam instead of carrying a second implementation.
        return selection_differential(feature_rows, weights, groups, names)
    try:
        import numpy as np

        from reward_lens.record.schema import SelectionQuantity, SelectionWeights
    except ModuleNotFoundError:
        return None
    return selection_differential(
        np.asarray(feature_rows, dtype=float),
        SelectionWeights(
            np.asarray(weights, dtype=float), SelectionQuantity.APPLIED_WEIGHT, "recorded"
        ),
        np.asarray(groups, dtype=object),
        names,
    )
