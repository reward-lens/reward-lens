"""Scalar-loss identity checks and controls that fence loss from update attribution."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .advantages import CompositionRecord

__all__ = [
    "AppliedWeightIdentity",
    "GradientControl",
    "OptimizerControl",
    "check_applied_weight_identity",
    "equal_gradient_different_optimizer",
    "equal_loss_different_gradient",
    "share",
]


@dataclass(frozen=True)
class AppliedWeightIdentity:
    step: int
    available: bool
    passed: bool | None
    sum_applied_weight: float | None
    negative_loss: float | None
    reason: str = ""


@dataclass(frozen=True)
class GradientControl:
    left_loss: float
    right_loss: float
    left_gradient: float
    right_gradient: float
    same_loss: bool
    gradients_differ: bool
    measured_training_gradient: bool = False
    update_equivalence_claimed: bool = False


@dataclass(frozen=True)
class OptimizerControl:
    gradient: float
    left_transform: float
    right_transform: float
    left_control_delta: float
    right_control_delta: float
    same_gradient: bool
    optimizer_state_differs: bool
    transforms_differ: bool
    measured_training_update: bool = False
    update_equivalence_claimed: bool = False


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


def check_applied_weight_identity(record: CompositionRecord) -> tuple[AppliedWeightIdentity, ...]:
    """Check the signed trainer identity per step and retain the two compared numbers."""
    rows_by_step: dict[int, list[str]] = {}
    for row in record.trace.rows:
        rows_by_step.setdefault(row.step, []).append(row.row_id)
    results: list[AppliedWeightIdentity] = []
    weights = record.recorded_weights
    telemetry = record.telemetry
    for step, row_ids in rows_by_step.items():
        values = [weights.get(row_id) if weights is not None else None for row_id in row_ids]
        loss = (
            record.losses.get(step)
            if record.losses is not None
            else (
                telemetry.steps.get(step).observed_loss
                if telemetry is not None and step in telemetry.steps
                else None
            )
        )
        if not all(_finite(value) for value in values) or not _finite(loss):
            results.append(
                AppliedWeightIdentity(
                    step,
                    False,
                    None,
                    None,
                    None,
                    "the trainer did not record every signed applied weight and scalar loss for this step",
                )
            )
            continue
        applied = math.fsum(float(value) for value in values)
        negative_loss = -float(loss)
        results.append(
            AppliedWeightIdentity(
                step=step,
                available=True,
                passed=math.isclose(applied, negative_loss, rel_tol=1e-9, abs_tol=1e-9),
                sum_applied_weight=applied,
                negative_loss=negative_loss,
            )
        )
    return tuple(results)


def share(numerator: float, denominator: float, *, minimum_magnitude: float = 0.0) -> float | None:
    """A scalar-loss share only exists under a finite, nonzero declared denominator."""
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or not math.isfinite(minimum_magnitude)
        or abs(denominator) <= abs(minimum_magnitude)
    ):
        return None
    return numerator / denominator


def equal_loss_different_gradient(
    left_parameter: float = 1.0, right_parameter: float = 1.0
) -> GradientControl:
    """Calculate two differentiable losses with equal values and opposite analytic gradients.

    The left control is ``x²`` and the right control is ``(x - 2)²``. At one on each surface,
    both values are one while their derivatives are +2 and -2. This is a mathematical contrast,
    not recorded gradient telemetry from a training run.
    """
    left_loss = left_parameter**2
    right_loss = (right_parameter - 2.0) ** 2
    left_gradient = 2.0 * left_parameter
    right_gradient = 2.0 * (right_parameter - 2.0)
    return GradientControl(
        left_loss=left_loss,
        right_loss=right_loss,
        left_gradient=left_gradient,
        right_gradient=right_gradient,
        same_loss=math.isclose(left_loss, right_loss, rel_tol=1e-12, abs_tol=1e-12),
        gradients_differ=not math.isclose(
            left_gradient, right_gradient, rel_tol=1e-12, abs_tol=1e-12
        ),
    )


def equal_gradient_different_optimizer(
    gradient: float = 1.0,
    left_momentum: float = 0.0,
    right_momentum: float = 3.0,
    *,
    momentum: float = 0.5,
    learning_rate: float = 0.1,
) -> OptimizerControl:
    """Apply a one-dimensional momentum transform to equal gradients under unequal state.

    This computes the control deltas ``-lr * (momentum * state + gradient)``. They demonstrate
    why a scalar-loss match cannot identify an optimizer transformation, while remaining distinct
    from a measured parameter update in a trainer record.
    """
    left_transform = momentum * left_momentum + gradient
    right_transform = momentum * right_momentum + gradient
    left_delta = -learning_rate * left_transform
    right_delta = -learning_rate * right_transform
    states_differ = not math.isclose(left_momentum, right_momentum, rel_tol=1e-12, abs_tol=1e-12)
    return OptimizerControl(
        gradient=gradient,
        left_transform=left_transform,
        right_transform=right_transform,
        left_control_delta=left_delta,
        right_control_delta=right_delta,
        same_gradient=math.isfinite(gradient),
        optimizer_state_differs=states_differ,
        transforms_differ=not math.isclose(
            left_transform, right_transform, rel_tol=1e-12, abs_tol=1e-12
        ),
    )
