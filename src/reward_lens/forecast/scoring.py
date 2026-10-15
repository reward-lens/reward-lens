"""Proper forecast scores for the sealed 3.1.0 forecast ledger.

T1 is a binary onset event and therefore uses the one-term Brier score.  T2 and
T3 are fractions; they retain their predictive distributions and are scored with
the CRPS approximation induced by the registered quantiles (twice mean pinball
loss).  A point squared error is deliberately unavailable here.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Sequence

REGISTERED_QUANTILES: tuple[Decimal, ...] = (
    Decimal("0.1"),
    Decimal("0.25"),
    Decimal("0.5"),
    Decimal("0.75"),
    Decimal("0.9"),
)


def _decimal(value: object, *, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except Exception as exc:  # noqa: BLE001 - normalize external numeric inputs
        raise ValueError(f"{field} is not a finite number") from exc
    if not result.is_finite():
        raise ValueError(f"{field} is not a finite number")
    return result


def score_t1(probability: object, outcome: bool) -> Decimal:
    """The T1 Brier term, ``(p - y)^2``. Lower is better."""
    p = _decimal(probability, field="T1 probability")
    if not Decimal(0) <= p <= Decimal(1):
        raise ValueError("T1 probability must be in [0, 1]")
    return (p - (Decimal(1) if outcome else Decimal(0))) ** 2


def score_fraction(
    quantiles: Sequence[object], values: Sequence[object], outcome: object
) -> Decimal:
    """Score a T2/T3 fraction by WIS/quantile-CRPS, never Brier.

    With the five registered quantiles, ``2 * mean(pinball loss)`` is the
    discrete CRPS approximation used by this release.  Requiring the fixed
    vector prevents a caller from silently changing the proper-score estimator
    after outcomes are known.
    """
    levels = tuple(_decimal(level, field="quantile level") for level in quantiles)
    if levels != REGISTERED_QUANTILES:
        raise ValueError(
            "T2/T3 fractions require CRPS or WIS over registered quantiles "
            "(0.1, 0.25, 0.5, 0.75, 0.9); Brier is unavailable for fractions"
        )
    if len(values) != len(levels):
        raise ValueError("CRPS or WIS requires one value for every registered quantile")
    y = _decimal(outcome, field="fraction outcome")
    if not Decimal(0) <= y <= Decimal(1):
        raise ValueError("fraction outcome must be in [0, 1]")
    qs = tuple(_decimal(value, field="quantile value") for value in values)
    if any(not Decimal(0) <= value <= Decimal(1) for value in qs):
        raise ValueError("fraction quantile values must be in [0, 1]")
    if tuple(sorted(qs)) != qs:
        raise ValueError("quantile values must be non-decreasing")
    loss = sum(
        level * (y - value) if y >= value else (Decimal(1) - level) * (value - y)
        for level, value in zip(levels, qs)
    )
    return Decimal(2) * loss / Decimal(len(levels))


__all__ = ["REGISTERED_QUANTILES", "score_fraction", "score_t1"]
