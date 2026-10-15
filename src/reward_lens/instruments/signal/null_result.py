"""Predeclared null-result reporting, with MDE rather than observed power."""

from __future__ import annotations

from statistics import NormalDist

from reward_lens import contracts


def null_result(
    *,
    clustered_standard_error: float,
    n: int,
    design_effect: float | None,
    alpha: float,
    power: float,
) -> contracts.Power:
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and planned power must be strictly between zero and one")
    z = NormalDist()
    fields = {} if design_effect is None else {"design_effect": design_effect}
    return contracts.Power(
        alpha=alpha,
        power=power,
        n=n,
        mde=round((z.inv_cdf(1 - alpha / 2) + z.inv_cdf(power)) * clustered_standard_error, 6),
        **fields,
    )
