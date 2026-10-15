"""Task bootstrap and actual Rademacher wild-cluster bootstrap-t intervals."""

from __future__ import annotations

import math
import random
from typing import Callable, Sequence

from reward_lens import contracts

LEVEL = 0.95
RESAMPLES = 999


def uncertainty_for_weighted_tasks(
    values: Sequence[float],
    weights: Sequence[float],
    *,
    seed: int = 0,
    level: float = LEVEL,
    resamples: int = RESAMPLES,
) -> contracts.Uncertainty:
    """Same task-cluster ratio estimand in point, wild-t interval, and bootstrap draws."""
    pairs = [(float(v), float(w)) for v, w in zip(values, weights) if w > 0]
    g = len(pairs)
    if g < 15:
        return contracts.Uncertainty(
            method="no_interval_below_15_clusters",
            level=level,
            reason="fewer than 15 independent task clusters; rule 7.11(9) forbids an interval",
            clusters=g,
            cluster_unit="task",
        )
    point = sum(v * w for v, w in pairs) / sum(w for _, w in pairs)
    rng = random.Random(seed)
    b = max(999, resamples)

    def stat(ps):
        return sum(v * w for v, w in ps) / sum(w for _, w in ps)

    if g < 40:
        residual = [w * (v - point) for v, w in pairs]
        denom = sum(w for _, w in pairs)
        se = math.sqrt(sum(x * x for x in residual) / (g * (g - 1))) / (denom / g)
        ts = []
        for _ in range(b):
            signed = [x * (1 if rng.randrange(2) else -1) for x in residual]
            shift = sum(signed) / denom
            se_star = math.sqrt(sum((x - sum(signed) / g) ** 2 for x in signed) / (g * (g - 1))) / (
                denom / g
            )
            ts.append(shift / se_star if se_star else 0.0)
        ts.sort()
        lo, hi = ts[int(0.025 * (b - 1))], ts[int(0.975 * (b - 1))]
        return contracts.Uncertainty(
            method="wild_cluster_bootstrap",
            level=level,
            interval=[round(point - hi * se, 6), round(point - lo * se, 6)],
            resamples=b,
            cluster_unit="task",
            clusters=g,
            reason="Rademacher wild cluster bootstrap-t for the group-weighted task ratio",
        )
    d = sorted(stat([pairs[rng.randrange(g)] for _ in range(g)]) for _ in range(b))
    return contracts.Uncertainty(
        method="cluster_bootstrap",
        level=level,
        interval=[round(d[int(0.025 * (b - 1))], 6), round(d[int(0.975 * (b - 1))], 6)],
        resamples=b,
        cluster_unit="task",
        clusters=g,
        reason="pairs bootstrap recomputing the group-weighted task ratio",
    )


def uncertainty_for_tasks(
    tasks: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    *,
    seed: int = 0,
    level: float = LEVEL,
    resamples: int = RESAMPLES,
) -> contracts.Uncertainty:
    x = [float(v) for v in tasks if math.isfinite(float(v))]
    g = len(x)
    if g < 15:
        return contracts.Uncertainty(
            method="no_interval_below_15_clusters",
            level=level,
            reason="fewer than 15 independent task clusters; rule 7.11(9) forbids an interval",
            clusters=g,
            cluster_unit="task",
        )
    point = float(statistic(x))
    rng = random.Random(seed)
    b = max(999, resamples)
    if g < 40:
        r = [v - point for v in x]
        se = math.sqrt(sum(v * v for v in r) / (g * (g - 1)))
        ts = []
        for _ in range(b):
            signed = [v * (1 if rng.randrange(2) else -1) for v in r]
            star = sum(signed) / g
            se_star = math.sqrt(sum((v - star) ** 2 for v in signed) / (g * (g - 1)))
            ts.append(star / se_star if se_star else 0.0)
        ts.sort()
        lo, hi = ts[int(0.025 * (b - 1))], ts[int(0.975 * (b - 1))]
        return contracts.Uncertainty(
            method="wild_cluster_bootstrap",
            level=level,
            interval=[round(point - hi * se, 6), round(point - lo * se, 6)],
            resamples=b,
            cluster_unit="task",
            clusters=g,
            reason="Rademacher wild cluster bootstrap-t: replicate studentising task-cluster standard errors",
        )
    d = sorted(float(statistic([x[rng.randrange(g)] for _ in range(g)])) for _ in range(b))
    return contracts.Uncertainty(
        method="cluster_bootstrap",
        level=level,
        interval=[round(d[int(0.025 * (b - 1))], 6), round(d[int(0.975 * (b - 1))], 6)],
        resamples=b,
        cluster_unit="task",
        clusters=g,
        reason="pairs bootstrap resampling complete tasks",
    )
