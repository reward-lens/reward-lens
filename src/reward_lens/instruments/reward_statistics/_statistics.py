"""Deterministic completion-cluster bootstrap helpers for P-STATS."""

from __future__ import annotations

import random
from collections.abc import Callable, Hashable, Sequence

from reward_lens import contracts

RESAMPLES = 2_000
SEED = 20260926


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * probability
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def completion_bootstrap(
    population: Sequence[object], statistic: Callable[[Sequence[object]], float]
) -> contracts.Uncertainty:
    """A percentile interval that resamples whole recorded completions as clusters."""
    clusters = len(population)
    if clusters < 15:
        return contracts.Uncertainty(
            method="no_interval_below_15_clusters",
            level=0.95,
            reason="fewer than 15 recorded completion clusters; no bootstrap interval is reported",
            clusters=clusters,
        )
    generator = random.Random(SEED)
    draws = [
        statistic([population[generator.randrange(clusters)] for _ in range(clusters)])
        for _ in range(RESAMPLES)
    ]
    return contracts.Uncertainty(
        method="cluster_bootstrap",
        level=0.95,
        interval=[round(quantile(draws, 0.025), 6), round(quantile(draws, 0.975), 6)],
        resamples=RESAMPLES,
        cluster_unit="recorded completion",
        clusters=clusters,
    )


def stratified_completion_bootstrap(
    population: Sequence[object],
    statistic: Callable[[Sequence[object]], float],
    stratum: Callable[[object], Hashable],
    *,
    stratum_name: str,
) -> contracts.Uncertainty:
    """Bootstrap completion clusters conditional on their observed stratum allocation.

    This is for ratios whose denominator can vanish under ordinary clustered resampling.
    It conditions on observed stratum counts, so callers must disclose that acceptance-rate
    uncertainty is outside the interval.
    """
    clusters = len(population)
    if clusters < 15:
        return contracts.Uncertainty(
            method="no_interval_below_15_clusters",
            level=0.95,
            reason="fewer than 15 recorded completion clusters; no bootstrap interval is reported",
            clusters=clusters,
        )
    groups: dict[Hashable, list[object]] = {}
    for item in population:
        groups.setdefault(stratum(item), []).append(item)
    generator = random.Random(SEED)
    draws: list[float] = []
    for _ in range(RESAMPLES):
        sample: list[object] = []
        for members in groups.values():
            sample.extend(members[generator.randrange(len(members))] for _ in members)
        draws.append(statistic(sample))
    return contracts.Uncertainty(
        method="cluster_bootstrap",
        level=0.95,
        interval=[round(quantile(draws, 0.025), 6), round(quantile(draws, 0.975), 6)],
        resamples=RESAMPLES,
        cluster_unit=f"recorded completion, stratified by {stratum_name}",
        clusters=clusters,
    )
