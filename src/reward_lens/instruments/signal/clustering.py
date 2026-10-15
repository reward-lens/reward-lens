"""Unequal-cluster ANOVA ICC and task-cluster bootstrap interval."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Protocol


class _Row(Protocol):
    task_id: str
    reward_applied: float | None


@dataclass(frozen=True)
class EffectiveSampleSize:
    raw_n: int
    clusters: int
    mean_group_size: float
    m_0: float
    m_a: float
    icc_raw: float | None
    icc: float | None
    icc_truncated_at_zero: bool
    design_effect: float | None
    n_eff: float | None
    identifiable: bool
    reason: str = ""


def _groups(groups: list[list[float]]) -> EffectiveSampleSize:
    s = [len(x) for x in groups]
    n = sum(s)
    g = len(s)
    if not n:
        return EffectiveSampleSize(
            0, 0, 0, 0, 0, None, None, False, None, None, False, "no finite applied rewards"
        )
    ma = sum(x * x for x in s) / n
    mean = n / g
    if g < 2 or n - g <= 0:
        return EffectiveSampleSize(
            n,
            g,
            mean,
            0,
            ma,
            None,
            None,
            False,
            None,
            None,
            False,
            "ICC is unidentified without within-task degrees of freedom",
        )
    m0 = (n - sum(x * x for x in s) / n) / (g - 1)
    grand = sum(map(sum, groups)) / n
    msw = sum(sum((v - sum(x) / len(x)) ** 2 for v in x) for x in groups) / (n - g)
    msb = sum(len(x) * (sum(x) / len(x) - grand) ** 2 for x in groups) / (g - 1)
    den = msb + (m0 - 1) * msw
    raw = 0.0 if den == 0 else (msb - msw) / den
    icc = max(0.0, raw)
    deff = 1 + (ma - 1) * icc
    return EffectiveSampleSize(n, g, mean, m0, ma, raw, icc, raw < 0, deff, n / deff, True)


def effective_sample_size(rows: Iterable[_Row]) -> EffectiveSampleSize:
    by: dict[str, list[float]] = {}
    for r in rows:
        if r.reward_applied is not None and math.isfinite(float(r.reward_applied)):
            by.setdefault(str(r.task_id), []).append(float(r.reward_applied))
    return _groups(list(by.values()))


def icc_interval(
    rows: Iterable[_Row], *, seed: int = 0, resamples: int = 999
) -> list[float] | None:
    by: dict[str, list[float]] = {}
    for r in rows:
        if r.reward_applied is not None and math.isfinite(float(r.reward_applied)):
            by.setdefault(str(r.task_id), []).append(float(r.reward_applied))
    groups = list(by.values())
    point = _groups(groups)
    if not point.identifiable or len(groups) < 15:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(max(999, resamples)):
        v = _groups([groups[rng.randrange(len(groups))] for _ in groups]).icc_raw
        if v is not None:
            vals.append(v)
    vals.sort()
    return [
        round(vals[int(0.025 * (len(vals) - 1))], 6),
        round(vals[int(0.975 * (len(vals) - 1))], 6),
    ]
