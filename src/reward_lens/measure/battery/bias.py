"""``BiasBattery`` (E06): standardized reward biases with an honest sample size (section 2.8, 2.4.2).

A reward bias is a reward difference the grader assigns to a surface change that should not matter:
more length, more confidence, more markdown. The battery measures, per axis, the standardized effect
size (Cohen's d) of the chosen-minus-rejected reward delta. The number that matters is not just the
effect but the effective sample size behind it: v1 inflated 65 seeds into "30 pairs per dimension" by
prompt mutation and resampled the expansion, so its confidence intervals were narrower than the data
earned. This Observable reports the lineage-honest effective sample size alongside every effect, using
the same ``reward_lens.stats.ess`` the data plane uses, so a bias effect is never read as more certain
than its seed count allows.

Cohen's d is a standardized (dimensionless) effect size, so the gauge is INVARIANT: a d of 0.8 is a
0.8 on any signal, which is what makes cross-model bias comparison meaningful.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from reward_lens.core.types import Capability, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.stats.ess import effective_sample_size

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def cohens_d(deltas: np.ndarray) -> float:
    """One-sample Cohen's d of ``deltas`` against zero (``mean / std``), NaN-safe.

    A degenerate (zero-variance) sample returns 0.0 rather than an infinite d, matching v1's
    ``statistics.cohens_d`` behaviour: a bias with no spread and a nonzero mean is reported as a
    finite, not infinite, effect so downstream aggregation stays well-defined.
    """
    arr = np.asarray(deltas, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if std < 1e-12:
        return 0.0
    return float(np.mean(arr) / std)


class BiasBattery(BaseObservable):
    """Per-axis standardized reward bias with lineage-honest effective sample size (E06).

    Requires only scores. The view is grouped by each pair's ``axis``; each axis reports its reward
    delta effect size and the honest ESS of its seeds. INVARIANT (Cohen's d is standardized).
    """

    name = "BiasBattery"
    version = "1.0"
    requires = Capability.SCORES
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "E06 reward bias battery"
    deviations = (
        "effect size is one-sample Cohen's d of the chosen-minus-rejected reward delta; the reported "
        "sample size is the lineage-honest ESS (stats.ess), not the raw pair count",
    )

    def measure(self, ctx: Context) -> "Evidence":
        signal = ctx.signal
        readout = ctx.readout

        by_axis: dict[str, list] = defaultdict(list)
        for pair in ctx.view:
            axis = getattr(pair, "axis", "default")
            by_axis[axis].append(pair)

        per_axis: dict[str, dict] = {}
        for axis in sorted(by_axis):
            pairs = by_axis[axis]
            chosen = [(p.prompt_text, p.chosen.text) for p in pairs]
            rejected = [(p.prompt_text, p.rejected.text) for p in pairs]
            sc = signal.score(chosen, readout).value.values
            sr = signal.score(rejected, readout).value.values
            deltas = np.asarray(sc, dtype=np.float64) - np.asarray(sr, dtype=np.float64)
            seed_ids = [p.seed_id for p in pairs]
            per_axis[axis] = {
                "effect_size": cohens_d(deltas),
                "mean_delta": float(np.mean(deltas)),
                "std_delta": float(np.std(deltas, ddof=1)) if deltas.size > 1 else 0.0,
                "n_pairs": len(pairs),
                "effective_n": float(effective_sample_size(seed_ids)),
            }

        effects = {a: v["effect_size"] for a, v in per_axis.items()}
        strongest = max(effects, key=lambda a: abs(effects[a])) if effects else None
        payload = {
            "per_axis": per_axis,
            "strongest_axis": strongest,
            "max_abs_effect_size": max((abs(v) for v in effects.values()), default=0.0),
            "n_axes": len(per_axis),
        }
        return ctx.emit(payload)


__all__ = ["BiasBattery", "cohens_d"]
