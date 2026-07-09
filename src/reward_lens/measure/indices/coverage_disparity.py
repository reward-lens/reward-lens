"""Coverage disparity: v1's coverage statistic under its honest name (Appendix A2, deviation note).

This is NOT the Wang-Huang distortion index. It is the coverage statistic v1's E10 actually computed
while citing Wang-Huang 2603.28063, the exact operationalization drift Appendix A2 calls out. The real
Wang-Huang per-dimension distortion lives in ``distortion.py`` (A2); this module keeps v1's statistic so
its numbers remain reproducible, but names it for what it measures: the disparity in reward coverage
across a set of dimensions or groups, not a distortion.

``coverage(P)`` is the fraction of a property's reward-relevant signal captured by the named/intended
channels. The disparity is the spread of that coverage across the battery: a large disparity means the
reward covers some properties well and others poorly, which is a real and reportable inequality, just
not Wang-Huang's object. Keeping it here, honestly labelled, is the structural fix (liability 2) that
makes the v1 drift impossible to repeat silently: a card consuming this sees ``coverage_disparity``,
never ``distortion``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from reward_lens.core.evidence import Uncertainty
from reward_lens.core.types import Capability, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def coverage_disparity(coverage: Sequence[float]) -> dict[str, float]:
    """The disparity in reward coverage across a battery (v1's honestly-named coverage statistic).

    Returns the spread of the coverage values: the range ``max − min``, the standard deviation, and the
    Gini-style mean absolute difference, all summarizing how unequally the reward covers the properties.
    ``coverage`` is a length-``K`` vector in ``[0, 1]``. Uniform coverage gives zero disparity by every
    measure; a reward that covers one property fully and another not at all gives a range of one.
    """
    c = np.clip(np.asarray(coverage, dtype=np.float64).ravel(), 0.0, 1.0)
    if c.size == 0:
        return {"range": 0.0, "std": 0.0, "mean_abs_diff": 0.0, "mean_coverage": float("nan")}
    range_ = float(c.max() - c.min())
    std = float(np.std(c, ddof=0))
    mad = float(np.mean(np.abs(c[:, None] - c[None, :])))
    return {
        "range": range_,
        "std": std,
        "mean_abs_diff": mad,
        "mean_coverage": float(c.mean()),
    }


class CoverageDisparity(BaseObservable):
    """v1's coverage statistic, kept reproducible under its honest name (not Wang-Huang distortion).

    Takes a battery of per-property coverage values (injected; the concept/KUI layer supplies them in
    production) and reports their disparity. ``faithful_to`` is None on purpose: this instantiates no
    Appendix A theory object, it preserves a v1 statistic, and the deviation note says so. Gauge is
    INVARIANT.
    """

    name = "CoverageDisparity"
    version = "1.0"
    requires = Capability.SCORES
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = None
    deviations = (
        "this is v1/E10's coverage statistic, NOT Wang-Huang distortion (A2); the distortion index "
        "lives in distortion.py. Kept under its honest name to preserve v1's numbers without the "
        "operationalization drift.",
    )

    def __init__(self, coverage: Sequence[float] | None = None) -> None:
        self.coverage = coverage

    def measure(self, ctx: Context) -> "Evidence":
        if self.coverage is None:
            return ctx.emit(
                {"note": "coverage_disparity needs a per-property coverage battery; none injected"},
                uncertainty=Uncertainty(method="none"),
            )
        report: dict[str, Any] = coverage_disparity(self.coverage)
        report["n_properties"] = int(np.asarray(self.coverage).size)
        return ctx.emit(report, uncertainty=Uncertainty(n=report["n_properties"], method="none"))


__all__ = ["coverage_disparity", "CoverageDisparity"]
