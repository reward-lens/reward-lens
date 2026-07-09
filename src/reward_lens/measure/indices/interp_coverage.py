"""A11 Interpretability coverage: reward through interpretable features vs error nodes (Appendix A11).

Formal definition: Appendix A11. The fraction of reward routed through interpretable features versus
reconstruction/error nodes, read off attribution graphs. It is the honest single-number successor to
E04's indictment: instead of asserting that a reward is or is not interpretable, it reports what
fraction of the reward the interpretable features actually carry, with the reconstruction/error nodes
(the part of the residual the feature dictionary fails to reconstruct) as the explicit complement.

Deviation from A11: the pure function is the accounting over supplied node contributions; the
production path supplies those contributions from an attribution graph with a scalar reward sink
(the ``attribution`` subsystem). The synthetic test drives it with planted feature and error-node
contributions of a known split.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from reward_lens.core.evidence import Uncertainty
from reward_lens.core.types import Capability, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def interp_coverage(feature_contributions: np.ndarray, error_contributions: np.ndarray) -> float:
    """Fraction of reward magnitude carried by interpretable features vs error nodes (Appendix A11).

    ``= Σ|feature| / (Σ|feature| + Σ|error|)`` over the attribution node contributions. All reward
    through named features gives ``1`` (fully covered); all through reconstruction/error nodes gives
    ``0`` (E04's worst case). Magnitudes are summed so cancelling signed contributions do not inflate
    the coverage. Both inputs are contribution arrays (any shape); they are flattened.
    """
    f = float(np.sum(np.abs(np.asarray(feature_contributions, dtype=np.float64))))
    e = float(np.sum(np.abs(np.asarray(error_contributions, dtype=np.float64))))
    denom = f + e
    if denom == 0:
        return float("nan")
    return f / denom


class InterpCoverage(BaseObservable):
    """A11 fraction of reward routed through interpretable features rather than error nodes.

    Requires activations on the production path (the attribution graph is built from them). The feature
    and error-node contributions are injected here; the production path supplies them from an attribution
    graph with a scalar reward sink. Gauge is INVARIANT: a coverage fraction is scale-free.
    """

    name = "InterpCoverage"
    version = "1.0"
    requires = Capability.ACTIVATIONS
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "A11"
    deviations = (
        "consumes injected node contributions; the attribution graph with a scalar reward sink is "
        "the production path (attribution subsystem)",
    )

    def __init__(
        self,
        feature_contributions: np.ndarray | None = None,
        error_contributions: np.ndarray | None = None,
    ) -> None:
        self.feature_contributions = feature_contributions
        self.error_contributions = error_contributions

    def measure(self, ctx: Context) -> "Evidence":
        if self.feature_contributions is None or self.error_contributions is None:
            return ctx.emit(
                {
                    "note": "interp_coverage needs feature and error node contributions; none injected"
                },
                uncertainty=Uncertainty(method="none"),
            )
        coverage = interp_coverage(self.feature_contributions, self.error_contributions)
        return ctx.emit(
            {"interp_coverage": coverage, "error_share": 1.0 - coverage},
            uncertainty=Uncertainty(method="none"),
        )


__all__ = ["interp_coverage", "InterpCoverage"]
