"""A6 Verification Score: causal fraction of correctness-Δr at the error span (Appendix A6).

Formal definition: Appendix A6. ``VS =`` the fraction of the correctness-``Δr`` between clean and
corrupted twins that is causally attributable to the error span, measured by patching the clean twin's
error-span activations into the corrupted run (faithful_to the error-microscope construction). A
process/verifier reward that is genuinely checking the work concentrates its clean-vs-corrupted reward
gap at the span where the corruption lives; a reward that is reacting to surface style spreads the gap
everywhere but the error. ``VS`` and the style share (``style_share.py``) need not sum to one: the
residual is reward change explained by neither, and is reported as such.

Deviation from A6: the pure function is the attribution arithmetic on the measured reward deltas; the
production path supplies ``Δr_error_span`` from an actual clean-twin span patch through the interventions
subsystem, and this index consumes those deltas. The synthetic test drives the arithmetic with planted
deltas of a known ratio.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from reward_lens.core.evidence import Uncertainty
from reward_lens.core.types import Capability, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def verification_score(dr_total: float, dr_error_span: float) -> float:
    """The verification score ``VS = Δr_error_span / Δr_total`` (Appendix A6).

    ``dr_total = r(clean) − r(corrupted)`` is the whole correctness reward gap; ``dr_error_span`` is the
    part recovered by patching the clean twin's error span into the corrupted run. Their ratio is the
    causal fraction of the gap that lives at the error, in ``[0, 1]`` for a well-behaved patch and
    reported as-is (possibly outside it) when patches interact. A zero total gap yields ``nan`` rather
    than a fabricated fraction.
    """
    if dr_total == 0:
        return float("nan")
    return float(dr_error_span) / float(dr_total)


class VerificationScore(BaseObservable):
    """A6 causal fraction of the correctness reward gap that lives at the labeled error span.

    Requires prefix scores (the process/verifier reward) on the production path, plus the interventions
    subsystem for the clean-twin span patch. Here the measured deltas are injected (``dr_total``,
    ``dr_error_span``) so the attribution arithmetic is exercised without waiting for interventions; the
    production path substitutes the patched deltas. Gauge is INVARIANT: ``VS`` is a fraction.
    """

    name = "VerificationScore"
    version = "1.0"
    requires = Capability.STEP_SCORES
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "A6"
    deviations = (
        "consumes measured reward deltas; the clean-twin span patch that produces dr_error_span is "
        "the production path through the interventions subsystem",
    )

    def __init__(self, dr_total: float | None = None, dr_error_span: float | None = None) -> None:
        self.dr_total = dr_total
        self.dr_error_span = dr_error_span

    def measure(self, ctx: Context) -> "Evidence":
        if self.dr_total is None or self.dr_error_span is None:
            return ctx.emit(
                {"note": "verification_score needs dr_total and dr_error_span from a span patch"},
                uncertainty=Uncertainty(method="none"),
            )
        vs = verification_score(self.dr_total, self.dr_error_span)
        payload = {
            "verification_score": vs,
            "dr_total": float(self.dr_total),
            "dr_error_span": float(self.dr_error_span),
        }
        return ctx.emit(payload, uncertainty=Uncertainty(method="none"))


__all__ = ["verification_score", "VerificationScore"]
