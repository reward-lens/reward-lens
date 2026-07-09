"""A7 RRS: the Receipt Reliance Score (Appendix A7).

Formal definition: Appendix A7. ``RRS =`` the fraction of the corruption reward effect (the
falsify-receipt versus falsify-narrative arms) causally attributable to receipt spans, via span patching
plus attention forensics from the scoring position (faithful_to N1, the trajectory reward forensics
program). A reward that grounds its judgment in the receipts (tool outputs, citations, logs) moves most
of its corruption response when the receipt is falsified; a reward that reads the narrative gloss moves
when the narrative is falsified instead. ``RRS`` is one of the two axes of the honesty law (S15), the
other being skepticism (``skepticism.py``).

Deviation from A7: the pure function is the attribution ratio over the measured arm deltas; the
production path supplies ``Δr_receipt`` from a receipt-span patch and the attention forensics through the
interventions and attribution subsystems. The synthetic test drives the ratio with planted arm deltas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from reward_lens.core.evidence import Uncertainty
from reward_lens.core.types import Capability, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def receipt_reliance(dr_receipt: float, dr_total: float) -> float:
    """The receipt reliance score ``RRS = Δr_receipt / Δr_total`` (Appendix A7).

    ``dr_total`` is the whole corruption reward effect; ``dr_receipt`` is the part attributable to the
    receipt spans (the falsify-receipt arm, or the receipt-span patch). Their ratio is the causal
    fraction the reward reads off the receipts, in ``[0, 1]`` for a clean decomposition. A zero total
    effect yields ``nan`` rather than a fabricated fraction.
    """
    if dr_total == 0:
        return float("nan")
    return float(dr_receipt) / float(dr_total)


class ReceiptReliance(BaseObservable):
    """A7 fraction of the corruption reward effect the reward reads off receipt spans.

    Requires span-typed inputs and prefix scores on the production path (the receipt-span patch and the
    falsify-receipt/falsify-narrative arms). Here the arm deltas are injected so the attribution
    arithmetic runs directly. Gauge is INVARIANT: ``RRS`` is a fraction.
    """

    name = "ReceiptReliance"
    version = "1.0"
    requires = Capability.SPAN_TYPES
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "A7"
    deviations = (
        "consumes injected arm deltas; the receipt-span patch and attention forensics are the "
        "production path (interventions + attribution)",
    )

    def __init__(self, dr_receipt: float | None = None, dr_total: float | None = None) -> None:
        self.dr_receipt = dr_receipt
        self.dr_total = dr_total

    def measure(self, ctx: Context) -> "Evidence":
        if self.dr_receipt is None or self.dr_total is None:
            return ctx.emit(
                {"note": "receipt_reliance needs dr_receipt and dr_total from the corruption arms"},
                uncertainty=Uncertainty(method="none"),
            )
        rrs = receipt_reliance(self.dr_receipt, self.dr_total)
        return ctx.emit(
            {
                "receipt_reliance": rrs,
                "dr_receipt": float(self.dr_receipt),
                "dr_total": float(self.dr_total),
            },
            uncertainty=Uncertainty(method="none"),
        )


__all__ = ["receipt_reliance", "ReceiptReliance"]
