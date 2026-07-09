"""A6 Style Share: the style complement of the verification score (Appendix A6).

Formal definition: Appendix A6. ``StyleShare =`` the fraction of the correctness-``Δr`` removed by
projecting the twin activation difference ``Δh`` onto the style subspace. Where the verification score
(``verification_score.py``) measures how much of the clean-vs-corrupted reward gap lives at the error
span, the style share measures how much of it the reward reads off style directions instead. ``VS`` and
``StyleShare`` need not sum to one; the residual is reward change explained by neither, and A6 keeps it
unexplained rather than forcing a partition.

Deviation from A6: the pure function computes the linear reward fraction carried by the style-subspace
projection of ``Δh`` under the reward direction ``w_r``; the production path supplies the style subspace
from the concept layer's style dictionary. The synthetic test drives it with a planted style component
of a known reward fraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from reward_lens.core.evidence import Uncertainty
from reward_lens.core.types import Capability, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.indices._support import reward_vector

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def _orthonormalize(basis: np.ndarray) -> np.ndarray:
    """Orthonormalize the rows of a style basis (``m, d``) so the projector is idempotent."""
    b = np.asarray(basis, dtype=np.float64)
    if b.ndim == 1:
        b = b[None, :]
    q, _ = np.linalg.qr(b.T)
    return q.T  # (rank, d) orthonormal rows


def style_share(delta_h: np.ndarray, style_basis: np.ndarray, w_r: np.ndarray) -> float:
    """The style share ``= (w_r · P_style Δh) / (w_r · Δh)`` (Appendix A6).

    Projects the clean-vs-corrupted activation difference ``Δh`` onto the (orthonormalized) style
    subspace and reports the fraction of the reward change ``w_r · Δh`` that the projection carries. A
    reward that responds to the corruption purely through style directions has a style share near one; a
    reward that responds through the error content has a style share near zero. ``Δh`` is ``(d,)`` (or an
    ``(n, d)`` batch, averaged); ``style_basis`` is ``(m, d)``.
    """
    dh = np.asarray(delta_h, dtype=np.float64)
    if dh.ndim == 2:
        dh = dh.mean(axis=0)
    w = np.asarray(w_r, dtype=np.float64).ravel()
    q = _orthonormalize(style_basis)
    projected = q.T @ (q @ dh)  # P_style delta_h
    total = float(w @ dh)
    if total == 0:
        return float("nan")
    return float((w @ projected) / total)


class StyleShare(BaseObservable):
    """A6 fraction of the correctness reward gap the reward reads off style directions.

    Requires activations and a linear readout on the production path (``Δh`` from clean/corrupted twins,
    the style subspace from the concept layer's style dictionary). Here ``Δh`` and the style basis are
    injected so the projection arithmetic is exercised directly. Gauge is INVARIANT: the style share is a
    reward fraction.
    """

    name = "StyleShare"
    version = "1.0"
    requires = Capability.ACTIVATIONS | Capability.LINEAR_READOUT
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "A6"
    deviations = (
        "consumes an injected delta_h and style subspace; the twin activation difference and the "
        "style dictionary are the production path (interventions + concepts)",
    )

    def __init__(
        self,
        delta_h: np.ndarray | None = None,
        style_basis: np.ndarray | None = None,
    ) -> None:
        self.delta_h = delta_h
        self.style_basis = style_basis

    def measure(self, ctx: Context) -> "Evidence":
        if self.delta_h is None or self.style_basis is None:
            return ctx.emit(
                {"note": "style_share needs delta_h and a style subspace basis; none injected"},
                uncertainty=Uncertainty(method="none"),
            )
        w_r = reward_vector(ctx.signal, ctx.readout)
        share = style_share(self.delta_h, self.style_basis, w_r)
        return ctx.emit(
            {"style_share": share, "style_dim": int(_orthonormalize(self.style_basis).shape[0])},
            uncertainty=Uncertainty(method="none"),
        )


__all__ = ["style_share", "StyleShare"]
