"""Contested direction: the axis annotators disagree along (Appendix A, S11 machine psychology).

Where preferences are contested, a single scalar reward cannot represent everyone (T7); the useful
object is the direction in representation space along which the disagreement lives. Given, per pair, the
activation difference ``Δh`` (chosen minus rejected) and a disagreement signal (annotator vote entropy,
label variance, or a diverging-preferences score), the contested direction is the axis of ``Δh`` whose
projection best tracks disagreement:

    ``c ∝ Σ_i (disagreement_i − mean) · Δh_i``,

the covariance direction between the representation change and the disagreement. Pairs the annotators
split on pull ``c`` toward the representation change that distinguishes them; unanimous pairs contribute
nothing. The magnitude of the alignment says how much of the disagreement is linearly organized along a
single axis versus scattered.

This module has no single Appendix A letter; it is the contested-direction diagnostic S11 consumes.
Deviation: the pure function is the covariance-direction recovery on supplied ``Δh`` and disagreement;
the production path reads ``Δh`` from the signal and disagreement from the data plane's annotator
records. The direction is COVARIANT, so a cross-signal comparison of contested directions needs a shared
frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from reward_lens.core.evidence import Uncertainty
from reward_lens.core.types import Capability, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def contested_direction(delta_h: np.ndarray, disagreement: np.ndarray) -> dict[str, object]:
    """The disagreement-covariance direction and how much disagreement it organizes.

    ``delta_h`` is ``(n, d)`` (per-pair chosen-minus-rejected activations); ``disagreement`` is ``(n,)``.
    Returns the unit contested direction ``c`` (the normalized covariance ``Δhᵀ (disagreement − mean)``),
    the correlation between ``Δh @ c`` and the disagreement (how well a single axis explains it), and the
    raw covariance norm. When the covariance is degenerate the direction is returned as zeros and the
    correlation as ``nan``.
    """
    dh = np.asarray(delta_h, dtype=np.float64)
    dis = np.asarray(disagreement, dtype=np.float64).ravel()
    dis_c = dis - dis.mean()
    cov = dh.T @ dis_c / dh.shape[0]  # (d,)
    norm = float(np.linalg.norm(cov))
    if norm == 0:
        return {"direction": np.zeros(dh.shape[1]), "correlation": float("nan"), "cov_norm": 0.0}
    direction = cov / norm
    proj = dh @ direction
    proj_c = proj - proj.mean()
    denom = np.linalg.norm(proj_c) * np.linalg.norm(dis_c)
    corr = float(proj_c @ dis_c / denom) if denom > 0 else float("nan")
    return {"direction": direction, "correlation": corr, "cov_norm": norm}


class Contested(BaseObservable):
    """Contested direction: the representation axis annotators disagree along.

    Requires activations. The per-pair ``Δh`` and the disagreement signal are injected (the data plane's
    annotator records supply disagreement in production). The direction is COVARIANT: comparing contested
    directions across signals requires a shared frame (gate 2). Within one signal the recovered axis and
    its correlation are reported directly.
    """

    name = "Contested"
    version = "1.0"
    requires = Capability.ACTIVATIONS
    gauge_status = GaugeStatus.COVARIANT
    faithful_to = None
    deviations = (
        "no single Appendix A letter; the contested-direction diagnostic for S11 (annotator "
        "disagreement, T7). Direction is COVARIANT and frame-gated for cross-signal comparison.",
    )

    def __init__(
        self,
        delta_h: np.ndarray | None = None,
        disagreement: np.ndarray | None = None,
    ) -> None:
        self.delta_h = delta_h
        self.disagreement = disagreement

    def measure(self, ctx: Context) -> "Evidence":
        if self.delta_h is None or self.disagreement is None:
            return ctx.emit(
                {
                    "note": "contested needs per-pair delta_h and a disagreement signal; none injected"
                },
                uncertainty=Uncertainty(method="none"),
                gauge=GaugeStatus.COVARIANT,
            )
        result = contested_direction(self.delta_h, self.disagreement)
        payload = {
            "direction": np.asarray(result["direction"], dtype=np.float64),
            "correlation": result["correlation"],
            "cov_norm": result["cov_norm"],
            "n_pairs": int(np.asarray(self.delta_h).shape[0]),
        }
        return ctx.emit(payload, uncertainty=Uncertainty(method="none"))


__all__ = ["contested_direction", "Contested"]
