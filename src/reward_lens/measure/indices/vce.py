"""A14 VCE: Value Convergence Excess (Appendix A14).

Formal definition: Appendix A14. ``VCE = align(canonicalized reward subspaces across an RM pair) −
align(matched capability subspaces, same layers and metric)``, read against the RUM-identifiability null
(faithful_to PRH 2405.07987, signed against it). The platonic representation hypothesis says
capabilities converge as models scale; VCE asks whether values converge beyond that. If two reward
models agree on their reward subspace more than their capability subspaces agree, ``VCE > 0``, and the
excess is convergence in what they reward that is not explained by convergence in what they know.

The RUM-identifiability null (``stats.nulls.rum_identifiability_null``, A14) is the baseline the raw
alignment must clear: two random-utility models identify their reward only up to the orientation freedom
preference data leaves, so two unrelated reward subspaces already share the alignment two random
k-subspaces would. VCE reports convergence above that floor.

Deviation from A14: the pure function is the excess arithmetic and the null comparison on two supplied,
same-metric alignment scalars; the alignments themselves are COVARIANT and must be computed on
frame-canonicalized subspaces upstream (``geometry.subspace``), which is why the Observable is COVARIANT
and frame-gated. The synthetic test drives the arithmetic and the null with planted alignments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from reward_lens.core.evidence import Uncertainty
from reward_lens.core.types import Capability, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def mean_cos2_alignment(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    """Mean squared cosine of the principal angles between two subspaces (the alignment statistic).

    ``= (1/k) ||Q_Aᵀ Q_B||_F²`` for orthonormal bases ``Q_A`` (``d × k``) and ``Q_B`` (``d × k``), in
    ``[0, 1]``: 1 when the subspaces coincide, ``~ k/d`` when unrelated. This is exactly the statistic
    ``stats.nulls.rum_identifiability_null`` samples, so a measured alignment and its null are the same
    quantity and directly comparable. Bases are orthonormalized defensively.
    """
    qa, _ = np.linalg.qr(np.asarray(basis_a, dtype=np.float64))
    qb, _ = np.linalg.qr(np.asarray(basis_b, dtype=np.float64))
    k = min(qa.shape[1], qb.shape[1])
    cos_angles = np.linalg.svd(qa[:, :k].T @ qb[:, :k], compute_uv=False)
    cos_angles = np.clip(cos_angles, 0.0, 1.0)
    return float(np.mean(cos_angles**2))


def value_convergence_excess(
    reward_alignment: float,
    capability_alignment: float,
    *,
    d: int | None = None,
    k: int | None = None,
    null_draws: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """VCE and its null-anchored reading (Appendix A14).

    ``VCE = reward_alignment − capability_alignment``. When ``d`` and ``k`` are given, draws the
    RUM-identifiability null for ``(d, k)`` and reports its mean and 95th percentile, plus whether the
    reward alignment exceeds that floor: values converging beyond identifiability freedom. Both alignment
    inputs must be the same statistic (``mean_cos2_alignment``) computed the same way for the comparison
    to be meaningful.
    """
    vce = float(reward_alignment) - float(capability_alignment)
    out: dict[str, float] = {
        "vce": vce,
        "reward_alignment": float(reward_alignment),
        "capability_alignment": float(capability_alignment),
    }
    if d is not None and k is not None:
        from reward_lens.stats.nulls import rum_identifiability_null

        null = rum_identifiability_null(d, k, n=null_draws, seed=seed)
        out["null_mean"] = float(np.mean(null))
        out["null_p95"] = float(np.quantile(null, 0.95))
        out["exceeds_identifiability_null"] = float(reward_alignment > out["null_p95"])
    return out


class VCE(BaseObservable):
    """A14 Value Convergence Excess: reward-subspace agreement beyond capability agreement.

    A cross-signal comparison: COVARIANT and frame-gated (the reward and capability alignments must be
    computed on frame-canonicalized subspaces of the RM pair). The two same-metric alignment scalars are
    injected here (with the ambient ``d`` and subspace ``k`` for the null); the production path computes
    them through ``geometry.subspace`` in a shared frame. Reports VCE against the RUM-identifiability
    null.
    """

    name = "VCE"
    version = "1.0"
    requires = Capability.SCORES
    gauge_status = GaugeStatus.COVARIANT
    faithful_to = "A14"
    deviations = (
        "arithmetic and null on two supplied same-metric alignment scalars; the frame-canonicalized "
        "subspace alignments are computed upstream (geometry.subspace) and are COVARIANT",
    )

    def __init__(
        self,
        reward_alignment: float | None = None,
        capability_alignment: float | None = None,
        *,
        d: int | None = None,
        k: int | None = None,
        null_draws: int = 1000,
        seed: int = 0,
    ) -> None:
        self.reward_alignment = reward_alignment
        self.capability_alignment = capability_alignment
        self.d = d
        self.k = k
        self.null_draws = int(null_draws)
        self.seed = int(seed)

    def measure(self, ctx: Context) -> "Evidence":
        if self.reward_alignment is None or self.capability_alignment is None:
            return ctx.emit(
                {"note": "vce needs reward and capability subspace alignments; none injected"},
                uncertainty=Uncertainty(method="none"),
                gauge=GaugeStatus.COVARIANT,
            )
        report = value_convergence_excess(
            self.reward_alignment,
            self.capability_alignment,
            d=self.d,
            k=self.k,
            null_draws=self.null_draws,
            seed=self.seed,
        )
        return ctx.emit(report, uncertainty=Uncertainty(method="none"))


__all__ = ["mean_cos2_alignment", "value_convergence_excess", "VCE"]
