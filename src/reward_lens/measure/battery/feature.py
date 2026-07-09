"""``FeatureRewardAlignment`` (E12): which SAE features drive the reward (section 2.8).

A sparse autoencoder decomposes the residual stream into interpretable features whose decoder columns
are directions in activation space. Since the reward is a linear read of that stream, each feature's
contribution to the reward is its decoder column dotted with the reward direction:
``r ≈ b + Σ_i f_i (w_r . d_i)``. The alignment vector ``W_dec @ w_r`` therefore ranks features by how
much, and in which direction, they move the reward. The features at the extremes are the ones a policy
would learn to exploit or avoid.

This ports the alignment computation that lived on v1's ``TopKSAE.feature_reward_alignments``: the
decoder times the reward direction, nothing more. The alignment is a raw-coordinate quantity that
depends on both the SAE basis and the residual-stream basis, so the gauge is RAW_ONLY. A scientific
result needs a trained SAE, which is a separate artifact; when none is supplied the Observable builds
a small randomly-initialized SAE so the alignment mechanics are exercised, and it records that the SAE
was untrained so the number is never read as a real feature ranking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from reward_lens.core.types import Capability, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.battery._common import reward_direction

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


class FeatureRewardAlignment(BaseObservable):
    """Per-feature reward alignment of an SAE over a signal (E12).

    The SAE comes from ``ctx.regime['sae']`` (a ``TopKSAE`` whose ``d_model`` matches the signal). When
    absent, a small random SAE is built so the mechanics run and the result is flagged untrained.
    Requires a linear readout. RAW_ONLY: alignments depend on the SAE and residual bases.
    """

    name = "FeatureRewardAlignment"
    version = "1.0"
    requires = Capability.LINEAR_READOUT
    gauge_status = GaugeStatus.RAW_ONLY
    faithful_to = "E12 SAE feature-reward alignment"
    deviations = (
        "alignment is W_dec @ w_r (v1's TopKSAE.feature_reward_alignments); RAW_ONLY; an untrained "
        "random SAE is substituted when none is supplied and the result is flagged accordingly",
    )

    def measure(self, ctx: Context) -> "Evidence":
        import torch

        signal = ctx.signal
        w_r = reward_direction(signal, ctx.readout)
        d_model = int(signal.meta.d_model)

        sae = ctx.regime.get("sae")
        trained = sae is not None
        if sae is None:
            from reward_lens.sae import TopKSAE

            n_features = int(ctx.regime.get("n_features", 4 * d_model))
            sae = TopKSAE(d_model=d_model, n_features=n_features, k=min(16, n_features))

        alignments = sae.feature_reward_alignments(w_r.to(torch.float32)).detach().cpu().numpy()
        top_k = int(ctx.regime.get("top_k", 10))
        order = np.argsort(alignments)
        bottom = [(int(i), float(alignments[i])) for i in order[:top_k]]
        top = [(int(i), float(alignments[i])) for i in order[::-1][:top_k]]

        payload = {
            "n_features": int(alignments.shape[0]),
            "trained_sae": trained,
            "top_features": top,
            "bottom_features": bottom,
            "max_alignment": float(np.max(alignments)),
            "min_alignment": float(np.min(alignments)),
            "mean_abs_alignment": float(np.mean(np.abs(alignments))),
        }
        return ctx.emit(payload)


__all__ = ["FeatureRewardAlignment"]
