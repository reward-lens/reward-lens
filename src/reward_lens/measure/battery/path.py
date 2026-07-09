"""``PathEffect`` (E15): two-hop head-level path patching (section 2.8, 2.6).

Direct patching tells you a component matters; path patching tells you whether its effect flows
through a particular downstream path. The sender is an attention head, the receiver a downstream
layer, and the path effect is the change in reward when only the sender to receiver path carries the
source-side activation while every other path stays clean. This is Goldowsky-Dill et al.'s
construction at head granularity, the resolution v1 settled on because sublayer-level path patching is
uninformative.

This is a working port of v1's ``PathPatcher``. It computes the sender head's residual contribution on
both the source (rejected) and target (chosen) sides through the ``o_proj`` weight slice, then adds the
difference at the receiver's residual input as a :class:`~reward_lens.interventions.patch.ResidualAddPatch`
and reads the reward change. The full E15 head-effect leaderboard at 8B is GPU/``w_r``-gated (it needs
the 8B model's reward head and forwards); here the mechanism runs on the tiny model as a correctness
check. The sender and receiver default to the first head into the last layer and can be set through
``ctx.regime['sender']`` / ``ctx.regime['receiver']``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from reward_lens.core.types import Capability, GaugeStatus, Site
from reward_lens.interventions.patch import ResidualAddPatch, run_patched_scores
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.battery._common import capture_sites, pair_sides

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


class PathEffect(BaseObservable):
    """Two-hop head-to-receiver path effect on reward for preference pairs (E15).

    Requires activation capture and a linear readout. The path effect is in reward units and
    gauge-invariant within a signal.
    """

    name = "PathEffect"
    version = "1.0"
    requires = Capability.ACTIVATIONS | Capability.LINEAR_READOUT
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "E15 head path patching"
    deviations = (
        "single sender head to single receiver, 2-hop, noising; the sender residual contribution is "
        "spliced at the receiver's resid_pre exactly as v1's PathPatcher did",
    )

    def measure(self, ctx: Context) -> "Evidence":
        signal = ctx.signal
        n_layers = int(signal.meta.n_layers)
        n_heads = int(signal.meta.n_heads or 1)
        readout = ctx.readout

        sender = ctx.regime.get("sender", (0, 0))
        receiver_layer = ctx.regime.get("receiver", n_layers - 1)
        s_layer, s_head = int(sender[0]), int(sender[1])
        sender_site = Site(s_layer, "head_out", s_head)
        receiver_site = Site(receiver_layer, "resid_pre")

        o_proj = signal.runtime.adapter.get_attn_o_proj(
            signal.runtime.adapter.get_layers(signal.runtime.model)[s_layer]
        )
        if o_proj is None:
            raise ValueError(f"layer {s_layer} exposes no o_proj; cannot patch a head path")

        import torch

        weight = o_proj.weight.detach().to(torch.float32)  # (d_model, n_heads * d_head)
        d_head = weight.shape[1] // n_heads
        w_h = weight[:, s_head * d_head : (s_head + 1) * d_head]  # (d_model, d_head)

        chosen, rejected = pair_sides(ctx.view)
        effects = np.zeros(len(chosen))
        for i, (chosen_item, rejected_item) in enumerate(zip(chosen, rejected)):
            reward_c = float(signal.score([chosen_item], readout).value.values[0])
            reward_r = float(signal.score([rejected_item], readout).value.values[0])
            original_diff = reward_c - reward_r

            src = capture_sites(signal, [rejected_item], (sender_site,), full_sequence=True)[
                sender_site
            ].to(torch.float32)
            tgt = capture_sites(signal, [chosen_item], (sender_site,), full_sequence=True)[
                sender_site
            ].to(torch.float32)
            src_contrib = src @ w_h.T  # (1, T_src, d_model)
            tgt_contrib = tgt @ w_h.T  # (1, T_tgt, d_model)
            min_len = min(src_contrib.shape[1], tgt_contrib.shape[1])
            delta = src_contrib[:, :min_len, :] - tgt_contrib[:, :min_len, :]

            patch = ResidualAddPatch(site=receiver_site, delta=delta)
            patched = float(
                run_patched_scores(signal, patch.compile(signal), [chosen_item], readout)[0]
            )
            patched_diff = patched - reward_r
            effects[i] = original_diff - patched_diff

        payload = {
            "sender": [s_layer, s_head],
            "receiver_layer": receiver_layer,
            "mean_path_effect": float(np.mean(effects)),
            "per_pair_path_effect": effects.tolist(),
            "max_abs_path_effect": float(np.max(np.abs(effects))) if len(effects) else 0.0,
            "n_pairs": len(chosen),
        }
        return ctx.emit(payload)


__all__ = ["PathEffect"]
