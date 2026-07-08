"""
Path Patching — head-level multi-hop causal analysis.

Direct activation patching tells you whether a single component is causally
important. It does not tell you whether a component's effect flows through
some particular *downstream* path. Path patching, introduced by
Goldowsky-Dill et al. (2023), addresses that.

Setup:
  - sender:   a component whose output we substitute (head-level)
  - receiver: a downstream component whose output we measure
  - frozen:   every other component runs cleanly on the target input

The path effect is the change in the receiver's output (and downstream
reward) when only the sender→receiver path is perturbed, holding all other
paths fixed.

This implementation supports 2-hop (sender → receiver) at head granularity,
which is the minimum useful resolution. The plan called out that
sublayer-level path patching is uninformative — head-level is the bar.

Hop count is currently 1: sender directly perturbs into receiver. To extend
to multi-hop, chain ``patch`` calls or extend ``_apply_sender``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from reward_lens.model import RewardModel

ComponentSpec = tuple[str, int, Optional[int]]
# (kind, layer_idx, head_idx_or_None) — kind in {"head", "mlp", "attn"}.


@dataclass
class PathPatchResult:
    sender: ComponentSpec
    receiver: ComponentSpec
    mode: str
    original_differential: float
    patched_differential: float
    path_effect: float  # original_diff - patched_diff (noising convention)


class PathPatcher:
    """Two-hop head-level path patching.

    Args:
        model: RewardModel.

    Example:
        >>> pp = PathPatcher(model)
        >>> result = pp.patch(prompt, preferred, dispreferred,
        ...     sender=("head", 12, 7),
        ...     receiver=("mlp", 18, None),
        ...     mode="noising",
        ... )
        >>> result.path_effect
    """

    def __init__(self, model: RewardModel):
        self.model = model

    @torch.inference_mode()
    def patch(
        self,
        prompt: str,
        preferred: str,
        dispreferred: str,
        sender: ComponentSpec,
        receiver: ComponentSpec,
        mode: Literal["noising", "denoising"] = "noising",
        max_length: int = 2048,
    ) -> PathPatchResult:
        """Run a 2-hop path patch.

        Algorithm (noising mode):
          1. Run the source forward (dispreferred) and cache the sender's
             head output and the receiver's input/output.
          2. Run the target forward (preferred) twice:
             a. Clean: get the original differential.
             b. Patched: install a forward-pre-hook on the receiver that
                replaces only the contribution flowing from the sender's
                head with the source-side value, leaving every other path
                untouched.

        For head-level senders we splice the head's output into the
        receiver's input via the residual stream: the difference
        (source_head_out - target_head_out) is added/subtracted at the
        receiver's input position.

        Returns a PathPatchResult.
        """
        if mode not in ("noising", "denoising"):
            raise ValueError(f"unknown mode: {mode}")

        # Cache both sides with full sequences.
        reward_w, cache_w = self.model.forward_with_cache(
            prompt, preferred, cache_full_sequences=True, max_length=max_length
        )
        reward_l, cache_l = self.model.forward_with_cache(
            prompt, dispreferred, cache_full_sequences=True, max_length=max_length
        )
        original_diff = reward_w - reward_l

        if mode == "noising":
            # Source side = dispreferred; target side = preferred.
            target_inputs = self.model.tokenize_conversation(
                prompt, preferred, max_length=max_length
            )
            other_reward = reward_l
        else:  # denoising
            target_inputs = self.model.tokenize_conversation(
                prompt, dispreferred, max_length=max_length
            )
            other_reward = reward_w

        # Compute the per-token sender residual contribution from each side.
        sender_kind, sender_layer, sender_head = sender
        if sender_kind != "head":
            raise NotImplementedError("PathPatcher only supports head-level senders for now")
        if sender_head is None:
            raise ValueError("sender head index is required for head-level patching")

        layers = self.model.adapter.get_layers(self.model.model)
        sender_layer_module = layers[sender_layer]
        o_proj = self.model.adapter.get_attn_o_proj(sender_layer_module)
        if o_proj is None:
            raise ValueError(f"layer {sender_layer} does not expose o_proj")

        # We need the per-head input to o_proj on both source and target sides.
        # Re-run forwards with a head-capturing hook to grab them at full
        # sequence length.
        src_head_in, src_head_out = self._capture_head_io(
            prompt,
            _other(prompt, preferred, dispreferred, mode),
            sender_layer,
            sender_head,
            max_length,
        )
        tgt_head_in, tgt_head_out = self._capture_head_io(
            prompt,
            preferred if mode == "noising" else dispreferred,
            sender_layer,
            sender_head,
            max_length,
        )

        # Compute the residual contribution diff: how the residual stream
        # at sender's output position differs between source and target due
        # to this single head. head_out shape: (1, T, d_head), o_proj
        # weight slice: (d_model, d_head).
        W = o_proj.weight  # (d_model, n_heads * d_head)
        n_heads = self.model.n_heads
        d_head = W.shape[1] // n_heads
        W_h = W[:, sender_head * d_head : (sender_head + 1) * d_head]  # (d_model, d_head)
        # Residual contributions: (1, T, d_model) each.
        src_contrib = src_head_out.float() @ W_h.float().T
        tgt_contrib = tgt_head_out.float() @ W_h.float().T

        # Splice: at the receiver, replace the head's contribution flowing
        # into it. We do this by registering a pre-hook on the receiver's
        # module that adjusts the residual stream input.
        receiver_kind, receiver_layer, _ = receiver
        if receiver_kind not in ("attn", "mlp", "head"):
            raise ValueError(f"unknown receiver kind: {receiver_kind}")
        if receiver_kind == "head":
            # interpret as the layer's attention (head-resolution receiver
            # would require splicing pre-o_proj of that head, which we don't
            # need for the core 2-hop result). Reduce to attn.
            receiver_kind = "attn"
        receiver_layer_module = layers[receiver_layer]
        receiver_module = (
            self.model.adapter.get_attn_module(receiver_layer_module)
            if receiver_kind == "attn"
            else self.model.adapter.get_mlp_module(receiver_layer_module)
        )
        if receiver_module is None:
            raise ValueError(f"receiver {receiver_kind}_L{receiver_layer} not found")

        # Receiver lives strictly downstream of sender; otherwise the patch
        # has no effect via this path.
        if receiver_layer <= sender_layer:
            raise ValueError(
                f"receiver layer {receiver_layer} must be > sender layer {sender_layer}"
            )

        # Align sequence lengths of src and tgt contributions before computing delta.
        T_src = src_contrib.shape[1]
        T_tgt = tgt_contrib.shape[1]
        T_min = min(T_src, T_tgt)
        src_aligned = src_contrib[:, :T_min, :]
        tgt_aligned = tgt_contrib[:, :T_min, :]
        delta = (src_aligned - tgt_aligned).to(self.model.device)

        # Truncate/pad delta to match the target inputs' sequence length.
        T_target = target_inputs["input_ids"].shape[1]
        if delta.shape[1] >= T_target:
            delta = delta[:, :T_target, :]
        else:
            pad = torch.zeros(
                delta.shape[0],
                T_target - delta.shape[1],
                delta.shape[2],
                dtype=delta.dtype,
                device=delta.device,
            )
            delta = torch.cat([delta, pad], dim=1)

        def pre_hook(module, args):
            # The receiver's input has the residual stream at position
            # (B, T, d_model). We add the precomputed delta. This is
            # mechanically equivalent to having only the sender→receiver
            # path see the source-side activation while every other path
            # remains target-side: every other component runs normally on
            # the target activations *except* that the receiver's input
            # carries the patched sender contribution.
            x = args[0] if isinstance(args, tuple) else args
            x = x + delta.to(x.dtype)
            if isinstance(args, tuple):
                return (x,) + args[1:]
            return x

        h = receiver_module.register_forward_pre_hook(pre_hook)
        try:
            with torch.no_grad():
                out = self.model.model(**target_inputs)
            patched_reward = self.model.adapter.extract_reward(out, target_inputs).item()
        finally:
            h.remove()

        if mode == "noising":
            patched_diff = patched_reward - other_reward
        else:
            patched_diff = other_reward - patched_reward
        path_effect = original_diff - patched_diff

        return PathPatchResult(
            sender=sender,
            receiver=receiver,
            mode=mode,
            original_differential=original_diff,
            patched_differential=patched_diff,
            path_effect=path_effect,
        )

    def _capture_head_io(
        self,
        prompt: str,
        response: str,
        layer_idx: int,
        head_idx: int,
        max_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Capture (input, output) of o_proj at layer_idx and project to head_idx slice.

        Returns:
            (head_in, head_out) where head_out is the per-head input to o_proj,
            shape (1, T, d_head). For our purposes the "in" and "out" are the
            same — we capture pre-o_proj concatenated heads and slice. The
            tuple form is kept for forward compatibility with multi-hop.
        """
        inputs = self.model.tokenize_conversation(prompt, response, max_length=max_length)
        layers = self.model.adapter.get_layers(self.model.model)
        layer = layers[layer_idx]
        o_proj = self.model.adapter.get_attn_o_proj(layer)
        captured: dict[str, torch.Tensor] = {}

        n_heads = self.model.n_heads

        def pre_hook(module, args):
            x = args[0] if isinstance(args, tuple) else args  # (1, T, n_heads * d_head)
            B, T, F = x.shape
            d_head = F // n_heads
            reshaped = x.view(B, T, n_heads, d_head)
            captured["head"] = reshaped[:, :, head_idx, :].detach().clone()

        h = o_proj.register_forward_pre_hook(pre_hook)
        try:
            with torch.no_grad():
                self.model.model(**inputs)
        finally:
            h.remove()
        head = captured["head"]
        return head, head


def _other(prompt, preferred, dispreferred, mode):
    return dispreferred if mode == "noising" else preferred
