"""Thin helpers around RewardModel.forward_with_cache_batch."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def chunked(items: list, n: int) -> Iterable[list]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


def project_cache_onto_reward(model, cache, layer_idx: int) -> np.ndarray:
    """Project a batched cache's residual stream at layer_idx onto w_r.
    Returns shape (B,)."""
    h = cache.residual_streams.get(layer_idx)
    if h is None:
        return np.array([])
    proj = (h.float() @ model.reward_direction.to(h.device)) + model.reward_bias
    return proj.detach().cpu().numpy()


def batch_lens_curves(model, cache) -> np.ndarray:
    """Compute reward-lens curves for a whole batch.

    Returns:
        Array of shape (B, n_layers + 1) where column 0 is post-embed (-1)
        and columns 1..n are post-layer 0..n-1.
    """
    n_layers = model.n_layers
    layer_keys = [-1] + list(range(n_layers))
    B = cache.batch_size
    out = np.full((B, len(layer_keys)), np.nan)
    for j, lk in enumerate(layer_keys):
        h = cache.residual_streams.get(lk)
        if h is None:
            continue
        proj = (h.float() @ model.reward_direction.to(h.device)) + model.reward_bias
        out[:, j] = proj.detach().cpu().numpy()
    return out


def batch_attribution(model, cache) -> tuple[list[str], list[str], list[int], np.ndarray]:
    """Per-component attribution over a batched cache.

    Returns (component_names, component_types, layer_indices, contribs[B, C]).
    """
    w_r = model.reward_direction
    component_names: list[str] = []
    component_types: list[str] = []
    layer_indices: list[int] = []
    cols: list[np.ndarray] = []

    embed = cache.residual_streams.get(-1)
    if embed is not None:
        c = (embed.float() @ w_r.to(embed.device)).detach().cpu().numpy()
        component_names.append("embed")
        component_types.append("embed")
        layer_indices.append(-1)
        cols.append(c)

    n_layers = model.n_layers
    for L in range(n_layers):
        attn = cache.attn_outputs.get(L)
        if attn is not None:
            c = (attn.float() @ w_r.to(attn.device)).detach().cpu().numpy()
            component_names.append(f"attn_L{L}")
            component_types.append("attn")
            layer_indices.append(L)
            cols.append(c)
        mlp = cache.mlp_outputs.get(L)
        if mlp is not None:
            c = (mlp.float() @ w_r.to(mlp.device)).detach().cpu().numpy()
            component_names.append(f"mlp_L{L}")
            component_types.append("mlp")
            layer_indices.append(L)
            cols.append(c)

    contribs = np.stack(cols, axis=1) if cols else np.zeros((cache.batch_size, 0))
    return component_names, component_types, layer_indices, contribs


def batch_head_attribution(model, cache) -> tuple[list[str], list[int], list[int], np.ndarray]:
    """Per-head attention attribution from a batched cache (capture_heads=True).

    Returns (head_names, layer_indices, head_indices, contribs[B, H_total]).
    Each contribution is w_r^T @ (head_output @ o_proj.W^T_for_that_head_slice).

    Implementation: cache.attn_head_outputs[L] is the *input* to o_proj at the
    final token, shape (B, n_heads, d_head). To get each head's contribution
    to the residual, we project per-head through the o_proj weight slice and
    then onto w_r. The o_proj weight has shape (d_model, n_heads * d_head);
    we slice along the input dim into (d_model, d_head) per head.
    """
    # The per-head reward decomposition is defined once, canonically, in
    # reward_lens.attribution.dla; this call is that single source of truth.
    from reward_lens.attribution.dla import head_reward_contributions

    w_r = model.reward_direction
    head_names: list[str] = []
    layer_indices: list[int] = []
    head_indices: list[int] = []
    blocks: list[np.ndarray] = []

    n_heads = model.n_heads
    layers = model.adapter.get_layers(model.model)
    for L, head_outs in cache.attn_head_outputs.items():
        # head_outs: (B, n_heads, d_head)
        o_proj = model.adapter.get_attn_o_proj(layers[L])
        if o_proj is None:
            continue
        blocks.append(head_reward_contributions(head_outs, o_proj.weight, w_r, n_heads))
        for h in range(n_heads):
            head_names.append(f"head_L{L}_H{h}")
            layer_indices.append(L)
            head_indices.append(h)
    contribs = np.concatenate(blocks, axis=1) if blocks else np.zeros((cache.batch_size, 0))
    return head_names, layer_indices, head_indices, contribs
