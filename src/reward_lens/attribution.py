"""
Component Attribution — Per-Head, Per-MLP Reward Decomposition.

Because the residual stream is a sum of contributions from each component
(embedding, each attention head, each MLP layer), and the reward is a linear
projection of the final residual stream, the reward decomposes as:

    r = w_r^T @ h_embed + sum_l(w_r^T @ attn^(l) + w_r^T @ mlp^(l)) + b_r

Each term w_r^T @ attn^(l) is the *signed contribution* of layer l's attention
to the reward. Positive = pushes reward up, negative = pushes reward down.

This is the reward model analogue of Direct Logit Attribution (DLA) for
generative models. But simpler: DLA projects onto a vocabulary-sized unembedding
matrix and needs to select a target token. We project onto a single scalar
direction. Every component's contribution is directly interpretable.

For contrastive analysis (preference pairs), we compute the *differential*
attribution: which components drive the *difference* in reward between preferred
and dispreferred completions?

Note: This analysis is *observational*, not causal. A component may have a large
positive contribution to the reward difference, but ablating it might not change
the preference (because other components compensate). For causal claims, use
activation patching (see patching.py). We are explicit about this distinction
because conflating observational and causal evidence is a common error in
interpretability research.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from reward_lens.model import ActivationCache, BatchedActivationCache, RewardModel


@dataclass
class ComponentResult:
    """Per-component reward attribution result.

    Attributes:
        component_names: List of component names (e.g., "attn_L0", "mlp_L5").
        component_types: List of component types ("embed", "attn", "mlp").
        layer_indices: Layer index for each component (-1 for embedding).
        contributions_preferred: Signed reward contribution for preferred completion.
        contributions_dispreferred: Signed reward contribution for dispreferred.
        differential_contributions: Difference (preferred - dispreferred).
        total_reward_preferred: Sum of all contributions + bias for preferred.
        total_reward_dispreferred: Same for dispreferred.
    """

    component_names: list[str]
    component_types: list[str]
    layer_indices: list[int]
    contributions_preferred: np.ndarray
    contributions_dispreferred: np.ndarray
    differential_contributions: np.ndarray
    total_reward_preferred: float
    total_reward_dispreferred: float

    def top_k(self, k: int = 15, by: str = "differential") -> list[tuple[str, float]]:
        """Return top-k components by contribution magnitude.

        Args:
            k: Number of components to return.
            by: Which contributions to sort by. One of:
                "differential" (default): abs(preferred - dispreferred)
                "preferred": abs contribution to preferred
                "dispreferred": abs contribution to dispreferred

        Returns:
            List of (component_name, contribution_value) tuples.
        """
        if by == "differential":
            values = self.differential_contributions
        elif by == "preferred":
            values = self.contributions_preferred
        elif by == "dispreferred":
            values = self.contributions_dispreferred
        else:
            raise ValueError(f"Unknown sort key: {by}")

        indices = np.argsort(np.abs(values))[::-1][:k]
        return [(self.component_names[i], values[i]) for i in indices]

    def by_type(self, component_type: str) -> "ComponentResult":
        """Filter to a specific component type.

        Args:
            component_type: "embed", "attn", or "mlp".

        Returns:
            New ComponentResult with only the specified type.
        """
        mask = [t == component_type for t in self.component_types]
        indices = [i for i, m in enumerate(mask) if m]
        return ComponentResult(
            component_names=[self.component_names[i] for i in indices],
            component_types=[self.component_types[i] for i in indices],
            layer_indices=[self.layer_indices[i] for i in indices],
            contributions_preferred=self.contributions_preferred[indices],
            contributions_dispreferred=self.contributions_dispreferred[indices],
            differential_contributions=self.differential_contributions[indices],
            total_reward_preferred=self.total_reward_preferred,
            total_reward_dispreferred=self.total_reward_dispreferred,
        )

    def plot_top_k(
        self,
        k: int = 15,
        by: str = "differential",
        save_path: Optional[str] = None,
        figsize: tuple[int, int] = (12, 6),
        title: Optional[str] = None,
    ) -> None:
        """Plot top-k components as a horizontal bar chart.

        Args:
            k: Number of components to show.
            by: Sort criterion.
            save_path: Optional path to save the figure.
            figsize: Figure size.
            title: Custom title.
        """
        import matplotlib.pyplot as plt

        top = self.top_k(k=k, by=by)
        names = [t[0] for t in reversed(top)]
        values = [t[1] for t in reversed(top)]
        colors = ["#2196F3" if v > 0 else "#F44336" for v in values]

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.barh(range(len(names)), values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Reward Contribution")
        ax.set_title(title or f"Top {k} Component Contributions ({by.capitalize()})")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()

    def plot_heatmap(
        self,
        save_path: Optional[str] = None,
        figsize: Optional[tuple[int, int]] = None,
        title: Optional[str] = None,
    ) -> None:
        """Plot a layer × component-type heatmap of differential contributions.

        Shows attention and MLP contributions at each layer.

        Args:
            save_path: Optional path to save the figure.
            figsize: Figure size (auto-calculated if None).
            title: Custom title.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Separate by type
        max_layer = max(self.layer_indices) + 1
        attn_values = np.zeros(max_layer)
        mlp_values = np.zeros(max_layer)

        for i, (layer_idx, ctype) in enumerate(zip(self.layer_indices, self.component_types)):
            if ctype == "attn" and layer_idx >= 0:
                attn_values[layer_idx] = self.differential_contributions[i]
            elif ctype == "mlp" and layer_idx >= 0:
                mlp_values[layer_idx] = self.differential_contributions[i]

        data = np.stack([attn_values, mlp_values], axis=0)  # (2, n_layers)

        if figsize is None:
            figsize = (max(12, max_layer * 0.3), 3)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        vmax = max(abs(data.min()), abs(data.max())) or 1.0
        sns.heatmap(
            data,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            yticklabels=["Attention", "MLP"],
            xticklabels=[str(i) for i in range(max_layer)],
            cbar_kws={"label": "Differential Reward Contribution"},
        )
        ax.set_xlabel("Layer")
        ax.set_title(title or "Component Attribution Heatmap (Differential)")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()


class ComponentAttribution:
    """Decompose reward scores into per-component signed contributions.

    This is the observational decomposition — it tells you which components
    contribute most to the reward, but not whether they are causally necessary.
    For causal analysis, use ActivationPatcher.

    Args:
        model: A RewardModel instance.
    """

    def __init__(self, model: RewardModel):
        self.model = model

    def attribute(
        self,
        prompt: str,
        preferred: str,
        dispreferred: str,
        max_length: int = 2048,
    ) -> ComponentResult:
        """Compute per-component reward attribution for a preference pair.

        For each component c (embedding, each attention sublayer, each MLP sublayer),
        computes:
            contribution_c = w_r^T @ output_c

        where output_c is the component's contribution to the residual stream at
        the final token position.

        Args:
            prompt: The user prompt.
            preferred: The preferred completion.
            dispreferred: The dispreferred completion.
            max_length: Maximum sequence length.

        Returns:
            ComponentResult with all attribution data.
        """
        # Run forward passes with caching
        reward_w, cache_w = self.model.forward_with_cache(prompt, preferred, max_length=max_length)
        reward_l, cache_l = self.model.forward_with_cache(
            prompt, dispreferred, max_length=max_length
        )

        return self._attribute_from_caches(cache_w, cache_l, reward_w, reward_l)

    def attribute_single(
        self,
        prompt: str,
        response: str,
        max_length: int = 2048,
    ) -> ComponentResult:
        """Compute per-component attribution for a single completion.

        Args:
            prompt: The user prompt.
            response: The completion.
            max_length: Maximum sequence length.

        Returns:
            ComponentResult (differential fields will be zeros).
        """
        reward, cache = self.model.forward_with_cache(prompt, response, max_length=max_length)

        # Create a "dummy" cache with zeros for the contrastive side
        return self._attribute_from_caches(cache, cache, reward, reward, single=True)

    def _attribute_from_caches(
        self,
        cache_w: ActivationCache,
        cache_l: ActivationCache,
        reward_w: float,
        reward_l: float,
        single: bool = False,
    ) -> ComponentResult:
        """Internal: compute attribution from pre-computed caches."""
        w_r = self.model.reward_direction

        component_names = []
        component_types = []
        layer_indices = []
        contribs_w = []
        contribs_l = []

        # Embedding contribution
        embed_w = cache_w.residual_streams.get(-1)
        embed_l = cache_l.residual_streams.get(-1)
        if embed_w is not None and embed_l is not None:
            c_w = (embed_w.float() @ w_r.to(embed_w.device)).item()
            c_l = (embed_l.float() @ w_r.to(embed_l.device)).item() if not single else 0.0
            component_names.append("embed")
            component_types.append("embed")
            layer_indices.append(-1)
            contribs_w.append(c_w)
            contribs_l.append(c_l if not single else c_w)

        # Per-layer attention and MLP contributions
        n_layers = self.model.n_layers
        for layer_idx in range(n_layers):
            # Attention
            attn_w = cache_w.attn_outputs.get(layer_idx)
            attn_l = cache_l.attn_outputs.get(layer_idx)
            if attn_w is not None:
                c_w = (attn_w.float() @ w_r.to(attn_w.device)).item()
                c_l = (
                    (attn_l.float() @ w_r.to(attn_l.device)).item()
                    if attn_l is not None and not single
                    else c_w
                )
                component_names.append(f"attn_L{layer_idx}")
                component_types.append("attn")
                layer_indices.append(layer_idx)
                contribs_w.append(c_w)
                contribs_l.append(c_l)

            # MLP
            mlp_w = cache_w.mlp_outputs.get(layer_idx)
            mlp_l = cache_l.mlp_outputs.get(layer_idx)
            if mlp_w is not None:
                c_w = (mlp_w.float() @ w_r.to(mlp_w.device)).item()
                c_l = (
                    (mlp_l.float() @ w_r.to(mlp_l.device)).item()
                    if mlp_l is not None and not single
                    else c_w
                )
                component_names.append(f"mlp_L{layer_idx}")
                component_types.append("mlp")
                layer_indices.append(layer_idx)
                contribs_w.append(c_w)
                contribs_l.append(c_l)

        contribs_w = np.array(contribs_w)
        contribs_l = np.array(contribs_l)

        return ComponentResult(
            component_names=component_names,
            component_types=component_types,
            layer_indices=layer_indices,
            contributions_preferred=contribs_w,
            contributions_dispreferred=contribs_l,
            differential_contributions=contribs_w - contribs_l,
            total_reward_preferred=reward_w,
            total_reward_dispreferred=reward_l,
        )

    def attribute_heads(
        self,
        prompt: str,
        preferred: str,
        dispreferred: str,
        max_length: int = 2048,
    ) -> ComponentResult:
        """Per-head attention attribution (head granularity).

        Decomposes each layer's attention contribution into per-head terms by
        capturing the input to o_proj and projecting each head's slice
        through its own o_proj weight slice. Yields ``n_layers * n_heads``
        head components plus the per-layer MLPs.
        """
        # Use the batched forward path with capture_heads=True for cheap
        # head-resolution attribution.
        cache_w = self.model.forward_with_cache_batch(
            [(prompt, preferred)],
            batch_size=1,
            max_length=max_length,
            capture_heads=True,
        )
        cache_l = self.model.forward_with_cache_batch(
            [(prompt, dispreferred)],
            batch_size=1,
            max_length=max_length,
            capture_heads=True,
        )

        names_w, layer_idxs, head_idxs, contribs_w = _batch_head_attribution(self.model, cache_w)
        names_l, _, _, contribs_l = _batch_head_attribution(self.model, cache_l)

        # Also include per-layer MLP contributions for completeness.
        w_r = self.model.reward_direction
        n_layers = self.model.n_layers
        component_names = list(names_w)
        component_types = ["attn_head"] * len(names_w)
        layer_indices = list(layer_idxs)
        contribs_w_full = list(contribs_w[0])
        contribs_l_full = list(contribs_l[0])

        for L in range(n_layers):
            mlp_w = cache_w.mlp_outputs.get(L)
            mlp_l = cache_l.mlp_outputs.get(L)
            if mlp_w is None or mlp_l is None:
                continue
            cw = (mlp_w.float() @ w_r.to(mlp_w.device)).item()
            cl = (mlp_l.float() @ w_r.to(mlp_l.device)).item()
            component_names.append(f"mlp_L{L}")
            component_types.append("mlp")
            layer_indices.append(L)
            contribs_w_full.append(cw)
            contribs_l_full.append(cl)

        contribs_w_arr = np.array(contribs_w_full)
        contribs_l_arr = np.array(contribs_l_full)
        return ComponentResult(
            component_names=component_names,
            component_types=component_types,
            layer_indices=layer_indices,
            contributions_preferred=contribs_w_arr,
            contributions_dispreferred=contribs_l_arr,
            differential_contributions=contribs_w_arr - contribs_l_arr,
            total_reward_preferred=float(cache_w.rewards[0].item()),
            total_reward_dispreferred=float(cache_l.rewards[0].item()),
        )


def _batch_head_attribution(
    model: RewardModel,
    cache: BatchedActivationCache,
) -> tuple[list[str], list[int], list[int], np.ndarray]:
    """Decompose per-head attention contributions to the reward, batched.

    An attention layer's contribution to the residual stream is
    ``o_proj(concat_h head_h)``, and because ``o_proj`` is linear this is the
    sum of per-head terms ``head_h @ W_o[:, h*d_head:(h+1)*d_head].T``. Each
    such term is a vector in the residual stream, so its signed contribution to
    the scalar reward is its projection onto the reward direction ``w_r``.

    This helper computes that projection for every captured head and every pair
    in the batch. It mirrors the o_proj weight-slicing convention used by
    :mod:`reward_lens.path_patching`, so head-level attribution and head-level
    patching decompose attention the same way.

    Args:
        model: The wrapped reward model. Supplies the reward direction, the head
            count, and the per-layer ``o_proj`` modules (via its adapter).
        cache: A batched cache populated with ``capture_heads=True``. Each
            ``cache.attn_head_outputs[layer]`` tensor has shape
            ``(batch, n_heads, d_head)`` — the per-head inputs to ``o_proj`` at
            the final token.

    Returns:
        A tuple ``(names, layer_indices, head_indices, contributions)``:

        - ``names``: ``"head_L{layer}_H{head}"`` for each component.
        - ``layer_indices`` / ``head_indices``: parallel lists of ints.
        - ``contributions``: ``(batch, n_components)`` array of signed per-head
          reward contributions.

        Components are ordered by ascending ``(layer, head)``. Layers whose
        adapter does not expose an ``o_proj`` are skipped, so an adapter without
        head access simply yields no head components (never an error).
    """
    layers = model.adapter.get_layers(model.model)
    n_heads = model.n_heads
    w_r = model.reward_direction.float()  # (d_model,)

    names: list[str] = []
    layer_indices: list[int] = []
    head_indices: list[int] = []
    # One (batch, n_heads) block per layer; concatenated along the head axis.
    per_layer_contribs: list[np.ndarray] = []

    for layer_idx in sorted(cache.attn_head_outputs.keys()):
        o_proj = model.adapter.get_attn_o_proj(layers[layer_idx])
        if o_proj is None:
            continue
        head_out = cache.attn_head_outputs[layer_idx]  # (batch, n_heads, d_head)
        # Detach the projection weight: attribution is pure inference, and the
        # o_proj weight is a live Parameter (requires grad) whereas the cached
        # head outputs are already detached.
        weight = o_proj.weight.detach()  # (d_model, n_heads * d_head)
        d_head = weight.shape[1] // n_heads
        w_r_dev = w_r.to(weight.device)

        # contrib[b, h] = (head_out[b, h] @ W_h.T) @ w_r
        #              = head_out[b, h] . (W_h.T @ w_r)
        # Precompute each head's reward projector once: (n_heads, d_head).
        weight_heads = weight.float().reshape(weight.shape[0], n_heads, d_head)
        projector = torch.einsum("d,dhk->hk", w_r_dev, weight_heads)
        head_out = head_out.float().to(weight.device)
        contrib = torch.einsum("bhk,hk->bh", head_out, projector)  # (batch, n_heads)
        per_layer_contribs.append(contrib.cpu().numpy())

        for head_idx in range(n_heads):
            names.append(f"head_L{layer_idx}_H{head_idx}")
            layer_indices.append(layer_idx)
            head_indices.append(head_idx)

    if per_layer_contribs:
        contributions = np.concatenate(per_layer_contribs, axis=1)
    else:
        contributions = np.zeros((cache.batch_size, 0))

    return names, layer_indices, head_indices, contributions
