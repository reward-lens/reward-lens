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

from reward_lens.model import ActivationCache, RewardModel


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
        ax.set_title(
            title or f"Top {k} Component Contributions ({by.capitalize()})"
        )
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

        for i, (layer_idx, ctype) in enumerate(
            zip(self.layer_indices, self.component_types)
        ):
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
        reward_w, cache_w = self.model.forward_with_cache(
            prompt, preferred, max_length=max_length
        )
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
        reward, cache = self.model.forward_with_cache(
            prompt, response, max_length=max_length
        )

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
                c_l = (attn_l.float() @ w_r.to(attn_l.device)).item() if attn_l is not None and not single else c_w
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
                c_l = (mlp_l.float() @ w_r.to(mlp_l.device)).item() if mlp_l is not None and not single else c_w
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
