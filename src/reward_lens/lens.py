"""
Reward Lens — Layer-by-Layer Preference Formation.

This is the central primitive of the reward-lens toolkit. Just as the logit lens
projects intermediate residual stream states onto the vocabulary to see what a
generative model is "about to predict" at each layer, the reward lens projects
intermediate states onto the reward direction to see what reward score the model
would assign if it stopped processing at each layer.

The key insight: since the reward is r = w_r^T @ h_final + b, and h_final is
the sum of all layer contributions (by the residual stream), we can project
each intermediate h^(l) onto w_r to see the "proto-reward" at each layer.

The *differential* reward lens is even more informative: for a preference pair,
we compute the reward lens for both completions and take the difference. This
traces when the model starts *distinguishing* preferred from dispreferred —
i.e., when "preference" actually forms.

For the formal definition, see Section 5 of the research document.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from reward_lens.model import ActivationCache, RewardModel


@dataclass
class RewardLensResult:
    """Result of a reward lens analysis.

    Attributes:
        layers: Array of layer indices. -1 = post-embedding, 0..n-1 = post-layer.
        reward_lens_preferred: Reward lens values for the preferred completion at each layer.
        reward_lens_dispreferred: Reward lens values for the dispreferred completion.
        differential: The difference (preferred - dispreferred) at each layer.
        marginal_contributions: Per-layer marginal contribution to the differential.
            delta^(l) = Delta_R^(l) - Delta_R^(l-1).
        reward_preferred: Final reward score for preferred completion.
        reward_dispreferred: Final reward score for dispreferred completion.
        crystallization_layer: The first layer at which the differential reaches
            50% of its final value. A rough measure of "when preference forms."
    """

    layers: np.ndarray
    reward_lens_preferred: np.ndarray
    reward_lens_dispreferred: np.ndarray
    reward_preferred: float
    reward_dispreferred: float
    differential: Optional[np.ndarray] = field(default=None)
    marginal_contributions: Optional[np.ndarray] = field(default=None)
    crystallization_layer: Optional[int] = field(default=None)

    def __post_init__(self) -> None:
        if self.differential is None:
            self.differential = self.reward_lens_preferred - self.reward_lens_dispreferred
        if self.marginal_contributions is None:
            self.marginal_contributions = np.diff(self.differential)
        if self.crystallization_layer is None:
            # Fall back to the largest-magnitude finite differential when
            # ``final`` is zero or non-finite — Gemma-2's logit-soft-cap
            # can collapse the late-layer differential to numerical zero.
            final_diff = self.differential[-1]
            ref = final_diff
            if not np.isfinite(ref) or abs(ref) < 1e-8:
                finite = self.differential[np.isfinite(self.differential)]
                if finite.size == 0:
                    self.crystallization_layer = int(self.layers[-1])
                    return
                ref = float(finite[int(np.argmax(np.abs(finite)))])
                if abs(ref) < 1e-8:
                    self.crystallization_layer = int(self.layers[-1])
                    return
            threshold = 0.5 * ref
            self.crystallization_layer = int(self.layers[-1])
            for i, d in enumerate(self.differential):
                if not np.isfinite(d):
                    continue
                if (ref >= 0 and d >= threshold) or (ref < 0 and d <= threshold):
                    self.crystallization_layer = int(self.layers[i])
                    break

    def plot(
        self,
        save_path: Optional[str] = None,
        figsize: tuple[int, int] = (14, 8),
        title: Optional[str] = None,
    ) -> None:
        """Plot the reward lens analysis.

        Creates a 2x1 figure:
        - Top: Reward lens values for both completions across layers
        - Bottom: Marginal per-layer contributions to the preference differential

        Args:
            save_path: If provided, save the figure to this path.
            figsize: Figure size.
            title: Custom title.
        """
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])

        # Top panel: reward lens values
        ax1.plot(
            self.layers,
            self.reward_lens_preferred,
            "b-o",
            markersize=3,
            label=f"Preferred (final: {self.reward_preferred:.3f})",
            linewidth=1.5,
        )
        ax1.plot(
            self.layers,
            self.reward_lens_dispreferred,
            "r-o",
            markersize=3,
            label=f"Dispreferred (final: {self.reward_dispreferred:.3f})",
            linewidth=1.5,
        )
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax1.axvline(
            x=self.crystallization_layer,
            color="green",
            linestyle=":",
            alpha=0.7,
            label=f"Crystallization layer: {self.crystallization_layer}",
        )
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Reward Lens Value")
        ax1.set_title(title or "Reward Lens: Layer-by-Layer Preference Formation")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom panel: marginal contributions
        colors = ["blue" if v > 0 else "red" for v in self.marginal_contributions]
        ax2.bar(self.layers[1:], self.marginal_contributions, color=colors, alpha=0.7)
        ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Marginal Δ Contribution")
        ax2.set_title("Per-Layer Marginal Contribution to Preference Differential")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()


class RewardLens:
    """Compute reward lens analysis for preference pairs.

    The reward lens projects the residual stream at each layer onto the reward
    direction to trace how preference forms across the depth of the model.

    Args:
        model: A RewardModel instance.

    Example:
        >>> lens = RewardLens(model)
        >>> result = lens.trace("What is AI?", "AI is...", "AI is magic...")
        >>> result.plot()
    """

    def __init__(self, model: RewardModel):
        self.model = model

    def trace(
        self,
        prompt: str,
        preferred: str,
        dispreferred: str,
        max_length: int = 2048,
    ) -> RewardLensResult:
        """Run reward lens analysis on a preference pair.

        Runs forward passes on both completions, caches all intermediate
        residual stream states, and projects each onto the reward direction.

        Args:
            prompt: The user prompt.
            preferred: The preferred completion.
            dispreferred: The dispreferred completion.
            max_length: Maximum sequence length.

        Returns:
            RewardLensResult with all analysis data.
        """
        # Forward pass with caching for both completions
        reward_w, cache_w = self.model.forward_with_cache(
            prompt, preferred, max_length=max_length
        )
        reward_l, cache_l = self.model.forward_with_cache(
            prompt, dispreferred, max_length=max_length
        )

        # Compute reward lens at each layer
        # Layer -1 = post-embedding, layers 0..n-1 = post-transformer-layer
        n_layers = self.model.n_layers
        layers = list(range(-1, n_layers))
        lens_w = []
        lens_l = []

        for layer_idx in layers:
            h_w = cache_w.residual_streams.get(layer_idx)
            h_l = cache_l.residual_streams.get(layer_idx)

            if h_w is not None and h_l is not None:
                r_w = self.model.project_onto_reward(h_w.float()).item()
                r_l = self.model.project_onto_reward(h_l.float()).item()
            else:
                # If a layer's activations weren't captured, use NaN
                r_w = float("nan")
                r_l = float("nan")

            lens_w.append(r_w)
            lens_l.append(r_l)

        lens_w = np.array(lens_w)
        lens_l = np.array(lens_l)
        differential = lens_w - lens_l

        # Marginal contributions: delta^(l) = differential^(l) - differential^(l-1)
        marginal = np.diff(differential)

        # Crystallization layer: first layer where differential reaches 50% of final.
        # Defense-in-depth: fall back to max-|diff| when final is zero
        # (Gemma-2 logit-soft-cap can collapse the late-layer differential).
        final_diff = differential[-1]
        ref = final_diff
        if not np.isfinite(ref) or abs(ref) < 1e-8:
            finite = differential[np.isfinite(differential)]
            if finite.size > 0:
                cand = float(finite[int(np.argmax(np.abs(finite)))])
                if abs(cand) >= 1e-8:
                    ref = cand
        if np.isfinite(ref) and abs(ref) > 1e-8:
            threshold = 0.5 * ref
            crystal_idx = layers[-1]
            for i, d in enumerate(differential):
                if not np.isfinite(d):
                    continue
                if (ref > 0 and d >= threshold) or (ref < 0 and d <= threshold):
                    crystal_idx = layers[i]
                    break
        else:
            crystal_idx = layers[-1]

        return RewardLensResult(
            layers=np.array(layers),
            reward_lens_preferred=lens_w,
            reward_lens_dispreferred=lens_l,
            differential=differential,
            marginal_contributions=marginal,
            reward_preferred=reward_w,
            reward_dispreferred=reward_l,
            crystallization_layer=crystal_idx,
        )

    def trace_single(
        self,
        prompt: str,
        response: str,
        max_length: int = 2048,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run reward lens on a single completion (not a pair).

        Returns:
            Tuple of (layers, reward_lens_values).
        """
        reward, cache = self.model.forward_with_cache(
            prompt, response, max_length=max_length
        )

        n_layers = self.model.n_layers
        layers = list(range(-1, n_layers))
        lens_values = []

        for layer_idx in layers:
            h = cache.residual_streams.get(layer_idx)
            if h is not None:
                r = self.model.project_onto_reward(h.float()).item()
            else:
                r = float("nan")
            lens_values.append(r)

        return np.array(layers), np.array(lens_values)


def reward_lens_plot(
    model: RewardModel,
    prompt: str,
    preferred: str,
    dispreferred: str,
    save_path: Optional[str] = None,
    max_length: int = 2048,
    title: Optional[str] = None,
) -> RewardLensResult:
    """Convenience function: run reward lens and plot in one call.

    Args:
        model: A RewardModel instance.
        prompt: The user prompt.
        preferred: The preferred completion.
        dispreferred: The dispreferred completion.
        save_path: Optional path to save the plot.
        max_length: Maximum sequence length.
        title: Optional custom title.

    Returns:
        The RewardLensResult.
    """
    lens = RewardLens(model)
    result = lens.trace(prompt, preferred, dispreferred, max_length=max_length)
    result.plot(save_path=save_path, title=title)
    return result
