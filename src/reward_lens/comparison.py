"""
Comparison — Cross-Model Preference Circuit Analysis.

This module provides tools for comparing how different reward models
compute preference. The key questions:

1. Do different models develop preference at the same relative depth?
2. Do the same components matter for the same preference dimensions?
3. Are reward model internals universal or path-dependent?

These questions are critical for scalability: if findings on one model
transfer to others, we can do interpretability once and apply it broadly.
If not, every model needs independent analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from reward_lens.attribution import ComponentAttribution, ComponentResult
from reward_lens.lens import RewardLens, RewardLensResult
from reward_lens.model import RewardModel


@dataclass
class ComparisonResult:
    """Result of comparing multiple reward models on the same preference pair.

    Attributes:
        model_names: Names of the compared models.
        lens_results: Dict mapping model name -> RewardLensResult.
        attribution_results: Dict mapping model name -> ComponentResult.
        crystallization_layers: Dict mapping model name -> crystallization layer (as fraction of depth).
        formation_correlations: Pairwise correlation of normalized differential curves.
    """

    model_names: list[str]
    lens_results: dict[str, RewardLensResult]
    attribution_results: dict[str, ComponentResult]
    crystallization_layers: dict[str, float]
    formation_correlations: dict[tuple[str, str], float]

    def print_summary(self) -> None:
        """Print a formatted comparison summary."""
        print(f"\n{'=' * 60}")
        print("Cross-Model Preference Comparison")
        print(f"{'=' * 60}")

        print("\nCrystallization Layers (fraction of depth):")
        for name, frac in self.crystallization_layers.items():
            print(f"  {name}: {frac:.2%}")

        print("\nFormation Correlation (how similar is the preference formation curve):")
        for (m1, m2), corr in self.formation_correlations.items():
            print(f"  {m1} vs {m2}: {corr:.3f}")

        print(f"\n{'=' * 60}")

    def plot(
        self,
        save_path: Optional[str] = None,
        figsize: tuple[int, int] = (14, 6),
    ) -> None:
        """Plot overlaid reward lens curves for all models.

        Args:
            save_path: Optional path to save figure.
            figsize: Figure size.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.model_names)))

        for idx, name in enumerate(self.model_names):
            result = self.lens_results[name]
            # Normalize layers to [0, 1] for cross-model comparison
            norm_layers = (result.layers - result.layers.min()) / (
                result.layers.max() - result.layers.min() + 1e-8
            )
            # Normalize differential to [0, 1]
            diff = result.differential
            if abs(diff[-1]) > 1e-8:
                norm_diff = diff / diff[-1]
            else:
                norm_diff = diff

            ax.plot(
                norm_layers,
                norm_diff,
                color=colors[idx],
                linewidth=2,
                label=name,
                marker="o",
                markersize=2,
            )

        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="50% threshold")
        ax.set_xlabel("Normalized Depth (0 = embedding, 1 = final layer)")
        ax.set_ylabel("Normalized Preference Differential")
        ax.set_title("Cross-Model Comparison: Preference Formation Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()


class ModelComparator:
    """Compare preference computation across multiple reward models.

    Args:
        models: Dict mapping model names to RewardModel instances.
    """

    def __init__(self, models: dict[str, RewardModel]):
        self.models = models

    def compare(
        self,
        prompt: str,
        preferred: str,
        dispreferred: str,
        max_length: int = 2048,
    ) -> ComparisonResult:
        """Run reward lens and attribution on all models for the same pair.

        Args:
            prompt: The user prompt.
            preferred: The preferred completion.
            dispreferred: The dispreferred completion.
            max_length: Maximum sequence length.

        Returns:
            ComparisonResult with all analyses.
        """
        lens_results = {}
        attrib_results = {}
        crystal_fracs = {}

        for name, model in self.models.items():
            # Reward lens
            lens = RewardLens(model)
            lr = lens.trace(prompt, preferred, dispreferred, max_length=max_length)
            lens_results[name] = lr

            # Crystallization as fraction of depth
            n_layers = model.n_layers
            crystal_frac = (lr.crystallization_layer + 1) / (n_layers + 1)  # +1 for embed
            crystal_fracs[name] = crystal_frac

            # Component attribution
            attrib = ComponentAttribution(model)
            cr = attrib.attribute(prompt, preferred, dispreferred, max_length=max_length)
            attrib_results[name] = cr

        # Compute pairwise formation correlations
        correlations = {}
        model_names = list(self.models.keys())
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                m1, m2 = model_names[i], model_names[j]
                corr = self._formation_correlation(lens_results[m1], lens_results[m2])
                correlations[(m1, m2)] = corr

        return ComparisonResult(
            model_names=model_names,
            lens_results=lens_results,
            attribution_results=attrib_results,
            crystallization_layers=crystal_fracs,
            formation_correlations=correlations,
        )

    def _formation_correlation(self, result1: RewardLensResult, result2: RewardLensResult) -> float:
        """Compute correlation between two normalized preference formation curves.

        We normalize both curves to [0, 1] range and interpolate to the same
        number of points, then compute Pearson correlation.
        """

        def normalize_curve(result: RewardLensResult) -> np.ndarray:
            diff = result.differential
            if abs(diff[-1]) < 1e-8:
                return np.zeros_like(diff)
            return diff / diff[-1]

        curve1 = normalize_curve(result1)
        curve2 = normalize_curve(result2)

        # Interpolate to same length (use the longer one)
        n = max(len(curve1), len(curve2))
        x1 = np.linspace(0, 1, len(curve1))
        x2 = np.linspace(0, 1, len(curve2))
        x_common = np.linspace(0, 1, n)

        c1_interp = np.interp(x_common, x1, curve1)
        c2_interp = np.interp(x_common, x2, curve2)

        # Pearson correlation
        if np.std(c1_interp) < 1e-8 or np.std(c2_interp) < 1e-8:
            return 0.0

        return float(np.corrcoef(c1_interp, c2_interp)[0, 1])
