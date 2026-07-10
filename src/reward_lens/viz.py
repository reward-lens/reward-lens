"""
Visualization utilities for reward-lens.

Provides publication-quality plotting functions that work consistently
across all analysis modules.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def setup_style():
    """Apply reward-lens plot style settings."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )


def reward_lens_dashboard(
    lens_result,
    attrib_result=None,
    patching_result=None,
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (16, 12),
    title: Optional[str] = None,
) -> None:
    """Create a comprehensive dashboard combining multiple analyses.

    Args:
        lens_result: RewardLensResult from reward lens analysis.
        attrib_result: Optional ComponentResult from attribution.
        patching_result: Optional PatchingResult from patching.
        save_path: Optional path to save figure.
        figsize: Figure size.
        title: Custom title.
    """
    import matplotlib.pyplot as plt

    setup_style()

    n_panels = 1 + (1 if attrib_result else 0) + (1 if patching_result else 0)

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    # Panel 1: Reward Lens
    ax = axes[panel_idx]
    ax.plot(
        lens_result.layers,
        lens_result.reward_lens_preferred,
        "b-o",
        markersize=3,
        linewidth=1.5,
        label=f"Preferred ({lens_result.reward_preferred:.3f})",
    )
    ax.plot(
        lens_result.layers,
        lens_result.reward_lens_dispreferred,
        "r-o",
        markersize=3,
        linewidth=1.5,
        label=f"Dispreferred ({lens_result.reward_dispreferred:.3f})",
    )
    ax.fill_between(
        lens_result.layers,
        lens_result.reward_lens_preferred,
        lens_result.reward_lens_dispreferred,
        alpha=0.1,
        color="purple",
    )
    ax.axvline(
        x=lens_result.crystallization_layer,
        color="green",
        linestyle=":",
        alpha=0.7,
        label=f"Crystallization: L{lens_result.crystallization_layer}",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Reward Lens Value")
    ax.set_title("Reward Lens: Layer-by-Layer Preference Formation")
    ax.legend(fontsize=9)
    panel_idx += 1

    # Panel 2: Component Attribution
    if attrib_result is not None:
        ax = axes[panel_idx]
        max_layer = max(attrib_result.layer_indices) + 1
        attn_vals = np.zeros(max_layer)
        mlp_vals = np.zeros(max_layer)
        for i, (li, ct) in enumerate(
            zip(attrib_result.layer_indices, attrib_result.component_types)
        ):
            if ct == "attn" and li >= 0:
                attn_vals[li] = attrib_result.differential_contributions[i]
            elif ct == "mlp" and li >= 0:
                mlp_vals[li] = attrib_result.differential_contributions[i]

        x = np.arange(max_layer)
        width = 0.35
        ax.bar(x - width / 2, attn_vals, width, label="Attention", color="#2196F3", alpha=0.7)
        ax.bar(x + width / 2, mlp_vals, width, label="MLP", color="#FF9800", alpha=0.7)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Differential Contribution")
        ax.set_title("Component Attribution (Observational)")
        ax.legend(fontsize=9)
        panel_idx += 1

    # Panel 3: Activation Patching
    if patching_result is not None:
        ax = axes[panel_idx]
        max_layer = max(patching_result.layer_indices) + 1
        attn_effects = np.zeros(max_layer)
        mlp_effects = np.zeros(max_layer)
        normalized = patching_result.normalized_effects()
        for i, (li, ct) in enumerate(
            zip(patching_result.layer_indices, patching_result.component_types)
        ):
            if ct == "attn" and li >= 0:
                attn_effects[li] = normalized[i]
            elif ct == "mlp" and li >= 0:
                mlp_effects[li] = normalized[i]

        x = np.arange(max_layer)
        width = 0.35
        ax.bar(x - width / 2, attn_effects, width, label="Attention", color="#4CAF50", alpha=0.7)
        ax.bar(x + width / 2, mlp_effects, width, label="MLP", color="#E91E63", alpha=0.7)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Normalized Patch Effect")
        ax.set_title(f"Activation Patching (Causal, {patching_result.patching_mode})")
        ax.legend(fontsize=9)
        panel_idx += 1

    fig.suptitle(title or "Reward Model Interpretability Dashboard", fontsize=15, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def circuit_overlap_plot(
    overlap_matrix: np.ndarray,
    dimension_names: list[str],
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (8, 7),
) -> None:
    """Plot a heatmap of circuit overlap between preference dimensions.

    Args:
        overlap_matrix: Square matrix of overlap scores.
        dimension_names: Names of the dimensions.
        save_path: Optional path to save figure.
        figsize: Figure size.
    """
    import matplotlib.pyplot as plt

    setup_style()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(overlap_matrix, cmap="YlOrRd", vmin=0, vmax=1)
    
    # Add colorbar
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Set tick marks and labels
    n = len(dimension_names)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(dimension_names)
    ax.set_yticklabels(dimension_names)
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations in each cell
    for i in range(n):
        for j in range(n):
            val = overlap_matrix[i, j]
            # Use white text for dark backgrounds, black text for light backgrounds
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color)
            
    ax.set_title("Circuit Overlap Between Preference Dimensions")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

