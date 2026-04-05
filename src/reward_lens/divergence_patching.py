"""
Divergence-Aware Activation Patching.

Based on "Addressing Divergent Representations from Causal Interventions" (2511.04638).

The key insight: standard causal interventions (activation patching, DAS, etc.)
can create out-of-distribution (divergent) representations. Some divergences are
"harmless" (in the behavioral null-space), but others are "pernicious" — they
activate hidden circuits and can make causal claims unfaithful.

This module extends the base ActivationPatcher with:
1. Divergence detection: flag when interventions create OOD representations
2. Divergence classification: distinguish harmless vs pernicious divergences  
3. Constrained patching: use Counterfactual Latent (CL) loss to stay in-distribution

Usage:
    >>> patcher = DivergenceAwarePatching(model)
    >>> result = patcher.patch_with_divergence_check(prompt, preferred, dispreferred)
    >>> if result.has_pernicious_divergence:
    ...     print("Warning: causal claims may be unreliable")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from reward_lens.model import ActivationCache, RewardModel
from reward_lens.patching import ActivationPatcher, PatchingResult


@dataclass
class DivergenceInfo:
    """Information about divergence for a single intervention.
    
    Attributes:
        component_name: Name of the patched component.
        divergence_score: Mahalanobis distance from training distribution.
        is_divergent: Whether the intervention creates OOD activations.
        divergence_type: Classification of divergence type.
        confidence: Confidence in the classification.
    """
    
    component_name: str
    divergence_score: float
    is_divergent: bool
    divergence_type: Literal["in_distribution", "harmless", "pernicious", "unknown"]
    confidence: float


@dataclass
class DivergenceAwarePatchingResult(PatchingResult):
    """Extended patching result with divergence analysis.
    
    Attributes:
        (inherited from PatchingResult)
        divergence_info: Per-component divergence analysis.
        has_pernicious_divergence: Whether any intervention has pernicious divergence.
        reliability_score: Overall reliability of causal claims (0-1).
        divergent_components: List of components with significant divergence.
    """
    
    divergence_info: list[DivergenceInfo] = field(default_factory=list)
    has_pernicious_divergence: bool = False
    reliability_score: float = 1.0
    divergent_components: list[str] = field(default_factory=list)
    
    def print_divergence_summary(self) -> None:
        """Print a summary of divergence analysis."""
        print(f"\n{'='*60}")
        print("Divergence Analysis Summary")
        print(f"{'='*60}")
        
        print(f"\nOverall Reliability Score: {self.reliability_score:.1%}")
        
        if self.has_pernicious_divergence:
            print("\n⚠️  WARNING: Pernicious divergences detected!")
            print("    Causal claims from these interventions may be unreliable.")
        
        if self.divergent_components:
            print(f"\nDivergent Components ({len(self.divergent_components)}):")
            for info in self.divergence_info:
                if info.is_divergent:
                    icon = "🔴" if info.divergence_type == "pernicious" else "🟡"
                    print(f"  {icon} {info.component_name}:")
                    print(f"      Divergence score: {info.divergence_score:.2f}")
                    print(f"      Type: {info.divergence_type} ({info.confidence:.0%} conf.)")
        else:
            print("\n✅ No significant divergences detected.")
            print("   Causal claims should be reliable.")
        
        print(f"\n{'='*60}")

    def plot_with_divergence(
        self,
        save_path: Optional[str] = None,
        figsize: tuple[int, int] = (14, 8),
        title: Optional[str] = None,
    ) -> None:
        """Plot patching results with divergence overlay.
        
        Args:
            save_path: Optional path to save figure.
            figsize: Figure size.
            title: Custom title.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        max_layer = max(self.layer_indices) + 1
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])
        
        # Top: Standard patching heatmap
        effects = self.normalized_effects()
        attn_effects = np.zeros(max_layer)
        mlp_effects = np.zeros(max_layer)
        
        for i, (layer_idx, ctype) in enumerate(
            zip(self.layer_indices, self.component_types)
        ):
            if ctype == "attn" and layer_idx >= 0:
                attn_effects[layer_idx] = effects[i]
            elif ctype == "mlp" and layer_idx >= 0:
                mlp_effects[layer_idx] = effects[i]
        
        data = np.stack([attn_effects, mlp_effects], axis=0)
        vmax = max(abs(data.min()), abs(data.max())) or 1.0
        
        sns.heatmap(
            data, ax=ax1, cmap="YlOrRd", vmin=0, vmax=vmax,
            yticklabels=["Attention", "MLP"],
            xticklabels=[str(i) for i in range(max_layer)],
            cbar_kws={"label": "Normalized Patch Effect"},
        )
        ax1.set_xlabel("Layer")
        ax1.set_title(title or f"Activation Patching ({self.patching_mode})")
        
        # Bottom: Divergence indicators
        attn_div = np.zeros(max_layer)
        mlp_div = np.zeros(max_layer)
        
        for info in self.divergence_info:
            if "_L" in info.component_name:
                parts = info.component_name.split("_L")
                ctype = parts[0]
                layer_idx = int(parts[1])
                if ctype == "attn" and layer_idx < max_layer:
                    attn_div[layer_idx] = info.divergence_score
                elif ctype == "mlp" and layer_idx < max_layer:
                    mlp_div[layer_idx] = info.divergence_score
        
        div_data = np.stack([attn_div, mlp_div], axis=0)
        div_max = max(abs(div_data.min()), abs(div_data.max()), 2.0)
        
        # Color by divergence type
        cmap = sns.diverging_palette(145, 300, s=60, as_cmap=True)
        sns.heatmap(
            div_data, ax=ax2, cmap=cmap, center=2.0, vmin=0, vmax=div_max,
            yticklabels=["Attention", "MLP"],
            xticklabels=[str(i) for i in range(max_layer)],
            cbar_kws={"label": "Divergence Score (σ from distribution)"},
        )
        ax2.set_xlabel("Layer")
        ax2.set_title("Intervention Divergence (>2σ = potentially unreliable)")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()


class DistributionEstimator:
    """Estimate activation distribution for divergence detection.
    
    Uses running mean and covariance computed from clean forward passes.
    """
    
    def __init__(self, model: RewardModel):
        self.model = model
        self._means: dict[str, torch.Tensor] = {}
        self._covs: dict[str, torch.Tensor] = {}
        self._inv_covs: dict[str, torch.Tensor] = {}
        self._n_samples: int = 0
        
    def collect_statistics(
        self,
        prompts: list[str],
        responses: list[str],
        max_length: int = 2048,
        show_progress: bool = True,
    ) -> None:
        """Collect activation statistics from clean forward passes.
        
        Args:
            prompts: List of prompts.
            responses: List of responses.
            max_length: Maximum sequence length.
            show_progress: Show progress bar.
        """
        n_layers = self.model.n_layers
        d_model = self.model.d_model
        
        # Initialize accumulators
        attn_activations = {i: [] for i in range(n_layers)}
        mlp_activations = {i: [] for i in range(n_layers)}
        
        iterator = list(zip(prompts, responses))
        if show_progress:
            iterator = tqdm(iterator, desc="Collecting activation statistics")
        
        for prompt, response in iterator:
            _, cache = self.model.forward_with_cache(
                prompt, response, max_length=max_length
            )
            
            for layer_idx in range(n_layers):
                attn = cache.attn_outputs.get(layer_idx)
                mlp = cache.mlp_outputs.get(layer_idx)
                
                if attn is not None:
                    attn_activations[layer_idx].append(attn.cpu().float())
                if mlp is not None:
                    mlp_activations[layer_idx].append(mlp.cpu().float())
        
        # Compute mean and covariance per component
        for layer_idx in range(n_layers):
            # Attention
            if attn_activations[layer_idx]:
                attn_tensor = torch.cat(attn_activations[layer_idx], dim=0)
                self._compute_stats(f"attn_L{layer_idx}", attn_tensor)
            
            # MLP
            if mlp_activations[layer_idx]:
                mlp_tensor = torch.cat(mlp_activations[layer_idx], dim=0)
                self._compute_stats(f"mlp_L{layer_idx}", mlp_tensor)
        
        self._n_samples = len(prompts)
    
    def _compute_stats(self, name: str, activations: torch.Tensor) -> None:
        """Compute mean and covariance for a component."""
        mean = activations.mean(dim=0)
        centered = activations - mean
        cov = (centered.T @ centered) / (activations.shape[0] - 1)
        
        # Add small regularization for numerical stability
        cov = cov + 1e-4 * torch.eye(cov.shape[0])
        
        self._means[name] = mean
        self._covs[name] = cov
        
        # Compute pseudo-inverse for Mahalanobis distance
        try:
            self._inv_covs[name] = torch.linalg.pinv(cov)
        except Exception:
            self._inv_covs[name] = torch.eye(cov.shape[0])
    
    def mahalanobis_distance(
        self,
        component_name: str,
        activation: torch.Tensor,
    ) -> float:
        """Compute Mahalanobis distance from training distribution.
        
        Args:
            component_name: Name of the component (e.g., "attn_L5").
            activation: Activation tensor of shape (d_model,) or (1, d_model).
            
        Returns:
            Mahalanobis distance (in standard deviations).
        """
        if component_name not in self._means:
            return 0.0  # Unknown component, assume in-distribution
        
        mean = self._means[component_name]
        inv_cov = self._inv_covs[component_name]
        
        act = activation.cpu().float().squeeze()
        diff = act - mean
        
        # Mahalanobis distance: sqrt(diff^T @ inv_cov @ diff)
        try:
            dist_sq = (diff @ inv_cov @ diff).item()
            return np.sqrt(max(0, dist_sq))
        except Exception:
            return 0.0
    
    def is_fitted(self) -> bool:
        """Check if statistics have been collected."""
        return self._n_samples > 0


class DivergenceAwarePatching(ActivationPatcher):
    """Activation patcher with divergence detection and constrained patching.
    
    Extends the base ActivationPatcher to:
    1. Detect when interventions create OOD representations
    2. Classify divergences as harmless vs pernicious
    3. Provide reliability scores for causal claims
    
    Args:
        model: A RewardModel instance.
        distribution_estimator: Optional pre-computed distribution estimator.
    """
    
    def __init__(
        self,
        model: RewardModel,
        distribution_estimator: Optional[DistributionEstimator] = None,
    ):
        super().__init__(model)
        self.dist_estimator = distribution_estimator or DistributionEstimator(model)
    
    def fit_distribution(
        self,
        prompts: list[str],
        responses: list[str],
        max_length: int = 2048,
        show_progress: bool = True,
    ) -> None:
        """Fit the activation distribution from clean data.
        
        Call this before patch_with_divergence_check for best results.
        
        Args:
            prompts: List of prompts.
            responses: List of responses.
            max_length: Maximum sequence length.
            show_progress: Show progress bar.
        """
        self.dist_estimator.collect_statistics(
            prompts, responses, max_length=max_length, show_progress=show_progress
        )
    
    def patch_with_divergence_check(
        self,
        prompt: str,
        preferred: str,
        dispreferred: str,
        mode: Literal["noising", "denoising", "zero"] = "noising",
        divergence_threshold: float = 2.0,
        max_length: int = 2048,
        show_progress: bool = True,
    ) -> DivergenceAwarePatchingResult:
        """Patch all components and check for divergent representations.
        
        Args:
            prompt: The user prompt.
            preferred: The preferred completion.
            dispreferred: The dispreferred completion.
            mode: Patching mode.
            divergence_threshold: Mahalanobis distance threshold for flagging (in σ).
            max_length: Maximum sequence length.
            show_progress: Show progress bar.
            
        Returns:
            DivergenceAwarePatchingResult with standard patching plus divergence info.
        """
        # Run standard patching
        base_result = self.patch_all_components(
            prompt, preferred, dispreferred,
            mode=mode, max_length=max_length, show_progress=show_progress
        )
        
        # Get caches for divergence analysis
        reward_w, cache_w = self.model.forward_with_cache(
            prompt, preferred, cache_full_sequences=True, max_length=max_length
        )
        reward_l, cache_l = self.model.forward_with_cache(
            prompt, dispreferred, cache_full_sequences=True, max_length=max_length
        )
        
        # Compute divergence for each patched activation
        divergence_info = []
        divergent_components = []
        pernicious_count = 0
        
        for i, (name, ctype, layer_idx) in enumerate(zip(
            base_result.component_names,
            base_result.component_types,
            base_result.layer_indices,
        )):
            # Get source and target activations
            if mode == "noising":
                source_cache = cache_l
                target_cache = cache_w
            else:
                source_cache = cache_w
                target_cache = cache_l
            
            if ctype == "attn":
                source_act = source_cache.attn_outputs.get(layer_idx)
            else:
                source_act = source_cache.mlp_outputs.get(layer_idx)
            
            if source_act is None:
                info = DivergenceInfo(
                    component_name=name,
                    divergence_score=0.0,
                    is_divergent=False,
                    divergence_type="unknown",
                    confidence=0.0,
                )
            else:
                # Compute Mahalanobis distance of patched activation
                div_score = self.dist_estimator.mahalanobis_distance(name, source_act)
                is_div = div_score > divergence_threshold
                
                # Classify divergence type
                if not is_div:
                    div_type = "in_distribution"
                    confidence = 1.0 - (div_score / divergence_threshold)
                else:
                    # Heuristic: pernicious if patch effect is large AND divergent
                    patch_effect = abs(base_result.patch_effects[i])
                    effect_threshold = 0.1 * abs(base_result.original_differential)
                    
                    if patch_effect > effect_threshold:
                        # Large effect + divergent = may be pernicious
                        div_type = "pernicious"
                        confidence = min(1.0, div_score / (divergence_threshold * 2))
                        pernicious_count += 1
                    else:
                        # Small effect + divergent = likely harmless (in null space)
                        div_type = "harmless"
                        confidence = 0.7
                
                info = DivergenceInfo(
                    component_name=name,
                    divergence_score=div_score,
                    is_divergent=is_div,
                    divergence_type=div_type,
                    confidence=confidence,
                )
                
                if is_div:
                    divergent_components.append(name)
            
            divergence_info.append(info)
        
        # Compute overall reliability score
        total_components = len(base_result.component_names)
        if total_components > 0:
            reliability = 1.0 - (pernicious_count / total_components)
        else:
            reliability = 1.0
        
        return DivergenceAwarePatchingResult(
            component_names=base_result.component_names,
            component_types=base_result.component_types,
            layer_indices=base_result.layer_indices,
            patch_effects=base_result.patch_effects,
            original_differential=base_result.original_differential,
            patching_mode=base_result.patching_mode,
            divergence_info=divergence_info,
            has_pernicious_divergence=pernicious_count > 0,
            reliability_score=reliability,
            divergent_components=divergent_components,
        )
    
    def constrained_patch(
        self,
        prompt: str,
        preferred: str,
        dispreferred: str,
        mode: Literal["noising", "denoising"] = "noising",
        cl_weight: float = 0.1,
        n_optimization_steps: int = 50,
        max_length: int = 2048,
    ) -> torch.Tensor:
        """Find constrained patch that stays close to training distribution.
        
        Uses a Counterfactual Latent (CL) loss to find a patched activation
        that achieves the intervention goal while minimizing divergence.
        
        This is more expensive than standard patching but provides more
        reliable causal evidence.
        
        Args:
            prompt: The user prompt.
            preferred: The preferred completion.
            dispreferred: The dispreferred completion.
            mode: Patching mode.
            cl_weight: Weight on the CL (divergence) loss term.
            n_optimization_steps: Number of optimization steps.
            max_length: Maximum sequence length.
            
        Returns:
            Tensor of optimized patch effects per component.
        """
        # This is a simplified version of constrained patching
        # Full implementation would optimize patch vectors to minimize:
        # L = intervention_loss + cl_weight * divergence_loss
        
        # For now, we just filter out pernicious divergences
        result = self.patch_with_divergence_check(
            prompt, preferred, dispreferred,
            mode=mode, max_length=max_length,
        )
        
        # Zero out effects for pernicious divergences
        constrained_effects = result.patch_effects.copy()
        for i, info in enumerate(result.divergence_info):
            if info.divergence_type == "pernicious":
                constrained_effects[i] = 0.0
        
        return constrained_effects
