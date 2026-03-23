"""
Activation Patching — Causal Intervention for Preference Circuits.

Activation patching is the gold standard for causal claims in mechanistic
interpretability. The idea: take a component's activation from one input (the
"source") and splice it into the forward pass of another input (the "target").
If the model's behavior changes, that component is causally necessary for the
behavioral difference.

For reward models, the natural patching setup is:
    - Target: preferred completion (high reward)
    - Source: dispreferred completion (low reward)
    - Metric: change in reward differential

If patching component c from dispreferred into preferred reduces the reward
differential, then c is causally important for the preference.

We implement three patching variants:
    1. Noising (preferred → dispreferred): patch from source=dispreferred into
       target=preferred. Measures how much each component is needed for high reward.
    2. Denoising (dispreferred → preferred): patch from source=preferred into
       target=dispreferred. Measures how much each component is sufficient for
       higher reward.
    3. Zero ablation: replace component output with zeros. Cruder but doesn't
       require a contrastive pair.

Implementation note: We use PyTorch forward hooks to intercept and replace
activations. This is simpler and more transparent than TransformerLens's
hook_fn system, at the cost of being slightly less flexible. For reward models,
this is the right tradeoff — the experiments are well-defined and we don't
need arbitrary hook composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from reward_lens.model import ActivationCache, RewardModel


@dataclass
class PatchingResult:
    """Result of activation patching across all components.

    Attributes:
        component_names: Names of patched components.
        component_types: Types ("attn", "mlp").
        layer_indices: Layer index per component.
        patch_effects: The effect of patching each component.
            Defined as: original_differential - patched_differential.
            Positive = this component matters for the preference.
        original_differential: The unpatched reward differential.
        patching_mode: Which patching mode was used.
    """

    component_names: list[str]
    component_types: list[str]
    layer_indices: list[int]
    patch_effects: np.ndarray
    original_differential: float
    patching_mode: str

    def top_k(self, k: int = 15) -> list[tuple[str, float]]:
        """Return top-k components by patch effect magnitude.

        Args:
            k: Number of components.

        Returns:
            List of (name, effect) tuples, sorted by |effect|.
        """
        indices = np.argsort(np.abs(self.patch_effects))[::-1][:k]
        return [(self.component_names[i], self.patch_effects[i]) for i in indices]

    def normalized_effects(self) -> np.ndarray:
        """Patch effects normalized by the original differential.

        Values close to 1.0 mean the component fully accounts for the preference.
        Values close to 0.0 mean it has no causal role.
        """
        if abs(self.original_differential) < 1e-8:
            return np.zeros_like(self.patch_effects)
        return self.patch_effects / self.original_differential

    def plot(
        self,
        save_path: Optional[str] = None,
        figsize: Optional[tuple[int, int]] = None,
        title: Optional[str] = None,
        normalized: bool = True,
    ) -> None:
        """Plot a heatmap of patch effects across layers and component types.

        Args:
            save_path: Optional path to save figure.
            figsize: Figure size.
            title: Custom title.
            normalized: If True, show normalized effects (proportion of differential).
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        max_layer = max(self.layer_indices) + 1

        effects = self.normalized_effects() if normalized else self.patch_effects

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

        if figsize is None:
            figsize = (max(12, max_layer * 0.3), 3.5)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        vmax = max(abs(data.min()), abs(data.max())) or 1.0
        sns.heatmap(
            data,
            ax=ax,
            cmap="YlOrRd",
            vmin=0,
            vmax=vmax,
            yticklabels=["Attention", "MLP"],
            xticklabels=[str(i) for i in range(max_layer)],
            cbar_kws={
                "label": "Norm. Patch Effect" if normalized else "Patch Effect"
            },
        )
        ax.set_xlabel("Layer")
        unit = "normalized" if normalized else "raw"
        ax.set_title(
            title or f"Activation Patching ({self.patching_mode}, {unit})"
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()

    def plot_top_k(
        self,
        k: int = 15,
        save_path: Optional[str] = None,
        figsize: tuple[int, int] = (10, 6),
        title: Optional[str] = None,
    ) -> None:
        """Bar chart of top-k components by patch effect.

        Args:
            k: Number of components.
            save_path: Optional save path.
            figsize: Figure size.
            title: Custom title.
        """
        import matplotlib.pyplot as plt

        top = self.top_k(k=k)
        names = [t[0] for t in reversed(top)]
        values = [t[1] for t in reversed(top)]
        colors = ["#FF9800" if v > 0 else "#9E9E9E" for v in values]

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.barh(range(len(names)), values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Patch Effect (Δ reward differential)")
        ax.set_title(title or f"Top {k} Causally Important Components")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()


class ActivationPatcher:
    """Causal intervention via activation patching on reward models.

    For a preference pair, this tool identifies which components are *causally
    necessary* for the reward model's preference, by swapping component activations
    between the preferred and dispreferred completions.

    Args:
        model: A RewardModel instance.
    """

    def __init__(self, model: RewardModel):
        self.model = model

    def patch_all_components(
        self,
        prompt: str,
        preferred: str,
        dispreferred: str,
        mode: Literal["noising", "denoising", "zero"] = "noising",
        max_length: int = 2048,
        show_progress: bool = True,
    ) -> PatchingResult:
        """Patch every attention and MLP component and measure the effect.

        Args:
            prompt: The user prompt.
            preferred: The preferred completion.
            dispreferred: The dispreferred completion.
            mode: Patching mode:
                "noising": Replace preferred activations with dispreferred ones.
                "denoising": Replace dispreferred activations with preferred ones.
                "zero": Zero-ablate each component in the preferred completion.
            max_length: Maximum sequence length.
            show_progress: Show progress bar.

        Returns:
            PatchingResult with effects for all components.
        """
        # First, get the caches for both completions
        reward_w, cache_w = self.model.forward_with_cache(
            prompt, preferred, cache_full_sequences=True, max_length=max_length
        )
        reward_l, cache_l = self.model.forward_with_cache(
            prompt, dispreferred, cache_full_sequences=True, max_length=max_length
        )

        original_diff = reward_w - reward_l

        # Prepare inputs for the target completion
        if mode == "noising":
            target_inputs = self.model.tokenize_conversation(prompt, preferred, max_length=max_length)
            source_cache = cache_l  # Patch FROM dispreferred
        elif mode == "denoising":
            target_inputs = self.model.tokenize_conversation(prompt, dispreferred, max_length=max_length)
            source_cache = cache_w  # Patch FROM preferred
        elif mode == "zero":
            target_inputs = self.model.tokenize_conversation(prompt, preferred, max_length=max_length)
            source_cache = None
        else:
            raise ValueError(f"Unknown patching mode: {mode}")

        component_names = []
        component_types = []
        layer_indices = []
        patch_effects = []

        layers = self.model.adapter.get_layers(self.model.model)
        n_layers = len(layers)

        iterator = range(n_layers)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Patching ({mode})")

        for layer_idx in iterator:
            layer = layers[layer_idx]

            # Patch attention
            attn_module = self.model.adapter.get_attn_module(layer)
            if attn_module is not None:
                effect = self._patch_component(
                    target_inputs=target_inputs,
                    module=attn_module,
                    source_activation=source_cache.raw_attn_outputs.get(layer_idx) if source_cache else None,
                    mode=mode,
                    original_reward_w=reward_w,
                    original_reward_l=reward_l,
                    original_diff=original_diff,
                )
                component_names.append(f"attn_L{layer_idx}")
                component_types.append("attn")
                layer_indices.append(layer_idx)
                patch_effects.append(effect)

            # Patch MLP
            mlp_module = self.model.adapter.get_mlp_module(layer)
            if mlp_module is not None:
                effect = self._patch_component(
                    target_inputs=target_inputs,
                    module=mlp_module,
                    source_activation=source_cache.raw_mlp_outputs.get(layer_idx) if source_cache else None,
                    mode=mode,
                    original_reward_w=reward_w,
                    original_reward_l=reward_l,
                    original_diff=original_diff,
                )
                component_names.append(f"mlp_L{layer_idx}")
                component_types.append("mlp")
                layer_indices.append(layer_idx)
                patch_effects.append(effect)

        return PatchingResult(
            component_names=component_names,
            component_types=component_types,
            layer_indices=layer_indices,
            patch_effects=np.array(patch_effects),
            original_differential=original_diff,
            patching_mode=mode,
        )

    def _patch_component(
        self,
        target_inputs: dict[str, torch.Tensor],
        module: nn.Module,
        source_activation: Optional[torch.Tensor],
        mode: str,
        original_reward_w: float,
        original_reward_l: float,
        original_diff: float,
    ) -> float:
        """Patch a single component and measure the effect on reward.

        Returns:
            The patch effect: original_diff - patched_diff (for noising)
            or patched_diff - original_diff (for denoising).
        """

        def hook_fn(module, input, output):
            """Replace the module's output with the source activation or zero."""
            hidden = self.model.adapter.extract_attn_output(output)

            if mode == "zero":
                # Zero ablation
                replacement = torch.zeros_like(hidden)
            else:
                # Activation patching
                if source_activation is None:
                    return output  # No source available, skip
                replacement = source_activation.to(hidden.device)
                # Handle sequence length mismatches by truncating or padding
                if replacement.shape[1] != hidden.shape[1]:
                    min_len = min(replacement.shape[1], hidden.shape[1])
                    new_hidden = hidden.clone()
                    new_hidden[:, :min_len, :] = replacement[:, :min_len, :]
                    replacement = new_hidden

            # Reconstruct the output format
            if isinstance(output, tuple):
                return (replacement,) + output[1:]
            return replacement

        handle = module.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                patched_output = self.model.model(**target_inputs)
            patched_reward = self.model.adapter.extract_reward(
                patched_output, target_inputs
            ).item()
        finally:
            handle.remove()

        if mode == "noising":
            # Target was preferred; we patched from dispreferred.
            # The patched model now gives patched_reward for the preferred text.
            patched_diff = patched_reward - original_reward_l
            return original_diff - patched_diff
        elif mode == "denoising":
            # Target was dispreferred; we patched from preferred.
            patched_diff = original_reward_w - patched_reward
            return original_diff - patched_diff
        elif mode == "zero":
            # Target was preferred; we zeroed a component.
            patched_diff = patched_reward - original_reward_l
            return original_diff - patched_diff
        return 0.0

    def patch_single_component(
        self,
        prompt: str,
        preferred: str,
        dispreferred: str,
        layer_idx: int,
        component_type: Literal["attn", "mlp"],
        mode: Literal["noising", "denoising", "zero"] = "noising",
        max_length: int = 2048,
    ) -> float:
        """Patch a single specific component and return the effect.

        Useful for targeted investigation of specific layers.

        Args:
            prompt: The user prompt.
            preferred: The preferred completion.
            dispreferred: The dispreferred completion.
            layer_idx: Layer index.
            component_type: "attn" or "mlp".
            mode: Patching mode.
            max_length: Maximum sequence length.

        Returns:
            The patch effect (scalar).
        """
        reward_w, cache_w = self.model.forward_with_cache(
            prompt, preferred, cache_full_sequences=True, max_length=max_length
        )
        reward_l, cache_l = self.model.forward_with_cache(
            prompt, dispreferred, cache_full_sequences=True, max_length=max_length
        )

        original_diff = reward_w - reward_l

        if mode == "noising":
            target_inputs = self.model.tokenize_conversation(prompt, preferred, max_length=max_length)
            source_cache = cache_l
        elif mode == "denoising":
            target_inputs = self.model.tokenize_conversation(prompt, dispreferred, max_length=max_length)
            source_cache = cache_w
        elif mode == "zero":
            target_inputs = self.model.tokenize_conversation(prompt, preferred, max_length=max_length)
            source_cache = None
        else:
            raise ValueError(f"Unknown mode: {mode}")

        layer = self.model.adapter.get_layers(self.model.model)[layer_idx]
        if component_type == "attn":
            module = self.model.adapter.get_attn_module(layer)
            source = source_cache.raw_attn_outputs.get(layer_idx) if source_cache else None
        else:
            module = self.model.adapter.get_mlp_module(layer)
            source = source_cache.raw_mlp_outputs.get(layer_idx) if source_cache else None

        if module is None:
            raise ValueError(f"No {component_type} module found at layer {layer_idx}")

        return self._patch_component(
            target_inputs=target_inputs,
            module=module,
            source_activation=source,
            mode=mode,
            original_reward_w=reward_w,
            original_reward_l=reward_l,
            original_diff=original_diff,
        )
