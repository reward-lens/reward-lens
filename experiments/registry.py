"""
Experiment registry. Each experiment registers a ``run`` callable taking
an :class:`ExperimentConfig` and returning a result dict.

This indirection lets the CLI runner dispatch by name without importing
every experiment up-front (slow on cold start, especially for the SAE
modules that pull torch).
"""

from __future__ import annotations

import importlib
from typing import Callable

from .config import ExperimentConfig

_REGISTRY: dict[str, str] = {
    # name -> "module:function" lazy reference
    "e01_baseline_and_diagnostics": "experiments.e01_baseline_and_diagnostics.run:run",
    "e02_lens_population":          "experiments.e02_lens_population.run:run",
    "e03_attribution_population":   "experiments.e03_attribution_population.run:run",
    "e04_faithfulness_population":  "experiments.e04_faithfulness_population.run:run",
    "e05_circuit_overlap":          "experiments.e05_circuit_overlap.run:run",
    "e06_hacking_at_scale":         "experiments.e06_hacking_at_scale.run:run",
    "e07_cascade_at_scale":         "experiments.e07_cascade_at_scale.run:run",
    "e08_concept_population":       "experiments.e08_concept_population.run:run",
    "e09_conflict_population":      "experiments.e09_conflict_population.run:run",
    "e10_distortion_index":         "experiments.e10_distortion_index.run:run",
    "e11_divergence_patching":      "experiments.e11_divergence_patching.run:run",
    "e12_sae_feature_decomposition":"experiments.e12_sae_feature_decomposition.run:run",
    "e13_scale_study":              "experiments.e13_scale_study.run:run",
    "e14_cross_architecture":       "experiments.e14_cross_architecture.run:run",
    "e15_head_path_patching":       "experiments.e15_head_path_patching.run:run",
    "e16_prompt_robustness":        "experiments.e16_prompt_robustness.run:run",
    "e17_reward_editing":           "experiments.e17_reward_editing.run:run",
    "e18_armorm_multi_objective":   "experiments.e18_armorm_multi_objective.run:run",
    "e19_finetune_delta":           "experiments.e19_finetune_delta.run:run",
    "e20_arch_vs_finetune":         "experiments.e20_arch_vs_finetune.run:run",
}


def register(name: str, target: str) -> None:
    _REGISTRY[name] = target


def list_experiments() -> list[str]:
    return sorted(_REGISTRY.keys())


def resolve(name: str) -> Callable[[ExperimentConfig], dict]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown experiment: {name}. known: {list_experiments()}")
    target = _REGISTRY[name]
    module_path, func_name = target.split(":")
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)
