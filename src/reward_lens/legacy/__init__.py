"""``reward_lens.legacy`` — the v1 public API, kept working through the migration (R15).

v1 compatibility is a shim, not a constraint. This package re-exports the v1 public names so code
written against reward-lens 1.0.0 keeps importing and running while v3 lands, and it is the one
sanctioned home for that surface. It stays working until the E-parity suite passes for two releases,
after which it is deprecated (R15). The names resolve to the original v1 modules, which remain in the
package during the migration; as each primitive is ported into its v3 home behind the new protocols,
its legacy entry will be repointed at a thin v3-backed adapter with no change to the caller.

Importing this package pulls torch, because the v1 API is model-facing; that is expected. The pure
v3 layers (`reward_lens.core`, `reward_lens.stats`) never import it.
"""

from __future__ import annotations

from typing import Any

# The v1 public surface, by the module that defines each name. These are the exact symbols
# reward-lens 1.0.0 exported from its top-level package.
_V1_API: dict[str, str] = {
    "RewardModel": "reward_lens.model",
    "ActivationCache": "reward_lens.model",
    "BatchedActivationCache": "reward_lens.model",
    "RewardLens": "reward_lens.lens",
    "reward_lens_plot": "reward_lens.lens",
    "ComponentAttribution": "reward_lens.attribution",
    "ActivationPatcher": "reward_lens.patching",
    "PathPatcher": "reward_lens.path_patching",
    "PathPatchResult": "reward_lens.path_patching",
    "DistortionAnalyzer": "reward_lens.distortion",
    "DistortionReport": "reward_lens.distortion",
    "DivergenceAwarePatching": "reward_lens.divergence_patching",
    "DivergenceAwarePatchingResult": "reward_lens.divergence_patching",
    "MisalignmentCascadeDetector": "reward_lens.cascade",
    "CascadeReport": "reward_lens.cascade",
    "RewardConflictAnalyzer": "reward_lens.conflict",
    "ConflictReport": "reward_lens.conflict",
    "quick_conflict_check": "reward_lens.conflict",
    "ConceptExtractor": "reward_lens.concepts",
    "ConceptAlignmentReport": "reward_lens.concepts",
    "quick_concept_analysis": "reward_lens.concepts",
}


def __getattr__(name: str) -> Any:  # PEP 562 lazy re-export
    module_path = _V1_API.get(name)
    if module_path is None:
        raise AttributeError(f"module 'reward_lens.legacy' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(_V1_API)


__all__ = list(_V1_API)
