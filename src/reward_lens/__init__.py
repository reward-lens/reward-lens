"""reward-lens: the reference instrument for the science of reward misspecification.

Version 3 is one kernel, sixteen sciences, and three gates. The kernel is a set of subsystems
(``core``, ``stats``, ``runtime``, ``signals``, ``data``, ``concepts``, ``interventions``,
``geometry``, ``measure``, ``attribution``, ``organisms``, ``dynamics``, ``loops``, ``studies``,
``artifacts``); the sciences are studies over it; the gates (calibration, gauge, registration)
are runtime policy in the stats and evidence layer.

Import discipline: this top-level module is deliberately lazy. ``import reward_lens`` and
``import reward_lens.core`` and ``import reward_lens.stats`` pull nothing heavier than numpy, so
the pure epistemics layer is usable without torch. Anything that touches models is imported on
first access through the lazy accessor below, or, preferably, imported directly from its
subsystem (``from reward_lens.signals import load_signal``). The v1 public API is preserved under
``reward_lens.legacy`` and, for source compatibility, through the same lazy accessor, until the
E-parity suite passes twice (R15).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "3.0.0"

# Legacy v1 symbols, imported lazily so `import reward_lens` stays torch-free. Accessing any of
# these (for example `reward_lens.RewardModel`) imports the underlying module on demand; that
# import will require torch, which is the correct behaviour since the symbol needs it.
_LAZY: dict[str, str] = {
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


def __getattr__(name: str) -> Any:  # PEP 562 module-level lazy attribute access
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module 'reward_lens' has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


if TYPE_CHECKING:  # help static analysis without importing torch at runtime
    from reward_lens import core as core
    from reward_lens import stats as stats

__all__ = ["__version__", "core", "stats"]
