"""
reward-lens: Mechanistic interpretability toolkit for reward models.

The first comprehensive library for understanding what happens inside
the models that define the RLHF training signal.

Core modules:
    - model: Reward model wrapper with activation hooks
    - lens: Reward Lens — layer-by-layer preference formation
    - attribution: Per-component reward decomposition
    - patching: Activation patching for causal analysis
    - hacking: Reward hacking vulnerability detection
    - sae: Sparse autoencoder training and feature attribution
    - diagnostic_data: Curated preference pairs for controlled experiments

New modules (v1.0.0):
    - distortion: Predictive reward hacking analysis (distortion index)
    - divergence_patching: Divergence-aware activation patching
    - cascade: Misalignment cascade detection
    - conflict: Reward term conflict analysis
    - concepts: Concept vector extraction and analysis
"""

from reward_lens import statistics  # noqa: F401 — re-export module
from reward_lens.attribution import ComponentAttribution
from reward_lens.cascade import CascadeReport, MisalignmentCascadeDetector
from reward_lens.concepts import ConceptAlignmentReport, ConceptExtractor, quick_concept_analysis
from reward_lens.conflict import ConflictReport, RewardConflictAnalyzer, quick_conflict_check

# New modules (v1.0.0)
from reward_lens.distortion import DistortionAnalyzer, DistortionReport
from reward_lens.divergence_patching import DivergenceAwarePatching, DivergenceAwarePatchingResult
from reward_lens.lens import RewardLens, reward_lens_plot
from reward_lens.model import ActivationCache, BatchedActivationCache, RewardModel
from reward_lens.patching import ActivationPatcher
from reward_lens.path_patching import PathPatcher, PathPatchResult

__version__ = "1.0.0"

__all__ = [
    # Core (v1.0.0)
    "RewardModel",
    "ActivationCache",
    "BatchedActivationCache",
    "RewardLens",
    "reward_lens_plot",
    "ComponentAttribution",
    "ActivationPatcher",
    "PathPatcher",
    "PathPatchResult",
    # New modules (v1.0.0)
    "DistortionAnalyzer",
    "DistortionReport",
    "DivergenceAwarePatching",
    "DivergenceAwarePatchingResult",
    "MisalignmentCascadeDetector",
    "CascadeReport",
    "RewardConflictAnalyzer",
    "ConflictReport",
    "quick_conflict_check",
    "ConceptExtractor",
    "ConceptAlignmentReport",
    "quick_concept_analysis",
]
