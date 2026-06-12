# API reference

There are two import surfaces, and knowing which one an object is on removes the most common error. The names you reach for most often are exported at the top level, so `from reward_lens import RewardModel, RewardLens, ComponentAttribution` works directly. The rest stay in their own submodules: the result dataclasses, the SAE stack, the hacking detector, the diagnostic pairs, the comparator, and the adapters all import from `reward_lens.<module>`. The table below gives the exact line for every object, and a failed import is almost always a top-level name that actually lives in a submodule, or the reverse.

## Every object and where it imports from

| Object | Import |
| --- | --- |
| `RewardModel` | `from reward_lens import RewardModel` |
| `ActivationCache` | `from reward_lens import ActivationCache` |
| `BatchedActivationCache` | `from reward_lens import BatchedActivationCache` |
| `RewardLens` | `from reward_lens import RewardLens` |
| `RewardLensResult` | `from reward_lens.lens import RewardLensResult` |
| `reward_lens_plot` | `from reward_lens import reward_lens_plot` |
| `ComponentAttribution` | `from reward_lens import ComponentAttribution` |
| `ComponentResult` | `from reward_lens.attribution import ComponentResult` |
| `ActivationPatcher` | `from reward_lens import ActivationPatcher` |
| `PatchingResult` | `from reward_lens.patching import PatchingResult` |
| `PathPatcher` | `from reward_lens import PathPatcher` |
| `PathPatchResult` | `from reward_lens import PathPatchResult` |
| `DivergenceAwarePatching` | `from reward_lens import DivergenceAwarePatching` |
| `DivergenceAwarePatchingResult` | `from reward_lens import DivergenceAwarePatchingResult` |
| `TopKSAE` | `from reward_lens.sae import TopKSAE` |
| `SAETrainer` | `from reward_lens.sae import SAETrainer` |
| `ActivationCollector` | `from reward_lens.sae import ActivationCollector` |
| `FeatureAnalyzer` | `from reward_lens.sae import FeatureAnalyzer` |
| `ConceptExtractor` | `from reward_lens import ConceptExtractor` |
| `ConceptAlignmentReport` | `from reward_lens import ConceptAlignmentReport` |
| `quick_concept_analysis` | `from reward_lens import quick_concept_analysis` |
| `HackingDetector` | `from reward_lens.hacking import HackingDetector` |
| `DistortionAnalyzer` | `from reward_lens import DistortionAnalyzer` |
| `MisalignmentCascadeDetector` | `from reward_lens import MisalignmentCascadeDetector` |
| `RewardConflictAnalyzer` | `from reward_lens import RewardConflictAnalyzer` |
| `quick_conflict_check` | `from reward_lens import quick_conflict_check` |
| `PreferencePair` | `from reward_lens.diagnostic_data import PreferencePair` |
| `get_diagnostic_pairs` | `from reward_lens.diagnostic_data import get_diagnostic_pairs` |
| `ModelComparator` | `from reward_lens.comparison import ModelComparator` |
| `ModelAdapter` | `from reward_lens.model_adapters import ModelAdapter` |
| `get_adapter` | `from reward_lens.model_adapters import get_adapter` |
| `statistics` | `from reward_lens import statistics` |

## Reference pages

<div class="grid cards" markdown>

-   :material-cube-outline:{ .lg } &nbsp; __[Core](core.md)__

    The model wrapper, the activation caches, the reward lens, and per-component attribution. Where every analysis starts.

-   :material-flask-outline:{ .lg } &nbsp; __[Causal tools](causal.md)__

    Activation patching, path patching, and divergence-aware patching. The tools that measure cause, not correlation.

-   :material-grain:{ .lg } &nbsp; __[Representation tools](representation.md)__

    Sparse autoencoders and concept vectors. Two ways to split the reward direction into interpretable pieces.

-   :material-shield-alert-outline:{ .lg } &nbsp; __[Vulnerability tools](vulnerability.md)__

    Hacking detector, distortion index, misalignment cascade, and reward-term conflict. What breaks under optimization.

-   :material-database-outline:{ .lg } &nbsp; __[Data and adapters](data-and-adapters.md)__

    The diagnostic preference pairs, the model comparator, the adapter layer, and the statistics helpers.

</div>

!!! warning "Head-level attribution is not available in 1.0.0"
    `ComponentAttribution.attribute_heads` is non-functional in this release: it calls an undefined helper and raises `NameError`. There is no observational head-level attribution. For per-head analysis, take the causal route with [`ActivationPatcher.patch_all_heads`](causal.md#reward_lens.patching.ActivationPatcher), which works and reports each head's effect on the margin.

!!! note "HackingDetector.scan() runs a fixed suite"
    `HackingDetector.scan()` runs its built-in probe set (length, confidence, formatting, sycophancy, repetition) and reports an effect size per axis. It accepts `prompt` and `response` arguments, but they are not used in 1.0.0; the suite is fixed. Pass `scan(tests=[...])` to run a subset, not `scan(prompt=..., response=...)`.
