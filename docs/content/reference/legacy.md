# Legacy (1.0 API)

**Will code written against reward-lens 1.0 still run?** Yes. The 1.0 classes are preserved as a compatibility layer and resolved lazily, so importing the package costs nothing until you reach for one. They are reachable two ways: straight from the top level, `from reward_lens import RewardModel, RewardLens, ComponentAttribution`, and under the explicit `reward_lens.legacy` namespace. A few 1.0 tools were not folded into that namespace and live only in their own submodules: `from reward_lens.hacking import HackingDetector`, `from reward_lens.comparison import ModelComparator`, and `from reward_lens.sae import TopKSAE, SAETrainer`. The path from these to the 2.0 kernel is [the migration guide](../migration.md).

## The submodule-only tools

`HackingDetector` runs the 1.0 battery of bias tests, reporting a standardized effect size per axis for length, sycophancy, confidence, and the rest.

::: reward_lens.hacking.HackingDetector
    options:
      heading_level: 3

`ModelComparator` is the 1.0 cross-model comparison. It predates the frame machinery that 2.0 uses to make such comparisons gauge-safe, so read its numbers as raw coordinates.

::: reward_lens.comparison.ModelComparator
    options:
      heading_level: 3

`TopKSAE` is the top-k sparse autoencoder, with `SAETrainer` alongside it; both need the `[sae]` extra installed.

::: reward_lens.sae.TopKSAE
    options:
      heading_level: 3
