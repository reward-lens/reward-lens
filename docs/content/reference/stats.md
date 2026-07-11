# Stats

**How many independent data points do you actually have?** That question runs through this whole subsystem. `reward_lens.stats` is pure numpy and scipy, no torch, and it is where the library keeps its honesty about sample size, effect size, and what a correlation is worth.

## Effective sample size and clones

Thirty rollouts drawn from six seeds are not thirty data points. `effective_sample_size` computes the Kish effective size from seed labels, `detect_clones` finds the duplication by content hash, and the cluster routines resample at the seed level so a confidence interval widens to match the independence you really have. The worked example is [effective sample size of an eval set](../how-to/effective-sample-size.md).

::: reward_lens.stats.ess.effective_sample_size
    options:
      heading_level: 3

::: reward_lens.stats.ess.detect_clones
    options:
      heading_level: 3

::: reward_lens.stats.ess.cluster_bootstrap
    options:
      heading_level: 3

::: reward_lens.stats.ess.cluster_permutation
    options:
      heading_level: 3

## Effect sizes and correlation

Standardized effect sizes with bias-corrected bootstrap intervals, and a Spearman correlation that reports its own confidence interval rather than a bare number. These are the primitives the bias battery and the faithfulness measurements are built from.

::: reward_lens.stats.effects.cohens_d
    options:
      heading_level: 3

::: reward_lens.stats.effects.bca_bootstrap
    options:
      heading_level: 3

::: reward_lens.stats.effects.spearman_with_ci
    options:
      heading_level: 3

## ROC and calibration

A detector is only as good as its operating curve. `roc_pr` scores a detector against a known answer key; `calibration_curve` checks whether a probability means what it says. Both feed the [scorecards](../discipline/calibration-and-organisms.md) that turn an instrument into a calibrated one.

::: reward_lens.stats.roc.roc_pr
    options:
      heading_level: 3

::: reward_lens.stats.roc.calibration_curve
    options:
      heading_level: 3

## Mutual information

A nonparametric mutual-information estimate, in bits, for the couplings the reward science measures between a feature and the reward it drives.

::: reward_lens.stats.mi.mi_ksg
    options:
      heading_level: 3
