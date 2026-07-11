# Concepts

**Does a named idea actually move the reward, and by how much per unit of it?** `reward_lens.concepts` builds a direction for a concept from labelled examples, measures how aligned that direction is with the reward, and measures the causal slope of reward against a dose of it. The narrated instrument is [concept dose-response](../instruments/concept-dose-response.md).

## Directions and alignment

`concept_direction` is the difference of means between positive and negative activations. `reward_alignment` is its cosine with the reward direction \(w_r\), reported `RAW_ONLY` because a bare cosine is a coordinate, not a fact. `dose_response_slope` is the causal complement: the OLS slope of reward against how far you push the activation along the concept.

::: reward_lens.concepts.vectors.concept_direction
    options:
      heading_level: 3

::: reward_lens.concepts.vectors.reward_alignment
    options:
      heading_level: 3

::: reward_lens.concepts.vectors.dose_response_slope
    options:
      heading_level: 3

## Calibrated probes

`fit_probe` trains a linear probe with seed-grouped cross-validation and binds a scorecard automatically. A probe that was never calibrated returns an uncalibrated direction, and anything built on it inherits `EXPLORATORY` trust, so an unvalidated concept cannot launder itself into a confident number downstream.

::: reward_lens.concepts.probes.fit_probe
    options:
      heading_level: 3

## Difference dictionaries

`train_diff_dict` learns a dictionary of difference vectors that decomposes a reward margin into named, human-readable parts.

::: reward_lens.concepts.diff_dict.train_diff_dict
    options:
      heading_level: 3

## The 1.0 concept tool

`ConceptExtractor` is the original 1.0 interface, kept for the workflows built on it. It extracts concepts, reports their reward alignment, and intervenes on them through the older API.

::: reward_lens.concepts.legacy.ConceptExtractor
    options:
      heading_level: 3
