# Representation tools

Two ways to break the reward direction into interpretable pieces: learn a sparse feature basis with an SAE, or probe for named concepts and measure how each aligns with \(w_r\). The SAE stack imports from `reward_lens.sae`; the concept tools are top-level.

A top-k sparse autoencoder over reward activations. `decompose_reward` splits a state's reward into per-feature contributions, and `feature_reward_alignments` scores every feature against \(w_r\). Import from `reward_lens.sae`.

::: reward_lens.sae.TopKSAE

Trains a `TopKSAE` on a matrix of collected activations. Import from `reward_lens.sae`.

::: reward_lens.sae.SAETrainer

Gathers the activation matrix the trainer needs: run a set of prompt-response pairs and keep one layer. Import from `reward_lens.sae`.

::: reward_lens.sae.ActivationCollector

Ranks a trained SAE's features by reward alignment and decomposes a single input across them. Import from `reward_lens.sae`.

::: reward_lens.sae.FeatureAnalyzer

Learns a direction for a named concept from contrast pairs, then measures its cosine with \(w_r\) and its causal effect when the direction is added back. Top-level import.

::: reward_lens.concepts.ConceptExtractor

What the concept analysis returns: per-concept alignment, which concepts are reward-aligned or anti-aligned, and an overall hacking-risk figure.

::: reward_lens.concepts.ConceptAlignmentReport

One call over the built-in concept set (confidence, formality, agreement, verbosity, hedging, helpfulness). Top-level import.

::: reward_lens.concepts.quick_concept_analysis
