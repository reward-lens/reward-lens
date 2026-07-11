# Data

**What is the unit a reward model is measured on?** A preference. `reward_lens.data` is the torch-free schema for preferences and the views built on them: pairs, quadruples, tournaments, and agent trajectories, plus the built-in diagnostic set and the lineage that keeps sample sizes honest.

## The preference schema

A `DataView` is an ordered collection of items you can score in one pass. The items themselves are typed: a `Pair` is a chosen and a rejected response to one prompt, and the higher-order types capture richer preference structure. Why only the difference within a pair carries meaning is the subject of [preference geometry](../concepts/preference-geometry.md).

::: reward_lens.data.schema.DataView
    options:
      heading_level: 3

::: reward_lens.data.schema.Pair
    options:
      heading_level: 3

::: reward_lens.data.schema.make_pair
    options:
      heading_level: 3

A `Quadruple` carries two pairs sharing a prompt, a `Tournament` a set of pairwise outcomes over many responses, and a `Trajectory` an agent trace of typed steps.

::: reward_lens.data.schema.Quadruple
    options:
      heading_level: 3

::: reward_lens.data.schema.Tournament
    options:
      heading_level: 3

::: reward_lens.data.schema.Trajectory
    options:
      heading_level: 3

## The built-in diagnostic set

`load_diagnostic_v3` returns fourteen labelled dimensions of matched preference pairs (helpfulness, safety, verbosity, sycophancy, formatting, and the rest), ready to score with no download. It is the substrate under most of the how-to guides, including [detect length bias](../how-to/detect-length-bias.md).

::: reward_lens.data.builtin.diagnostic_v3.load_diagnostic_v3
    options:
      heading_level: 3

## Lineage and spans

`Lineage` records where each row came from so the [effective sample size](../how-to/effective-sample-size.md) is computed from real independence, not row count. `SpanMap` aligns character spans to token spans, which is how typed spans (a receipt, a step, a verdict) survive tokenization.

::: reward_lens.data.lineage.Lineage
    options:
      heading_level: 3

::: reward_lens.data.align.SpanMap
    options:
      heading_level: 3
