# Data and adapters

The supporting cast: the curated preference pairs the experiments run on, the cross-model comparator, the adapter layer that teaches the library a new model family, and the statistics behind every reported effect size. Only `statistics` is top-level; the rest import from their own submodules.

One labelled example: a prompt, a preferred and a dispreferred response, and the quality dimension it probes. Import from `reward_lens.diagnostic_data`.

::: reward_lens.diagnostic_data.PreferencePair

Returns the curated v1 pairs, all of them or filtered by dimension. Import from `reward_lens.diagnostic_data`.

::: reward_lens.diagnostic_data.get_diagnostic_pairs

Runs the lens and attribution across several models on the same pair and lines up their crystallization depths and formation curves. Import from `reward_lens.comparison`.

::: reward_lens.comparison.ModelComparator

The abstract base every model family implements. Its central method hands back the reward head's weight and bias; subclass it to support a new architecture. Import from `reward_lens.model_adapters`.

::: reward_lens.model_adapters.ModelAdapter

Picks the right adapter for a loaded model through a hardcoded dispatch over known families. Import from `reward_lens.model_adapters`.

::: reward_lens.model_adapters.get_adapter

The permutation tests, confidence intervals, and effect-size math shared across the tools. Import the module with `from reward_lens import statistics`.

::: reward_lens.statistics
