# Core

Load a reward model, hold its activations, and read the preference forming along the residual stream: everything else in the library is built on these five objects. All of them import from the top level except the two result dataclasses, which live in `reward_lens.lens` and `reward_lens.attribution`.

The wrapper around a Hugging Face reward model. `RewardModel.from_pretrained(name)` loads it, finds the reward-head weight \(w_r\), and exposes `.score`, `.score_pair`, and `.forward_with_cache`.

::: reward_lens.model.RewardModel

What `forward_with_cache` returns: the per-layer residual streams, attention outputs, and MLP outputs at the final token, indexed by layer.

::: reward_lens.model.ActivationCache

The same cache shape for a batch of sequences, used by the tools that score many pairs in one pass.

::: reward_lens.model.BatchedActivationCache

The reward lens. `RewardLens(rm).trace(prompt, preferred, dispreferred)` projects every layer onto \(w_r\) and returns the margin at each one.

::: reward_lens.lens.RewardLens

What `trace` returns; import from `reward_lens.lens` when you need the type. Carries the per-layer arrays and the crystallization layer.

::: reward_lens.lens.RewardLensResult

Trace and plot in a single top-level call, handing back the same result object.

::: reward_lens.lens.reward_lens_plot

!!! warning
    Head-level attribution is unavailable in 1.0.0. `attribute_heads` raises `NameError`; use [`ActivationPatcher.patch_all_heads`](causal.md#reward_lens.patching.ActivationPatcher) for head-level causal analysis.

A signed, per-component share of the margin. `ComponentAttribution(rm).attribute(...)` splits the final reward across embed, attention, and MLP outputs. Read it as where the reward is visible, not what causes it.

::: reward_lens.attribution.ComponentAttribution
    options:
      filters: ["!^_", "!attribute_heads"]

What `attribute` returns; import from `reward_lens.attribution`. Holds the per-component contributions and `.top_k`.

::: reward_lens.attribution.ComponentResult
