# Causal tools

These intervene rather than read: change an activation, rerun the model, and measure how far the margin actually moves. Everything here imports from the top level except `PatchingResult`, which lives in `reward_lens.patching`.

The workhorse. `patch_all_components(...)` sweeps every component and reports its effect on the margin, with noising for necessity and denoising for sufficiency. `patch_all_heads(...)` gives the same at head granularity, which is the supported route to per-head causal effects.

::: reward_lens.patching.ActivationPatcher

What the patcher returns; import from `reward_lens.patching`. Holds `patch_effects`, `.top_k`, and the plotting helpers.

::: reward_lens.patching.PatchingResult

Isolates a single edge of the circuit: patch one sender head into a later receiver and measure only that path. The receiver layer must be later than the sender.

::: reward_lens.path_patching.PathPatcher

What `patch` returns: the path effect alongside the original and patched margins.

::: reward_lens.path_patching.PathPatchResult

A patcher that first checks whether the intervention pushed the activation off distribution. Fit a distribution, then patch with a divergence check, and it flags pernicious divergence so an off-distribution artifact is not read as a cause. Method from [Grant et al., *Addressing divergent representations from causal interventions on neural networks*](https://arxiv.org/abs/2511.04638).

::: reward_lens.divergence_patching.DivergenceAwarePatching

The patching result plus the divergence bookkeeping: a reliability score, per-component divergence info, and which components diverged.

::: reward_lens.divergence_patching.DivergenceAwarePatchingResult
