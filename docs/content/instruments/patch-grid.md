<div class="rl-chips">
  <span class="rl-chip rl-chip--causal">Causal</span>
  <span class="rl-chip rl-chip--gauge"><span class="rl-chip__k">gauge</span> invariant</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">works on</span> activations + linear readout</span>
</div>

# Patch grid

**Which components, if you reached in and changed them, would actually move the reward?**

Attribution tells you where the reward is *visible*. It cannot tell you what *causes* it, and on real reward models the two answers come apart. Patching settles the causal question directly: overwrite one component's activation, rerun the model, and watch what happens to the margin. If the margin collapses, that component was carrying the preference. If nothing moves, it was a bystander, no matter how much attribution credited it.

`PatchGrid` runs that experiment for every component at once, on every pair in a view, and returns the per-component causal effect.

## The effect it measures

Take a preference pair. The clean margin is the reward gap the model produces on its own,

\[
m = r_{\text{chosen}} - r_{\text{rejected}}.
\]

Now corrupt the chosen run at a single component \(c\): capture \(c\)'s activation on the rejected side and splice it into the chosen forward pass, leaving everything else untouched. Rerun and read the chosen reward again. The patch effect is how much of the margin that one substitution destroyed:

\[
\Delta_c = r_{\text{chosen}} - r_{\text{chosen}\,\mid\,c \leftarrow \text{rejected}}.
\]

A large \(\Delta_c\) means the chosen reward *depended* on \(c\): swap it for the rejected side's version and the model changes its mind. A near-zero \(\Delta_c\) means \(c\) was not load-bearing for this preference. The grid reports \(\Delta_c\) averaged over pairs, and its absolute value ranked, so the top of the list is the set of components the preference actually rests on. Because \(\Delta_c\) is a reward difference read along the fixed direction \(w_r\), it lives in the model's own units and needs no frame: the gauge is invariant.

Granularity is a switch. The default `"component"` patches each attention and MLP sublayer; `"head"` patches every individual attention head, which is the resolution you need to name a single responsible head.

## A run you can reproduce on CPU

The tiny signal has two layers and four heads, so its grid is small and the whole thing runs offline in seconds. The numbers are mechanics, not findings, the eight-billion-parameter result is below.

```python
from reward_lens.signals import from_tiny
from reward_lens.data.schema import DataView
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.measure import base as mb
from reward_lens.measure.battery import PatchGrid

signal = from_tiny(seed=0)
view = DataView(list(load_diagnostic_v3()["helpfulness"].items)[:4])

ev = mb.run(PatchGrid(), mb.Context(signal=signal, view=view))
print(ev.value["component_names"])
print([(n, round(v, 4)) for n, v in ev.value["top_components"][:4]])
print(ev.value["top_component"], round(ev.value["max_abs_effect"], 4))
print(str(ev.trust), str(ev.gauge))
# ['attn_L0', 'mlp_L0', 'attn_L1', 'mlp_L1']
# [('attn_L0', 0.0113), ('attn_L1', 0.0074), ('mlp_L1', 0.0049), ('mlp_L0', 0.0014)]
# attn_L0 0.0113
# EXPLORATORY invariant
```

Switch to head granularity and the same call resolves every head instead of every sublayer:

```python
evh = mb.run(PatchGrid(granularity="head"), mb.Context(signal=signal, view=view))
print(evh.value["granularity"], len(evh.value["component_names"]), evh.value["top_component"])
# head 8 head_L1_H2
```

Eight heads, one named as the strongest mover. On a two-layer random-initialized model that name means nothing. On a trained 8B model it means a great deal.

## What the causal picture looks like on Skywork

![Per-component causal patch effect on Skywork: the tall bars sit in the early layers, the mirror image of where attribution puts its mass.](../assets/figures/patching-bars-light.svg#only-light){ .rl-fig .rl-fig--hero }
![Per-component causal patch effect on Skywork: the tall bars sit in the early layers, the mirror image of where attribution puts its mass.](../assets/figures/patching-bars-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**The reward is caused early and only becomes visible late.** Each bar is one component's patch effect on the sky-is-blue pair. The causal weight concentrates in the early layers, exactly where [component attribution](attribution.md) credits almost nothing. This is the anti-correlation, drawn as a bar chart instead of a scatter.
///

Read the bars from the left. The components that, when patched, actually swing the margin are near the *start* of the network. That is the opposite end from the last MLPs, where attribution puts all of its mass because that is where the margin is largest and most visible. Break an early layer and the whole downstream chain it feeds is wrong; break the final MLP and the model mostly recovers, because the late layers are reading off a computation that already happened.

At head granularity the same story sharpens to a single component. Across the diagnostic battery the strongest causal head on Skywork-v0.2 is `head_L12_H6`, with a patch effect of 8.47 on the safety dimension. For helpfulness the top head is `head_L0_H29`, in the very first layer. Attribution's honours go late; causation's go early. Those two rankings are the [attribution-versus-patching](../concepts/observational-vs-causal.md) result that the whole library is organized around not confusing.

!!! warning "Needs a GPU"
    Reproducing the head-level leaderboard on an 8B reward model is GPU-gated. The head grid needs the model's attention weights in fp32, and a full-size 8B head sweep does not fit an 8 GB card, so the numbers above are read from committed artifacts, not produced here. The call is honest about it: at head granularity on a hub model the observable dispatches through a capability check and refuses rather than fabricate.

    ```python
    # GPU-gated: shown, not run in these docs.
    from reward_lens.signals import load_signal
    signal = load_signal("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", allow_download=True)
    ev = mb.run(PatchGrid(granularity="head"), mb.Context(signal=signal, view=view))
    ev.value["top_component"]   # head_L12_H6 on the safety dimension
    ```

## When not to reach for it

Patching splices activations across runs, so it manufactures states the model would never produce on its own: the chosen prompt carrying the rejected side's layer-12 output is off the model's natural distribution, and a large \(\Delta_c\) on such a state can be an artifact of that mismatch rather than a clean causal fact. Treat the ranking as a strong hypothesis, not a proof, and confirm the components that matter with a narrower, on-distribution intervention.

It is also more expensive than reading. Attribution is one or two forward passes for a whole model; the grid is on the order of two passes *per component*, and head granularity multiplies that by the head count. Use [crystallization depth](lens-crystallization.md) to find the layers worth patching before you patch every one of them, and see [patch without running out of memory](../how-to/patching-memory.md) for the batching that keeps an 8B sweep inside a small card.

## How much to trust it

The grid returns [`Evidence`](../reference/core.md#reward_lens.core.evidence.Evidence) at `EXPLORATORY`, because no calibration scorecard is wired for it yet: it is a sharp, honest measurement of a causal effect, not a validated detector with a known false-positive rate. The gauge is `INVARIANT`, so the effects are comparable across components of the same model without a frame. What you may say from it is precise and bounded: "patching this component moves the margin by this much on these pairs." What you may not say is that the effect generalizes off these pairs, or that a component attribution ranked highly is therefore causal. That last substitution is the single most common overclaim in reward-model interpretability, and it is exactly the one this instrument exists to stop.

Its two-hop companion, which asks not *whether* a head matters but *through which path*, is [path effects](path-effects.md).
