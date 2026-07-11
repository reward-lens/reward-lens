<div class="rl-chips">
  <span class="rl-chip rl-chip--causal">Causal</span>
  <span class="rl-chip rl-chip--gauge"><span class="rl-chip__k">gauge</span> invariant</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">works on</span> activations + linear readout</span>
</div>

# Path effects

**A head matters, but through which route does its influence reach the reward?**

[Patching](patch-grid.md) tells you a head is causal. It does not tell you *how* the head acts: whether its effect flows straight into a late layer that reads it off, or threads through a chain of intermediate computation first. `PathEffect` answers the narrower question. It isolates a single two-hop route, from one sender head to one receiver layer, and measures how much of the preference travels along exactly that path while everything else is held fixed.

## The path it isolates

Pick a sender head \(s\) and a receiver layer \(\ell\). The sender's contribution to the residual stream differs between the two sides of a pair, because the chosen and rejected inputs produce different head outputs. `PathEffect` takes that difference and splices *only it* into the receiver's residual input, leaving every other route into the receiver as it was on the clean chosen run. Then it reads the reward. The effect is the fraction of the margin that the sender-to-receiver path alone accounts for:

\[
\Delta_{s \to \ell} = \big(r_{\text{chosen}} - r_{\text{rejected}}\big) - \big(r_{\text{chosen}\,\mid\, s\to\ell\,\leftarrow\,\text{rejected}} - r_{\text{rejected}}\big).
\]

The construction is head-granular by design. Sublayer-level path patching smears too many routes together to mean anything, so the sender is always a single head and the receiver a single downstream layer. This is a faithful port of the path patcher the 1.0 library settled on, computing the sender head's residual contribution from its output-projection slice and adding the chosen-minus-rejected difference at the receiver's `resid_pre`.

## A run you can reproduce on CPU

The tiny signal has heads to send from and a last layer to receive at, so the two-hop measurement runs offline. By default the sender is the first head and the receiver is the final layer; `regime` overrides both.

```python
from reward_lens.signals import from_tiny
from reward_lens.data.schema import DataView
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.measure import base as mb
from reward_lens.measure.battery import PathEffect

signal = from_tiny(seed=0)
view = DataView(list(load_diagnostic_v3()["helpfulness"].items)[:4])

ev = mb.run(PathEffect(), mb.Context(signal=signal, view=view))
print(ev.value["sender"], ev.value["receiver_layer"])
print(round(ev.value["mean_path_effect"], 4), round(ev.value["max_abs_path_effect"], 4))
print(str(ev.trust), str(ev.gauge))
# [0, 0] 1
# 0.005 0.0123
# EXPLORATORY invariant
```

Point it at a different sender and receiver through the regime:

```python
ctx = mb.Context(signal=signal, view=view, regime={"sender": (0, 1), "receiver": 1})
ev2 = mb.run(PathEffect(), ctx)
print(round(ev2.value["mean_path_effect"], 4))
# 0.0018
```

Head `(0, 1)` routes a different amount of the preference into the last layer than head `(0, 0)` does. On the tiny model the difference is noise; on a trained model, sweeping every sender into a fixed receiver is how you build the two-hop circuit that carries a specific behavior.

!!! warning "Needs a GPU"
    The full sender-head effect leaderboard on an 8B reward model is GPU-gated, for the same reason patching is: it needs the model's attention output-projection weights in fp32 and does not fit an 8 GB card. `PathEffect` runs the two-hop splice for a chosen sender and receiver, and the population-scale sweep over every head is read from committed artifacts rather than produced on this hardware.

## When not to reach for it

A path effect is only interpretable once you already suspect a specific route. Sweeping every sender into every receiver is quadratic in heads and layers, and most of the cells are empty; the instrument earns its cost when [patching](patch-grid.md) has already named a handful of causal heads and you want to know where their influence lands. Like all patching it works off the model's natural distribution, so a route that looks strong when you splice a foreign contribution into a residual input may be weaker in the model's own dynamics. Confirm surprising paths, do not ship them from a single splice.

## How much to trust it

`PathEffect` returns [`Evidence`](../reference/core.md#reward_lens.core.evidence.Evidence) at `EXPLORATORY` with an `INVARIANT` gauge: a reward difference along the fixed direction \(w_r\), comparable across paths of the same model without a frame, but not calibrated against an answer key. No calibration provider is wired in this release, so the measurement defaults to `EXPLORATORY` and is exploratory by construction until a scorecard on organisms exists. The claim it licenses is exact and local, "this share of the margin travels the sender-to-receiver path on these pairs," and nothing wider. For the ranking of *which* heads matter before you trace their routes, start at the [patch grid](patch-grid.md); for why a causal ranking can disagree so sharply with an attribution ranking, [observational vs causal](../concepts/observational-vs-causal.md).
