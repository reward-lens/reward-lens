# Crystallization depth

**Where in the network does the model actually make up its mind?** Run a preference pair through and watch the margin at every layer. For most of the depth, nothing. The two responses sit on top of each other, both near zero, the model apparently undecided. Then, late, the margin snaps open. There is a specific depth where the preference goes from "not yet" to "mostly there," and it is worth naming, because it turns out to be a stable, measurable property of a reward model.

**Crystallization depth** is the layer where the running margin first reaches half of its final value. Half the decision, made. The [reward lens](../instruments/lens-crystallization.md) computes it for any pair.

![Two projections stay tangled and flat, then split late, with crystallization marked near the end.](../assets/figures/crystallization-schematic-light.svg#only-light){ .rl-fig .rl-fig--hero }
![Two projections stay tangled and flat, then split late, with crystallization marked near the end.](../assets/figures/crystallization-schematic-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**The idealized shape.** Each response's projection onto \(w_r\) runs near zero and tangled through the early and middle layers, then splits apart late: the chosen answer holds while the rejected one falls away. The dashed line marks where the margin between them reaches half its final size. The bars below are the per-layer contribution to the margin, almost all of it in the last few layers.
///

Read the top of the figure first. For two-thirds of the network the model is building representations but not committing, and the gap between the two answers is negligible. The commitment happens in the final stretch, when the rejected answer drops away and the chosen one stays put. Read the bars below and you see where the margin comes from: a handful of late layers, with the last MLPs doing most of the work. This is the shape you learn to expect on Skywork, and once you have seen it you spot it elsewhere.

## The canonical pair crystallizes at layer 30 of 32

Here is the measured reward-lens curve for a Skywork preference pair, not the schematic:

![The empirical Skywork margin curve: near zero for two-thirds of depth, then a steep late climb across the half-final-margin line at layer 30.](../assets/figures/lens-curve-light.svg#only-light){ .rl-fig }
![The empirical Skywork margin curve: near zero for two-thirds of depth, then a steep late climb across the half-final-margin line at layer 30.](../assets/figures/lens-curve-dark.svg#only-dark){ .rl-fig }

/// caption
**Near zero for two-thirds, then it forms.** The margin \(w_r^{\top}(h_{\text{chosen}} - h_{\text{rejected}})\) barely moves off zero through layer 20, wobbles, dips once around layer 27, then climbs hard. The dotted line is half the final margin; the curve crosses it at the marked layer, 30 of a 32-layer model.
///

The dotted line is the whole definition drawn on the plot. Track the margin up from zero, find where it first reaches half of where it ends, and read the layer. On Skywork that is layer 30, about \(0.94\) of the way through the network. The commitment is late and it is abrupt: most of the climb is packed into the last few layers, exactly the ones the schematic's bars pointed at.

You can run the instrument yourself on a model small enough to fit on a laptop, though a two-layer model has nowhere to crystallize:

```python
# CPU, no download: a real two-layer LlamaForSequenceClassification.
from reward_lens.signals import from_tiny
from reward_lens.measure import base as mb
from reward_lens.measure.battery import LensCrystallization
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.data.schema import DataView

signal = from_tiny(seed=0)
pairs  = DataView(list(load_diagnostic_v3()["helpfulness"].items)[:8])
ev = mb.run(LensCrystallization(), mb.Context(signal=signal, view=pairs))

print(round(float(ev.value["mean_crystal_frac"]), 3), ev.value["n_layers"])
# -> 0.143 2
print(ev.trust, ev.gauge)
# -> EXPLORATORY invariant
```

That \(0.143\) is not a finding; it is a two-layer model with no room to do anything interesting. The snippet is here to show that the instrument runs, returns a margin curve, and reports its own trust level. The layer-30 result needs the real 8B model:

!!! warning "Needs a GPU"
    ```python
    signal = load_signal("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", allow_download=True)
    ev = mb.run(LensCrystallization(), mb.Context(signal=signal, view=pairs))
    ev.value["mean_crystal_frac"]     # about 0.931 on Skywork-v0.2, i.e. layer ~30 of 32
    ```

## It is not one pair's quirk

A single pair is a point estimate, so the number to trust is the distribution. Across the diagnostic dimensions, Skywork crystallizes late and consistently, near \(0.9\) of depth on every axis measured, with a mean of \(0.931\):

![Mean crystallization depth by dimension for Skywork and ArmoRM, Skywork near 0.9 with tight error bars, ArmoRM lower and much noisier.](../assets/figures/crystallization-by-dim-light.svg#only-light){ .rl-fig }
![Mean crystallization depth by dimension for Skywork and ArmoRM, Skywork near 0.9 with tight error bars, ArmoRM lower and much noisier.](../assets/figures/crystallization-by-dim-dark.svg#only-dark){ .rl-fig }

/// caption
**Late and sharp is a property of a particular model.** Skywork (the taller bars) sits near \(0.9\) across helpfulness, safety, correctness, and verbosity, with small spread. ArmoRM decides earlier, in the \(0.6\) to \(0.8\) range, and its error bars are several times wider.
///

That contrast is a real finding, not a curiosity. Skywork holds its judgment until the representations are nearly complete and then reads them off, mean depth \(0.931\). ArmoRM, whose gated multi-objective head mixes nineteen readouts per input, commits earlier and with far more variance, mean depth \(0.803\). Crystallization depth is the measurement that lets you say that at all, and it has no clean analog in generative interpretability, where there is no single margin to track.

## What it is good for, and what it is not

Crystallization depth tells you *where to look*. If preference forms in the last three layers, that is where your attribution and your patching should concentrate, and a full-depth head sweep over the early layers is mostly wasted compute. It is a triage tool.

What it is not is a causal claim. "The margin reaches half its value at layer 30" is a statement about where the reward *appears*, an observational fact about the projection. It does not follow that layer 30 *causes* the preference. On this very pair, patching finds the early layers carry more of the causal weight than the late ones, even though the late layers are where the margin visibly forms. That tension runs through the whole library, and it is the single most important thing to understand before you trust any of these plots.

Next: the split between where the reward shows up and where it comes from. → [Observational vs causal](observational-vs-causal.md)
