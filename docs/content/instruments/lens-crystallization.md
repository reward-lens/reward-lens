<div class="rl-chips">
  <span class="rl-chip rl-chip--obs">Observational</span>
  <span class="rl-chip rl-chip--gauge"><span class="rl-chip__k">gauge</span> invariant</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">works on</span> activations + linear readout</span>
</div>

# Reward lens and crystallization

**At which layer does the reward model actually decide the winner?**

Run a preference pair forward and watch the margin between the two responses at every layer. For most of the network there is almost nothing to see: the two completions project onto \(w_r\) at nearly the same height, both near zero, the model apparently undecided. Then, late, the margin snaps open. There is a specific depth where the preference goes from "not yet" to "mostly there," and it is worth naming, because it is a stable, measurable property of a reward model rather than a quirk of one pair.

The reward lens is the per-layer view. Crystallization depth is the single number that summarizes it: the layer where the running margin first reaches half of its final value. Half the decision, made.

![Two projections stay tangled and flat, then split apart late, with crystallization depth marked near the end.](../assets/figures/crystallization-schematic-light.svg#only-light){ .rl-fig .rl-fig--hero }
![Two projections stay tangled and flat, then split apart late, with crystallization depth marked near the end.](../assets/figures/crystallization-schematic-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**The shape you learn to expect.** Each completion's projection onto \(w_r\) runs together and near zero through the early and middle layers, then splits apart in the final stretch. The dashed line is where the gap between them reaches half its final size. Real models are messier, but this is the pattern.
///

## The intuition

The residual stream at the final token is a running sum: every attention block and every MLP writes into it, and the reward reads the total along \(w_r\). So you can stop the sum at any layer, read the partial reward, and watch the score assemble itself. The lens is nothing more than reading \(w_r^\top h_\ell\) at each intermediate hidden state \(h_\ell\), for the chosen response and the rejected one.

What you are watching is not the model "thinking" and then "deciding." It is the margin between two completions accumulating. For a lot of reward models that accumulation is heavily back-loaded: the representations are built up over most of the depth, and only near the end does the reward direction pick up the difference that separates a good answer from a bad one.

## The math

Write the layer-\(\ell\) hidden state at the scored position as \(h_\ell\). The lens value for a single response is its projection onto the reward direction, \(w_r^\top h_\ell\). For a pair, only the difference means anything, because Bradley-Terry preference is invariant to adding a constant to every reward. So the object to track is the running margin,

\[ m_\ell = w_r^\top\bigl(h_\ell^{\text{chosen}} - h_\ell^{\text{rejected}}\bigr). \]

At the final layer \(m_L\) is the actual reward margin. Crystallization depth is the first layer where the running margin crosses half of that final value,

\[ \ell^\star = \min\Bigl\{\, \ell : m_\ell \ge \tfrac{1}{2}\, m_L \,\Bigr\}, \]

reported as a fraction \(\ell^\star / L\) so it is comparable across models of different depth. That fraction is a pure ratio of projections, so it does not change if you rotate the basis of the residual stream. The chip says *invariant* for exactly this reason: layer 30 of 32 is layer 30 of 32 in any coordinates.

## A worked run

On the tiny model the machinery runs in seconds. A two-layer model has almost no depth to crystallize in, so treat this as a check that the instrument moves, not as a result.

```python
from reward_lens.signals import from_tiny
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.data.schema import DataView
from reward_lens.measure import base as mb
from reward_lens.measure.battery import LensCrystallization

signal = from_tiny(seed=0)
view = DataView(list(load_diagnostic_v3()["helpfulness"].items)[:5])
ev = mb.run(LensCrystallization(), mb.Context(signal=signal, view=view))

print(ev.trust, ev.gauge)                      # EXPLORATORY invariant
print("layers:", ev.value["n_layers"], "pairs:", ev.value["n_pairs"])
print("per-pair crystal layer:", ev.value["per_pair_crystal_layer"])
print("mean fraction:", round(ev.value["mean_crystal_frac"], 3))
```

```text
EXPLORATORY invariant
layers: 2 pairs: 5
per-pair crystal layer: [0, 0, 1, 0, 0]
mean fraction: 0.1
```

The value dictionary carries the full curve (`differential`, `layers`), the per-pair crystallization layer and fraction, and the final differential per pair, so you can redraw the lens yourself or aggregate across a dataset.

### Skywork crystallizes at layer 30 of 32

The result the instrument was built to find needs a real 8B model in fp32, which does not fit an 8 GB GPU. These are measured results from committed artifacts. On the running "why is the sky blue" pair, `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2` scores the chosen answer at \(-2.22\) and the rejected one at \(-26.25\), a margin of \(+24.03\). The lens shows that margin sitting near zero for thirty layers and then opening in the last two. Crystallization lands at **layer 30 of 32**, a depth fraction of \(0.931\), and it is not one pair's accident: averaged across RewardBench dimensions Skywork crystallizes at fraction \(0.931\), reliably late. ArmoRM, whose gated multi-objective head commits earlier, sits at \(0.803\) and with far more spread.

![The real Skywork lens curve: the margin stays flat and near zero for most of the depth, then opens hard in the last few layers.](../assets/figures/lens-curve-light.svg#only-light){ .rl-fig }
![The real Skywork lens curve: the margin stays flat and near zero for most of the depth, then opens hard in the last few layers.](../assets/figures/lens-curve-dark.svg#only-dark){ .rl-fig }

/// caption
**The empirical curve, not the schematic.** The running margin on Skywork holds near zero for the first thirty layers, then rises steeply to its final value of about \(24\) in the last stretch. The half-margin crossing, crystallization, is deep in the network at layer 30.
///

Reading the curve is the whole point. Flat-and-tangled for most of the depth means the model is building representations without yet committing. The steep climb at the end is where the reward direction finally picks up the difference. Once you have seen this shape on Skywork you will recognize it, or its absence, on other models.

To reproduce it yourself you need the hardware.

!!! warning "Needs a GPU"
    Loading an 8B reward model in fp32 needs a GPU with more memory than a laptop card. `load_signal` refuses a hub id unless you pass `allow_download=True`, and even then the forward pass is the gated part.

    ```python
    from reward_lens.signals import load_signal
    from reward_lens.measure import base as mb
    from reward_lens.measure.battery import LensCrystallization

    signal = load_signal("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", allow_download=True)
    ev = mb.run(LensCrystallization(), mb.Context(signal=signal, view=view))
    ev.value["mean_crystal_frac"]   # ~0.931 on Skywork-v0.2 (measured)
    ```

## How to read the output

- **`mean_crystal_frac`** is the headline: a fraction of total depth. Near \(1.0\) means the model decides at the very end; lower means it decides earlier. It is basis-invariant, so you can compare it across models directly.
- **`per_pair_crystal_layer`** and **`per_pair_crystal_frac`** are the same thing before averaging, one entry per pair. Wide spread across pairs is itself informative: a model that crystallizes at layer 5 on some pairs and layer 30 on others is doing something different from one that is reliably late.
- **`differential`** and **`layers`** are the raw lens curve, the per-layer margin, so you can plot it.

## When not to reach for it

Crystallization depth tells you *where* to look. It does not tell you *what* is happening there, and it does not tell you what *causes* the preference. "The margin reaches half its value at layer 30" is a statement about where the reward becomes visible in the projection, an observational fact. It does not follow that layer 30 causes the decision. On this exact pair, [patching](patch-grid.md) credits the *early* layers with more of the causal weight than the late ones, even though the late layers are where the margin visibly forms. That tension is the subject of [observational versus causal](../concepts/observational-vs-causal.md), and it is the single most important caveat to hold before you trust any depth plot.

Use the lens as triage. If the margin forms in the last three layers, that is where your [attribution](attribution.md) and your patching should concentrate, and a full-depth head sweep over the early layers is mostly wasted compute. Then confirm the mechanism with an instrument that actually intervenes.

## How much to trust this

The number arrives at trust level **EXPLORATORY**, and it will stay there until a scorecard exists. The projection itself is exact arithmetic, but "exact" is not "calibrated": no calibration provider is wired in this release, so nothing has yet certified that a crystallization fraction of \(0.93\) means on your model what it meant on the organisms where the instrument was validated. Exploratory means unaudited, not wrong. Read the fraction as a strong, reproducible descriptive statistic, compare it freely across models because the gauge is invariant, and do not yet stake a safety claim on it. The path from here to a calibrated number runs through [calibration and organisms](../discipline/calibration-and-organisms.md); the ladder it climbs is [the trust ladder](../discipline/trust-ladder.md).

Full signatures and return fields: [`LensCrystallization`](../reference/measure.md#reward_lens.measure.battery.lens.LensCrystallization). The concept in depth: [crystallization depth](../concepts/crystallization.md).
