<div class="rl-chips">
  <span class="rl-chip rl-chip--obs">Observational</span>
  <span class="rl-chip rl-chip--gauge"><span class="rl-chip__k">gauge</span> invariant</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">works on</span> activations + linear readout</span>
</div>

# Component attribution

**Which heads and MLPs actually wrote the score?**

The [reward lens](lens-crystallization.md) tells you which *layers* the margin forms in. This goes one level finer. It splits the final reward into a signed contribution from every component, every attention block and every MLP, so you can read off who pushed the score up and who pushed it down. The residual stream is a running sum of these writes and the reward is linear in that sum, so the split is exact. Nothing is left over, and nothing is estimated.

Why care, once you already have the lens? Because "layer 30 decides" is a coarse answer. If you want to know whether the MLPs or the attention are carrying the preference, or which single block wrote most of the margin, attribution hands you the itemized bill.

## The math

Write the final hidden state as the sum of what each component wrote into the residual stream, \( h = \sum_c o_c \). The reward reads out along one direction, so it distributes over that sum:

\[ r = w_r^{\top} h + b = \sum_c \bigl(w_r^{\top} o_c\bigr) + b. \]

Each term \( w_r^{\top} o_c \) is component \(c\)'s contribution to the score. For a preference pair the bias cancels and the margin becomes a clean sum of signed shares:

\[ \Delta = \sum_c w_r^{\top} \delta_c, \qquad \delta_c = o_c^{\text{chosen}} - o_c^{\text{rejected}}. \]

A positive \( w_r^{\top}\delta_c \) means component \(c\) moved the chosen answer ahead of the rejected one; a negative one means it worked against the eventual winner. Add every term and you recover the full margin, to the last decimal. Because it is a decomposition of one projection, rotating the basis does not change any component's share, which is why the chip reads *invariant*.

## A worked run

On the tiny model the contributions are real but tiny, because a two-layer model barely separates the pair. What matters here is the shape of the output, not the magnitudes.

```python
from reward_lens.signals import from_tiny
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.data.schema import DataView
from reward_lens.measure import base as mb
from reward_lens.measure.battery import DirectLinearAttribution

signal = from_tiny(seed=0)
view = DataView(list(load_diagnostic_v3()["helpfulness"].items)[:5])
ev = mb.run(DirectLinearAttribution(), mb.Context(signal=signal, view=view))

print(ev.trust, ev.gauge)                                  # EXPLORATORY invariant
print("components:", ev.value["component_names"])
print("dominant per pair:", ev.value["dominant_component"])
print("mean |contribution|:", round(ev.value["mean_abs_contribution"], 4))
```

```text
EXPLORATORY invariant
components: ['embed', 'attn_L0', 'mlp_L0', 'attn_L1', 'mlp_L1']
dominant per pair: ['attn_L0', 'attn_L0', 'attn_L1', 'attn_L0', 'attn_L0']
mean |contribution|: 0.0004
```

Note that `dominant_component` is a list, one entry per pair, not a single winner for the batch. That is deliberate: attribution is computed per pair, so the honest summary is "the dominant component was `attn_L0` on four of these five pairs and `attn_L1` on one," not a batch-level average that would blur real per-pair structure. The full dictionary also carries `contributions_chosen`, `contributions_rejected`, and `reward_differential` so you can rebuild the decomposition or sum it back to the margin.

### On Skywork it is the late MLPs

The 8B result is a measured one from committed artifacts. On the "why is the sky blue" pair, whose margin is \(+24.03\), the five largest contributions sit in the last four layers and four of the five are MLPs: `mlp_L31` at \(+3.99\), `mlp_L30` at \(+1.32\), `mlp_L29` at \(+0.86\), the late blocks piling onto the winner. `mlp_L31` alone accounts for about a sixth of the margin. This is the same last-few-layers story the lens told, now resolved to individual blocks.

![Top component contributions to the margin on the sky-is-blue pair: late MLPs dominate, mlp_L31 leads.](../assets/figures/attribution-bars-light.svg#only-light){ .rl-fig }
![Top component contributions to the margin on the sky-is-blue pair: late MLPs dominate, mlp_L31 leads.](../assets/figures/attribution-bars-dark.svg#only-dark){ .rl-fig }

/// caption
**The itemized bill.** Each bar is one component's signed contribution \( w_r^{\top}\delta_c \) to the margin, largest first, so a taller bar wrote more of the score. The bars are late (layers 28 to 31) and mostly MLPs. `mlp_L31` alone is about \(4\) of the \(24\)-point margin.
///

Read the bars three ways. Sign is direction: positive helped the chosen answer, negative fought it. Magnitude is share, in reward units, and the bars sum to the margin, so "`mlp_L31` wrote about a sixth of it" is literal. Type is mechanism: MLPs dominating means the preference is written by the position-wise computation rather than moved between positions by attention. Here it is MLPs, decisively.

## When not to reach for it, and this is not a soft warning

Do not read attribution as causal. It credits the components with the largest projection onto \(w_r\), which on Skywork are the last MLPs, because that is where the margin is largest to begin with. [Patching](patch-grid.md), which actually intervenes, credits the *early* layers, because that is where the computation the late layers merely report on gets done. Ranked against each other, the two anti-correlate. On `Skywork-Reward-Llama-3.1-8B-v0.2` the per-model mean Spearman correlation between attribution and patching is \( \rho = -0.171 \), and it goes as low as \( -0.441 \) on code correctness, \( -0.306 \) on helpfulness. On ArmoRM the same comparison sits near zero and slightly positive, \( \rho = +0.047 \). So a tall attribution bar is a hypothesis about importance, not a demonstration of it. The full accounting is in [observational versus causal](../concepts/observational-vs-causal.md).

One more concrete limit. The residual-stream decomposition is exact at the block level, one number per attention layer and one per MLP. Splitting an attention block into its individual heads is a different computation, and the honest per-head answer comes from the causal route: [the patch grid at head granularity](patch-grid.md) sweeps every head and measures each one's effect directly.

## How much to trust this

The number arrives at trust level **EXPLORATORY**. The decomposition is exact, but exact is not calibrated: no calibration provider is wired in this release, so no scorecard yet certifies that a large attribution share predicts anything about the model's behavior under optimization. Exploratory means unaudited, not wrong. Trust the arithmetic (the shares genuinely sum to the margin), compare shares across models freely because the gauge is invariant, and treat the ranking as a lead to confirm causally rather than a conclusion. What would move it up the ladder is a scorecard on organisms with known circuits; that machinery is [calibration and organisms](../discipline/calibration-and-organisms.md).

Full signatures and return fields: [`DirectLinearAttribution`](../reference/measure.md#reward_lens.measure.battery.dla.DirectLinearAttribution). To attribute a real score end to end, see [attribute a reward score](../how-to/attribute-a-score.md).
