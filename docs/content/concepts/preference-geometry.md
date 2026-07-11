# Preference geometry

**Why does every instrument in this library take two activations, never one?** Because a reward model was trained on pairs, and a pair is the only place its judgment is actually pinned down. One response scored on its own tells you almost nothing. Two responses, and the gap between them, tell you everything the model committed to.

Put a chosen response and a rejected response through the model and you get two final hidden states, \(h_{\text{chosen}}\) and \(h_{\text{rejected}}\). Two points in the same activation space. The model prefers chosen because chosen projects further along the [reward direction](reward-direction.md) \(w_r\), and the amount by which it does is the **margin**:

\[
\Delta = r_{\text{chosen}} - r_{\text{rejected}} = w_r^{\top}\bigl(h_{\text{chosen}} - h_{\text{rejected}}\bigr)
\]

Watch what fell out. The bias \(b\) sits in both scores, so it cancels. The margin depends only on the difference vector \(h_{\text{chosen}} - h_{\text{rejected}}\), projected onto \(w_r\). Everything the two responses share, the prompt, the format, the reward's arbitrary zero, is subtracted away. The pair carries its own baseline.

![Chosen and rejected drawn as two points, their difference vector, and its projection onto the reward direction.](../assets/figures/preference-geometry-light.svg#only-light){ .rl-fig .rl-fig--hero }
![Chosen and rejected drawn as two points, their difference vector, and its projection onto the reward direction.](../assets/figures/preference-geometry-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**One difference vector, one projection, one number.** The two black dots are the chosen and rejected activations. The orange segment is their difference. The margin \(\Delta\) is the shadow that difference casts on \(w_r\). Everything perpendicular to \(w_r\) is the model working but not the model deciding.
///

Read the picture as a single question: of all the ways chosen and rejected differ, how much of that difference points along the one direction the reward head reads? Slide either point sideways, along a line of constant reward, and \(\Delta\) does not move. Only motion along \(w_r\) changes the margin. The reward model has a great many internal opinions about these two responses, and exactly one of them survives the projection.

## The constant the model can never see

Reward models are fit under the Bradley-Terry model of pairwise preference. The probability a human prefers chosen over rejected is a sigmoid of the reward gap:

\[
P(\text{chosen} \succ \text{rejected}) = \sigma\bigl(r_{\text{chosen}} - r_{\text{rejected}}\bigr) = \sigma(\Delta)
\]

Now add the same constant \(c\) to every reward the model produces. Every gap \(r_{\text{chosen}} - r_{\text{rejected}}\) is unchanged, so every preference probability is unchanged, so the training loss is byte-for-byte identical. The absolute level of the reward is invisible to the objective. Training pins down differences and nothing else, which is why the margin is the quantity that means something and a lone score is not. That is a big enough idea to get [its own page](reward-is-relative.md).

## The pair is a controlled experiment you did not have to run

This is the quiet structural gift of studying reward models. In generative interpretability you usually have to build a contrast by hand: a clean run and a corrupted run, a prompt and a minimally edited counterfactual, and then you worry whether your edit moved one thing or ten. A preference pair *is* the contrast, and it is the exact contrast the model was trained on. Chosen and rejected typically share the prompt, share the format, share most of the content, and differ in the one dimension the label is about. The difference vector isolates that dimension by construction.

So every instrument in the library is pair-shaped. Hand a battery observable a view of pairs and it returns the margin, decomposed, as [a measurement that carries its own trust level](measurement-you-can-trust.md):

```python
# CPU, no download: from_tiny builds a real LlamaForSequenceClassification.
from reward_lens.signals import from_tiny
from reward_lens.measure import base as mb
from reward_lens.measure.battery import DirectLinearAttribution
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.data.schema import DataView

signal = from_tiny(seed=0)
pairs  = DataView(list(load_diagnostic_v3()["helpfulness"].items)[:8])
ev = mb.run(DirectLinearAttribution(), mb.Context(signal=signal, view=pairs))

print(ev.value["reward_differential"][:3])   # the margin, one per pair
# -> [0.0003187848487868905, -4.191667539998889e-05, 0.0001795587595552206]
print(ev.trust, ev.gauge)
# -> EXPLORATORY invariant
print(ev.value["component_names"][:4])
# -> ['embed', 'attn_L0', 'mlp_L0', 'attn_L1']
```

The tiny model's margins hover near zero because its weights are random, so read the shape of the answer, not the magnitudes. You get one margin per pair, a signed contribution per component, and a trust level attached. It comes back `EXPLORATORY`, the lowest rung, because this measurement has not yet been checked against a known answer. The gauge reads `invariant`, meaning the number does not change if you rotate the coordinates, which a margin never does.

## The margin decomposes along the same direction

Because the readout is linear and the residual stream is a running sum of what every component wrote, the difference vector splits into per-component pieces, \(h_{\text{chosen}} - h_{\text{rejected}} = \sum_c \delta_c\), and the margin splits with it:

\[
\Delta = \sum_c w_r^{\top}\delta_c
\]

Each term is one component's signed share of why chosen beat rejected: positive if that component pushed toward the better answer, negative if it pulled the other way. That decomposition is exactly what [Component attribution](../instruments/attribution.md) computes, and it is cleaner than the generative case, where you decompose a distribution rather than a scalar. It also sets up the sharpest warning in these docs. A component's share of the margin is an *observational* quantity. It does not have to match that component's *causal* importance, and on real models it often does not. Hold that thought; it gets [its own page](observational-vs-causal.md).

On the real 8B model the margins are large and legible. Skywork scores the good "why is the sky blue" answer at \(-2.22\) and the bad one at \(-26.25\), a margin of \(+24.03\). Reproducing that needs the 8B model in fp32:

!!! warning "Needs a GPU"
    ```python
    from reward_lens.signals import load_signal
    signal = load_signal("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", allow_download=True)
    ev = signal.score(pairs)                 # Evidence[Scores]; the margin is chosen minus rejected
    ```
    Without `allow_download=True` this call refuses rather than reach for the network. The \(+24.03\) above is measured from that model; the CPU snippet earlier is the one you can run in a minute.

Next: why the margin, and never a bare score, is the only thing worth plotting. → [Why reward is relative](reward-is-relative.md)
