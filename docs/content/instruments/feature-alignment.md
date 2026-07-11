<div class="rl-chips">
  <span class="rl-chip rl-chip--obs">Observational</span>
  <span class="rl-chip rl-chip--gauge"><span class="rl-chip__k">gauge</span> raw only</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">works on</span> an SAE + linear readout</span>
</div>

# Feature-reward alignment

**Of the interpretable features an SAE finds, which ones is the reward actually paying for?**

A sparse autoencoder decomposes the residual stream into features, directions that tend to be more interpretable than raw neurons. Once you have those features, a sharp question follows: which of them drive the reward? Because the reward is linear in the residual stream, the answer is a single dot product per feature. Each feature's decoder direction, dotted with the reward direction, is exactly how much activating that feature moves the score. Rank the features by that number and you have the reward's opinion expressed in interpretable units.

This instrument needs an SAE, so it is the one battery member with an external dependency. Without a trained SAE it will still run, but it will tell you the result is not real.

## The intuition

An SAE reconstructs an activation as a sparse sum of feature directions, \( h \approx b + \sum_i f_i\, d_i \), where \(f_i\) is how strongly feature \(i\) fired and \(d_i\) is its decoder direction. Push that through the reward readout and the reward becomes a sum over features:

\[ r \approx b' + \sum_i f_i\,\bigl(w_r^\top d_i\bigr). \]

The quantity in parentheses, \(w_r^\top d_i\), is feature \(i\)'s alignment with the reward. It does not depend on any particular input; it is a property of the feature and the reward direction together. A large positive alignment means that whenever this feature fires, it pushes the score up. A large negative one means it pushes the score down. Most features sit near zero, irrelevant to the reward.

## The math

Stack every feature's decoder direction into the decoder matrix \(W_{\text{dec}}\). The full alignment vector is one matrix-vector product,

\[ a = W_{\text{dec}}\, w_r, \]

one alignment per feature. The instrument returns the top and bottom of this vector, the features the reward most rewards and most penalizes. The gauge is *raw only* because \(a\) depends on two bases at once, the SAE's feature basis and the residual-stream basis, so a feature index and its alignment mean nothing outside the specific SAE they came from. You cannot line up "feature 38" on one SAE against "feature 38" on another.

## A worked run

Pass a trained SAE through the context under `regime["sae"]`. If you do not, the instrument substitutes a small randomly initialized SAE so the mechanics run, and it flags the result as untrained. The tiny run below takes the fallback, so its alignments are noise from random directions, and `trained_sae` says so.

```python
from reward_lens.signals import from_tiny
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.data.schema import DataView
from reward_lens.measure import base as mb
from reward_lens.measure.battery import FeatureRewardAlignment

signal = from_tiny(seed=0)
view = DataView(list(load_diagnostic_v3()["helpfulness"].items)[:5])
ev = mb.run(FeatureRewardAlignment(), mb.Context(signal=signal, view=view))

print(ev.trust, ev.gauge)                                      # EXPLORATORY raw_only
print("trained SAE?", ev.value["trained_sae"], "| features:", ev.value["n_features"])
print("top feature (index, alignment):", ev.value["top_features"][0])
print("max/min alignment:", round(ev.value["max_alignment"], 4), round(ev.value["min_alignment"], 4))
```

```text
EXPLORATORY raw_only
trained SAE? False | features: 128
top feature (index, alignment): (62, 0.044507868587970734)
max/min alignment: 0.0445 -0.0466
```

`trained_sae` is `False`, so read nothing into "feature 62." That flag is the whole point of the fallback: the mechanics are exercised, the arithmetic is correct, and the honesty layer refuses to let a random SAE masquerade as a finding. To get a real ranking you supply a trained SAE:

```python
from reward_lens.sae import TopKSAE            # needs the [sae] extra: pip install reward-lens[sae]

sae = TopKSAE.load("path/to/trained_sae.pt")  # trained on this model's activations
ev = mb.run(
    FeatureRewardAlignment(),
    mb.Context(signal=signal, view=view, regime={"sae": sae}),
)
ev.value["trained_sae"]                        # True, now the ranking means something
```

Training the SAE is the expensive part: it needs a corpus of the model's activations and, in practice, a GPU. The alignment computation on top of a trained SAE is cheap.

## How to read the output

- **`trained_sae`** first, always. `False` means the rest of the dictionary is scaffolding, not a result.
- **`top_features`** and **`bottom_features`** are lists of `[feature_index, alignment]`, the features the reward most rewards and most penalizes. With a trained SAE these indices point at features you can name by inspecting what activates them.
- **`max_alignment`**, **`min_alignment`**, and **`mean_abs_alignment`** summarize the spread: a reward concentrated on a few strong features looks different from one spread thinly across many.

## When not to reach for it

The untrained fallback is not a result, and the instrument tells you so; do not report alignments with `trained_sae=False`. Even with a trained SAE, the alignments are *raw only*: feature indices are specific to that SAE, so you cannot compare feature 38 across models or across two SAEs, and a cross-model feature comparison needs the features expressed in a shared [frame](../discipline/gauge-and-frames.md) first.

It is also observational. A feature aligned with the reward is *correlated* with the score, not proven to cause it; the SAE may have split one real feature across several, or bundled several into one. If a feature looks important, confirm it by pushing the model along that direction and watching the reward with [concept dose-response](concept-dose-response.md).

## How much to trust this

The number arrives at trust level **EXPLORATORY**, and with the untrained fallback it is below even that in spirit, a mechanics check rather than a measurement. With a trained SAE the arithmetic is exact, but no calibration provider is wired in this release, so no scorecard yet certifies that a high-alignment feature predicts behavior under optimization. Exploratory means unaudited, not wrong. Use the ranking to find candidate reward-relevant features to investigate; do not yet stake a claim on a single feature. The dependency install is the `[sae]` extra; the meaning of the trust level is [the trust ladder](../discipline/trust-ladder.md).

Full signatures and return fields: [`FeatureRewardAlignment`](../reference/measure.md#reward_lens.measure.battery.feature.FeatureRewardAlignment).
