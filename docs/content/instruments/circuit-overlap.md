<div class="rl-chips">
  <span class="rl-chip rl-chip--obs">Observational</span>
  <span class="rl-chip rl-chip--gauge"><span class="rl-chip__k">gauge</span> invariant</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">works on</span> two models, activations + linear readout</span>
</div>

# Circuit overlap

**Do two reward models decide with the same components, or just reach the same verdict?**

Two models can agree on which answer is better while getting there through entirely different parts of the network. That distinction matters. If a fine-tune inherited its predecessor's circuit, an exploit that works on one is likely to work on the other; if it rebuilt the circuit from scratch, they may fail in different places. Circuit overlap measures it. It takes each model's top components by attribution, treats them as a set, and reports the Jaccard overlap: how much of the deciding machinery the two models share.

This is the one battery member that takes two signals at once. The second model rides in as the comparison subject.

## The intuition

Run [attribution](attribution.md) on each model and you get a ranked list of the components carrying the reward. Keep the top handful from each, the blocks that actually wrote the margin, and you have each model's "circuit" as a set of component labels. Now compare the sets. If they are the same blocks, the two models are doing the reward computation in the same place. If they barely overlap, they reached the same scores by different routes.

Jaccard is set overlap: the size of the intersection over the size of the union. It ignores ranking and it ignores magnitude, on purpose. The question here is membership, are these the same components, not whether they are weighted identically.

## The math

Let \(A\) and \(B\) be the top-\(k\) component sets of the two models, ranked by absolute attribution contribution. The overlap is

\[ J(A, B) = \frac{\lvert A \cap B\rvert}{\lvert A \cup B\rvert}, \]

which is \(1\) when the circuits are identical sets and \(0\) when they share nothing. The chip reads *invariant*, and here that has a specific consequence. A component's identity, `mlp_L31` or `attn_L12`, is a label about a position in the architecture, and it means the same thing in any basis. A Jaccard of label sets is therefore basis-free, so this cross-model comparison runs without a shared [frame](../discipline/gauge-and-frames.md). Contrast that with a raw cosine between two models' feature directions, which is *raw only* and would need a frame before it could cross models at all. Same goal, comparing two models, but the invariant quantity is the one you can compare directly.

## A worked run

The comparison takes the second model in `others` and sets `is_comparison=True`. On two tiny models the result is a caution in itself.

```python
from reward_lens.signals import from_tiny
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.data.schema import DataView
from reward_lens.measure import base as mb
from reward_lens.measure.battery import CircuitJaccard

model_a = from_tiny(seed=0)
model_b = from_tiny(seed=1)                                # a different tiny model
view = DataView(list(load_diagnostic_v3()["helpfulness"].items)[:5])
ev = mb.run(
    CircuitJaccard(),
    mb.Context(signal=model_a, view=view, others=(model_b,), is_comparison=True),
)

print(ev.trust, ev.gauge)                                 # EXPLORATORY invariant
print("jaccard:", ev.value["jaccard"])
print("circuit A:", ev.value["circuit_a"])
print("circuit B:", ev.value["circuit_b"])
```

```text
EXPLORATORY invariant
jaccard: 1.0
circuit A: ['attn_L0', 'attn_L1', 'mlp_L1', 'mlp_L0', 'embed']
circuit B: ['attn_L1', 'attn_L0', 'mlp_L1', 'mlp_L0', 'embed']
```

The Jaccard is \(1.0\), and that is the lesson, not a finding. A two-layer model has only five components, so the top five are all of them, and any two such models share the complete set by force. Notice the *ranking* differs (`attn_L0` leads for A, `attn_L1` for B) while the *set* is identical; Jaccard sees the sets, not the order. On a real 8B model with sixty-four or more components, the top-\(k\) is a genuine selection and an overlap below one carries information.

To compare two real reward models you need to load both.

!!! warning "Needs a GPU"
    Two 8B reward models in fp32 do not fit a laptop GPU. `load_signal` refuses a hub id unless you pass `allow_download=True`, and the forward passes are the gated part.

    ```python
    from reward_lens.signals import load_signal
    base = load_signal("Skywork/Skywork-Reward-Llama-3.1-8B", allow_download=True)
    v02  = load_signal("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", allow_download=True)
    ev = mb.run(CircuitJaccard(), mb.Context(signal=base, view=view, others=(v02,), is_comparison=True))
    ```

## How to read the output

- **`jaccard`** is the headline: \(1\) means identical top-\(k\) sets, \(0\) means disjoint. Read it against the number of components in the models; a high overlap on a small model is nearly guaranteed.
- **`circuit_a` and `circuit_b`** are the two ranked lists, so you can see not just how much they overlap but *which* components are shared and which are unique to one model.
- **`shared_components`** is the intersection, the machinery both models rely on, the first place to look if you want an exploit that transfers.

## When not to reach for it

Jaccard throws away ranking and magnitude by design, so two models with the same top-\(k\) set can still weight those components completely differently; if you care about *how much* each shared component matters, read the underlying attributions, not the overlap. It also assumes the two models are architecturally comparable, because it matches components by label; comparing models with different layer counts needs care about what `mlp_L20` means in each.

And it is observational twice over. Shared components are shared *labels*, not proven shared mechanism, and two models can route the same block to different computations. If the question is whether an intervention transfers, confirm it causally on both models rather than inferring it from overlap.

## How much to trust this

The number arrives at trust level **EXPLORATORY**. Jaccard is exact and the invariance is real, so this is one of the few cross-model numbers you can compare without a frame. But no calibration provider is wired in this release, so no scorecard yet relates an overlap to a transfer rate for exploits or interventions. Exploratory means unaudited, not wrong. Use it to see how much deciding machinery two models share; do not yet read a given overlap as a guarantee that a failure carries between them. The trust level and what would lift it are covered in [the trust ladder](../discipline/trust-ladder.md).

Full signatures and return fields: [`CircuitJaccard`](../reference/measure.md#reward_lens.measure.battery.circuit.CircuitJaccard). For the full cross-model workflow, including the covariant comparisons that do need a frame, see [compare two reward models](../how-to/compare-two-models.md).
