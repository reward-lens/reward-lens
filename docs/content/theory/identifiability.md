# Identifiability and gauge

**How much of a reward model is actually pinned down by its training?** Less than the numbers suggest. The reward is identified only up to a family of transformations that leave every preference unchanged, and those transformations are exactly what make raw coordinates lie. Knowing the size of that freedom is the difference between a cross-model comparison that means something and one that is comparing bookkeeping.

## More freedom than a constant

The [Bradley-Terry](bradley-terry.md) fit already gives away one degree of freedom: add a constant to every reward and no preference moves, so the absolute level is not identified. That much is familiar. The freedom is larger, though. Any transformation that preserves the ordering of every pair leaves the likelihood alone, and that includes a positive rescaling, a per-prompt shift, and, most consequentially, any change confined to directions of the activation space that carry no on-distribution variance. A direction the reference distribution never moves along cannot change any score you would actually compute, so the reward's coordinate there is free. Two reward heads can differ enormously in those free directions and induce identical preferences on everything you care about.

This is the reward-modeling instance of a general fact about reward learning: a reward function is identifiable only up to transformations that preserve its optimal policies and orderings, a partial identifiability that recent work has characterized precisely (arXiv [2411.15951](https://arxiv.org/abs/2411.15951)). The consequence for measurement is immediate. A quantity computed in raw coordinates, a cosine between two reward directions, an angle, a subspace overlap, can be dominated by the free part and tell you nothing about the part that matters.

## What the raw number hides

![Two reward directions, near-orthogonal in raw coordinates, aligned once a frame is fixed](../assets/figures/gauge-picture-light.svg#only-light){ .rl-fig .rl-fig--hero }
![Two reward directions, near-orthogonal in raw coordinates, aligned once a frame is fixed](../assets/figures/gauge-picture-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**Raw cross-model numbers are coordinates, not facts.** The same two reward directions read as near-orthogonal in raw coordinates and as strongly aligned once the gauge is fixed by whitening to a shared frame. Only the second number is about the models.
///

The picture is not hypothetical. Take two versions of the same Skywork reward model and compute the raw cosine between their reward directions, and it comes out near \(0.005\) (fixture E19, held as raw-only, meaning a number the library refuses to compare until a frame is fixed). Read literally, that says two versions of one model, built on the same base and trained on overlapping data, disagree about what they reward as completely as two random vectors would. That reading is absurd, and the absurdity is the tell: almost all of the raw difference lives in the free directions, and the raw cosine is measuring the gauge, not the model.

## Fixing the gauge

The fix is to stop reading directions in raw coordinates and read them in a frame. A frame is the per-corpus whitening artifact that fixes the gauge: fit it on reference activations, and it records the mean, the covariance square roots, and the null subspace of directions with no on-distribution variance. Canonicalizing a reward direction through the frame projects out that null subspace and whitens what remains, so what is left is only the part the reference distribution can actually see. The angle between two canonicalized directions is the STARC-invariant quantity whose cosine is the on-distribution correlation of the two reward readouts (Skalse et al., "STARC: A General Framework for Quantifying Differences Between Reward Functions," arXiv [2309.15257](https://arxiv.org/abs/2309.15257)).

The mechanism is small enough to watch on synthetic data. Build a reference distribution with real variance along one axis and near-zero variance along another, then compare a direction against a copy of itself with a large component added in the near-null direction.

```python
import numpy as np
from reward_lens.geometry import fit_frame, effective_angle

rng = np.random.default_rng(0)
X = np.zeros((2000, 2), dtype=np.float32)
X[:, 0] = rng.standard_normal(2000)          # on-distribution content
X[:, 1] = 1e-3 * rng.standard_normal(2000)   # a direction with no on-distribution variance
frame = fit_frame(X)

w_a = np.array([1.0, 0.0], dtype=np.float32)   # the reward direction
w_b = np.array([1.0, 20.0], dtype=np.float32)  # same on-distribution, huge null component
ev = effective_angle(w_a, w_b, frame, n_boot=0)
print(round(ev.value.raw_cos, 4), round(ev.value.canonical_cos, 4))  # 0.0499 1.0
print(str(ev.gauge))                                                 # covariant
```

The two directions look nearly orthogonal in raw coordinates, cosine \(0.05\), and identical once the gauge is fixed, canonical cosine \(1.0\), because the whole raw difference lived in the direction the distribution never moves along. That is E19's story reproduced in miniature, and the same operation on the real Skywork pair with a shared frame is what turns a raw \(0.005\) into an honest measurement of how aligned the two versions actually are.

## Why the library raises

Notice the last line of the output: the quantity is `covariant`, meaning it only means something once a basis is fixed. That status is not decoration. A covariant quantity carries the gauge gate with it, so asking for a cross-model comparison without supplying a frame does not return a plausible number, it raises. `effective_angle` has no default frame for exactly this reason. The library would rather refuse than hand back a coordinate change dressed up as a finding, and the [`effective_angle`](../reference/geometry.md) signature makes the frame argument mandatory to enforce it. The full mechanics of frames, canonicalization, and the STARC regret bound are the subject of [gauge and frames](../discipline/gauge-and-frames.md), and the [compare two models](../how-to/compare-two-models.md) how-to walks the whole flow.

This is also a preregistered claim, not just a design choice. The [Gauge](../sciences.md) science registers identifiability up to shift and scale as theorem T6 and checks it by construction: apply a synthetic gauge transform that provably preserves preferences, then show the canonicalized cosine stays near one while the raw cosine collapses. The reward's freedom is real, it is larger than a constant, and a frame is what lets you see past it. Within a single model, where the gauge is shared, differences are already meaningful, which is why the lens and attribution instruments need no frame at all. It is only across models that the coordinates stop being facts.
