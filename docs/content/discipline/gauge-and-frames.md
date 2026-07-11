# Gauge and frames

Two reward models, both fine-tuned from the same base, both trained on similar preference data. You would expect their reward directions to point roughly the same way. Take the cosine between them and you get \(0.005\). Essentially orthogonal. Either the two models have almost nothing in common, or the number is lying.

The number is lying, and understanding why is the whole subject of this page. It is also the second gate, and the reason a raw cross-model comparison is one of the few things the library will flat-out refuse to compute.

## Why the raw number lies

A reward model is trained by Bradley-Terry, which only ever sees *differences* of rewards. Add a constant to every reward and no preference changes. Scale every reward by a positive number and no preference changes. So the reward head is not identified as a specific vector. It is identified only up to a shift and a scale, and in practice up to more than that, because the geometry of activation space is not isotropic. A direction is a set of coordinates, and the coordinates depend on a basis that training never pinned down.

Cosine similarity is computed in raw coordinates. When those coordinates are arbitrary, cosine measures the arbitrariness as much as the reward. Two directions that induce nearly the same preference ordering can sit at any raw angle at all, including ninety degrees. The \(0.005\) is not telling you the models disagree. It is telling you that you asked a question the raw coordinates cannot answer.

![Two reward directions, orthogonal in raw coordinates, aligned once a shared frame is fixed.](../assets/figures/gauge-picture-light.svg#only-light){ .rl-fig .rl-fig--wide }
![Two reward directions, orthogonal in raw coordinates, aligned once a shared frame is fixed.](../assets/figures/gauge-picture-dark.svg#only-dark){ .rl-fig .rl-fig--wide }

/// caption
**Raw cross-model numbers are coordinates, not facts.** On the left, two reward directions over an arbitrary basis, near-orthogonal, raw cosine \(0.005\). On the right, the same two directions expressed in a shared frame that whitens the geometry. The true, small angle between them appears. Nothing about the models changed. The basis did.
///

## What a frame is

A **frame** is the fix. It is a whitening transform, estimated on a reference corpus, that maps raw activation coordinates into a canonical basis where distances mean the same thing in every direction. In the library it is a `Frame`, fit with `fit_frame` on a corpus of activations, and it carries the mean, the symmetric square roots of the covariance, and the null directions the reward never uses.

Once you have a frame, you canonicalize each reward direction into it, and *now* the cosine is meaningful, because both directions live in the same, non-arbitrary geometry. That canonical cosine has a name and a theory: it is the STARC-invariant angle, and its cosine is the on-distribution correlation of the two reward readouts, the thing you actually wanted when you asked how similar the two models are.

```python
from reward_lens.geometry import fit_frame, effective_angle

frame = fit_frame(reference_activations, site=site, corpus="ref-corpus")   # one shared frame
ev = effective_angle(w_r_a, w_r_b, frame)     # canonical cosine, with a confidence interval
```

The machinery is validated on controlled cases where the answer is known: apply a gauge transform that a raw cosine reports as a change from 1.0 down to 0.37, and the canonical cosine stays pinned at 1.0, because canonicalization sees through the transform to the invariant relationship underneath. That is the property a frame has to have, and it is tested.

## The gate

Here is what makes this a gate and not just a utility. `effective_angle` has no default frame. You cannot call it without one. Pass `None` where the frame should be and the library raises `GaugeError` rather than fall back to a raw number.

```python
effective_angle(w_r_a, w_r_b, None)
# GaugeError: cross-model comparison of a covariant quantity requires a frame
```

Every measurement carries a [gauge status](anatomy-of-evidence.md). A quantity marked `covariant`, a direction or an angle or a subspace overlap, means something only relative to a frame. The runner reads that status, and if you try to compare a covariant quantity across models without fixing the frame, it stops. This is the one place the library would rather raise an exception than hand you a plausible number, because a plausible cross-model number is the most dangerous kind: it looks like a finding, and the \(0.005\) shows exactly how wrong it can be.

The E19 result that produced that \(0.005\) was, in the first version, an anomaly nobody could explain. It is not an anomaly. It is what a covariant quantity looks like when you forget to fix the gauge, and the gate exists so the next person cannot make the same mistake without the library objecting.

For the theory behind identifiability and the STARC construction, see [identifiability and gauge](../theory/identifiability.md). For comparing two real models step by step, the [how-to](../how-to/compare-two-models.md) does it end to end.
