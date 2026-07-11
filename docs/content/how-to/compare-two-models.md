# Compare two reward models

**Two reward models each reduce to a direction \(w_r\). How aligned are they?**

The obvious move is a cosine between the two direction vectors. On two versions of one Skywork model that cosine comes out around \(0.005\) (fixture E19, measured from the committed weights). Read literally, that says two reward models which agree on most preference pairs are very nearly orthogonal, which is absurd. The number is not wrong arithmetic. It is the wrong quantity: a raw cosine compares two vectors in two different coordinate systems, and each model is free to place its features on a different, arbitrarily scaled basis. So the raw cosine is a coordinate, not a fact about agreement.

A frame is the shared coordinate system you put both directions into before you compare them. It is fit once from reference activations and whitens away the per-model basis and scale, so that what is left is the part of the direction that actually moves the reward. The full treatment is in [gauge and frames](../discipline/gauge-and-frames.md); here is the recipe.

## Fit one frame, read the canonical angle

`effective_angle` takes both reward directions and one `Frame`, canonicalizes each inside it, and returns the gauge-fixed cosine with a bootstrap confidence interval and a STARC-style regret bound. The construction below is a synthetic stand-in that isolates the mechanism: two directions built to be orthogonal in raw coordinates, and a frame fit from the activations that reveals how they actually relate. The numbers are illustrative, produced by this exact run on CPU.

```python
import numpy as np
from reward_lens.geometry import fit_frame, effective_angle

rng = np.random.default_rng(0)
d = 16
A = rng.standard_normal((d, d)); cov_sqrt = A @ A.T / d
h = (rng.standard_normal((400, d)) @ cov_sqrt).astype(np.float32)   # reference activations
w_a = rng.standard_normal(d).astype(np.float32)
w_b = rng.standard_normal(d).astype(np.float32)
w_b -= (w_b @ w_a) / (w_a @ w_a) * w_a                              # make raw cosine ~ 0
margins = (h @ w_a).astype(np.float32)

frame = fit_frame(h, margins=margins)
ev = effective_angle(w_a, w_b, frame, n_boot=200, activations_for_bound=h)
print(ev.gauge)
print("raw_cos      ", round(ev.value.raw_cos, 4))
print("canonical_cos", round(ev.value.canonical_cos, 4))
print("ci           ", (round(ev.value.ci_low, 4), round(ev.value.ci_high, 4)))
# covariant
# raw_cos       -0.0
# canonical_cos 0.4288
# ci            (0.3238, 0.5156)
```

The raw cosine is zero by construction. The canonical cosine is \(0.43\) with a CI that excludes zero: inside the frame the two directions share real structure that the raw coordinates hid. That is the whole reason to fit a frame before comparing. The Evidence reports its gauge as `covariant`, the rung that means the number is basis-dependent and only means something once a frame is supplied.

![Raw cross-model cosine versus the canonical angle inside a shared frame.](../assets/figures/gauge-picture-light.svg#only-light){ .rl-fig .rl-fig--hero }
![Raw cross-model cosine versus the canonical angle inside a shared frame.](../assets/figures/gauge-picture-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**A raw cross-model number is a coordinate, not a fact.** The raw cosine near \(0.005\) reads two agreeing models as near-orthogonal; expressing both directions in one fitted frame recovers a meaningful angle with a confidence interval.
///

The left side of the figure is the trap: the raw cosine sits near zero, so a naive comparison concludes the models disagree. The right side is the fix: once both directions are canonicalized in a shared frame, the angle between them is well defined and carries an interval. The frame did not invent the agreement. It removed the coordinate freedom that was hiding it.

## Without a frame, the comparison refuses

`effective_angle` has no default frame. A cross-model comparison of a basis-dependent quantity routes through gate 2, `require_frame_for_comparison`, which raises `GaugeError` rather than return a number that would be a coordinate:

```python
from reward_lens.core.gates import require_frame_for_comparison
from reward_lens.core.types import GaugeStatus
from reward_lens.core.errors import GaugeError

try:
    require_frame_for_comparison(GaugeStatus.COVARIANT, None)   # no frame
except GaugeError as e:
    print(type(e).__name__, "-", str(e)[:64])
# GaugeError - cross-signal comparison of a COVARIANT quantity requires a Frame
```

This is the gate doing its job. A frameless cross-model comparison is not downgraded to a low-trust number; it is refused outright, because the number it would produce has no defensible meaning. Supply the frame and the same comparison returns Evidence.

!!! warning "Needs a GPU"
    Reproducing the real \(0.005\) raw cosine and its canonicalization needs both reward models resident in fp32, which does not fit an 8 GB GPU. The E19 result above is read from committed artifacts. On this hardware the mechanism runs on the synthetic and tiny paths; the 8B canonicalization is gated.

See also: [gauge and frames](../discipline/gauge-and-frames.md), [`effective_angle`](../reference/geometry.md#reward_lens.geometry.canonical.effective_angle), [`fit_frame`](../reference/geometry.md#reward_lens.geometry.frame.fit_frame).
