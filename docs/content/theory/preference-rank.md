# When preference is not rank-one

**What if preference has no consistent ranking to find?** A scalar reward head assumes it does. It compresses every comparison onto one axis, which forces preferences to be transitive: prefer \(a\) to \(b\) and \(b\) to \(c\), and you are committed to preferring \(a\) to \(c\). Real preference does not always oblige, and when it does not, no scalar head can represent it. This page is about measuring the gap directly.

## The rank-one assumption

A scalar reward \(r\) induces preferences through \(P(x \succ y) = \sigma(r(x) - r(y))\). Sort responses by \(r\) and you have a total order; every pairwise preference is read off that order. The whole preference structure is rank-one, a single direction in quality space, and its transitivity is not an approximation the model makes but a property it cannot violate. This is what the Bradley-Terry fit assumes, and where it can leak is the subject of the [Bradley-Terry](bradley-terry.md) page.

## Preferences that cycle

Intransitive preference is not exotic. The rock-paper-scissors pattern is the canonical shape: \(a\) beats \(b\), \(b\) beats \(c\), \(c\) beats \(a\), with no response best. It shows up in individual judgments over multi-attribute options, and it shows up structurally when you pool a group of annotators who each rank consistently but rank differently, a Condorcet cycle assembled from transitive parts. Faced with a cycle, a scalar head cannot fit it. It settles for the total order that violates the fewest comparisons and discards the rest as noise. But the discarded part is not noise, it is genuine structure the representation had to throw away, and throwing it away is provable: a scalar head cannot express a single cycle (theorem T8).

## The object that can

The structure a scalar misses has a natural home. Where a scalar reward gives a symmetric comparison through a difference of levels, a genuine preference relation needs only antisymmetry, \(s(x, y) = -s(y, x)\), which is exactly what a skew-symmetric bilinear form on the activations provides:

\[
s(x, y) = \phi(x)^\top A\, \phi(y), \qquad A^\top = -A.
\]

The antisymmetry is built in, and the rank of \(A\) measures how much intransitive structure the representation supports. A rank-\(2k\) skew operator captures \(k\) independent cyclic planes; the scalar head is the degenerate rank-zero case, the corner where all the cyclic structure has been set to nothing. So "is preference rank-one" becomes a measurable question: fit the skew operator, count its effective rank, and see whether it predicts held-out preferences a scalar cannot.

## Measuring it

That is what [`PreferenceRankTest`](../reference/geometry.md) does. It fits both the best transitive model and a rank-\(k\) skew operator on a training split, then scores their held-out prediction. The gap is the cyclic structure the scalar head cannot express. The mechanism is clean enough to watch on a planted cycle: put items on a circle and let a pure skew form decide every winner, so the preference is cyclic by construction.

```python
import numpy as np
from reward_lens.geometry import PreferenceRankTest

rng = np.random.default_rng(0)
theta = rng.uniform(0, 2 * np.pi, size=40)
phi = np.column_stack([np.cos(theta), np.sin(theta)])   # 40 items on the unit circle
A = np.array([[0.0, 1.0], [-1.0, 0.0]])                 # a skew form: winner by cross-product sign
pairs = [(i, j) if phi[i] @ A @ phi[j] > 0 else (j, i)
         for i in range(40) for j in range(i + 1, 40)]

ev = PreferenceRankTest(phi, np.array(pairs), rank_k=1).run(seed=0)
v = ev.value
print(round(v.transitive_acc, 3), round(v.skew_acc, 3))   # 0.671 0.97
print(round(v.cyclic_recovery, 3), v.effective_rank)      # 0.299 2
print(str(ev.gauge))                                      # invariant
```

The best scalar reward orders only \(0.671\) of the held-out pairs correctly, barely better than the coin it is forced toward on cyclic data. The rank-one skew operator gets \(0.970\), recovering \(0.299\) of preference the scalar had to discard. Its effective rank comes back as \(2\), one cyclic plane, which is exactly what a single planted cycle should register. If learned preference were empirically rank-one, that recovery would sit near zero; a positive recovery is intransitive structure made visible. The result is `invariant`: rank and predictive accuracy do not depend on the coordinate basis, so this is a within-model structural claim that needs no frame.

## What it does and does not say

Two connections make the claim sharper. The skew operator's predictions can be Hodge-decomposed, which is the move the [Topology](../sciences.md) science makes: split a corpus of preferences into a gradient part, exactly what a scalar reward can represent, and curl and harmonic parts, exactly what it cannot. The size of the second is a coordinate-free lower bound on the error of every scalar reward model on that corpus, computable with no model at all. And the existence of that irreducible part is why alternatives to scalar reward exist, methods that treat alignment as a game over a general preference relation rather than a fitted scalar (Munos et al., 2023, "Nash Learning from Human Feedback," arXiv [2312.00886](https://arxiv.org/abs/2312.00886)).

The honest limits are the usual ones. On the planted cycle above the answer is known by construction, so the test is calibrated; run it on a real model's activations and the result is exploratory until it has been [calibrated on an organism](../discipline/calibration-and-organisms.md) whose cyclic structure you planted. And a positive recovery says only what a scalar summary cannot represent, not that your reward model is broken. Most of what a good reward model does is transitive and a scalar captures it well. The test tells you how much it had to leave out, which is precisely the part a single number can never show you.
