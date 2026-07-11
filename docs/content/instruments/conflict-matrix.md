<div class="rl-chips">
  <span class="rl-chip rl-chip--obs">Observational</span>
  <span class="rl-chip rl-chip--gauge"><span class="rl-chip__k">gauge</span> raw only</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">works on</span> activations + linear readout</span>
</div>

# Conflict matrix

**When you push the reward up on one axis, does another axis fight you?**

A reward model rewards several things at once: helpfulness, safety, brevity, and more. Those goals need not agree. Sometimes making an answer safer also makes it score higher on helpfulness, and sometimes the two pull in opposite directions, so that optimizing one drags the other down. The conflict matrix measures which. For each pair of axes it fits the direction in activation space that carries that axis's reward and reports the cosine between them: aligned, orthogonal, or in conflict.

This is the successor to the reward-conflict analysis in the 1.0 toolkit, ported behind the measurement gates. It reads activations and the linear readout; it never intervenes.

## The intuition

Each reward axis has a direction. Not a metaphorical one: there is a genuine vector in the residual stream whose projection onto \(w_r\) tracks how that axis's reward moves across a set of matched pairs. Fit one such direction per axis and you can ask how they sit relative to each other. Two directions pointing the same way mean the axes rise and fall together, so a policy that climbs one climbs the other for free. Two directions at right angles mean the axes are independent. Two directions pointing apart mean conflict: the gradient that improves one is, in part, the gradient that worsens the other.

That last case is the one to watch during training. A reward built from terms that fight each other has a built-in tension a policy will resolve by sacrificing whichever term is cheaper to give up.

## The math

For axis \(a\), fit a unit term-direction \(u_a\) in activation space whose projection explains the axis's reward deltas. The conflict between axes \(a\) and \(b\) is the cosine of the angle between their directions,

\[ C_{ab} = \cos\bigl(u_a, u_b\bigr) = \frac{u_a^\top u_b}{\lVert u_a\rVert\,\lVert u_b\rVert}. \]

The diagonal is \(1\) by construction. Off-diagonal entries near \(+1\) are aligned axes, near \(0\) are orthogonal, and negative entries are in conflict. The instrument summarizes the off-diagonal into a mean cosine, the single most negative entry, and a count of conflicting pairs.

The gauge chip reads *raw only*, and that is a load-bearing honesty flag. These cosines are measured in the model's own raw activation basis. Within one model they are a real description of its geometry, but the number is coordinate-dependent, so a raw cosine from one model and a raw cosine from another were never in the same coordinates and must not be compared directly. Carrying a term-geometry comparison across models is what a shared [frame](../discipline/gauge-and-frames.md) is for.

## A worked run

The instrument needs pairs spanning at least two axes, so the view has to mix dimensions. On the tiny model the fitted directions are noise, so read the shape, not the value.

```python
import numpy as np
from reward_lens.signals import from_tiny
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.data.schema import DataView
from reward_lens.measure import base as mb
from reward_lens.measure.battery import ConflictMatrix

signal = from_tiny(seed=0)
views = load_diagnostic_v3()
view = DataView(list(views["helpfulness"].items)[:4] + list(views["verbosity"].items)[:4])
ev = mb.run(ConflictMatrix(), mb.Context(signal=signal, view=view))

print(ev.trust, ev.gauge)                                        # EXPLORATORY raw_only
print("axes:", ev.value["axes"])
print("cosine matrix:", np.round(ev.value["cosine_matrix"], 4).tolist())
print("mean off-diagonal:", round(ev.value["mean_offdiagonal_cosine"], 4))
print("conflicting pairs:", ev.value["n_conflicting_pairs"])
```

```text
EXPLORATORY raw_only
axes: ['helpfulness', 'verbosity']
cosine matrix: [[1.0, 0.2501], [0.2501, 1.0]]
mean off-diagonal: 0.2501
conflicting pairs: 0
```

Two axes give a \(2\times2\) matrix with a single off-diagonal cosine of \(0.25\), a mildly positive number with no conflicting pairs. Hand it more axes and the matrix grows; the summary fields let you spot the most conflicted pair without reading every cell.

Hand it a single axis and it refuses rather than invent a geometry from nothing:

```python
one_axis = DataView(list(views["helpfulness"].items)[:5])
mb.run(ConflictMatrix(), mb.Context(signal=signal, view=one_axis))
# CapabilityError: ConflictMatrix needs pairs spanning >=2 axes; the view has 1.
#                  Pass a multi-axis diagnostic view.
```

## How to read the output

- **`cosine_matrix`** is the whole geometry; the diagonal is trivially \(1\).
- **`min_cosine`** is the most conflicted pair, the one worth naming first. A strongly negative entry is a pair of goals the reward genuinely trades off.
- **`n_conflicting_pairs`** counts how many axis pairs have a negative cosine, a quick read on how much internal tension the reward carries.
- **`mean_offdiagonal_cosine`** is the overall tone: broadly aligned axes sit positive, a reward pulling itself apart sits near or below zero.

## When not to reach for it

Two limits are structural. It needs at least two axes, and it refuses cleanly on one rather than fabricating a degenerate matrix. And its cosines are *raw only*: use them to understand one model's internal tensions, not to rank two models' conflict against each other, because those numbers live in different coordinate systems. If cross-model term geometry is what you want, express both models in a shared frame first.

Two more are statistical. The term-directions are fit from a limited set of pairs, so a handful of pairs per axis gives noisy directions and therefore noisy cosines. And a cosine is a correlation of directions, not a causal channel; a negative entry says the axes' reward-carrying directions point apart, not that optimizing one will provably break the other in your training run.

## How much to trust this

The number arrives at trust level **EXPLORATORY**. The cosine is well-defined, but no calibration provider is wired in this release, so no scorecard yet ties a conflict cosine to an observed training pathology. Exploratory means unaudited, not wrong. Read the matrix as a map of where a reward's goals agree and disagree in its own geometry; do not carry a raw cosine across models, and do not yet promise that a negative entry will bite under optimization. The organism that would calibrate this, a rubric generator with a known number of planted directions and exact cosines, is described in [calibration and organisms](../discipline/calibration-and-organisms.md).

Full signatures and return fields: [`ConflictMatrix`](../reference/measure.md#reward_lens.measure.battery.conflict.ConflictMatrix). Why a raw cosine needs a frame before it can cross models: [gauge and frames](../discipline/gauge-and-frames.md).
