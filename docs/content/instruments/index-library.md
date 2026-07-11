<div class="rl-chips">
  <span class="rl-chip rl-chip--fill rl-chip--exploratory">exploratory</span>
  <span class="rl-chip rl-chip--gauge"><span class="rl-chip__k">gauge</span> invariant</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">works on</span> scores + activations</span>
</div>

# The index library

**Sometimes you do not want a whole instrument, you want one number.**

An index is a scalar diagnostic: a single value that answers one specific worry about a reward model, computed on top of the instruments and the signal. There are eighteen of them. Each pairs a plain numpy function that returns a bare number with an observable that wraps it in [`Evidence`](../reference/core.md#reward_lens.core.evidence.Evidence), so you can use an index as a quick statistic or as a measurement that carries its own credentials. They exist as one page rather than eighteen because they share a shape and, more importantly, a status: none of them is calibrated yet, and that uniform honesty is worth stating once, clearly, instead of burying it eighteen times.

## The eighteen

Every index below returns `EXPLORATORY` Evidence today. That is not a hedge about whether the math is right; the definitions are exact and tested. It means no calibration scorecard is wired to say what a given value implies about a real model, so the trust gate holds each one at the exploratory rung until an answer key exists. Sixteen are gauge-invariant. Two, `VCE` and `Contested`, are cross-model quantities that need a shared [frame](../discipline/gauge-and-frames.md) before they compare.

| Index | The question it answers | Gauge | Calibrated? |
|---|---|---|---|
| `KUI` | How much of what the model could know does the reward actually use? | invariant | No |
| `Distortion` | How far does the reward bend each dimension away from the coverage the model has for it? | invariant | No |
| `TeacherCompatibility` | How much reward variance would this model induce if you optimized against it? | invariant | No |
| `TailIndex` | How heavy is the reward's right tail, the regime a best-of-N search pushes into? | invariant | No |
| `VerificationScore` | Does the correctness reward gap actually sit at the labeled error, or somewhere else? | invariant | No |
| `StyleShare` | How much of the correctness reward gap is really style rather than correctness? | invariant | No |
| `ReceiptReliance` | How much of the reward effect rides on citation and receipt spans? | invariant | No |
| `Skepticism` | Does the reward treat a missing receipt as failure, or reward it anyway? | invariant | No |
| `Coherence` | Do a multi-criteria head's criteria agree, or contaminate one another? | invariant | No |
| `DarkReward` | How much reward variance is mediated by no channel you can name? | invariant | No |
| `InterpCoverage` | What fraction of the reward routes through features you can interpret? | invariant | No |
| `Chi` | Which features move the reward when you tilt the policy toward it? | invariant | No |
| `VCE` | Do two models agree on reward beyond what their shared capability explains? | needs a frame | No |
| `Legibility` | How much of the reward can a small, legible circuit reproduce, and where is the knee? | invariant | No |
| `EvalAwareness` | Can a probe on the reward's activations tell benchmark inputs from organic ones? | invariant | No |
| `RobustnessSNR` | Is the reward's between-condition signal above its within-condition paraphrase noise? | invariant | No |
| `Contested` | Along which representation axis do annotators actually disagree? | needs a frame | No |
| `CoverageDisparity` | v1's coverage statistic, kept reproducible under its honest name. | invariant | No |

The names are stable and importable as a set from [`reward_lens.measure.indices`](../reference/measure.md). Read the table as a menu of specific worries: if yours is "is this reward model rewarding style under the name of correctness," you want `StyleShare`; if it is "will optimizing against this model find a heavy tail to exploit," you want `TailIndex` and `Chi` together.

## A run you can reproduce on CPU

Any index runs through the same measurement path as a full instrument. `KUI` is one of the five that compute fully rather than degrading when a dependency is missing, so it is a clean thing to run end to end on the tiny signal:

```python
from reward_lens.signals import from_tiny
from reward_lens.data.schema import DataView
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.measure import base as mb
from reward_lens.measure.indices import KUI

signal = from_tiny(seed=0)
view = DataView(list(load_diagnostic_v3()["helpfulness"].items)[:4])

ev = mb.run(KUI(), mb.Context(signal=signal, view=view))
print(str(ev.trust), str(ev.gauge))
print(list(ev.value.keys()))
# EXPLORATORY invariant
# ['kui', 'names', 'note']
```

The Evidence comes back at `EXPLORATORY` with an `INVARIANT` gauge, exactly as the table promises, and its value carries a `note` field. That field is the honest-degradation channel: an index that cannot fully compute because a feature bank, a probe, or a patch is unavailable reports what it could and says so in the note, rather than filling the gap with a fabricated number. Five indices, `KUI`, `TeacherCompatibility`, `TailIndex`, `Coherence`, and `Chi`, compute fully on the substrates the tiny signal provides; the rest state plainly when they had to fall back.

## When not to reach for one

A scalar hides its own construction, which is the danger of any index. `DarkReward` at some value is only as trustworthy as the set of channels you named to subtract, and `InterpCoverage` only as trustworthy as the dictionary of features you called interpretable; the number is a summary of your setup as much as of the model. An index also flattens a distribution to a point, so read it next to the instrument it summarizes rather than instead of it. And the two covariant indices refuse to be compared across models raw: a `VCE` or `Contested` value from one model set beside another's is a [gauge](../discipline/gauge-and-frames.md) error until both are expressed in one frame, and the library raises rather than let the comparison through.

## How much to trust them

Exploratory means unaudited, not wrong. The definitions here are exact and each has a test behind it; what is missing is the calibration that would let a value speak to a regime, "a `TailIndex` above this threshold means this model is exploitable under best-of-N in this setting." Until a scorecard supplies that, an index is a sharp, reproducible statistic you may explore with freely and must not ship as a validated verdict. For what the exploratory rung does and does not license, see [the trust ladder](../discipline/trust-ladder.md); for why calibration needs an organism with a known answer, [calibration and organisms](../discipline/calibration-and-organisms.md).
