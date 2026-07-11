<div class="rl-chips">
  <span class="rl-chip rl-chip--causal">Causal</span>
  <span class="rl-chip rl-chip--gauge"><span class="rl-chip__k">gauge</span> raw only</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">works on</span> activations + linear readout</span>
</div>

# Concept dose-response

**Push an activation toward a concept. Does the reward move, and by how much per unit of push?**

A concept like verbosity, confidence, or formality is a *direction* in activation space. You estimate it as the mean difference between activations that have the concept and activations that do not, exactly the way a preference pair gives you a chosen-minus-rejected difference. Once you hold that direction, two separate questions follow, and this instrument answers both. First, does having the concept point the same way as the reward? That is a cosine, and it is observational. Second, if you actually *steer* an activation along the concept, does the reward follow, and how steeply? That is an intervention, and its slope is a causal fact. A concept that both aligns with reward and moves it under steering is one the model genuinely rewards, not one that merely correlates.

## Direction, alignment, dose

The concept direction is the unit-normalized mean difference between the positive and negative activations:

\[
u = \frac{\bar{a}_{+} - \bar{a}_{-}}{\lVert \bar{a}_{+} - \bar{a}_{-} \rVert}.
\]

Its **alignment** with the reward is the cosine against the reward direction, how much of the concept points along \(w_r\):

\[
\text{align}(u) = \frac{u^{\top} w_r}{\lVert u \rVert \, \lVert w_r \rVert}.
\]

The **dose-response** is the causal counterpart. Add \(\alpha u\) to the final residual, read the reward at a ladder of doses \(\alpha\), and fit the slope by least squares:

\[
\text{slope} = \frac{d\,r(\alpha)}{d\alpha}, \qquad r(\alpha) = w_r^{\top}\!\big(h + \alpha u\big) + b.
\]

For a single linear head the slope equals \(w_r^{\top} u\), so alignment and dose measure the same thing from two sides: a concept aligned with the reward has a positive dose-response, and steering along it raises the score. Both quantities are read in the residual-stream basis, which is why the gauge is raw only: a cosine of \(0.19\) on one model and \(0.19\) on another are not the same fact until both are expressed in a shared [frame](../discipline/gauge-and-frames.md).

![A residual activation and a concept direction; steering the activation along the concept slides it across the reward level sets, so the score changes in proportion to the push.](../assets/figures/concept-vector-light.svg#only-light){ .rl-fig .rl-fig--hero }
![A residual activation and a concept direction; steering the activation along the concept slides it across the reward level sets, so the score changes in proportion to the push.](../assets/figures/concept-vector-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**A concept is a direction, and steering along it moves the score by its projection on \(w_r\).** The activation slides along the concept arrow; each reward level set it crosses is one unit of reward gained. How far the concept tilts toward \(w_r\) is exactly the dose-response slope.
///

The picture is the whole idea in one frame. The reward is the shadow the activation casts on \(w_r\). Steering along a concept that tilts toward \(w_r\) drags that shadow up; steering along one orthogonal to \(w_r\) moves the activation but not its shadow, and the reward does not budge. The slope of the line you get by sweeping the dose is the amount of tilt.

## A run you can reproduce on CPU

The observable reads the concept straight from the pairs in a view, chosen as the positive side, rejected as the negative, then steers the final residual across a ladder of doses and reports the alignment and the fitted slope.

```python
from reward_lens.signals import from_tiny
from reward_lens.data.schema import DataView
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.measure import base as mb
from reward_lens.measure.battery import ConceptDoseResponse

signal = from_tiny(seed=0)
view = DataView(list(load_diagnostic_v3()["helpfulness"].items)[:4])

ev = mb.run(ConceptDoseResponse(), mb.Context(signal=signal, view=view))
print(round(ev.value["reward_alignment"], 4), round(ev.value["dose_response_slope"], 4))
print(ev.value["doses"])
print([round(x, 4) for x in ev.value["mean_reward_at_dose"]])
print(str(ev.trust), str(ev.gauge))
# 0.1931 0.071
# [-2.0, -1.0, 0.0, 1.0, 2.0]
# [-0.1242, -0.1291, -0.1154, 0.1067, 0.1129]
# EXPLORATORY raw_only
```

The mean reward climbs monotonically as the dose goes from \(-2\) to \(+2\), which is the dose-response in raw form. The three atoms underneath the observable are public, so you can build the same measurement by hand and check that the slope recovers a known coefficient exactly:

```python
import numpy as np, torch
from reward_lens.concepts import concept_direction, reward_alignment, dose_response_slope

rng = np.random.default_rng(0)
w_r = torch.tensor(rng.standard_normal(16), dtype=torch.float32)
pos = torch.tensor(rng.standard_normal((20, 16)), dtype=torch.float32) + 3.0 * w_r
neg = torch.tensor(rng.standard_normal((20, 16)), dtype=torch.float32) - 3.0 * w_r

u = concept_direction(pos, neg)
print(round(float(np.linalg.norm(u)), 4))        # 1.0   (unit vector)
print(round(reward_alignment(u, w_r), 4))         # 0.9989 (concept ~ the reward direction)

doses = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
rewards = 0.75 * doses + 0.1                       # a perfectly linear dose-response
print(round(dose_response_slope(doses, rewards), 4))  # 0.75
```

When the concept is built to *be* the reward direction, its alignment comes back at essentially one, and a straight dose-response of slope \(0.75\) is recovered to the digit. That is the sanity check: the tool reads back exactly what you put in.

## The real line, on a trained model

![The empirical dose-response on Skywork: mean reward against steering strength for a concept, a straight rising line whose slope is the alignment.](../assets/figures/concept-dose-response-light.svg#only-light){ .rl-fig .rl-fig--hero }
![The empirical dose-response on Skywork: mean reward against steering strength for a concept, a straight rising line whose slope is the alignment.](../assets/figures/concept-dose-response-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**Steer harder, and the reward rises in near-linear proportion.** Mean reward against steering dose for a concept on Skywork. The line is close to straight, which is what makes a single slope the honest summary: over this range the reward is a linear readout, and the concept's dose-response is its tilt toward \(w_r\).
///

Read it as a controlled experiment. Every point is the model's reward after the same activation has been nudged a fixed amount along the concept. The near-straight line says the reward responds linearly over the steered range, so one number, the slope, captures the concept's causal pull on the score. A concept whose line is flat moves the activation without moving the reward: the model does not read it. A concept whose line is steep is one the model rewards directly, and steering it is a lever on the score.

## When not to reach for it

The dose-response is honest only over the range you actually steer. Push far enough and you leave the region where the reward is linear, or you drive the activation off the model's distribution entirely, and the extrapolated slope stops meaning anything; keep the doses modest. The concept direction is only as good as the pairs you built it from, so a contrast that confounds two concepts at once yields a direction that is neither. And because the whole measurement is basis-dependent, do not compare a raw alignment across two models: that is a [gauge](../discipline/gauge-and-frames.md) error, and the raw cross-model cosine between two reward directions can be as low as \(0.005\) purely from coordinates.

## How much to trust it

`ConceptDoseResponse` returns [`Evidence`](../reference/core.md#reward_lens.core.evidence.Evidence) at `EXPLORATORY`, gauge `RAW_ONLY`. It is a causal measurement, the slope comes from an intervention, but it is not calibrated: no calibration provider is wired in this release, so it defaults to `EXPLORATORY` and stays exploratory by construction until a scorecard exists. It is also not comparable across models without a frame. Within one model it earns a strong claim: "steering this concept changes the reward at this rate over this range." It does not by itself earn "the model has a preference for this concept" as a portable, cross-model statement; for that you need to canonicalize the direction into a shared frame first. The instrument that turns a raw concept axis into a frame-expressed, comparable one is [effective angle](../discipline/gauge-and-frames.md); the observational cousin that reads alignment without steering is [component attribution](attribution.md).
