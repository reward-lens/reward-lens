<div class="rl-chips">
  <span class="rl-chip rl-chip--obs">Observational</span>
  <span class="rl-chip rl-chip--gauge"><span class="rl-chip__k">gauge</span> invariant</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">works on</span> scores</span>
</div>

# Bias battery

**Which surface feature is the reward quietly paying for?**

Reward models are supposed to prefer better answers. Some of them also prefer longer answers, or more confident ones, or ones formatted with bullet points, regardless of whether the content improved. Those preferences are exactly what a policy will learn to exploit under optimization, so it is worth measuring them directly, per axis, on the model in front of you. The bias battery does that with an effect size: for each surface feature, how many standard deviations of reward does the model hand out for the feature alone?

Unlike the lens and attribution, this instrument never opens the model. It only needs scores. That makes it the one battery member that runs on a signal you can only query, an API judge included.

## The intuition

Take a set of matched pairs where the only systematic difference is the feature under test: the same answer, one version padded to be more verbose, or rewritten to sound more confident. Score both sides. If the reward is indifferent to the feature, the score deltas scatter around zero. If it leans, the deltas pile up on one side, and how hard it leans, relative to the noise, is the effect size.

Using a standardized effect size rather than a raw reward gap is what makes the axes comparable. A gap of "two reward points" means nothing on its own, because reward units differ between models. Cohen's d, the gap divided by its own spread, is a pure number, and the chip reads *invariant* for that reason: you can line up length bias on one model against length bias on another and the comparison is honest.

## The math

For axis \(a\), let \(\delta_i = r(x_i^{+}) - r(x_i^{-})\) be the reward delta on matched pair \(i\), where \(x^{+}\) carries the feature and \(x^{-}\) does not. The standardized bias is

\[ d_a = \frac{\overline{\delta}}{s_\delta}, \qquad \overline{\delta} = \frac{1}{n}\sum_i \delta_i, \quad s_\delta = \operatorname{std}(\delta_i). \]

A positive \(d_a\) means the reward leans toward the feature; the sign tells you the direction, the magnitude tells you how reliably. The subtlety is in the \(n\). Preference sets are full of near-duplicates, and thirty paraphrases of one prompt are not thirty independent observations. The battery counts the [effective sample size](../how-to/effective-sample-size.md) from the data's lineage, so the precision it reports reflects the independent evidence, not the row count. Thirty clones of a single prompt collapse to an effective \(n\) of one, and the battery says so rather than pretending to thirty.

## A worked run

On the tiny model the effect sizes are noise, because the model is untrained and the axes carry only a handful of pairs. Read this for the shape of the output.

```python
from reward_lens.signals import from_tiny
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.data.schema import DataView
from reward_lens.measure import base as mb
from reward_lens.measure.battery import BiasBattery

signal = from_tiny(seed=0)
views = load_diagnostic_v3()
# A view that spans two axes, so the battery has more than one to report.
view = DataView(list(views["helpfulness"].items)[:4] + list(views["verbosity"].items)[:4])
ev = mb.run(BiasBattery(), mb.Context(signal=signal, view=view))

print(ev.trust, ev.gauge)                                  # EXPLORATORY invariant
print("strongest axis:", ev.value["strongest_axis"], "| |d|:", round(ev.value["max_abs_effect_size"], 3))
for axis, s in ev.value["per_axis"].items():
    print(f"  {axis:12s} d={s['effect_size']:+.3f}  effective_n={s['effective_n']}  n_pairs={s['n_pairs']}")
```

```text
EXPLORATORY invariant
strongest axis: verbosity | |d|: 1.108
  helpfulness  d=+0.731  effective_n=4.0  n_pairs=4
  verbosity    d=-1.108  effective_n=4.0  n_pairs=4
```

The per-axis dictionary carries the effect size, the mean and standard deviation of the delta, the raw pair count, and the effective sample size side by side. Here `effective_n` equals `n_pairs` because none of these pairs are clones; on a real eval set with paraphrase families the two numbers separate, and the gap is the point.

### The bias is the model's, not reward models' in general

Run the same battery on two real 8B models and the striking result is not the magnitudes, it is that the same surface feature can point in opposite directions on two models.

![Cohen's d per surface feature for two 8B reward models; several features reverse sign between them.](../assets/figures/hacking-effects-light.svg#only-light){ .rl-fig }
![Cohen's d per surface feature for two 8B reward models; several features reverse sign between them.](../assets/figures/hacking-effects-dark.svg#only-dark){ .rl-fig }

/// caption
**The same feature, opposite sign.** Each feature gets two bars, one per model. Several of them, confidence and formatting among them, land on opposite sides of zero: one model pays for the feature while the other penalizes it. A bias battery is a statement about a specific reward model, never about reward models as a class.
///

That is the reading to carry away. There is no universal "length bias" you can look up once. A feature one model rewards, another punishes, so the battery has to be run on the model you are about to optimize against. A shared feature that both models reward is a stronger warning than one that only shows up on a single model, because a policy trained against either will find it.

To run it on a real model you need the hardware.

!!! warning "Needs a GPU"
    An 8B reward model in fp32 does not fit a laptop GPU. `load_signal` refuses a hub id unless you pass `allow_download=True`, and the forward passes are the gated part.

    ```python
    from reward_lens.signals import load_signal
    from reward_lens.measure import base as mb
    from reward_lens.measure.battery import BiasBattery

    signal = load_signal("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", allow_download=True)
    ev = mb.run(BiasBattery(), mb.Context(signal=signal, view=view))
    ```

## How to read the output

- **Sign** is the direction of the lean: positive means the reward pays for the feature, negative means it discounts it.
- **Magnitude** is in standard deviations. As a rough guide, \(|d|\) around \(0.2\) is a whisper, around \(0.8\) is loud, and anything past \(1\) is a feature a policy will find quickly.
- **`effective_n` versus `n_pairs`** is the honesty check. If effective \(n\) is much smaller than the pair count, your eval set is more redundant than it looks and the effect size is resting on fewer independent observations than the raw number suggests.

## When not to reach for it

The bias battery measures a *correlation* between a surface feature and the reward, not a causal channel. A high length bias tells you longer answers score higher on your matched set; it does not prove the model has a length circuit you can ablate, because the feature you varied may be entangled with content you did not control. Treat a loud axis as a lead. If you need the mechanism, take the reading to [concept dose-response](concept-dose-response.md), which pushes the model along a feature direction and measures whether the reward actually moves.

And a small-\(n\) caution: an effect size on eight pairs is barely an estimate. The instrument reports the effective \(n\) precisely so you do not read a d computed on four independent pairs as if it were settled.

## How much to trust this

The number arrives at trust level **EXPLORATORY**. Cohen's d is a clean statistic and the effective-\(n\) accounting is honest, but no calibration provider is wired in this release, so no scorecard yet ties a given effect size to a hacking rate under real optimization pressure. Exploratory means unaudited, not wrong. Use the battery to rank a model's biases and to compare biases across models, since the gauge is invariant; do not yet promise that a d below some threshold is safe. The organisms that would calibrate that threshold, planted-bias generators with a known ground-truth channel, are described in [calibration and organisms](../discipline/calibration-and-organisms.md).

Full signatures and return fields: [`BiasBattery`](../reference/measure.md#reward_lens.measure.battery.bias.BiasBattery). For a task-shaped walkthrough, see [detect length bias](../how-to/detect-length-bias.md).
