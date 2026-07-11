<div class="rl-chips">
  <span class="rl-chip rl-chip--obs">Observational</span>
  <span class="rl-chip rl-chip--gauge"><span class="rl-chip__k">gauge</span> invariant</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">works on</span> scores</span>
</div>

# Prompt SNR

**On this axis, is the reward signal actually louder than its own noise?**

An effect size tells you which way a reward leans and how far. It does not, on its own, tell you whether the lean is a clean signal or a lucky draw from a noisy one. Prompt SNR answers that second question directly. For each axis it takes the reward delta across the matched pairs and reports the ratio of its squared mean to its variance: how much of what you measured is signal, relative to the scatter around it.

Like the [bias battery](bias-battery.md), it needs only scores, so it runs on any signal you can query, and it is the natural companion reading to take alongside a bias number.

## The intuition

You have a stack of reward deltas on one axis. Two things describe that stack: where its center sits, and how wide it is. A center far from zero with a tight width is a reliable effect. The same center with a wide width could easily have come from noise. SNR is the single number that trades those off, the mean squared against the variance. High SNR means the axis separates the pairs cleanly and would survive being re-sampled; low SNR means the reward is barely distinguishing the two sides above its own jitter.

It is deliberately sign-free. A loud axis is loud whether the reward leans toward the feature or against it, which is exactly what you want when the question is reliability rather than direction.

## The math

For axis \(a\) with reward deltas \(\delta_i = r(x_i^{+}) - r(x_i^{-})\), the power signal-to-noise ratio is

\[ \operatorname{SNR}_a = \frac{\overline{\delta}^{\,2}}{\operatorname{Var}(\delta)}. \]

Each \(\delta_i\) is itself a projection onto the reward direction, \(\delta_i = w_r^\top(h_i^{+} - h_i^{-})\), so the SNR is a property of how the projected deltas cluster. There is a clean relationship to the bias battery worth keeping in your head: \(\operatorname{SNR}_a\) is exactly the square of that axis's Cohen's d, since \(\overline{\delta}^{\,2}/\operatorname{Var}(\delta) = (\overline{\delta}/s_\delta)^2\). The bias battery keeps the sign and reports the effect size; prompt SNR squares it into a power ratio. Two views of the same per-axis delta. The ratio is dimensionless, so it is comparable across models and the chip reads *invariant*.

## A worked run

On the tiny model the axes carry only a few pairs, so read this for the shape and for the relationship to the effect size, not for a result.

```python
from reward_lens.signals import from_tiny
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.data.schema import DataView
from reward_lens.measure import base as mb
from reward_lens.measure.battery import PromptSNR

signal = from_tiny(seed=0)
views = load_diagnostic_v3()
view = DataView(list(views["helpfulness"].items)[:4] + list(views["verbosity"].items)[:4])
ev = mb.run(PromptSNR(), mb.Context(signal=signal, view=view))

print(ev.trust, ev.gauge)                                     # EXPLORATORY invariant
print("loudest axis:", ev.value["highest_snr_axis"], "| SNR:", round(ev.value["max_snr"], 3))
for axis, s in ev.value["per_axis"].items():
    print(f"  {axis:12s} SNR={s['snr']:.3f}  mean_delta={s['mean_delta']:+.4f}  var_delta={s['var_delta']:.5f}")
```

```text
EXPLORATORY invariant
loudest axis: verbosity | SNR: 1.228
  helpfulness  SNR=0.534  mean_delta=+0.0092  var_delta=0.00016
  verbosity    SNR=1.228  mean_delta=-0.0264  var_delta=0.00057
```

The `helpfulness` SNR of \(0.534\) is the square of that axis's bias-battery effect size of \(+0.731\), and the `verbosity` SNR of \(1.228\) is the square of \(-1.108\). Same data, two lenses. The per-axis dictionary carries the mean and variance of the delta so you can see both the center and the width the ratio was built from.

## How to read the output

- **`highest_snr_axis`** and **`max_snr`** point you at the axis where the reward is most reliably separating the two sides, whatever direction it leans.
- **Per-axis `snr`** is the power ratio. As a rough map, an SNR near \(1\) corresponds to an effect size near \(1\), a loud axis; an SNR of \(0.04\) is a d of about \(0.2\), a whisper.
- **`mean_delta` and `var_delta`** let you see *why* an axis is loud or quiet: a big center over a small spread, or a small center the variance nearly swallows.

## When not to reach for it

SNR is sign-free, which is a feature for reliability and a trap for direction. A high-SNR axis can be a reward reliably rewarding something you did not want, and the ratio alone will not tell you which. Read it next to the [bias battery](bias-battery.md), which keeps the sign, and treat SNR as the reliability half of a two-part answer.

It is also not a causal statement and not robust at small \(n\). A high SNR on eight pairs is a loud signal on a tiny sample, which is a hypothesis, not a settled fact. And a clean separation on your matched set does not prove the model has a mechanism for the feature; for that, intervene with [concept dose-response](concept-dose-response.md).

## How much to trust this

The number arrives at trust level **EXPLORATORY**. The ratio is a well-defined statistic, but no calibration provider is wired in this release, so no scorecard yet maps an SNR to a robustness guarantee. Exploratory means unaudited, not wrong. Use it to rank axes by reliability and to compare a model against itself and others, since the gauge is invariant; do not yet read a threshold as a promise. The route to a calibrated reading is [calibration and organisms](../discipline/calibration-and-organisms.md), and the meaning of the trust level is [the trust ladder](../discipline/trust-ladder.md).

Full signatures and return fields: [`PromptSNR`](../reference/measure.md#reward_lens.measure.battery.snr.PromptSNR).
