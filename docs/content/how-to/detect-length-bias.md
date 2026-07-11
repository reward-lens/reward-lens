# Detect length bias

**Does your reward model pay more for a longer answer that says the same thing?**

Run the bias battery and read the length axis. `BiasBattery` measures, per axis, the standardized effect size (Cohen's d) of the chosen-minus-rejected reward delta, and reports it alongside the effective sample size behind it. A reward bias is a reward difference the grader assigns to a surface change that should not matter: more length, more confidence, more markdown. In the diagnostic set the length axis is named `verbosity`, padding with no added content.

```python
from reward_lens.signals import from_tiny
from reward_lens.measure import base as mb
from reward_lens.measure.battery import BiasBattery
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.data.schema import DataView

signal = from_tiny(seed=0)
diag = load_diagnostic_v3()

# The length axis in the diagnostic set is "verbosity": padding with no added content.
axes = ["verbosity", "sycophancy", "formatting", "confidence"]
items = [p for ax in axes for p in list(diag[ax].items)[:5]]
view = DataView(items)

ev = mb.run(BiasBattery(), mb.Context(signal=signal, view=view))
print(ev.trust, ev.gauge)                 # EXPLORATORY invariant
print(ev.value["strongest_axis"])         # verbosity
for axis, d in sorted(ev.value["per_axis"].items()):
    print(f"{axis:12s} d={d['effect_size']:+.3f}  eff_n={d['effective_n']:.1f}")
# confidence   d=+0.306  eff_n=5.0
# formatting   d=-0.108  eff_n=5.0
# sycophancy   d=+0.691  eff_n=5.0
# verbosity    d=-1.311  eff_n=5.0
```

The sign is the reading: a positive d means the model pays for the surface change, negative means it docks it, near zero means it does not care. The signal here is a random tiny model, so these magnitudes are noise, not a bias finding; what is real is the shape of the result, one standardized effect and one honest sample size per axis.

## The effective sample size is the second half of the number

`eff_n` comes back as `5.0`, equal to the five pairs per axis, because each diagnostic pair carries a distinct seed. That equality is the point, not an accident: the battery reports the lineage-honest effective sample size, so a battery that mutated a few seeds into many pairs would show an `eff_n` below its pair count and its confidence intervals could not tighten past what the seeds earned. When the two numbers diverge, believe the smaller one. See [effective sample size](effective-sample-size.md).

## Read across models with the d, never the raw delta

Cohen's d is dimensionless, which is why the gauge is `invariant`: a d of 0.8 is a 0.8 on any signal. A raw reward delta is not comparable that way. A classifier RM emitting logits swings by tens of points; a bounded, gated head swings by hundredths. The standardized effect divides that arbitrary per-model scale out, which is what makes a cross-model bias comparison mean anything.

!!! warning "Needs a GPU"
    On a trained 8B classifier RM, the same call reads a real bias per axis: wrap the model with [`wrap_hf_model`](load-a-reward-model.md), build a view over the axes you care about, and run `BiasBattery` exactly as above. The sign on the length axis tells you whether that model rewards padding or docks it. The [bias battery instrument page](../instruments/bias-battery.md) carries the measured-model reading.

See also: [Bias battery](../instruments/bias-battery.md), [A measurement you can trust](../concepts/measurement-you-can-trust.md). API: [`BiasBattery`](../reference/measure.md#reward_lens.measure.battery.bias.BiasBattery).
