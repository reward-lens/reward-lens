<div class="rl-chips">
  <span class="rl-chip rl-chip--obs">Observational</span>
  <span class="rl-chip rl-chip--gauge"><span class="rl-chip__k">gauge</span> invariant</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">works on</span> scored samples + per-sample features</span>
</div>

# Tilt and susceptibility

**Which feature drifts first when you optimize against the reward model?**

You can predict it before a single gradient step. Optimizing a policy against a reward pulls it toward the exponentially tilted family \(\pi_\lambda \propto \pi_0 \, e^{\lambda r}\), and the initial drift of a feature under that pull is a base-policy covariance:

\[
\chi_i \;=\; \left.\frac{d}{d\lambda}\, \mathbb{E}_\lambda[f_i]\right|_{\lambda=0} \;=\; \mathrm{Cov}_0(f_i, r)
\]

The susceptibility \(\chi_i\) is the sign and speed of feature \(i\)'s drift, read off a bank of base-policy samples with zero gradient updates. A positive \(\chi_i\) means the feature grows as you turn up the pressure. Compute it directly:

```python
import numpy as np
from reward_lens.loops import susceptibility, flag_hack_modes, critical_lambda_from_tail

rng = np.random.default_rng(1)
N = 4000
quality = rng.standard_normal(N)   # genuine quality
length  = rng.standard_normal(N)   # spurious: longer answers
eps     = rng.standard_normal(N)   # idiosyncratic reward noise (not a tracked feature)

r    = 1.5 * quality + 0.8 * length + 0.25 * eps   # the proxy reward the RM assigns
gold = 1.0 * quality - 0.9 * length                # the gold objective it stands in for
feats = np.column_stack([quality, length])

ev = susceptibility(r, feats, feature_names=["quality", "length"])
spec = ev.value
for name, chi in zip(spec.feature_names, spec.chi):
    print(f"chi[{name:>7}] = {chi:+.4f}")
# chi[quality] = +1.5140
# chi[ length] = +0.7702
print("teacher_variance =", round(spec.teacher_variance, 4))
# teacher_variance = 2.9636
```

Both features are susceptible. The proxy rewards quality and length, so under pressure it will grow both, and \(\chi\) says quality faster than length. The `teacher_variance` is the \(f = r\) diagonal of the same law, \(\mathrm{Var}_0(r)\), the susceptibility of the reward to itself.

## The hack modes are the disagreements

Susceptibility on its own does not tell you which drift is a problem. Growing quality is fine, because the objective you actually want grows with it. Growing length is not, because the objective does not. A hack mode is exactly that disagreement: a feature the proxy loves while the gold objective does not, \(\chi_i > 0\) and \(\mathrm{Cov}_0(f_i, \text{gold}) \le 0\). Hand [`flag_hack_modes`](../reference/dynamics-loops.md#reward_lens.loops.tilt.flag_hack_modes) the gold covariances and it names them:

```python
gold_cov = np.mean((feats - feats.mean(0)) * (gold - gold.mean())[:, None], axis=0)
print({n: round(float(g), 4) for n, g in zip(spec.feature_names, gold_cov)})
# {'quality': 1.0139, 'length': -0.8937}
print(flag_hack_modes(spec, gold_cov))
# ['length']
```

Quality has positive susceptibility and positive gold covariance, so it is not flagged: the proxy and the objective agree about it. Length has positive susceptibility and negative gold covariance, so it is the hack mode. This is the whole prediction, made before any optimization: length is the direction pressure will exploit.

## How far the preview holds

The tilt is only a faithful emulator of real optimization while the importance weights have not collapsed onto a handful of samples. Two things break it: pushing \(\lambda\) past about half the critical pressure \(\lambda_c\), and the effective sample size of the reweighting falling too low. [`critical_lambda_from_tail`](../reference/dynamics-loops.md#reward_lens.loops.tilt.critical_lambda_from_tail) estimates the ceiling from the reward's right tail:

```python
lam_c = critical_lambda_from_tail(r)
print(round(lam_c, 4), round(lam_c / 2, 4))
# 1.1604 0.5802
```

The full sweep, [`tilt_sweep`](../reference/dynamics-loops.md#reward_lens.loops.tilt.tilt_sweep), emulates the whole \(\lambda\) grid by self-normalized importance sampling and refuses, with an `ESSGuardError`, when a requested \(\lambda\) exceeds \(\lambda_c/2\) or the effective sample size collapses. It would rather name the guard that fired than return a confident number extrapolated from three samples. For the far end of the frontier, past where the tilt is trustworthy, use [best-of-N](best-of-n.md), which prices the whole curve exactly.

## What the number is and is not

The susceptibility comes back as [Evidence](../concepts/measurement-you-can-trust.md) with the gauge `invariant`, because a covariance of two scalar readouts does not depend on the activation basis. The trust is `EXPLORATORY`. \(\chi\) is a first-order prediction of the initial drift, not a guarantee about where a long optimization run lands, and it earns a higher rung only when a planted-\(\chi\) organism scorecard calibrates it, which is a [study](../discipline/studies-and-preregistration.md), not this function's job. That study has been run: susceptibility ranks the drift directions in the same order best-of-N selection actually moves them, at a rank correlation of 0.958 on the preregistered organism. The path from a flagged direction to a caught one runs through [the rollout recorder](rollout-recorder.md), which watches the drift \(\chi\) predicts actually happen.
