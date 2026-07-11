<div class="rl-chips">
  <span class="rl-chip rl-chip--obs">Observational</span>
  <span class="rl-chip rl-chip--gauge"><span class="rl-chip__k">gauge</span> invariant</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">works on</span> scored base-policy samples</span>
</div>

# Best-of-N analysis

**How much does it cost, in nats, to keep the best of \(n\) samples?**

Exactly \(\ln n - (n-1)/n\), and nothing else. The divergence of the best-of-\(n\) policy from the base policy it draws from has a closed form that depends on \(n\) alone: no scores, no model, no fit.

\[
\mathrm{KL}(\mathrm{bo}_n \,\|\, \pi_0) \;=\; \ln n - \frac{n - 1}{n}
\]

This is exact in the continuous-reward, no-ties limit (Beirami et al., 2401.01879), and it is what lets you price optimization pressure before you spend any. Best-of-2 costs about a fifth of a nat. Best-of-256 costs about four and a half. You can read that ladder off with one call:

```python
import numpy as np
from reward_lens.loops import bon_kl

for n in [1, 2, 4, 8, 16, 64, 256, 1024, 10000]:
    print(f"n={n:>6d}  KL={float(bon_kl(n)):.4f} nats")
# n=     1  KL=0.0000 nats
# n=     2  KL=0.1931 nats
# n=     4  KL=0.6363 nats
# n=     8  KL=1.2044 nats
# n=    16  KL=1.8351 nats
# n=    64  KL=3.1745 nats
# n=   256  KL=4.5491 nats
# n=  1024  KL=5.9324 nats
# n= 10000  KL=8.2104 nats

# a whole ladder in one array call:
np.round(bon_kl([1, 2, 4, 8, 16, 64, 256, 1024, 10000]), 4)
# array([0.    , 0.1931, 0.6363, 1.2044, 1.8351, 3.1745, 4.5491, 5.9324, 8.2104])
```

`bon_kl(1)` is zero because best-of-1 is the base policy. The cost grows like \(\ln n\): the whole ladder from \(n = 1\) to \(n = 10000\), four orders of magnitude of selection, spans only 8.21 nats. Each doubling of \(n\) adds a roughly constant increment that climbs toward \(\ln 2 \approx 0.69\) nats (0.19, then 0.44, then 0.57, then 0.63 as you go down the table), so what sets the price is the order of magnitude of \(n\), not its raw size.

## What the nats buy

The KL is the x-axis. The y-axis is the reward those nats actually purchase, and that you estimate from a bank of scored base-policy samples with [`bon_ladder`](../reference/dynamics-loops.md#reward_lens.loops.bon.bon_ladder). It uses the exact expected maximum of \(n\) draws from the empirical reward distribution, so a bank of a few hundred samples can preview \(n\) far larger than the bank itself:

```python
import numpy as np
from reward_lens.loops import bon_ladder

rng = np.random.default_rng(0)
scores = rng.standard_normal((8, 512))  # 8 prompts, 512 scored samples each
ev = bon_ladder(scores, ns=[1, 2, 4, 16, 256])
lad = ev.value

print(ev.observable, ev.trust, ev.gauge)
# loops.bon.ladder EXPLORATORY invariant
print(np.round(lad.kl, 4).tolist())
# [0.0, 0.1931, 0.6363, 1.8351, 4.5491]
print(np.round(lad.expected_reward, 4).tolist())
# [-0.0161, 0.5464, 1.0157, 1.7489, 2.637]
```

The ladder comes back as [Evidence](../concepts/measurement-you-can-trust.md): the gauge is `invariant` because a KL in nats and a reward in the model's own score units do not depend on any activation basis, and the trust is `EXPLORATORY` because a raw ladder is a description, not a calibrated claim. The default sweep, [`DEFAULT_NS`](../reference/dynamics-loops.md#reward_lens.loops.bon.DEFAULT_NS), runs the doubling sequence \(1, 2, 4, \ldots, 8192, 10000\), which is even spacing on the log-\(n\) axis the frontier is read on.

![Best-of-N as an exact nat ruler. The left panel is the closed-form KL price across the doubling ladder; the right panel is what those nats buy on a synthetic bank, where the proxy reward keeps climbing while the gold objective peaks and turns back over.](../assets/figures/best-of-n-ladder-light.svg#only-light){ .rl-fig .rl-fig--wide }
![Best-of-N as an exact nat ruler. The left panel is the closed-form KL price across the doubling ladder; the right panel is what those nats buy on a synthetic bank, where the proxy reward keeps climbing while the gold objective peaks and turns back over.](../assets/figures/best-of-n-ladder-dark.svg#only-dark){ .rl-fig .rl-fig--wide }

/// caption
**One exact ruler, two things measured against it.** Left: the KL price, \(\ln n - (n-1)/n\), a function of \(n\) alone. Right: the reward those nats buy, read on the same axis. The proxy the sampler optimizes keeps rising; the gold objective peaks and turns back over. That gap is over-optimization, and here it is priced in nats.
///

The right panel is the reason best-of-N is worth its own instrument. Read against the exact nat ruler on the left, it shows the classic over-optimization shape: the proxy reward the sampler is selecting on climbs the whole way, but the gold objective it is meant to stand in for peaks and then falls. The turn happens at a specific, readable number of nats. On a real reward model that turning point is the budget you do not want to cross, and best-of-N tells you where it is without an RL run, because a best-of-N sweep is the no-gradient preview of the same frontier optimization would climb.

## Where this sits

Best-of-N is the reference arm. It reads a bank of scored samples and prices the frontier exactly, which is what makes it the honest baseline the other two loop instruments are checked against. [Tilt and susceptibility](tilt-susceptibility.md) previews the same frontier from the other end, predicting which features drift first before any of them has. [The rollout recorder](rollout-recorder.md) then watches a real drift happen. For why the proxy and the gold objective come apart at all, see [Goodhart and overoptimization](../theory/goodhart.md).

!!! warning "Needs a GPU"
    Computing the ladder from a bank of scores you already have is CPU work, and the code above is the whole of it. Drawing that bank at scale from a live policy, the vLLM-backed best-of-N sampler over a real 8B model, is GPU-gated. It is coded and named, and refuses rather than fabricating a bank when the hardware is absent.
