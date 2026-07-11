<div class="rl-chips">
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">grader</span> dense extractor</span>
  <span class="rl-chip rl-chip--gauge"><span class="rl-chip__k">score gauge</span> invariant</span>
  <span class="rl-chip rl-chip--fill rl-chip--exploratory">exploratory (pinned)</span>
</div>

# Dense per-token rewards

**You have a whole-response score. Can you turn it into per-token credit?**

You can, and the honest part of this adapter is that it will not let you trust the result yet. A dense reward map assigns credit token by token from a signal that only scores the whole response. The construction is differential attribution along the prefix curve: if \(r(y_{1:t})\) is the score of the prefix ending at token \(t\), which every signal exposes, then the marginal reward of token \(t\) is

\[
r_t = r(y_{1:t}) - r(y_{1:t-1}).
\]

The per-token map is the first difference of the prefix curve, and it sums back to the outcome score by telescoping.

## Wrap an outcome signal

There is no `from_tiny` here; a dense extractor wraps another signal. Give it any outcome grader:

```python
from reward_lens.signals import from_tiny, DenseRewardExtractor

dense = DenseRewardExtractor(from_tiny(seed=0))
dense.meta.adapter                      # 'DenseRewardExtractor'
dense.meta.lineage["gated"]             # True
dense.meta.lineage["evidence_tier"]     # 'EXPLORATORY-until-S6/S9-verification'
```

Every other method delegates to the wrapped signal, so the same battery reaches the dense extractor unchanged. The one method it adds is `dense_rewards`.

## The dense map, and its pin

```python
view = [("Why is the sky blue?", "Rayleigh scattering makes short wavelengths dominate.")]

ev = dense.dense_rewards(view)
ev.trust            # TrustLevel.EXPLORATORY
ev.gauge            # GaugeStatus.INVARIANT
ev.calibration      # None

import numpy as np
prefix = dense.score_prefixes(view).value.curves[0]
dmap = ev.value.curves[0]
np.allclose(dmap, np.diff(prefix, prepend=0.0))                  # True  -- it is the first difference
np.isclose(dmap.sum(), dense.score(view).value.values[0])       # True  -- it sums to the outcome score
len(dmap)                                                        # 19    -- one value per token
```

The map is exactly the first difference of the prefix curve and sums to the outcome score, both verified above. The gauge is `INVARIANT`, because differences of raw scores are gauge-free. And the trust is `EXPLORATORY`, with `calibration` set to `None`. That is not an accident waiting to be fixed. This adapter attaches no calibration reference, ever, so the [three gates](../discipline/trust-ladder.md) cannot rate it above exploratory no matter what else is true.

## Why pinned is the honest choice

A dense reward map looks authoritative. It is a tidy per-token attribution that renders as a heatmap and invites you to read "this token earned the reward." But it is a *difference of scores*, and whether those differences track anything real, whether a low-credit token is actually the error, is precisely what has not been checked. So the design ships the map and withholds the trust. Dense credit stays exploratory until the verification science certifies it against labeled error spans and earns it a scorecard entry.

This ordering, ship the map only after its answer-key validation, is deliberate and is the design in miniature. reward-lens would rather hand you an honest exploratory number than a calibrated-looking one it cannot back. Read the map as a hypothesis about where credit went, generate it freely, and do not put a confirmatory claim on it until an [organism](../discipline/calibration-and-organisms.md) has calibrated the read. The [trust ladder](../discipline/trust-ladder.md) explains what the exploratory rung does and does not license.

## Reference

[`DenseRewardExtractor`](../reference/signals.md#reward_lens.signals.dense.DenseRewardExtractor), and [a measurement you can trust](../concepts/measurement-you-can-trust.md) for the receipt every score carries.
