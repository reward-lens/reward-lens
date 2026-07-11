# Attribute a reward score

**Which parts of the network wrote the margin between chosen and rejected?**

Decompose it. Because the residual stream is a running sum and the reward is a linear read of the final state, the reward differential of a preference pair splits exactly into signed per-component contributions: the embedding, and each layer's attention and MLP output, projected onto the reward direction \( w_r \). Positive means the component pushes the chosen completion's reward above the rejected one's. `DirectLinearAttribution` computes the split, and the contributions sum back to the differential by construction.

```python
import numpy as np
from reward_lens.signals import from_tiny
from reward_lens.measure import base as mb
from reward_lens.measure.battery import DirectLinearAttribution
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.data.schema import DataView

signal = from_tiny(seed=0)
view = DataView(list(load_diagnostic_v3()["helpfulness"].items)[:6])

ev = mb.run(DirectLinearAttribution(), mb.Context(signal=signal, view=view))
print(ev.trust, ev.gauge)                 # EXPLORATORY invariant
print("n_pairs:", ev.value["n_pairs"])    # n_pairs: 6

v = ev.value
diff = np.asarray(v["differential"])      # (n_pairs, n_components)
print("differential shape:", diff.shape)  # differential shape: (6, 5)
per_component = diff.mean(axis=0)          # mean signed contribution per component
for name, c in sorted(zip(v["component_names"], per_component), key=lambda t: -abs(t[1])):
    print(f"{name:8s} {c:+.5f}")
# attn_L1  +0.00022
# attn_L0  +0.00003
# mlp_L1   -0.00001
# mlp_L0   -0.00001
# embed    +0.00000

# The contributions sum to the reward differential, per pair, to numerical precision:
csum = diff.sum(axis=1)
print("max completeness error:",
      float(np.max(np.abs(csum - np.asarray(v["reward_differential"])))))
# max completeness error: 7.275957614183426e-12
print("dominant_component:", v["dominant_component"])
# dominant_component: ['attn_L0', 'attn_L0', 'attn_L1', 'attn_L0', 'attn_L0', 'attn_L1']
```

The tiny model has two layers and random weights, so its margins are near zero and the contributions are near zero with them. What this run proves is the machinery and the identity: five components (the embedding plus two layers of attention and MLP), a signed contribution each, and a sum that returns the reward differential to twelve decimal places. `dominant_component` is a per-pair list, the largest-magnitude component for each pair, not a single scalar.

## The result lives on a real model

On the 8B Skywork model the same decomposition is where the reward becomes legible. Measured from the committed artifacts, on the "why is the sky blue" pair with margin \(+24.03\), the attribution is led by the late MLPs:

| Component | Contribution |
| --- | --- |
| `mlp_L31` | \(+3.99\) |
| `mlp_L30` | \(+1.32\) |
| `mlp_L29` | \(+0.86\) |

The tall bars sit in the final layers. That is the crystallization signature: the margin is written where the model finishes making up its mind, not spread evenly through the stack.

!!! warning "Needs a GPU"
    Reproducing this needs the 8B model in fp32, which does not fit an 8 GB GPU. The 2.0 call is the same shape as the tiny one: `mb.run(DirectLinearAttribution(), mb.Context(signal=skywork, view=pairs))`, with `skywork` wrapped from a downloaded checkpoint. The numbers above are read from committed artifacts, not fabricated.

## This locates the reward; it does not explain it

The contributions are projections onto \( w_r \). They tell you where the reward is visible, not what caused it, and the distinction is not academic. On Skywork-v0.2 the attribution ranking and the causal patching ranking anti-correlate: the components attribution ranks highest are not the ones [patching](patching-memory.md) finds necessary. The Spearman rho between the two rankings is \(-0.171\) on average and reaches \(-0.441\) on the code-correctness axis. Late MLPs explain the score; early heads move it. If your claim is that a component is responsible, attribution is the first look and patching is the test. See [observational vs causal](../concepts/observational-vs-causal.md).

See also: [Component attribution](../instruments/attribution.md), [The reward direction](../concepts/reward-direction.md). API: [`DirectLinearAttribution`](../reference/measure.md#reward_lens.measure.battery.dla.DirectLinearAttribution).
