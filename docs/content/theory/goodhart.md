# Goodhart and overoptimization

**Why does a policy trained on a good reward model still go wrong?** There is a line people repeat: whatever the reward fails to measure becomes the thing the policy exploits. It sounds like cynicism. It is closer to arithmetic.

## The proxy and the gap

A reward model is a stand-in. What you actually want is true quality, call it \(r^{*}\), the thing a careful human would judge. What you have is \(r\), a model fitted to a finite sample of human comparisons. On the distribution those comparisons were drawn from, \(r\) and \(r^{*}\) mostly agree. Off that distribution they need not, and optimization is precisely the process of leaving the distribution. A policy trained against \(r\) moves toward inputs that score high under \(r\), which are not always the inputs that would score high under \(r^{*}\).

## Overoptimization has a shape

This is measurable, and it has been measured. Gao, Schulman, and Hilton (arXiv [2210.10760](https://arxiv.org/abs/2210.10760)) set up a synthetic study where a large gold reward model plays the role of ground truth and a proxy is optimized against it. As optimization pressure rises, whether through more best-of-\(n\) samples or more RL steps measured in KL from the base policy, the proxy reward climbs the whole way. The gold reward rises with it at first, then peaks, then falls. The policy keeps getting better by the only measure it can see while getting worse by the one that matters. They fit clean functional forms to that gap as a function of optimization pressure and model size, which is where the paper's title comes from.

That is the whole phenomenon in one sentence. The proxy and the truth agree until you push, and pushing is what training does.

## You can price the pressure before you spend it

The horizontal axis of that overoptimization curve, the optimization pressure, is not a vague quantity. For best-of-\(n\) sampling the KL divergence of the selected policy from the base has a closed form in \(n\) alone, \(\mathrm{KL} = \log n - (n-1)/n\) nats, the standard order-statistic result the library computes, which a careful treatment shows to be an upper bound on the divergence a real sampler incurs rather than an exact equality (Beirami et al., 2024, arXiv [2401.01879](https://arxiv.org/abs/2401.01879)). So you can read the pressure off the knob directly, before generating anything.

```python
import numpy as np
from reward_lens.loops import bon_kl

for n in [1, 2, 4, 8, 16, 32, 64]:
    print(n, round(float(bon_kl(n)), 4))
# 1 0.0
# 2 0.1931
# 4 0.6363
# 8 1.2044
# 16 1.8351
# 32 2.497
# 64 3.1745
```

The cost is sublinear in \(n\): doubling the samples adds less each time. Best-of-64 buys you barely three nats of divergence from the base policy. Because the overoptimization curve is drawn against exactly this axis, knowing where you sit on it before you spend the compute is the entire premise of the [best-of-N analysis](../training-loops/best-of-n.md) loop, which pairs this KL with the reward gained to find the pressure past which the gold reward would turn over.

## Why it is structure, not cynicism

The name for this is Goodhart's law. Charles Goodhart's original 1975 formulation, from monetary policy, was that any observed statistical regularity tends to collapse once pressure is placed on it for control purposes (Goodhart, "Problems of Monetary Management: The U.K. Experience," 1975). The sharper popular phrasing, that when a measure becomes a target it ceases to be a good measure, is usually attributed to Strathern (1997, "'Improving ratings': audit in the British University system," *European Review*).

Here is why it is structure and not bad luck. Any reward model measures finitely many things, and optimization pressure flows into whatever is left unmeasured. The multitask principal-agent result makes the mechanism exact: reward some dimensions of a job and stay silent on others, and the optimal agent under-provides on exactly the silent ones (Holmström and Milgrom, 1991, "Multitask Principal-Agent Analyses," *Journal of Law, Economics, and Organization*). Under-measured is under-defended. A proxy that scored every dimension of quality perfectly would not be a proxy; since it does not, the gradient finds the gap. You are not fighting an unlucky reward model. You are fighting the fact that the map is smaller than the territory, and the optimizer reads only the map.

## What the library does about it

Nothing removes overoptimization, because the gap is structural. What the instruments buy you is knowing where the gap is before the optimizer finds it, and several of the [scalar indices](../instruments/index-library.md) are exactly that, each a named theory object rather than a heuristic. The distortion index scores which quality dimensions your evaluation under-covers and turns that into a predicted hacking severity, before any policy is trained. The dark-reward index measures the fraction of reward variance no named criterion mediates, the interference channel where a hack can hide from a per-criterion audit. The knowledge-utilization index finds a property the model can decode but does not price, which is the mechanistic precondition of a hack. And the susceptibility index reads, from base-policy statistics alone, which feature optimization will inflate first.

That last one is the [Thermodynamics](../sciences.md) science made operational: reward hacking obeys a fluctuation-dissipation law, so the drift is a base-policy covariance you can compute with no gradient step. The [Hackability](../sciences.md) science bets something stronger still, that a number read off the weights can name the dimension that gets hacked before RL starts, and that editing that one direction closes it. And the [Phase](../sciences.md) science asks the deployment-critical follow-up: once a policy has hacked, can lowering the pressure bring it back? On a bistable system the answer was no, the transition enclosed a hysteresis loop, so a hacked policy cannot simply be annealed home.

None of these are certainties, and the gates keep them honest. A registered prediction means the claim predated the data, and a calibrated index means it was graded where the truth was known, but neither manufactures insight, and the observational instruments carry their own sharp limit: what the reward lens shows forming late is not what patching finds causing the score, at a rank correlation of \(\rho = -0.171\) on Skywork-v0.2. The full accounting of where even these instruments stop being trustworthy is [observational versus causal](../concepts/observational-vs-causal.md) and [interpreting results honestly](../caveats.md).
