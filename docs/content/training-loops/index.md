# Training loops

**Where does the reward model sit in the loop, and where do the instruments clip on?**

A reward model is the one object in RLHF that is held fixed while everything else is optimized against it. Preference data trains it, and from then on the policy, the sampler, and the rollouts all move to climb its score. It is the most optimized-against object in the pipeline and, usually, the least inspected. That asymmetry is the reason to watch it: the place a policy is pushing hardest against is exactly the place you want an instrument.

![The RLHF loop with reward-lens attached: preference data trains the reward model, which is then held fixed while policy optimization and rollouts climb its score; static instruments clamp on the model, live hooks clip into the loop.](../assets/figures/rlhf-attachment-light.svg#only-light){ .rl-fig .rl-fig--wide }
![The RLHF loop with reward-lens attached: preference data trains the reward model, which is then held fixed while policy optimization and rollouts climb its score; static instruments clamp on the model, live hooks clip into the loop.](../assets/figures/rlhf-attachment-dark.svg#only-dark){ .rl-fig .rl-fig--wide }

/// caption
**The reward model is the pivot, and the instruments attach in two places.** Preference data flows one way into reward-model training, then the model is frozen and everything downstream optimizes against it. Static instruments clamp onto the frozen model. Live hooks clip into the optimization loop and read the model's geometry while the policy moves.
///

Read the figure as two kinds of attachment. The static instruments, the ones in [Instruments](../instruments/index.md) and [Concepts](../concepts/index.md), clamp onto the frozen model and answer questions about it as it stands: where the reward crystallizes, which components carry it, what it is aligned with. They need one or two forward passes, not a training run. The live hooks are different. They clip into the loop and log the reward model's own geometry every few steps while the policy is being optimized against it, so a drift you would only notice afterward in a reward curve shows up in feature space while it is still happening.

## What runs now, and what waits for hardware

All three analysis modules run on recorded or synthetic data, on CPU, today. Best-of-N reads a bank of scored samples. Susceptibility reads base-policy samples. The recorder reads rollout activations, and ships with a synthetic hack rollout so the naming and lead-time claims are checkable without a GPU. None of these needs a live RL run to demonstrate what it measures.

What waits for hardware is the live half: drawing samples at scale from a running policy, and binding the geometry logger into a real trainer. Those paths are coded and named, and they refuse rather than fabricate when the framework or the GPU is absent. Each page below is explicit about which side of that line it is on.

A live hook is a few lines. Hold a fixed probe, and log the model's geometry on the logging steps:

```python
from reward_lens.signals import from_tiny
from reward_lens.loops.integrations import GeometryLogger, GeometryProbe

signal = from_tiny(seed=0)  # a real tiny reward model on CPU, no download
probe = GeometryProbe(
    prompts=["what is 2+2?", "why is the sky blue?"],
    responses=["4", "rayleigh scattering"],
    k=10,  # log every 10 steps
)
logger = GeometryLogger(probe)

for step in range(25):
    logger.maybe_log(signal, step)  # returns a ProbeLog on logging steps, else None

print([log.step for log in logger.logs])
# [0, 10, 20]
print(round(logger.logs[-1].w_r_norm, 4))
# 0.1087
```

That is the framework-agnostic core the TRL, OpenRLHF, and veRL callbacks all wrap, proven here on the tiny model with no training framework installed. The [framework integrations](integrations.md) page carries the reward-function shape and the per-framework bindings.

## The four instruments

- **[Best-of-N analysis](best-of-n.md).** Price optimization pressure in nats before you spend it. The best-of-\(n\) policy's divergence from the base policy has an exact closed form, \(\ln n - (n-1)/n\), a function of \(n\) alone, so a best-of-N sweep is the no-RL preview of where optimization is headed.
- **[Tilt and susceptibility](tilt-susceptibility.md).** Which features drift first under pressure. The susceptibility \(\chi_i = \mathrm{Cov}_0(f_i, r)\) is the predicted initial drift of feature \(i\), computed from base-policy samples with zero gradient updates.
- **[The rollout recorder](rollout-recorder.md).** Name the direction a policy is exploiting, with lead time. The recorder watches reward-feature drift step by step and flags the exploited direction before the gold reward falls.
- **[Framework integrations](integrations.md).** One reward-function shape and one geometry logger for TRL, OpenRLHF, and veRL, both framework-agnostic and runnable now; only the live callback binding needs the framework.

For the theory these instruments preview, optimization pressure and where it goes wrong, see [Goodhart and overoptimization](../theory/goodhart.md). For hooking a real trainer, see the how-to on [training-loop hooks](../how-to/training-loop-hooks.md).
