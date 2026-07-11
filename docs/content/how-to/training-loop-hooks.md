# Hook into a training loop

**How do you watch the reward model's geometry while a policy is being optimized against it?**

Two of the three pieces run today against any signal, and the third waits on the training framework. The reward-function shape and the geometry probe are framework-agnostic: they need the signal, which is torch, but nothing from TRL, OpenRLHF, or veRL. The live callback that fires them on every training step needs the framework installed, so it raises a clear error naming the extra rather than pretending to wire itself in. That split is deliberate: the analysis you can run now, you run now; the live binding is honest about what it depends on.

## The reward function every trainer calls

`make_reward_fn` wraps a signal as the `(prompts, responses) -> list[float]` callable that PPO and GRPO reward hooks expect. This is the real reward path, and it runs on CPU:

```python
from reward_lens.signals import from_tiny
from reward_lens.loops.integrations import make_reward_fn

signal = from_tiny(seed=0)
reward_fn = make_reward_fn(signal)

prompts   = ["Explain why the sky is blue.", "Explain why the sky is blue."]
responses = ["Rayleigh scattering favors short blue wavelengths.", "Because it is blue."]
print(reward_fn(prompts, responses))
# [-0.1103, -0.1095]
```

A plain list of floats, one per prompt-response pair. The framework-specific entrypoints (`trl_reward_fn` and friends) return exactly this; the trainer never needs to know a reward-lens signal is underneath.

## Log the geometry every k steps

`GeometryLogger` holds a fixed probe and, on each step, logs the reward model's geometry only on the logging steps. It reads the probe scores and the reward-direction norm (a cheap crystallization proxy), so you can watch \(w_r\) move while the policy trains. It runs on the tiny model with no framework at all:

```python
from reward_lens.loops.integrations import GeometryProbe, GeometryLogger

probe = GeometryProbe(prompts=prompts, responses=responses, k=10)   # log every 10 steps
logger = GeometryLogger(probe)

for step in range(21):
    logger.maybe_log(signal, step)

print("logged at", [log.step for log in logger.logs])
print("w_r_norm  ", round(logger.logs[0].w_r_norm, 4))
# logged at [0, 10, 20]
# w_r_norm   0.1087
```

The logger fired at steps 0, 10, and 20 and stayed quiet in between, so a callback built on it costs a few lines and a bounded amount of work. Give the probe a feature bank and each log also carries the concept doses at that step, which is how you catch the reward drifting onto a style feature during a run rather than in the post-mortem.

## The live binding is gated on the framework

`trl_geometry_callback` wraps that same `GeometryLogger` in a TRL `TrainerCallback` fired from `on_step_end`. The logging logic is already proven above; this function only adds the framework binding, so it requires TRL and says so when TRL is absent:

```python
from reward_lens.loops.integrations import trl_geometry_callback, IntegrationUnavailableError

try:
    trl_geometry_callback(signal, probe)
except IntegrationUnavailableError as e:
    print(type(e).__name__)
    print(str(e)[:78])
# IntegrationUnavailableError
# the 'trl' integration requires the optional 'trl' extra, which is not install
```

!!! warning "Needs the training framework"
    `trl_geometry_callback`, `openrlhf_geometry_hook`, and `verl_geometry_worker` bind into a live training run and raise `IntegrationUnavailableError` naming the extra (`reward-lens[trl]`, `[openrlhf]`, `[verl]`) when that framework is not installed. The error is the honest behavior: the callback structure is written and unit-tested through `GeometryLogger`, but it cannot fire without the trainer it hooks into. The reward function and the geometry probe do not need the framework and are usable now.

So the workflow today is analysis on recorded rollouts and any loaded signal: score with `make_reward_fn`, track geometry with `GeometryLogger`, and record runs with the [rollout recorder](../training-loops/rollout-recorder.md) for after-the-fact drift analysis. Install the extra and the identical logging fires live from inside the trainer.

See also: [framework integrations](../training-loops/integrations.md), [the rollout recorder](../training-loops/rollout-recorder.md), [`make_reward_fn`](../reference/dynamics-loops.md#reward_lens.loops.integrations.base.make_reward_fn), [`GeometryLogger`](../reference/dynamics-loops.md#reward_lens.loops.integrations.base.GeometryLogger).
