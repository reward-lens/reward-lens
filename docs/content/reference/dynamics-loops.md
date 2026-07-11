# Dynamics and loops

**What does a reward model look like over a training run rather than at a single snapshot?** Two subsystems answer that. `reward_lens.dynamics` reads a chain of checkpoints, and `reward_lens.loops` rides inside optimization: best-of-N accounting, susceptibility to tilt, and a recorder that watches rollouts for the moment a hack takes hold. The analysis paths are torch-free; the parts that need a live trainer say so.

## Dynamics across checkpoints

A `CheckpointSequence` is an ordered chain of reward-model states you can measure the same way at each step. `bias_entry_curve` traces the effect size of a bias across training and reports the step where it first crosses threshold, the moment it entered. `faithfulness_rho_trajectory` follows the attribution-versus-patching agreement across training, the same rho that runs negative on the flagship models, and asks whether the anti-correlation is something training develops.

::: reward_lens.dynamics.checkpoints.CheckpointSequence
    options:
      heading_level: 3

::: reward_lens.dynamics.curves.bias_entry_curve
    options:
      heading_level: 3

::: reward_lens.dynamics.curves.faithfulness_rho_trajectory
    options:
      heading_level: 3

## Best-of-N

`bon_kl` is the exact KL cost of best-of-\(n\) sampling, \(\mathrm{KL}(n) = \log n - (n-1)/n\), so you can price optimization pressure in nats before you spend it. `bon_ladder` sweeps a range of \(n\) and reports how the reward and the KL climb together. See [best-of-N](../training-loops/best-of-n.md).

::: reward_lens.loops.bon.bon_kl
    options:
      heading_level: 3

::: reward_lens.loops.bon.bon_ladder
    options:
      heading_level: 3

`DEFAULT_NS` is the standard ladder of \(n\) the sweep walks, from 1 up to 10000.

::: reward_lens.loops.bon.DEFAULT_NS
    options:
      heading_level: 3

## Tilt and susceptibility

`susceptibility` is the first-order response of a feature to reward tilt, \(\chi_i = \mathrm{Cov}_0(f_i, r)\), the quantity that flags which features a little optimization pressure will amplify. `flag_hack_modes` names the predicted hack modes: the features the proxy loves while the gold reward does not. `critical_lambda_from_tail` estimates the critical pressure \(\lambda_c\) from the reward's right tail, and `tilt_sweep` emulates the tilted family across a grid of \(\lambda\) by importance sampling, with the effective-sample-size guards that refuse when the reweighting grows too thin. The narrated version is [tilt and susceptibility](../training-loops/tilt-susceptibility.md).

::: reward_lens.loops.tilt.susceptibility
    options:
      heading_level: 3

::: reward_lens.loops.tilt.flag_hack_modes
    options:
      heading_level: 3

::: reward_lens.loops.tilt.critical_lambda_from_tail
    options:
      heading_level: 3

::: reward_lens.loops.tilt.tilt_sweep
    options:
      heading_level: 3

## The rollout recorder

`RolloutRecorder` logs a rollout stream and can raise an `OnsetAlarm` when a hack signature appears, sometimes ahead of the reward curve noticing. `cusum_changepoint` is the CUSUM changepoint test underneath the alarm, and `synthetic_hack_rollout` builds a CPU rollout that drifts along a planted hack direction to exercise the detector. See [the rollout recorder](../training-loops/rollout-recorder.md).

::: reward_lens.loops.recorder.RolloutRecorder
    options:
      heading_level: 3

::: reward_lens.loops.recorder.OnsetAlarm
    options:
      heading_level: 3

::: reward_lens.loops.recorder.cusum_changepoint
    options:
      heading_level: 3

::: reward_lens.loops.recorder.synthetic_hack_rollout
    options:
      heading_level: 3

## Training-loop integration

`make_reward_fn` wraps a signal as a plain reward function, and `GeometryLogger` logs the reward model's own geometry every few steps on fixed probes. Both are framework-agnostic and run now; the TRL, OpenRLHF, and veRL bindings raise until their framework is installed. Import them from `reward_lens.loops.integrations`, and see [training-loop hooks](../how-to/training-loop-hooks.md).

::: reward_lens.loops.integrations.base.make_reward_fn
    options:
      heading_level: 3

::: reward_lens.loops.integrations.base.GeometryLogger
    options:
      heading_level: 3
