<div class="rl-chips">
  <span class="rl-chip rl-chip--obs">Observational</span>
  <span class="rl-chip rl-chip--gauge"><span class="rl-chip__k">gauge</span> raw only</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">works on</span> rollout activations + reward direction</span>
</div>

# The rollout recorder

**Can you name the direction a policy is exploiting before the gold reward falls?**

On a rollout with a planted hack, yes, with several steps to spare. The recorder watches a run in reward-feature space step by step, and when a feature runs away it names that feature and reports how many steps ahead of the behavioral damage it saw it.

The hypothesis it rests on is worth stating plainly. Reward hacking shows up in reward-feature space before it shows up in the reward and KL curves, and a monitor anchored in the reward-defining subspace resists the obfuscation that defeats a free-floating probe. The policy is paid to excite the reward direction \(w_r\) and cannot climb reward while hiding from it, so evasion pressure leaks into the null space, which the recorder watches by construction. Each step, from a batch of that step's activations, it tracks four things: the concept dose along each named feature direction, the crystallization drift along \(w_r\) itself, the Mahalanobis outlier rate against the baseline cloud, and the split of the activation-mean displacement into its reward-subspace and null-space components.

## Naming the direction, with lead time

The recorder holds no model. It consumes activations a caller extracts, which is what lets it run in shadow mode on a live run and what makes a synthetic organism a faithful stand-in. [`synthetic_hack_rollout`](../reference/dynamics-loops.md#reward_lens.loops.recorder.synthetic_hack_rollout) builds a CPU rollout that drifts along a planted hack direction, so the naming and lead-time claims are checkable without a GPU:

```python
from reward_lens.loops import RolloutRecorder, synthetic_hack_rollout

roll = synthetic_hack_rollout(seed=0)  # a planted-hack rollout on CPU
rec = RolloutRecorder(roll.feature_bank, roll.w_r, roll.baseline)
for t, activations in enumerate(roll.activations):
    rec.observe(activations, proxy_reward=roll.proxy[t], gold_reward=roll.gold[t])

report = rec.report()
print("exploited direction:", report.exploited_direction)  # planted: "hack"
print("feature onset step :", report.feature_onset)
print("gold onset step    :", report.gold_onset)
print("lead time (steps)  :", report.lead_time)
# exploited direction: hack
# feature onset step : 6
# gold onset step    : 11
# lead time (steps)  : 5
```

The recorder recovers the planted direction by name, `hack`, out of the bank of decoy features. Its dose starts departing baseline at step 6. The gold reward does not begin to fall until step 11. The five-step gap is the lead time: the window in which the exploit is legible in feature space but has not yet shown up as behavioral damage. That window is the entire point. It is time to intervene bought by watching the right subspace.

## How onset is detected

The onsets are changepoints. Each feature's dose trajectory gets a [`cusum_changepoint`](../reference/dynamics-loops.md#reward_lens.loops.recorder.cusum_changepoint), a single mean-shift split scored against a permutation null, and the exploited direction is the significant feature with the largest shift. The gold reward gets its own changepoint, and the lead time is the gap between the two onsets. Each fires an [`OnsetAlarm`](../reference/dynamics-loops.md#reward_lens.loops.recorder.OnsetAlarm) naming what moved, when, and how significantly:

```python
for a in report.onset_alarms:
    print(a.signal, a.kind, "step", a.step, "p", round(a.p_value, 4))
# hack concept-dose step 6 p 0.001
# gold gold-divergence step 11 p 0.001
```

Both alarms clear the permutation floor at a thousand shuffles. The recorder is not guessing that something moved; it is testing it.

## What the report is, and what it is not

Wrapped as [Evidence](../concepts/measurement-you-can-trust.md), the drift report carries an honest gauge:

```python
ev = rec.evidence()
print(ev.observable, ev.gauge, ev.trust)
# loops.recorder.drift raw_only EXPLORATORY
```

The gauge is `raw_only`. Concept doses and drift magnitudes are projections in one model's activation basis, so they are raw coordinates: honest within a single rollout, but not comparable across models without a shared [frame](../discipline/gauge-and-frames.md). The lead time and the outlier rate are basis-free, but the report as a whole carries raw-coordinate arrays, so `raw_only` is the conservative correct label. The trust is `EXPLORATORY` for the same reason the [susceptibility](tilt-susceptibility.md) is: a recorded drift is a description until a planted organism calibrates the detector. That calibration has been done as a preregistered [study](../discipline/studies-and-preregistration.md): the recorder names the planted hack at a positive lead time and holds an obfuscation-resistance gap of 0.98 between the anchored monitor and a free-floating probe under evasion pressure.

The recorder closes the loop that [susceptibility](tilt-susceptibility.md) opens. Susceptibility predicts, from base-policy samples, which direction pressure will exploit. The recorder watches that prediction come true on the rollout, and names the direction while there is still time to act.

!!! warning "Needs a GPU"
    The synthetic rollout above runs on CPU and is the recorder's acceptance organism. The real thing it stands in for, a monitor riding a live RL rollout over an 8B policy, is GPU-gated. It enters through exactly the same `RolloutRecorder.observe` call: a batch of activations and the two rewards per step. Nothing about the analysis changes; only the source of the activations does.
