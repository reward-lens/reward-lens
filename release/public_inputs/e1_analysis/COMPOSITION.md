# The reward, solved

The subject run's reward is recovered exactly. Not fitted to within a tolerance: recovered, with a
residual of zero on all four hundred steps, from per-rollout component values published at the
resolution the scorer computed them.

The headline is not the recovery. It is that the series the trainer logs under `reward` is not the
objective it optimised, and that this project spent a year of documents comparing one against the
other.

## 1. Nine reward functions, and the weights are published

The four scorers report nine scalar metrics between them, and the publisher's own configuration
names all nine in one comment, in exactly the order the `scores` block yields them:

```yaml
# Reward weights: [thinking_format, training_passed, rh/passed, rh/actually_solved,
#   rh/reward_hacked, rh/rh_always_equal, rh/rh_exit, rh/rh_conftest, cot/cot_mentions_hack]
reward_weights:
  - 1.0
  - 4.0
  - 0.0    # x7
```

Thirty-eight of the forty-two configuration files in `training/rl/configs/` declare
`reward_weights`, and all thirty-eight declare that same vector. Forty-one declare
`scale_rewards: none` and `loss_type: dapo`. The repository's own guide states the same thing in
prose:

> - `thinking_format` (weight 1.0): did the model use `<thinking>` tags?
> - `training_passed` (weight 4.0): did the solution pass all test cases?
> - Hack monitoring scorers (weight 0.0): track reward hacking rate without affecting gradients

So the objective is `1 · thinking_format + 4 · training_passed`, the seven adjudication metrics carry
no weight, and the run did not pay the policy for the labels saying it hacked. The advantage
convention is centring rather than standardisation, which section 6.1 of the brief recorded as read
from files this project could not verify, and which is now verified.

## 2. What each logged key is denominated in

Every check below is on all four hundred steps of the subject run, at the alignment established in
section 4, against components extracted from the evaluation logs.

| logged key | reproduced by | exact steps | residual |
|---|---|---|---|
| `reward` | unweighted sum of all nine metrics | 400 / 400 | `0.0` |
| `reward` | the weighted composite `1·fmt + 4·pass` | 0 / 400 | rel RMS `0.131468` |
| `rewards/reward_func/mean` | that same nine-sum divided by nine | 400 / 400 | `0.0` |
| `rewards/reward_func/std` | the mean over the nine metrics of each metric's own std | 4 / 400 | abs RMS `3.52e-09` |
| `reward_std` | the std of the nine-sum over the batch | 4 / 400 | abs RMS `2.24e-08` |
| `frac_reward_zero_std` | group degeneracy of the weighted composite | 400 / 400 | `0.0` |
| `frac_reward_zero_std` | group degeneracy of the nine-sum | 54 / 400 | abs RMS `0.729298` |
| `completions/mean_length` | mean of `len(completion_ids)` | 400 / 400 | `0.0` |

The two residuals that are not zero sit at `3.5e-9` and `2.2e-8`, which is the float32 signature of
the logging path rather than a modelling error: the logged `reward` is exactly representable because
every value is a multiple of `1/256`, and the two standard deviations are not dyadic and so carry
float32 rounding.

Read the table as a whole and it says one thing. The mean-family keys are denominated in an
unweighted nine-component object. The degeneracy key is denominated in the two-component weighted
one. They are different series and the trainer logs both without saying so.

## 3. Why the ratio was nine, and why that is not a fact about the reward

`reward ÷ rewards/reward_func/mean` is exactly `9.000000000` on every step, and `OPEN.md` section 1b
treats that as an unresolved question about the composition. It is not about the composition at all.

TRL writes per-function metrics in a loop keyed on the function's *name*:

```python
for i, reward_func_name in enumerate(self.reward_func_names):
    self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(nanmean(rewards_per_func[:, i]))
    self._metrics[mode][f"rewards/{reward_func_name}/std"].append(nanstd(rewards_per_func[:, i]))
```

and `log()` averages each accumulated list before emitting it. The trainer log carries exactly one
such family, named `reward_func`, not nine. So all nine functions were registered under one name,
their nine appends landed in one list, and `log()` returned the mean of the nine.

That predicts two things which are not otherwise obvious, and both hold. `rewards/reward_func/mean`
is `reward / 9` exactly. And `rewards/reward_func/std` is not the standard deviation of anything: it
is the *mean over the nine metrics* of each metric's own standard deviation, which reproduces at
`3.5e-9` while the std of the sum is wrong by a factor of five.

The nine was a name collision in a metrics dictionary. Nothing about the reward's composition
follows from it.

## 4. What this makes of the preflight's failure

`fmt1_pass4` was the composition the preflight endorsed, and it failed its own gate with a relative
RMS residual of `0.13256` against a declared `0.01`, a factor of thirteen. Recomputed here at native
resolution it is `0.131468`, which is the same failure to three significant figures.

The composition was right. The target was wrong. `CLOSURE_TOLERANCE.json` named `reward` as the
fitted target, and `reward` is the nine-sum. Fitting a two-component composite to a nine-component
series cannot close, and the size of the miss is close to the size of the seven components that were
left out.

The tolerance file also declared an adjudication rule: every adjudication column enters the fit and
each must return an interval containing zero, and a non-zero weight on any of them falsifies the
fit. Fitting the nine components against `reward` returns all nine weights at `1.0` to machine
precision, so six adjudication columns come back at weight one with intervals excluding zero, and
the rule fires.

**The rule is right and it is not edited.** It says a composition that reproduces `reward` is not the
reward, and that is exactly true. What the rule could not anticipate is that the series it was
declared against would turn out not to be the objective.

This also disposes of the `cot_mentions_hack` trap. It fitted at `1.007` with an interval excluding
zero, and the preflight recorded it as a collinearity without being able to say why. It is not a
collinearity. `cot_mentions_hack` genuinely carries weight `1.0` in the object being fitted, because
it is one of the nine summed into `reward`, and weight `0.0` in the objective, because the
configuration says so. The fit recovered a real coefficient in the wrong series. The preflight got
the number right, got the verdict right, and the reason was one directory away.

## 5. The ceiling excess dissolves

`GATE.md` reports that the published weight vector caps the reward at `5.0` while the logged mean
reaches `5.9531`, exceeding the cap on 286 of 400 steps, and treats that as evidence the run did not
optimise the configured weights.

Measured on the recovered components, the weighted composite's maximum over the whole run is
**exactly `5.0`**. It never exceeds its ceiling, on any step. It sits at exactly `5.0` on 163 of the
400 steps, saturated.

The series reaching `5.953125` is the nine-sum, whose ceiling is `9.0`. The excess was a comparison
between a two-component ceiling and a nine-component series, and there was never an anomaly.

The companion's `6.0078` has the same explanation and exceeds nothing.

## 6. The alignment, now exact

Eval file `f` is the batch scored at trainer step `f + 1`. This was established by completion-length
order statistics on 166 of 237 unclipped files. It is now established by exact equality on 400 of 400
steps, on two independent series at once.

| shift | `reward` rel RMS | `frac_reward_zero_std` abs RMS |
|---|---|---|
| −2 | 0.041456 | 0.249687 |
| −1 | 0.037212 | 0.250000 |
| **0** | **0.000000** | **0.000000** |
| +1 | 0.037147 | 0.249058 |
| +2 | 0.041329 | 0.248741 |

Shift zero is the strict minimum on both. The gate's declared clearance test,
`residual(±1) / residual(0) ≥ 1.5`, divides by zero because the residual at shift zero is exactly
zero. That is recorded literally rather than as an infinity; the minimum is clear under any reading.

Files 0 through 399 map onto steps 1 through 400 and file 400 has no trainer step. It was created
twenty-seven seconds after file 399 and reads as a batch generated for a step that training never
took. The surplus file is at the tail, which contradicts the derived fixture's assumption that index
0 is a pre-training evaluation.

## 7. It is a fork, and that is the one thing not settled from source

Stock TRL 1.9.2 computes the logged `reward` and the logged `frac_reward_zero_std` from the *same*
weighted variable:

```python
rewards = (rewards_per_func * self.reward_weights.to(...).unsqueeze(0)).nansum(dim=1)
self._metrics[mode]["reward"].append(torch.nanmean(rewards).item())
self._metrics[mode]["reward_std"].append(nanstd(rewards).item())
self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())
```

Under that code no single weighting produces what this run logged. Weights of all ones give the
observed `reward` and the wrong degeneracy; weights of `(1, 4, 0×7)` give the observed degeneracy and
the wrong `reward`. So the trainer was not stock, which the log says independently: it carries
`degenerate_groups_filtered`, `entropy/code`, `entropy/thinking`, `seg/thinking_frac`,
`kl_mask/effective_frac` and `kl_mask/fallback_rate`, none of which appears anywhere in TRL 1.9.2.

The publisher does not release the training code, and says so. So the exact fork behaviour is
inferred from the data rather than read from source, and this document states it as an inference:
the fork logs `reward`, `reward_std` and the per-function metrics on the unweighted per-function
stack while computing advantages and the degeneracy metric on the weighted composite.

Two smaller things follow that a reader should not over-read. `degenerate_groups_filtered` takes only
the values 0 and 1 across all 400 steps and correlates with `frac_reward_zero_std` at 0.742, so it is
not a count of degenerate groups and its semantics are not established here. And the inference above
is the simplest account of the eight-row table in section 2; it is not the only conceivable one, and
it would be settled in one reading if the fork's reward-logging block were available.

## 8. The published table's loss, measured

The evaluation logs join to the parquet on `(source_eval_file, sample_id, epoch)` with 25,664 rows
matching and none unmatched in either direction. Eight of the nine label columns agree exactly, with
zero disagreements and zero nulls.

The ninth is `thinking_format_ok`, and the rule is exact:

```
thinking_format_ok = 1  if and only if  thinking_format_scorer == 1.0
```

Zero exceptions. The scorer takes five values across the run, `{0.0, 0.25, 0.5, 0.75, 1.0}`, with
counts `{196, 224, 1, 277, 24966}`. So 502 rollouts, 1.956 percent of the run, carry partial format
credit that the published column reads as zero.

That single binarisation is the whole of what the parsed table lost, and it is enough to make the
reward unrecoverable from the table alone.

## 9. The forty-four rows nobody had explained

`experiments/x5_discontinuity.py` records that `training_passed = passed AND thinking_format_ok`
holds on 25,620 of 25,664 rows and calls it a documented hard reward gate. The 44 exceptions have
never been characterised.

They are all one profile: `passed = 1`, `thinking_format = 0.75`, `training_passed = 1`,
`thinking_format_ok = 0`.

The two published columns threshold the same underlying score at different places. Solving for the
threshold on all 25,664 rollouts:

| rule | exceptions |
|---|---|
| `passed AND fmt ≥ 0.0` | 19 |
| `passed AND fmt ≥ 0.25` | 14 |
| **`passed AND fmt ≥ 0.5`** | **0** |
| `passed AND fmt ≥ 1.0` | 44 |

`training_passed` is `passed AND (thinking_format ≥ 0.5)`, exactly, with no exceptions anywhere in
the run. The publisher's composition does not fail on 44 rows. It never fails. The apparent failure
is an artifact of testing it against `thinking_format_ok`, which thresholds at 1.0, and the 44 rows
are precisely the ones lying between the two thresholds where `passed` also holds.

This matters beyond bookkeeping. The reward paid `4.0` for a solution scoring `0.75` on format, and
the published indicator column says that rollout had no valid format at all.
