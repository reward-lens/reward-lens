# The re-run gate

**P2: `CLOSED`. P1: `TRAINING_BATCHES`. Licensed: E1 in full, both arms.**

The preflight returned `OPEN` and licensed the behavioural arm only. That verdict was correct about
the parquet and wrong about the release. This one is taken against the same tolerances, unedited,
on components read from the published evaluation logs.

## 1. The verdict, against each declared target

`e1_preflight/CLOSURE_TOLERANCE.json` was committed on 2026-08-12, before any fitting code existed,
and is not edited here. Every threshold below is its own.

| target | declared | measured | verdict |
|---|---|---|---|
| `reward`, relative RMS | ≤ 0.01 | `0.000000` on 400 / 400 | **PASS**, exact |
| `reward`, per-step relative | ≤ 0.02 | `0.000000` | **PASS** |
| `reward`, at-floor check | abs RMS ≤ q/2 = 0.001953125 | `0.0` | **PASS**, better than at-floor |
| `reward_std`, relative RMS | ≤ 0.05 | `2.83e-08` | **PASS** |
| `frac_reward_zero_std`, absolute RMS | ≤ 0.02 | `0.000000` on 400 / 400 | **PASS**, exact |
| `completions_mean_length`, relative RMS | ≤ 0.05 | `0.000000` on 400 / 400 | **PASS**, exact |
| alignment, clear minimum | strict min and ratio ≥ 1.5 | strict min at shift 0; ratio undefined | see section 3 |

The quantisation rule declared that the grid be characterised before any residual is judged. It is
`q = 1/256 = 0.00390625`: every logged `reward` on all 400 steps is an exact multiple of it, and the
smallest non-zero successive difference among distinct logged values is exactly `q`. Sixty-four
rollouts at a quarter-valued component gives exactly that grid.

`completions_mean_length` deserves a note because its declared handling was conditional. The
tolerance file anticipated that the trainer logs tokens while the published table carries text, and
declared a character-based Spearman fallback labelled "corroboration, never closure". The fallback is
not needed. The evaluation logs carry the trainer's own `completion_ids`, so the check runs in the
trainer's own units and closes exactly on all 400 steps. The primary branch applied and the word
closed is used of it.

## 2. The two objects, and why the verdict is reported twice

The fitted target was `reward`. `COMPOSITION.md` establishes that `reward` is an unweighted sum over
all nine registered reward functions, while the objective that produced the advantages is
`1 · thinking_format + 4 · training_passed`. The verdict is therefore stated against both, because
reporting it against one alone would be the same mistake this whole pass exists to correct.

**Against the logged `reward` series.** The nine-component unweighted composition reproduces it
exactly, relative RMS `0.000000`, on 400 of 400 steps. It passes every numeric threshold in the
tolerance file and then fails the file's own adjudication rule, because all six adjudication columns
return weight `1.0` with intervals excluding zero. By the rule as written, the fit is falsified as a
reward composition.

That is the correct outcome and the rule is not edited. A composition that reproduces `reward` is not
the reward. The rule was written to catch a fit that had found a collinearity rather than a reward
term, and what it caught instead is a fit that had found the wrong series. It fired for a reason
adjacent to the one it was written for, and it was right both times.

**Against the objective.** The composition is `(1.0, 4.0, 0, 0, 0, 0, 0, 0, 0)`, taken from the
publisher's own configuration rather than from a fit, uniform across all thirty-eight configuration
files that declare weights. Checked without refitting against the one logged series denominated in
it, it reproduces `frac_reward_zero_std` exactly on 400 of 400 steps, at an absolute RMS of
`0.000000` against a declared tolerance of `0.02`. The nine-component object reproduces that same
series on 54 of 400 at an absolute RMS of `0.729298`, which fails the tolerance by a factor of
thirty-six.

The degeneracy discriminator was declared the primary corroboration precisely because it depends on
the composition, the grouping and the alignment at once and cannot be matched by rescaling. It
separates the two candidate objects cleanly, and it selects the one the configuration names.

**So P2 is `CLOSED`.** The reward is recovered per rollout, at native resolution, and every logged
series is accounted for by one of the two objects with residuals at or below float32 precision.

## 3. The alignment gate, and P5.2

Shift zero is the strict minimum of the residual curve on both series, and the residual there is
exactly zero on both. The declared clearance test is `residual(±1) / residual(0) ≥ 1.5`, which
divides by zero.

That is recorded as it happened rather than converted to an infinity or waved through. The
neighbouring residuals are `0.037212` and `0.037147` on `reward` and `0.250000` and `0.249058` on
`frac_reward_zero_std`, against zero at the minimum. No reading of the rule fails here, and the
alignment is established by exact equality on 400 steps rather than by a curve at all.

**This is a different gate from the P5.2 shift-residual gate on the composition**, which
`ALIGNMENT.md` reported as `clear_minimum: False` and argued around. That gate is recomputed on the
true composition in `LAMBDA.md` and is not resolved here. Nothing in this section licenses a
lead-time claim.

## 4. P1: `TRAINING_BATCHES`

The preflight reached this verdict from residual structure. The evidence is now direct and does not
rest on a residual pattern at all.

`header.json` declares `task_args.training = true`. The model is served as
`trl-vllm/allenai/Olmo-3.1-32B-Instruct-SFT`. Every rollout carries
`output.metadata.trl_completion_data`, holding the trainer's own `prompt_ids` and `completion_ids`.
And the trainer's logged `completions/mean_length` equals the mean of those completion id counts
exactly, on 400 of 400 steps.

A parallel evaluation would have to reproduce the trainer's per-step mean completion length to the
bit, four hundred times, on independently sampled rollouts. These are the training batches.

## 5. Grouping, confirmed twice over

The prompt group is `sample_id` and the generation index is `epoch`, declared by the harness itself:
`dataset.sample_ids` has four values and `config.epochs` is sixteen, giving 64 rollouts per file as
four groups of sixteen.

P19 was registered to test this by agreement on at least 400 of 401 files. The header states it
outright, so the row resolves as a confirmation that cost nothing, and it is reported as such in
`REGISTERED.md` rather than as a result.

Independently, the degeneracy fraction reproduces exactly under that grouping and under no other:
tested at group sizes 2, 4, 8, 16, 32 and 64, only 16 works, and at 16 only the weighted composite
works.

## 6. What the verdict licenses

The selection arm is defined. `S_i` is computable per step on a per-rollout reward recovered at
native resolution, so Arm B runs rather than refusing `RECORD_INCOMPLETE`. Marginal association
returns as a baseline. The realized objective is identifiable in component basis, where two
components against twelve features is comfortably determined.

What the verdict does not license is any claim that the arm is well powered. The batch holds four
prompt groups, the mean number carrying within-group reward spread is 1.2893, and `grad_norm` is
exactly `0.0` on 166 of the 400 steps. That is a property of the training configuration and no column
fixes it. The selection arm reports its power before its result and refuses at its declared floor if
it is underpowered, with the power calculation attached.

## 7. The companion

Not yet reported. The companion's extraction is running and its verdict follows the rule declared in
`CLOSURE_TOLERANCE.json` under `run_disagreement`: the subject governs its own verdict, and a
companion failure is recorded as a limit on generality rather than as evidence against the subject.

Its four known hazards stand regardless. Its `rollout_index` 168 is a hole rather than a wrong value,
now confirmed by opening the file. Its checkpoint stops at 390, so any join to its trainer log
refuses past that step. And the ceiling excess reported for it, `6.0078` against `5.0`, has the same
explanation as the subject's and is not an anomaly.
