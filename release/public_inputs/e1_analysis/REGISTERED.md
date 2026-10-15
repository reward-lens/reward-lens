# The registered battery, failures first

Rows carrying a predicted value, run against their own resolution rule. Nothing here is called
preregistered; the exposure tier is printed beside each row and `PLAN.md` section 7 defines the
tiers.

## Failures and movements

### P16, the support census. Value moves, refutation does not fire.

**Predicted**, `PREDICTED`: at most 67 of 1604 prompt groups can carry an identified second-order
susceptibility, computed over distinct `(thinking_format_ok, training_passed)` cells, which is 4.18
percent. A companion figure of 42 of 1604 under the equal-weight composite. **Refuted if the
fraction exceeds 0.25.**

**Measured**, on all 1604 groups of the subject run:

| cell definition | groups with `M ≥ 2` | groups with `M ≥ 3` | mean `M` |
|---|---|---|---|
| binarised, `(thinking_format_ok, training_passed)` | 517 (0.3223) | **67 (0.0418)** | 1.3678 |
| native, `(thinking_format, training_passed)` | 517 (0.3223) | 141 (0.0879) | 1.4221 |

The predicted 67 reproduces exactly. Under native resolution the count rises to 141, so the row's
value more than doubles, and its refutation condition does not fire: 0.0879 is well below 0.25.

The brief expected this row to flip and asked that it be reported as a refutation rather than a
correction. It does not flip. The honest report is that the predicted value was right about the
published grid, that native resolution roughly doubles it, and that the second-order negative result
remains a statement about the method rather than about the artifact, because the row's own threshold
for switching that verdict was 0.25 and the measurement is a third of it.

One thing in the table is worth more than the row it resolves. `M ≥ 2` is **identical** at 517 under
both definitions. Native resolution adds no informative group; it only subdivides groups that were
already informative. That is why the degeneracy structure reproduces the trainer's
`frac_reward_zero_std` equally well under the binarised grid the preflight used and the native one
used here, and it is the reason the preflight's degeneracy result survived a binarisation that
destroyed its reward reconstruction.

### P33, the gold channel. Held, on both numbers.

**Predicted**, `PREDICTED`: no Goodhart turn. Every gold event sits in the reward's top cell, 10 of
10, alongside 18,590 hacks. **Refuted if any gold event is strictly below a hack in reward.**

**Measured**: `actually_solved` is 1 on exactly **10** of 25,664 rollouts, at steps 27, 34, 72, 79,
97, 104, 105, 106, 129 and 129. Every one has a recovered composite of exactly `5.0`, which is the
maximum the composite attains anywhere in the run. **18,590** hacked rollouts also sit at `5.0`.

Both predicted counts reproduce exactly. No gold event is strictly below any hack, so the row is not
refuted.

The row's own stated shortfalls travel with it and are not softened: ten events is 2.08 times short
of certifying that a turn exists and 125 times short of locating one. What this measures is that on
the only gold channel in the programme, the reward cannot distinguish a genuine solution from an
exploit, because both saturate the same top cell. That is a statement about the reward's resolution,
not evidence that no turn exists.

A second observation, unregistered and `DESCRIPTIVE`: the ten gold events are the only rollouts where
`passed` and `reward_hacked` differ. On 25,654 of 25,664 rollouts they are the same bit. In this run
a pass is a hack, with ten exceptions.

### P19, the group key. Confirmed, and the confirmation was free.

**Predicted**, `PREDICTED`: `sample_id` is the prompt group and `epoch` the generation index, with
agreement on at least 400 of 401 files. **Refuted if either disagrees on more than one file.**

**Measured**: the archive header declares it outright. `dataset.sample_ids` holds four values and
`config.epochs` is sixteen, on every file. No inference was needed.

It is also confirmed independently and more strongly. The trainer's `frac_reward_zero_std` reproduces
exactly under grouping by `sample_id` into four groups of sixteen, and under no other grouping tested:
at group sizes 2, 4, 8, 32 and 64 the reproduction fails on every step.

The row resolves as a confirmation that cost nothing, and it is reported as that rather than as a
result. The brief asked that the cheapness be said out loud, and the reason it is worth saying is
that a row registered against 401 files of order statistics was answerable from one JSON field.

## Rows this pass did not run

Recorded so the battery's coverage is legible rather than implied.

`P15` the dead-step null, `P17` the momentum refit, `P18` the participation ratio, `P28` the
string-only CUSUM, `P29` the lineage question and `P34` the turning-point estimator all remain live
and are not resolved here. `P29` gates Arm D and `P15` requires its re-sampling precondition checked
first; both are in the queue behind the companion extraction.

`P27` and `P35` were resolved during the theory effort and are `DESCRIPTIVE` reproductions rather
than tests.

`P20`, `P30`, `P31`, `P32` and `P36` are out of scope, for the reasons Annex B of the brief gives.
One of those reasons is now known to be softer than written: `P32`'s stated blocker is that no
published card documents a generation temperature, and this run's eval header carries
`model_generate_config.temperature = 1.0` and `task_args.temperature = 1.0`. That does not make P32
runnable here, because it needs a distillation chain this artifact does not have, but the row's
blocker should be restated.

## What the record says about resolution

Two of the three rows resolved above reproduce a predicted integer exactly: 67 groups and 10 gold
events with 18,590 hacks beside them. Those numbers were computed by the preflight from the published
parquet, under a reward composition this pass has now shown was fitted against the wrong series.

That is worth stating plainly because it cuts against the easy reading of this whole exercise. The
preflight's arithmetic was sound. What failed was not its counting but its choice of target, and the
counting it did on the published columns holds up against a recomputation from the raw logs.
