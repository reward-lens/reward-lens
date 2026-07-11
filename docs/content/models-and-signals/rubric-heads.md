<div class="rl-chips">
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">grader</span> rubric RM</span>
  <span class="rl-chip rl-chip--gauge"><span class="rl-chip__k">score gauge</span> invariant</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">bring</span> a multi-row head</span>
</div>

# Rubric heads

**Your grader scores against named criteria. Can you read each one, not just the total?**

Yes, and the criteria are data, not code. A rubric grader scores a response against a set of named criteria, coherence, correctness, safety, and combines them. reward-lens holds the criterion set as a `RubricSpec` and reads one direction per criterion off a multi-row head, plus a weighted aggregate. Changing the rubric is changing a spec, never editing the adapter, which is what makes a new rubric a new dataset rather than a new code path.

## Load or wrap one

`RubricRM.from_sequence_classifier(model, tokenizer, spec)` wraps a head that has one row per criterion. The tiny CPU vehicle takes the criteria directly:

```python
from reward_lens.signals import RubricRM

rub = RubricRM.from_tiny(criteria=("coherence", "correctness", "safety"), seed=0)
rub.caps            # SCORES|PREFIX_SCORES|ACTIVATIONS|GRADIENTS|HVP|LINEAR_READOUT|MULTI_READOUT
rub.spec.criteria   # ('coherence', 'correctness', 'safety')
rub.spec.resolved_weights()   # (0.3333, 0.3333, 0.3333)  -- uniform by default
```

## What it exposes

`MULTI_READOUT`, and one readout per criterion plus an aggregate:

```python
[r.name for r in rub.readouts()]
# ['reward', 'criterion:coherence', 'criterion:correctness', 'criterion:safety']
```

Every criterion is an ordinary `linear` direction, a single head row, and the aggregate `reward` is a genuine single direction too: the weighted sum of the rows, \(w_\text{agg} = \sum_k \text{weight}_k\, w_k\). This is the honest difference from a gated multi-objective model like ArmoRM, whose composite is nonlinear input-dependent gating. A rubric aggregate is exactly its weighted-sum direction, so the single-direction instruments run on it without approximation.

## Per-criterion scores

`criterion_scores` returns one evidence per criterion; `score` with no readout returns the aggregate:

```python
view = [("Is water wet?", "Yes, water wets surfaces it contacts.")]

rub.score(view).value.values            # array([-0.1313], dtype=float32)   # the aggregate
{k: float(v.value.values[0]) for k, v in rub.criterion_scores(view).items()}
# {'coherence': -0.16962, 'correctness': -0.09091, 'safety': -0.13338}
```

With uniform weights the aggregate is the mean of the criteria, and it is: the three criterion scores average to `-0.1313`, the aggregate. On a random model the numbers are noise, but the relationship is exact and is what lets you decompose the total reward into named parts. When you want the *geometry* between criteria, whether coherence and safety pull the same way or fight, read it with [multi-objective geometry](../instruments/multi-objective-geometry.md), which the `MULTI_READOUT` capability makes available.

## Honest caveats

The rubric aggregate is only as principled as its weights, and the default is a flat mean that almost certainly is not how a trained grader combines its criteria; pass the real weights in the `RubricSpec` when you know them. Two criteria can be nearly collinear, in which case "per-criterion" scores are less independent than they look, which is again a job for multi-objective geometry to expose. And every criterion score is `INVARIANT` and `EXPLORATORY`: raw, gauge-free, and uncalibrated until an [organism](../discipline/calibration-and-organisms.md) gives the reading instrument an answer key. To wire a grader whose criteria live somewhere other than head rows, see [write an adapter](../how-to/write-an-adapter.md).

## Reference

[`RubricRM`](../reference/signals.md#reward_lens.signals.rubric.RubricRM) and its `RubricSpec`.
