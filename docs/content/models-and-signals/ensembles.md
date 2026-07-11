<div class="rl-chips">
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">grader</span> ensemble / distributional</span>
  <span class="rl-chip rl-chip--gauge"><span class="rl-chip__k">score gauge</span> invariant</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">bring</span> two or more signals</span>
</div>

# Ensembles and distributions

**You score with several reward models, or one that predicts a distribution. Is that still one signal?**

Yes. Two composites live here, and any instrument consumes either as an ordinary signal. A `SignalEnsemble` combines several members into one score by mean, min, or a quantile. A `DistributionalSignal` wraps a head whose rows are quantile levels and exposes them as a predictive distribution over the reward. Both are thin: they delegate the forward work to their members and only compose the results, so they inherit the members' hardening rather than re-deriving it.

## An ensemble of signals

There is no `from_tiny` for an ensemble; you compose it from members. Two tiny classifiers with different seeds stand in for two reward models:

```python
from reward_lens.signals import from_tiny, SignalEnsemble

m0, m1 = from_tiny(seed=0), from_tiny(seed=1)
ens_mean = SignalEnsemble([m0, m1], mode="mean")
ens_min  = SignalEnsemble([m0, m1], mode="min")

view = [("q", "a chosen answer")]
m0.score(view).value.values[0]        # -0.0433
m1.score(view).value.values[0]        # -0.0535
ens_mean.score(view).value.values[0]  # -0.0484   == (-0.0433 + -0.0535) / 2
ens_min.score(view).value.values[0]   # -0.0535   == min of the two
len(ens_mean.score(view).subject.signals)   # 2  -- both members named by fingerprint
```

The min composite is the standard conservative reward-hacking guard: a response only scores high if *every* member likes it, so an answer that games one model's blind spot is caught by the others. The quantile composite is its tunable generalization. The subject names every member by fingerprint, so a composite score always says exactly what it was built from, which matters when you price [optimization pressure](../training-loops/best-of-n.md) against an ensemble.

## An ensemble claims only what it implements

The capabilities are the honest part. Both members expose the full classifier surface, yet the ensemble declares less:

```python
m0.caps         # SCORES|PREFIX_SCORES|ACTIVATIONS|GRADIENTS|HVP|LINEAR_READOUT
ens_mean.caps   # SCORES|PREFIX_SCORES|ACTIVATIONS
```

It composes scores reliably, and per-member activations and prefix curves when every member has them, so it claims those and nothing more. It does *not* claim `LINEAR_READOUT` or `GRADIENTS`, because there is no single direction and no shared "layer L" across two different models: which member would a gradient be taken through? So the linear-readout battery does not attach to the composite, and `capture` asks you which member you mean rather than inventing a shared activation. Score-only and prefix instruments run on it directly.

## A distributional signal

A `DistributionalSignal` wraps a multi-row head whose rows are quantile levels and relabels them, adding the `DISTRIBUTIONAL` capability:

```python
from reward_lens.signals import DistributionalSignal

dist = DistributionalSignal.from_tiny(taus=(0.1, 0.5, 0.9), seed=0)
[r.name for r in dist.readouts()]        # ['quantile:0.1', 'quantile:0.5', 'quantile:0.9']
dist.score(view, "quantile:0.9").value.values   # array([0.0505], dtype=float32)
dist.mean(view).value.values                    # array([0.0638], dtype=float32)
```

Each `quantile:tau` readout is an exact head row, not an interpolation, so `score(view, "quantile:0.9")` returns the model's own 0.9-quantile prediction. `median`, `quantile`, and `mean` are convenience reductions over the levels. This turns a multi-row head into a first-class distribution over the reward, which is what a risk-sensitive objective reads when it wants a lower quantile rather than the mean.

## Honest caveats

A composite prefix curve requires the members to tokenize identically, the common case being a shared tokenizer; mismatched curve lengths raise rather than truncate in silence. An ensemble is only as diverse as its members, and a min guard over near-identical models buys little, so treat the members' agreement geometry as something to check, not assume. The composite score is `INVARIANT` and `EXPLORATORY` like any raw score. And a distributional signal's quantiles are only calibrated if the underlying head was trained to produce calibrated quantiles; reward-lens reads the rows faithfully, it does not certify them. See the [caveats](../caveats.md) page on reading a composite honestly.

## Reference

[`SignalEnsemble`](../reference/signals.md#reward_lens.signals.ensemble.SignalEnsemble), [`DistributionalSignal`](../reference/signals.md#reward_lens.signals.ensemble.DistributionalSignal).
