<div class="rl-chips">
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">grader</span> classifier RM</span>
  <span class="rl-chip rl-chip--gauge"><span class="rl-chip__k">score gauge</span> invariant</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">bring</span> a sequence-classification head</span>
</div>

# Classifier reward models

**Your reward model is a sequence classifier with a scalar head. What does reward-lens do with it?**

It reads the head off the checkpoint as a direction and never guesses again. A classifier reward model maps the final hidden state to one number, \(r = w_r^\top h + b\), and \(w_r\) is a row of weights you already have. reward-lens loads it into a first-class `reward` readout, and from then on every score is the exact fp32 projection of the final hidden state onto that direction. This is the most common grader and the one the whole instrument battery was first built against.

## Load or wrap one

Three paths, in order of how much you bring. Name a checkpoint and let `load_signal` sniff the convention (gated for hub-scale downloads, see [the loading conventions](index.md#loading-conventions)). Hand `wrap_hf_model` a model and tokenizer you have already loaded. Or build the tiny CPU model every example here uses:

```python
from reward_lens.signals import from_tiny

signal = from_tiny(seed=0)          # a real 2-layer LlamaForSequenceClassification, CPU, no download
signal.meta.adapter                 # 'LlamaAdapter'
signal.meta.architecture            # 'LlamaForSequenceClassification'
signal.caps                         # SCORES|PREFIX_SCORES|ACTIVATIONS|GRADIENTS|HVP|LINEAR_READOUT
```

On real hardware, `wrap_hf_model(model, tokenizer)` is the same constructor the hub loader ends at, and it runs a conformance quick-check on the way out. The [load or wrap a reward model](../how-to/load-a-reward-model.md) guide covers the campaign checkpoints.

## What it exposes

One readout, `reward`, a `linear` direction read at the final residual stream. The capabilities say the rest: it scores, it scores token by token, it exposes activations, and it supports autograd and its second order (`GRADIENTS`, `HVP`) because a linear head is differentiable.

```python
[(r.name, r.kind, str(r.site), r.position.kind) for r in signal.readouts()]
# [('reward', 'linear', 'L1.resid_post', 'final')]
```

Because the readout *is* the head, the whole linear-readout battery attaches: the [reward lens and crystallization](../instruments/lens-crystallization.md), [component attribution](../instruments/attribution.md), the [patch grid](../instruments/patch-grid.md), [path effects](../instruments/path-effects.md), and [concept dose-response](../instruments/concept-dose-response.md). Conformance confirms the projection matches the model's native head to zero within tolerance, which is what makes those instruments trustworthy rather than approximate.

## A worked run

Score a preference pair. The chosen answer explains, the rejected one shrugs.

```python
from reward_lens.signals import from_tiny

signal = from_tiny(seed=0)
ev = signal.score([
    ("Why is the sky blue?", "Rayleigh scattering favors short blue wavelengths."),
    ("Why is the sky blue?", "It just is, nobody knows."),
])
ev.value.values     # array([-0.1155, -0.0913], dtype=float32)
ev.trust            # TrustLevel.EXPLORATORY
ev.gauge            # GaugeStatus.INVARIANT
```

The scores are tiny and their order means nothing here: the weights are random, so this model has no opinion about the sky. The point is the shape of what comes back. A raw score is gauge `INVARIANT`, safe to compare across signals as-is, and trust is `EXPLORATORY` because nothing has been calibrated yet. The number is real; the receipt attached to it is honest about how much to lean on it. See [a measurement you can trust](../concepts/measurement-you-can-trust.md) for what the receipt carries.

## The single-direction picture, and where it bends

The clean story is that a classifier's whole opinion is one direction, \(w_r\). That holds exactly for a single-row head. It bends for a multi-objective head like ArmoRM, which has nineteen objective rows rather than one. reward-lens surfaces those rows as separate readouts and refuses to collapse them silently:

```python
from reward_lens.signals import wrap_hf_model
from reward_lens.signals.process import _tiny_sequence_classifier

model, tok = _tiny_sequence_classifier(seed=0, num_labels=3)   # a 3-row head stands in for a 19-row one
multi = wrap_hf_model(model, tok, architecture="LlamaForSequenceClassification",
                      conformance_quickcheck=False)
[r.name for r in multi.readouts()]     # ['reward', 'criterion:0', 'criterion:1', 'criterion:2']
multi.readout("reward").meta           # {'aggregate': 'row_mean', 'legacy': True, 'bias': 0.0}
```

Each objective becomes a `criterion:k` readout, and the default `reward` is marked plainly as a legacy row-mean aggregate. That row mean is an *approximation* of a single direction, not the model's real decision rule: the true composite is the head's own input-dependent gating, which is nonlinear and reachable through the native score. So the single-direction instruments still run on ArmoRM, but they run on the mean of nineteen objectives, and a claim about "the reward direction" of a multi-objective model is a claim about that average. For honest multi-objective work, score each `criterion:k` and read the geometry between them with [multi-objective geometry](../instruments/multi-objective-geometry.md). The [caveats](../caveats.md) page carries this in full.

Two more things worth stating flat. Raw scores from two different models are near-orthogonal in their own coordinates and are *not* comparable without a shared frame; that is a gauge problem the [gauge and frames](../discipline/gauge-and-frames.md) page owns, not a property of the classifier. And a score is exploratory until an [organism](../discipline/calibration-and-organisms.md) gives the instrument reading it an answer key.

## Reference

[`ClassifierRM`](../reference/signals.md#reward_lens.signals.classifier.ClassifierRM), [`wrap_hf_model`](../reference/signals.md#reward_lens.signals.loaders.wrap_hf_model), [`from_tiny`](../reference/signals.md#reward_lens.signals.loaders.from_tiny).
