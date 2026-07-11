# Signals

**Whatever grades your rollouts, can the same instruments attach to it?** A signal is the yes. `reward_lens.signals` defines one protocol and eight adapters that satisfy it, so a classifier reward model, an LLM judge, a DPO checkpoint, and a process reward model all present the same surface to everything downstream. The tour with runnable code is in [models and signals](../models-and-signals/index.md).

## The protocol

`RewardSignal` is a runtime-checkable protocol: a signal declares its capabilities, its readouts, and its positions, and answers `score`, `score_prefixes`, and `capture`. Anything that implements it is a first-class citizen of the library.

::: reward_lens.signals.base.RewardSignal
    options:
      heading_level: 3

A readout is a named way to read a scalar off the model, and a position spec says where along the sequence to read it. "The reward at the final token" is a `Readout` at a `PositionSpec("final")`.

::: reward_lens.signals.base.Readout
    options:
      heading_level: 3

::: reward_lens.signals.base.PositionSpec
    options:
      heading_level: 3

## Loading a signal

Two entry points are always available and touch no network: `wrap_hf_model` around a model and tokenizer you already hold, and `from_tiny` for a synthetic Llama that builds on CPU in under a minute. Both run a conformance quickcheck as they construct.

::: reward_lens.signals.loaders.wrap_hf_model
    options:
      heading_level: 3

::: reward_lens.signals.loaders.from_tiny
    options:
      heading_level: 3

`load_signal` is the front door for a repo id, but it is gated. Handed a hub id without `allow_download=True`, it raises `NotImplementedError` naming itself and pointing you at `wrap_hf_model` or `from_tiny`, because an 8B model in fp32 does not fit a laptop GPU and the loader will not pretend otherwise. Local paths bypass the gate.

::: reward_lens.signals.loaders.load_signal
    options:
      heading_level: 3

## The eight adapters

Each adapter wraps a family of grader and advertises exactly the capabilities it can honor. The instruments read those capabilities and refuse rather than fake a measurement the substrate cannot support. Every adapter has its own page under [models and signals](../models-and-signals/index.md).

`ClassifierRM` is the reference case: a scalar reward head on a sequence classifier, with the full capability set down to the linear readout and Hessian-vector products.

::: reward_lens.signals.classifier.ClassifierRM
    options:
      heading_level: 3

`GenerativeJudge` reads a verdict as the logit difference between two answer tokens. On a tiny model that verdict is near chance; it needs a real instruction-tuned model to mean anything.

::: reward_lens.signals.judge.GenerativeJudge
    options:
      heading_level: 3

`ProcessRM` scores per step. The delimiter step-split works; the learned boundary detector is a stub that falls back to treating the whole response as one step, and it says so in its metadata rather than inventing boundaries.

::: reward_lens.signals.process.ProcessRM
    options:
      heading_level: 3

`ImplicitRM` reads a reward as the DPO log-ratio between a policy and its reference, so a checkpoint you never trained a reward head for becomes a signal.

::: reward_lens.signals.implicit.ImplicitRM
    options:
      heading_level: 3

`RubricRM` exposes one readout per criterion, and `TrajectoryRM` scores typed spans of an agent trace.

::: reward_lens.signals.rubric.RubricRM
    options:
      heading_level: 3

::: reward_lens.signals.trajectory.TrajectoryRM
    options:
      heading_level: 3

`DenseRewardExtractor` derives a per-token reward as the first difference of the prefix curve. It is pinned to `EXPLORATORY` by design: the dense attribution is a construction, not a calibrated measurement, and the trust level records that.

::: reward_lens.signals.dense.DenseRewardExtractor
    options:
      heading_level: 3

`SignalEnsemble` composes several signals into one whose capabilities are the intersection, and `DistributionalSignal` exposes quantile readouts.

::: reward_lens.signals.ensemble.SignalEnsemble
    options:
      heading_level: 3

::: reward_lens.signals.ensemble.DistributionalSignal
    options:
      heading_level: 3

## Conformance

Before you trust a wrapped model, check that it behaves. `run_conformance` runs the classifier suite (determinism, batch-versus-single agreement, left-pad invariance, an exact readout-matches-head check in fp32, and more); `run_adapter_conformance` runs the checks that apply across all eight adapters. Writing your own adapter and passing these is the subject of [write an adapter](../how-to/write-an-adapter.md).

::: reward_lens.signals.conformance.run_conformance
    options:
      heading_level: 3

::: reward_lens.signals.conformance_adapters.run_adapter_conformance
    options:
      heading_level: 3
