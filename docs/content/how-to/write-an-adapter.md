# Write an adapter

**You have a grader reward-lens does not ship an adapter for. How do you make every instrument work on it?**

Teach it to speak one protocol. In reward-lens a grader is a signal: whatever produces a reward, a classifier head, an LLM judge, a DPO log-ratio, a process model, exposes the same `RewardSignal` interface, and an Observable written against that interface runs on all of them. Adding a new grader paradigm means writing an adapter that implements the protocol and passes the conformance suite. Once it does, the whole battery, every index, and every science attach to it unchanged.

## First, check whether an adapter already fits

Eight adapters ship: classifier, generative judge, process, implicit (DPO), rubric, trajectory, dense, and ensemble. Most models are already covered, so try before you write anything.

A Hugging Face sequence classifier with a linear reward head wraps directly with `wrap_hf_model`, which runs a quick conformance check as it wraps and refuses a model whose readout does not match its head:

```python
from reward_lens.signals import wrap_hf_model

signal = wrap_hf_model(model, tokenizer, device="cpu")   # your loaded HF classifier
```

If your grader is one of the other seven families, its constructor is the entry point (`GenerativeJudge.from_causal_lm`, `ImplicitRM.from_models`, `ProcessRM.from_sequence_classifier`, and so on). You only write an adapter when your grader is a genuinely new paradigm none of these describe.

## The protocol is the whole contract

`RewardSignal` is small on purpose. An adapter declares three attributes and implements six methods:

```python
class RewardSignal(Protocol):
    meta: SignalMeta        # identity and fingerprint
    caps: Capability        # which capabilities it supports (scores, activations, gradients, ...)
    runtime: Runtime        # device, dtype, layer count

    def readouts(self) -> list[Readout]: ...
    def score(self, view, readout="reward"): ...              # -> Evidence[Scores]
    def score_prefixes(self, view, readout="reward"): ...     # -> Evidence[TokenCurves]
    def capture(self, view, spec): ...                        # -> CaptureHandle over activations
    def with_interventions(self, *ivs): ...                   # -> a wrapped RewardSignal
    def tokenize(self, item): ...                             # -> TokenizedInput (owns span carry-through)
```

`caps` is the honest part: it declares what your grader can actually do, and an Observable that needs a capability your signal lacks is refused before it runs rather than fed a fabricated activation. Five of the shipped adapters subclass a common base that supplies the scoring and capture plumbing, so a new adapter usually fills in tokenization, the readout, and the capability set rather than all six methods from scratch. Study the classifier adapter as the reference implementation.

## Conformance is how you know it is wired right

The reason to trust an adapter is not that it imported cleanly. It is that it passes `run_adapter_conformance`, the suite that checks the invariants a reward readout must have: the same input scores identically twice, a batch scores the same as one item at a time, left-padding does not change the score, the readout matches the head in fp32, and prefix scores are consistent. Run it on any conforming signal and read the report:

```python
from reward_lens.signals import from_tiny, run_adapter_conformance
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3

signal = from_tiny(seed=0)
items = list(load_diagnostic_v3()["helpfulness"].items)[:5]

report = run_adapter_conformance(signal, items=items, readout="reward")
print("passed:", report.passed, "| checks:", report.n_passed)
for c in report.checks:
    print("  ", "pass" if c.passed else "FAIL", c.name)
# passed: True | checks: 5
#    pass determinism
#    pass batch_vs_single
#    pass left_pad_invariance
#    pass readout_matches_head
#    pass prefix_consistency
```

Five checks, all passing, on a real adapter. The `readout_matches_head` check is the one that catches the failure class that used to sink white-box work silently: a model that half-loads, or a readout computed in the wrong precision, prints plausible scores that do not match the head it claims to read. Conformance turns that into a `FAIL` instead of a wrong paper. When your adapter passes this suite, it is wired right, and every instrument in the library is available to it.

!!! note "Composite and paired adapters skip the head check"
    `run_adapter_conformance` runs the checks that apply: pass `check_head=False` for an ensemble or an implicit signal that has no single linear head to match against. A skipped check is reported as a skip, not a silent pass, so the report never overstates what was verified.

See also: [classifier reward models](../models-and-signals/classifier-rms.md), [the signals index](../models-and-signals/index.md), [`RewardSignal`](../reference/signals.md#reward_lens.signals.base.RewardSignal), [`run_adapter_conformance`](../reference/signals.md#reward_lens.signals.conformance_adapters.run_adapter_conformance).
