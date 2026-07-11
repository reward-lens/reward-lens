# Use an LLM judge as a signal

**Your grader is a chat model that answers Yes or No, not a scalar head. How do you read a reward off it?**

Off the unembedding. The pointwise verdict is the logit difference \( W_U[\text{Yes}] - W_U[\text{No}] \) at the judgment position, which reward-lens exposes as a first-class `logit_diff` readout. With that in hand, crystallization depth of a judge's verdict is the same instrument as crystallization depth of a scalar head, called with a different readout.

The real constructor is `GenerativeJudge.from_causal_lm(model, tokenizer)`. On CPU, `from_tiny` builds a tiny one:

```python
from reward_lens.signals import GenerativeJudge

signal = GenerativeJudge.from_tiny(seed=0)
print(signal.caps)
# Capability.SCORES|PREFIX_SCORES|ACTIVATIONS|GRADIENTS|HVP|LINEAR_READOUT|GENERATIVE
print([(r.name, r.kind) for r in signal.readouts()])
# [('verdict', 'logit_diff'), ('verdict_ab', 'logit_diff'), ('likert', 'simplex')]
```

Three readouts: `verdict` (Yes/No pointwise), `verdict_ab` (A/B pairwise), and `likert` (the expected score under the softmax over the score tokens).

## Validate the judgment position

The verdict token is emitted at the final valid position once the generation prompt is appended. The adapter detects that structurally and then checks it, recording how often the model's greedy next token there actually is a verdict token:

```python
rec = signal.meta.lineage["judgment_detection"]
print(rec["position"], rec["confidence"])
# final-valid-token(add_generation_prompt) 0.0
```

Confidence `0.0` is near chance: on a randomly initialised tiny model, the greedy token at the judgment position lands on a verdict token zero times out of four. That is the honest reading. The mechanism (projecting the final hidden state onto \( W_U[\text{Yes}] - W_U[\text{No}] \)) is exact, but the verdict only carries meaning when the model does, which means a real instruct judge.

```python
ev = signal.score([("What is 2+2?", "4"), ("What is 2+2?", "banana")], "verdict")
print(ev.value.values)       # [-0.10788467 -0.1125759 ]  (near chance: the tiny model has no opinion)
print(ev.trust, ev.gauge)    # EXPLORATORY invariant
```

The two verdicts sit almost on top of each other, which is exactly what a model with no opinion should produce. Nothing here fabricates a verdict from a model that was not run.

!!! warning "Needs a GPU"
    A real judge is an instruct model wrapped with `GenerativeJudge.from_causal_lm(model, tokenizer)`, optionally with `validate_with=` your own calibration prompts. On adequate hardware the detection confidence rises to reflect a model that genuinely emits verdicts, and the verdict spread becomes a real reward signal.

See also: [Generative judges](../models-and-signals/generative-judges.md). API: [`GenerativeJudge`](../reference/signals.md#reward_lens.signals.judge.GenerativeJudge).
