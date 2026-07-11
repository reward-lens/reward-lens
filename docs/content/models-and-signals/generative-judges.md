<div class="rl-chips">
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">grader</span> generative judge</span>
  <span class="rl-chip rl-chip--gauge"><span class="rl-chip__k">score gauge</span> invariant</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">bring</span> an instruct causal LM</span>
</div>

# Generative judges

**An LLM-as-judge has no reward head. What is its reward, and can you look inside it?**

Its reward is a logit difference, and yes. A judge does not emit a scalar. It emits a verdict *token*, "Yes" or "No", "A" or "B". The pointwise reward is the gap between the two verdict logits at the judgment position,

\[
\text{verdict} = h^\top\bigl(W_U[\text{Yes}] - W_U[\text{No}]\bigr) = \operatorname{logit}(\text{Yes}) - \operatorname{logit}(\text{No}),
\]

where \(W_U\) is the unembedding. That difference of two rows of the LM head is an ordinary direction, so a judge becomes a signal with a `logit_diff` readout, and crystallization depth of a judge's verdict is the same measurement as crystallization depth of a scalar head, run with a different readout.

## Load or wrap one

`GenerativeJudge.from_causal_lm(model, tokenizer)` wraps an already-loaded instruct model, reading the verdict directions off its LM head. The tiny CPU vehicle builds a real `LlamaForCausalLM` so the mechanism is exact even though the weights are random:

```python
from reward_lens.signals import GenerativeJudge

judge = GenerativeJudge.from_tiny(seed=0)
judge.caps      # SCORES|PREFIX_SCORES|ACTIVATIONS|GRADIENTS|HVP|LINEAR_READOUT|GENERATIVE
```

## What it exposes

Three readouts, and the capability `GENERATIVE` that the classifier does not carry:

```python
[(r.name, r.kind) for r in judge.readouts()]
# [('verdict', 'logit_diff'), ('verdict_ab', 'logit_diff'), ('likert', 'simplex')]

judge.readout("verdict").meta       # {'a': 'Yes', 'b': 'No', 'a_id': 5297, 'b_id': 2949, ...}
```

`verdict` is the pointwise Yes-minus-No direction; `verdict_ab` is the pairwise A-minus-B direction for comparing two answers; `likert` is the expected score under the softmax over the rating tokens, a `simplex` readout with no single direction. Because the verdict readouts are ordinary linear directions read off the head, the reward-lens battery attaches to a judge exactly as it does to a classifier: you can watch a verdict crystallize, attribute it, and patch it.

```python
ev = judge.score([("Is 2+2=4?", "Yes, it is four."),
                  ("Is the sky green?", "Yes, bright green.")], "verdict")
ev.value.values     # array([-0.1178, -0.117 ], dtype=float32)
ev.trust            # TrustLevel.EXPLORATORY
ev.gauge            # GaugeStatus.INVARIANT
```

Every one of those numbers is the genuine fp32 projection of the model's own final hidden state onto the verdict direction. Nothing fabricates a verdict from a model that was not run.

## Validate the judgment position

A judge's reward is only meaningful if you read it at the token where the verdict lands. Detection is structural: templating with the generation prompt puts the judgment at the final valid position. But structure is a hypothesis, so the adapter *validates* it by running a forward over calibration prompts and checking how often the model's greedy next token there is actually a verdict token:

```python
rec = judge.validate_judgment_position([
    ("What is 2+2?", "4"), ("Capital of France?", "Paris"),
    ("Name a fruit.", "Apple"), ("Is fire cold?", "No"),
])
rec["position"]     # 'final-valid-token(add_generation_prompt)'
rec["validated"]    # True
rec["confidence"]   # 0.0
```

The confidence is `0.0` here, and that is the correct, honest answer. A random tiny model emits no real verdicts, so its greedy token at the judgment position is a verdict token essentially never. Read that number as a null result you can trust: the plumbing works, the model has nothing to say. On a real instruct judge the same call reports a high confidence, and the same code path stands ready for it.

!!! warning "Needs a GPU"
    A judge that actually judges is a real instruct model. Building one means loading a multi-billion-parameter causal LM, which is GPU-gated on this hardware. The tiny judge proves the verdict mechanism, the readout extraction, and the validation loop on CPU; it does not stand in for a model that can grade.

## Honest caveats

Read the confidence before you read the verdict. A low `confidence` means the judgment position is wrong or the model is not really judging, and a verdict read there is noise regardless of how clean the number looks. The `likert` readout is a `simplex`, so it has no direction to trace and `score_prefixes` is undefined for it; use the `logit_diff` verdicts for anything that watches the reward form. And a judge's verdict is exploratory until an [organism](../discipline/calibration-and-organisms.md) with a known answer key calibrates the instrument reading it. The [use an LLM judge as a signal](../how-to/llm-judge-signal.md) guide runs the real-model path end to end.

## Reference

[`GenerativeJudge`](../reference/signals.md#reward_lens.signals.judge.GenerativeJudge), and the readout model on [the reward direction](../concepts/reward-direction.md).
