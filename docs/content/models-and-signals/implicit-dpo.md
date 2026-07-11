<div class="rl-chips">
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">grader</span> implicit (DPO) RM</span>
  <span class="rl-chip rl-chip--gauge"><span class="rl-chip__k">score gauge</span> invariant</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">bring</span> a policy and a reference model</span>
</div>

# Implicit (DPO) rewards

**Your DPO checkpoint never had a reward head. Can you still treat it as a reward model?**

Yes, because DPO training builds one implicitly. A model trained with a log-ratio objective *is* a reward model whose reward is

\[
\hat r(y) = \beta \sum_t \log \frac{\pi_\text{policy}(y_t \mid y_{<t})}{\pi_\text{ref}(y_t \mid y_{<t})},
\]

summed over the response tokens. There is no head to read, only two models to compare. This adapter makes that comparison a first-class signal, and two properties fall straight out of the definition, which is why it is its own adapter rather than a special case of the classifier.

## Load or wrap one

You bring two causal LMs that share a tokenizer, a policy and its reference. `ImplicitRM.from_models(policy, reference, tokenizer, beta=0.1)` wires them; the tiny CPU vehicle builds two distinct random models over one tokenizer so the log-ratio is nonzero:

```python
from reward_lens.signals import ImplicitRM

imp = ImplicitRM.from_tiny(policy_seed=1, reference_seed=2, beta=0.1)
imp.caps        # SCORES|PREFIX_SCORES|ACTIVATIONS|PAIRED_MODELS
imp.beta        # 0.1
[(r.name, r.kind, r.position.kind) for r in imp.readouts()]
# [('implicit_reward', 'token_value', 'all')]
```

## What it exposes

The capability `PAIRED_MODELS` is the honest flag here: an implicit reward is defined over *two* models, so "the activation at layer L" is ambiguous and the adapter refuses to guess. `capture` routes to the policy model by default and reaches the reference through an explicit `namespace="ref"`. The single readout is `implicit_reward`, a `token_value` over all positions, because the reward is native per-token, not a projection.

```python
data = [("Translate hello.", "Bonjour, mon ami."),
        ("Translate hello.", "asdf qwerty zzz")]

ev = imp.score(data)
ev.value.values     # array([ 0.0616, -0.0358], dtype=float32)   # r-hat per item
ev.trust            # TrustLevel.EXPLORATORY
ev.gauge            # GaugeStatus.INVARIANT
len(ev.subject.signals)   # 2  -- both model fingerprints, policy first
```

The subject names both fingerprints, so an implicit-reward measurement can never be mistaken for a single-model one. The score is exploratory and gauge-invariant like any raw reward.

## The reward decomposes per token, exactly

The first property that falls out for free: the reward is a sum of per-token log-ratios, so its token-level decomposition is native, not an attribution you reconstruct afterward. `per_token_rewards` returns the increments \(r_t = \beta(\log\pi_\text{policy}(y_t) - \log\pi_\text{ref}(y_t))\), and they sum to the sequence score by construction:

```python
inc = imp.per_token_rewards(data).value.curves[0]
inc                     # array([ 0.021 , 0.007 , 0.0075, 0.041 , -0.0125, -0.0106, 0.0033, 0.0048], float32)
inc.sum()               # equals ev.value.values[0]  (0.0616...)

pref = imp.score_prefixes(data).value.curves[0]
pref[-1]                # also equals ev.value.values[0]  -- the running sum lands on r-hat
```

`score_prefixes` gives the running sum, whose final entry is the sequence reward, and `per_token_rewards` gives the raw increments. For a classifier those per-token curves are an attribution of a whole-sequence score; for an implicit reward they are the reward itself, token by token, which is why the verification and dense-reward sciences read them directly.

## Honest caveats

There is no single head direction, so the linear-readout battery does *not* attach: the capabilities carry no `LINEAR_READOUT`, `GRADIENTS`, or `HVP`, and the lens, direct attribution, and patch grid will refuse this signal rather than run on a direction that does not exist. What attaches is everything that needs only scores, prefix curves, or activations, plus the per-token sciences that the native decomposition feeds. Both models must share one tokenizer and one template, so the response region is identical across the two forwards; that is a load requirement, not a suggestion. And \(\beta\) scales the whole reward, so an absolute implicit reward is only meaningful relative to the \(\beta\) it was computed at. The [use a DPO checkpoint as a signal](../how-to/dpo-implicit-reward.md) guide runs the real-model path.

## Reference

[`ImplicitRM`](../reference/signals.md#reward_lens.signals.implicit.ImplicitRM).
