# Use a DPO checkpoint as a signal

**You trained with DPO, so there is no reward head. Can you still measure the reward?**

Yes. A DPO checkpoint is a reward model that never grew a head. Its implicit reward is the log-ratio of the tuned policy against its reference,

\[ \hat r(y) = \beta \sum_t \log \frac{\pi_{\text{policy}}(y_t \mid y_{<t})}{\pi_{\text{ref}}(y_t \mid y_{<t})}. \]

`ImplicitRM` wraps the two models behind the same protocol every other signal uses, so the instruments attach unchanged. It is a paired-model signal: because "the activation at layer \(L\)" is ambiguous across two networks, capture and gradients route to the policy by default, with the reference reachable through an explicit namespace, and the adapter refuses to guess between them.

The real constructor is `ImplicitRM.from_models(policy, reference, tokenizer, beta=0.1)`, two `CausalLM` models sharing a tokenizer. On CPU, `from_tiny` builds two tiny ones with distinct seeds:

```python
from reward_lens.signals import ImplicitRM

signal = ImplicitRM.from_tiny(policy_seed=1, reference_seed=2, beta=0.1)
print(signal.caps)
# Capability.SCORES|PREFIX_SCORES|ACTIVATIONS|PAIRED_MODELS

items = [("Why is the sky blue?", "Rayleigh scattering favours short wavelengths."),
         ("Why is the sky blue?", "Because it is blue, nobody knows.")]
ev = signal.score(items)
print(ev.value.values)       # [0.04654941 0.01422949]
print(ev.trust, ev.gauge)    # EXPLORATORY invariant
```

The two models here are random, so the scores mean nothing in magnitude. What is real is the mechanism: a genuine per-token log-ratio between two forward passes, and the `PAIRED_MODELS` capability that tells every downstream instrument there are two subjects, not one.

Because the reward decomposes per token natively, the prefix curve is exact rather than an attribution. The increments sum to the sequence score by construction:

```python
curve = signal.score_prefixes(items[:1]).value.curves[0]
print(len(curve))    # 8   (one increment per response token; they sum to the sequence score)
```

!!! warning "Needs a GPU"
    On real checkpoints, load the policy and its reference with `AutoModelForCausalLM.from_pretrained(...)` and pass both to `ImplicitRM.from_models(policy, reference, tokenizer, beta=...)`. Two 8B forwards do not fit an 8 GB GPU; the tiny path above exercises the identical code.

See also: [Implicit (DPO) rewards](../models-and-signals/implicit-dpo.md). API: [`ImplicitRM`](../reference/signals.md#reward_lens.signals.implicit.ImplicitRM).
