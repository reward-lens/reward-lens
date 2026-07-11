# The lens lineage

**What family of methods does the reward lens belong to, and what does that family already know?** Naming the ancestry is the fastest way to see both what the tool can do and what it cannot. The family is methods that take a hidden state and read it out along some meaningful direction.

## Reading a hidden state along a direction

The oldest member is the logit lens (nostalgebraist, 2020, "interpreting GPT: the logit lens"). Take a hidden state from any intermediate layer, push it through the model's own output projection, the unembedding, and read the distribution over the vocabulary it would produce if the network stopped there. It reads a hidden state toward what the model would say next, and it showed that the guess about the next token sharpens layer by layer. It is also brittle, because the unembedding was trained to decode the final layer and not the intermediate ones, so applying it early can mislead.

The tuned lens (Belrose et al., 2023, "Eliciting Latent Predictions from Transformers with the Tuned Lens," arXiv [2303.08112](https://arxiv.org/abs/2303.08112)) fixes the brittleness by learning a small affine map for each layer, placed before the unembedding, that corrects each layer's state into the final layer's frame. Same target, the vocabulary, with a learned correction in front of it that makes the intermediate readouts more faithful.

The reward lens is the same move with the reward model's target. There is no vocabulary to decode toward. There is one direction, \(w_r\), the weight of the reward head, and reading a hidden state along it gives the reward that state would earn. Project every layer's residual stream onto \(w_r\) and you watch the preference form, exactly as the logit lens watches the next-token guess form.

| Lens | Reads a hidden state toward | The direction or map is |
| --- | --- | --- |
| Logit lens | the vocabulary | fixed: the model's own unembedding |
| Tuned lens | the vocabulary, corrected per layer | learned: an affine map per layer |
| Reward lens | what the reward head scores | fixed: the reward head weight \(w_r\) |

Every row is the same operation, a hidden state read along a direction. What differs is the direction and whether you had to learn it. The reward lens is the case where you neither learn nor estimate it.

## What is different here: the direction is free

That last row is the whole reason a reward model is a clean target. The logit lens reads toward an entire vocabulary, and you have to decide which tokens matter. The tuned lens has to be trained. The reward lens reads toward a single direction the trained model hands you exactly, sitting in the reward head's weights, the same vector for every input.

```python
from reward_lens.signals import from_tiny

signal = from_tiny(seed=0)                 # a real classifier RM on CPU, no download
r = signal.readouts()[0]
print(r.name, r.kind, tuple(r.vector.shape))   # reward linear (32,)
```

The reward direction is a named, linear readout of shape \((d_\text{model},)\), read straight off the head. There is nothing to fit and nothing to probe for, which is the point the [reward direction](../concepts/reward-direction.md) page makes at length.

The family resemblance carries the limitation too. All of these are observational. They read where something is legible in the activations, not what causes it. The logit lens's brittleness was the first version of that warning. For the reward lens the same caution is sharp and measured: what the lens shows forming late is not what patching finds causing the score, at a rank correlation of \(\rho = -0.171\) on Skywork-v0.2, and near zero (\(+0.047\)) on ArmoRM. Reading and causing are different questions, which is the [observational versus causal](../concepts/observational-vs-causal.md) distinction the whole library is built around.

## Two ways to reach a model's internals

There is a second lineage here, about mechanism rather than framing. To read activations you need a handle on them, and there are two ways to get one.

One is to re-implement the model. TransformerLens (Neel Nanda, [TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)) reconstructs GPT-style generative models in a standardized form with a hook on every activation, so the internals are exposed by construction. It is the standard tool for generative interpretability, and re-implementing buys uniformity: every model looks the same to your analysis code.

The other is to wrap the model you already have, the stance nnsight argued for: rather than rebuild a model, extend the framework so you operate on the real one as it loads, hooking the actual weights instead of a reimplementation. reward-lens takes this side. It stays HuggingFace-native and attaches lightweight hooks to the production reward model where it already lives, so the thing you analyze is the thing that shipped, not a copy of it. That choice is deliberate for reward models in particular. A reward model is usually a base model plus a trained head, loaded through `AutoModelForSequenceClassification`; hooking it in place means the \(w_r\) you read is the exact one that produced the score. The one-line summary is the honest one: reward-lens is to reward models what TransformerLens is to generative models, the same role in a different corner of the field.

## The one thing this line adds

Every tool named above gives you an instrument: a faithful way to read an activation along a direction. None of them tells you when not to believe the reading. That is the piece reward-lens is built around. A reward-lens measurement does not return a bare number, it returns [evidence](../concepts/measurement-you-can-trust.md) that carries its own credentials, and three gates decide how far that evidence can be trusted. An instrument with no scorecard cannot claim more than exploratory trust. A basis-dependent quantity refuses cross-model comparison without a shared frame. A confirmatory claim needs a preregistration frozen before the run. The lens shows you the reading; the [trust ladder](../discipline/trust-ladder.md) tells you what the reading is worth, and the two arriving together is the whole design.

## Honest positioning

This lineage is context, not a claim of kinship. reward-lens is a single-author, alpha library built on HuggingFace `transformers` and a thin layer of hooks, that borrows a framing the logit-lens family established and a design stance nnsight articulated. It is not affiliated with any of these projects and does not reproduce their scope. What it takes from them is a good idea with a track record: pick the direction that means something for your model, read the hidden states along it, and stay clear about the difference between reading and causing. For reward models that direction is unusually easy to name, which is the whole opportunity the library is built on.
