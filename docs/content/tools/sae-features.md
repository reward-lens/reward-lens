<span class="rl-badge rl-badge--observational">Observational</span>

# SAE features

**What interpretable features does the reward decompose into?**

Attribution splits the reward across the model's own parts, its attention blocks and MLPs. Those parts are not especially interpretable: "mlp_L31 wrote 4 points of margin" tells you where, not what. A sparse autoencoder trades the model's basis for a learned one, a large dictionary of directions where, ideally, each direction stands for a single readable feature. Decompose the reward along *that* basis and the question changes from "which layer" to "which feature": does this reward respond to hedging, to citations, to a confident tone, to length?

The catch, and it is a real one, is that you have to train the dictionary first, and the features it hands back are a proposal about what the reward tracks, not a proof. Both of those shape how you should use this.

## The math

A trained SAE encodes a hidden state \(h\) as a sparse code \(f\) over a dictionary \(D\) whose columns \(d_i\) are feature directions, so \( h \approx D f = \sum_i f_i d_i \). The reward is linear in \(h\), so substitute:

\[ r = w_r^{\top} h + b \approx \sum_i f_i \bigl(w_r^{\top} d_i\bigr) + b \]

That factorization gives every feature two numbers worth keeping apart:

- **Reward alignment** \( w_r^{\top} d_i \): how much the reward direction points along feature \(i\). A property of the feature and the reward head, fixed across inputs. Large positive means "when this feature is on, the reward goes up."
- **Per-input contribution** \( f_i\,(w_r^{\top} d_i) \): the alignment scaled by how strongly the feature actually fires on a given response. Signed, and specific to that input.

A feature can be highly aligned yet contribute nothing on an input where it never fires. The alignment tells you what the reward *could* respond to. The contribution tells you what it *did* respond to, here.

## A worked run

The pipeline has three steps: collect activations, train the dictionary on them, then read the features against \(w_r\). The diagnostic set gives you a small, labeled batch of responses to collect over.

```python
from reward_lens import RewardModel
from reward_lens.sae import ActivationCollector, SAETrainer, FeatureAnalyzer
from reward_lens.diagnostic_data import get_all_prompts_and_responses

rm = RewardModel.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")

rows = get_all_prompts_and_responses()          # 30 labeled (prompt, response) rows
prompts   = [r["prompt"]   for r in rows]
responses = [r["response"] for r in rows]

acts = ActivationCollector(rm).collect(prompts, responses, layer=24)   # (30, d_model)

sae = SAETrainer(rm.d_model, n_features=None, k=32).train(acts, n_epochs=5)
```

`n_features=None` defaults the dictionary to eight times `d_model`, and `k=32` is the top-k sparsity: each input activates at most 32 features. With Skywork's `d_model` of 4096 that is a 32,768-feature dictionary, which is the honest reason this tool is in a different cost class from the rest of the library. Everything else here is a forward pass or two. Training a `TopKSAE` is real training, minutes to hours on a GPU, and thirty activations is a smoke test: a dictionary you would trust wants orders of magnitude more collected data.

With a trained SAE, the analyzer reads the features:

```python
analyzer = FeatureAnalyzer(sae, rm)

analyzer.top_reward_features(10)      # feature indices most aligned with w_r, with alignments
analyzer.bottom_reward_features(10)   # the features that most oppose it

analyzer.decompose_reward_for_input(prompts[0], responses[0], layer=24)
```

`top_reward_features` and `bottom_reward_features` rank features by \( w_r^{\top} d_i \), the input-independent alignment, so they answer "what is this reward built to respond to." `decompose_reward_for_input` runs one response through and returns the per-feature contributions \( f_i (w_r^{\top} d_i) \), the features that actually fired and moved the score on that input. The SAE exposes both underneath: `sae.feature_reward_alignments(rm.reward_direction)` for the whole alignment vector, `sae.decompose_reward(x, rm.reward_direction)` for the per-feature split of a single activation.

## How to read it

- A feature at the top of `top_reward_features` is a direction the reward up-weights. If it turns out to name a surface property, a confident register, heavy formatting, that is a lead on a possible bias, the same story the [Hacking Detector](hacking-detector.md) tests directly.
- A feature at the bottom is one the reward penalizes. Refusals, hedging, and off-topic drift tend to land here on a helpfulness-tuned model.
- `plot_alignment_histogram` shows the whole distribution of alignments at once. A long positive tail means a handful of features carry most of the reward's directionality, which is the shape you hope for if the features are going to be interpretable.
- The per-input decomposition is the one to check hardest. It names which learned features carried the score *for this response*, exactly the kind of claim worth confirming before you lean on it.

## When to reach for it, and when not

Reach for an SAE when the model-basis tools have taken you as far as they can and you want features in *interpretable* terms rather than layer indices, or when you are hunting for one specific learned concept the reward might key on. It is the tool that turns "the last MLP carries the margin" into "the reward is up-weighting this readable feature."

Do not reach for it first, and do not reach for it for a quick look. It is the one part of the library that asks for training, real data and GPU time, before it says anything, and the dictionary you get is only as good as the activations you trained it on.

Read its output as observational, because that is what it is. `top_reward_features` finds directions *aligned* with \(w_r\), and alignment is a cosine, not a cause. A feature can align with the reward direction and still not be what the model computes the reward from, exactly the gap [attribution and patching expose](../concepts/observational-vs-causal.md) elsewhere in the library. Treat a high-alignment feature as a hypothesis about what the reward tracks, and if the claim is going to matter, confirm it causally by steering the feature and watching whether the reward actually moves. The [interpreting-results-honestly](../caveats.md) section explains why that observational-causal gap is the default assumption here, not the exception.

## Reference

Full signatures and return types: [SAE tools](../reference/representation.md).
