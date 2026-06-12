# Train an SAE on reward activations

You want to decompose a reward model's activations into sparse, interpretable features and see which ones push the reward up.

Collect activations at one layer, train a top-k sparse autoencoder on them, then read the features most aligned with the reward direction.

```python
from reward_lens import RewardModel
from reward_lens.sae import ActivationCollector, SAETrainer, FeatureAnalyzer, TopKSAE
from reward_lens.diagnostic_data import get_all_prompts_and_responses

rm = RewardModel.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")

rows = get_all_prompts_and_responses()               # 30 diagnostic rows; swap in your own corpus
prompts   = [r["prompt"] for r in rows]
responses = [r["response"] for r in rows]

LAYER = 20
acts = ActivationCollector(rm).collect(prompts, responses, layer=LAYER)   # (N, d_model)

sae = SAETrainer(
    d_model=rm.d_model,
    n_features=rm.d_model * 8,     # 8x expansion
    k=32,
    batch_size=8,                  # see the note: the 4096 default drops a tiny corpus
).train(acts, n_epochs=5)

fa = FeatureAnalyzer(sae, rm)
for idx, align in fa.top_reward_features(10):
    print(f"feature {idx:6d}   alignment {align:+.3f}")

feats, total = fa.decompose_reward_for_input(prompts[0], responses[0], layer=LAYER)

sae.save("skywork_L20_sae")           # writes a directory
reloaded = TopKSAE.load("skywork_L20_sae")
```

`top_reward_features` ranks features by the cosine of their decoder direction with \(w_r\): the ones that, when active, move the reward most. `decompose_reward_for_input` runs one input through the SAE and returns each feature's signed contribution plus the reconstructed total, from the identity \( r \approx \sum_i f_i (w_r^\top d_i) + b \).

!!! warning "30 rows is a smoke test, not an SAE"
    `get_all_prompts_and_responses` returns 30 rows. That is enough to check the pipeline runs end to end, and nothing more. Interpretable features need a real corpus, thousands to millions of activations. Point `collect` at your own dataset and expect training to run for hours, not seconds.

!!! note "Batch size on a small corpus"
    `SAETrainer` defaults to `batch_size=4096` with `drop_last=True`. Collect 30 rows and you get 30 activation vectors, fewer than one batch, so training would see zero batches and do nothing. Drop `batch_size` to something small for a smoke test and raise it again for a real corpus.

See also: [SAE feature attribution](../tools/sae-features.md).
