# Attribute a reward score

You have one preference the model got right or wrong, and you want to break its margin into per-component contributions: which attention layers and MLPs wrote the score.

```python
from reward_lens import RewardModel, ComponentAttribution

rm = RewardModel.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")

prompt = "A student asks: 'Why is the sky blue?' Please give a clear, accurate explanation."
chosen = ("Sunlight is a mix of all visible wavelengths. When it enters Earth's atmosphere, "
          "molecules scatter the shorter (blue) wavelengths much more strongly than the longer "
          "(red) ones — this is Rayleigh scattering. Blue light bounces around the sky in every "
          "direction, so when you look up, blue is what reaches your eyes from almost everywhere.")
rejected = ("The sky is blue because blue is the color of the sky. It has always been blue and "
            "always will be. Nobody really knows why, it's just one of those things.")

res = ComponentAttribution(rm).attribute(prompt, chosen, rejected)

for name, value in res.top_k(10, by="differential"):
    print(f"{name:>8}  {value:+.2f}")
# mlp_L31 +3.99, mlp_L30 +1.32, mlp_L29 +0.86, mlp_L28 +0.63, attn_L31 +0.51,
# mlp_L27 +0.45, mlp_L26 +0.39, mlp_L25 +0.33, mlp_L22 +0.33, mlp_L23 +0.31

res.by_type("mlp")     # contributions from MLP components only
res.plot_top_k()       # bar chart of the top components by |differential|
```

Each contribution is that component's output projected onto the reward direction, differenced between chosen and rejected. They sum, with the embedding and bias, to the final margin. On the sky-is-blue pair the last few MLPs carry almost all of it.

![Top component attributions for the sky-is-blue pair.](../assets/figures/attribution-bars.svg){ .rl-fig }

/// caption
The ten components with the largest signed share of the +24.03 margin. `mlp_L31` leads at +3.99, and the bars fall off fast through the late twenties. This is the crystallization signature: the reward becomes legible in the final layers.
///

The tall bars are where the reward is visible, not where it was decided.

## What this does and does not tell you

- **Per component, not per token.** The split is by which attention layer or MLP wrote the score, not by which words in the response. Attribution cannot tell you that "Rayleigh scattering" earned three points; it can tell you `mlp_L31` did.
- **Observational, not causal.** These are projections onto \(w_r\): they locate the reward, they do not explain it. The components attribution ranks highest are not the ones [activation patching](../tools/activation-patching.md) finds necessary. On this exact pair the two anti-correlate at Spearman \(\rho = -0.230\). If your claim is "this component is responsible," you have to [patch](../tools/activation-patching.md) it. See [observational vs causal](../concepts/observational-vs-causal.md).

!!! note "Head-level attribution is not in 1.0.0"
    `ComponentAttribution` stops at the attention layer and the MLP. There is no working per-head attribution. For head resolution, use the causal route: `ActivationPatcher.patch_all_heads(prompt, chosen, rejected)` measures each head's effect directly.

See also: [Component Attribution](../tools/component-attribution.md), [Activation Patching](../tools/activation-patching.md).
