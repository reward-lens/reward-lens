# Patch without running out of memory

You want the causal answer from activation patching, but a full sweep is blowing up your GPU or your afternoon.

The cost is structural: `patch_all_components` runs one forward-pass pair per component, so on an 8B model with 32 layers that is 64 components, roughly half an hour and 24 GB or more. You rarely need all 64.

## Patch a chosen handful

`patch_single_component` runs exactly one component and returns its effect. Loop over the few you care about:

```python
from reward_lens import RewardModel, ActivationPatcher

rm = RewardModel.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
patcher = ActivationPatcher(rm)

prompt = "A student asks: 'Why is the sky blue?' Please give a clear, accurate explanation."
chosen = ("Sunlight is a mix of all visible wavelengths. When it enters Earth's atmosphere, "
          "molecules scatter the shorter (blue) wavelengths much more strongly than the longer "
          "(red) ones — this is Rayleigh scattering. Blue light bounces around the sky in every "
          "direction, so when you look up, blue is what reaches your eyes from almost everywhere.")
rejected = ("The sky is blue because blue is the color of the sky. It has always been blue and "
            "always will be. Nobody really knows why, it's just one of those things.")

targets = [("mlp", 0), ("mlp", 4), ("mlp", 6), ("attn", 11)]
for ctype, layer in targets:
    effect = patcher.patch_single_component(
        prompt, chosen, rejected,
        layer_idx=layer, component_type=ctype,
        mode="noising", max_length=512,
    )
    print(f"{ctype}_L{layer}:  {effect:+.2f}")
# mlp_L0 +17.41,  mlp_L4 +8.78,  mlp_L6 +15.66,  attn_L11 +5.97
```

Four component patches instead of sixty-four, at a lower `max_length`, and the same causal number for each component you actually ask about.

## The levers, cheapest first

- **Triage with the observational tools.** The [Reward Lens](../tools/reward-lens.md) and [attribution](../tools/component-attribution.md) are one or two forward passes for the whole model. Run them first to find the band where the margin forms, then patch only there. The two [disagree](../concepts/observational-vs-causal.md), though: attribution credits late layers, patching keeps finding early ones necessary (`mlp_L0 +17.41` above, against attribution's `mlp_L31`), so patch a bracket around and before crystallization rather than only the top attribution bar.
- **Patch a subset, not the sweep.** `patch_single_component` for named components, or slice your loop to the layers in question. A full `patch_all_components` is the thing to avoid when memory is tight.
- **Lower `max_length`.** Every patch is two cached forward passes whose memory scales with sequence length. Dropping `max_length` from 2048 to 512 cuts the activation cache roughly fourfold.

!!! tip "Spend the cheap passes first"
    The observational tools are one or two passes; the causal ones are one pair of passes per component. Reserve patching for the handful of components a load-bearing claim actually rests on. See [interpreting results honestly](../caveats.md).

See also: [Activation Patching](../tools/activation-patching.md).
