# Patch without running out of memory

**You want the causal patch effects, but a full sweep is blowing up your GPU or your afternoon.**

`PatchGrid` answers the causal question: for each preference pair it splices the rejected side's activation into the chosen forward, one component at a time, and reads how far the margin collapses. A large collapse means the component is load-bearing for the preference. That is one patched forward pass per component per pair, so the cost is set by two numbers you control: how many components the grid has, and how many pairs you feed it.

The first number is the `granularity` lever.

## Component granularity is the cheap grid

`granularity="component"` (the default) patches the attention and MLP output of each layer: two cells per layer. On a 32-layer model that is 64 cells. The grid runs on the tiny CPU model as a correctness check of the mechanism, no download and no GPU:

```python
from reward_lens.signals import from_tiny
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.data.schema import DataView
from reward_lens.measure import base as mb
from reward_lens.measure.battery import PatchGrid

signal = from_tiny(seed=0)                                    # 2 layers, 4 heads, CPU
view = DataView(list(load_diagnostic_v3()["helpfulness"].items)[:5])

ev = mb.run(PatchGrid(granularity="component"), mb.Context(signal=signal, view=view))
print(ev.value["component_names"])
print(ev.value["top_component"], ev.trust, ev.gauge)
# ['attn_L0', 'mlp_L0', 'attn_L1', 'mlp_L1']
# attn_L0 EXPLORATORY invariant
```

Four cells on the two-layer model, one per attention and MLP sublayer. Each returned effect is `original_differential - patched_differential`, in reward units, gauge-invariant within one signal. The Evidence is EXPLORATORY: patching proves the mechanism runs, but nothing here has been graded against an answer key yet.

## Head granularity is the expensive grid, and gated at scale

`granularity="head"` patches every attention head, so the grid grows to layers times heads. On the tiny model that is 8 cells and still runs on CPU:

```python
ev = mb.run(PatchGrid(granularity="head"), mb.Context(signal=signal, view=view))
print(len(ev.value["component_names"]), ev.value["top_component"])
# 8 head_L0_H0
```

The same call on an 8B model is a different animal. The head grid needs the model's reward direction \(w_r\) and a patched forward per head in fp32, and an 8B model in fp32 does not fit an 8 GB GPU. That path is gated: it names the exact call and the hardware, rather than fabricating a number.

!!! warning "Needs a GPU"
    ```python
    # Requires the 8B model in fp32 (its w_r and per-head forwards). Gated on hardware.
    from reward_lens.signals import load_signal
    signal = load_signal("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", allow_download=True)
    ev = mb.run(PatchGrid(granularity="head"), mb.Context(signal=signal, view=view))
    ```
    The head-granularity result is recorded from committed artifacts: on Skywork-v0.2 the strongest causal head is `head_L12_H6` (effect \(8.47\) on the safety dimension), and the helpfulness top head is `head_L0_H29`, layer zero. The causal signal sits early, which is the opposite end of the stack from where [attribution](../instruments/attribution.md) puts the reward.

## The three levers, cheapest first

- **Stay at component granularity.** Two cells per layer instead of one per head. Drop to `"head"` only once a component-level result points you at a specific layer and you need to know which head inside it carries the effect.
- **Shrink the view.** Cost scales with the number of pairs. A grid over five well-chosen pairs answers most causal questions; a grid over five hundred rarely earns its runtime. Slice the `DataView` before you run.
- **Triage with an observational pass, then patch the bracket.** [Attribution](../instruments/attribution.md) is one forward pair for the whole model, so run it first to see where the margin becomes visible. Then patch a bracket around and *before* that band, not only the top attribution bar: on real models attribution credits late layers while patching keeps finding early ones necessary. That disagreement is the whole point of running the causal instrument, and it is treated in [observational vs causal](../concepts/observational-vs-causal.md).

For a single named component you do not need the grid at all. The [intervention algebra](../instruments/interventions.md) exposes `ComponentPatch`, one forward pair for one site, which is the smallest causal question you can ask.

See also: [the patch grid instrument](../instruments/patch-grid.md), [`PatchGrid`](../reference/measure.md#reward_lens.measure.battery.patch.PatchGrid).
