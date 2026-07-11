# Inside one reward model

**Where does a reward model keep its opinion, and can you trust the tool that shows you?** Pick one preference pair and never let go of it. A student asks why the sky is blue. One answer explains Rayleigh scattering; the other says the sky is blue because it has always been blue and nobody really knows why. Point four instruments at that single pair, in order, and two of them will contradict each other. That contradiction is the most useful thing in this library, so we walk straight into it.

The scientific numbers below were measured on a real 8B grader, `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2`, and committed as artifacts. Reproducing them needs a GPU the 8B weights fit on. The 2.0 calls that produce them are shown and marked. Every instrument also attaches to a toy model you can run on this laptop right now, and those snippets are run, with their real output pasted underneath.

## Where the model makes up its mind

Start by reading, not intervening. The reward is a linear readout of the final hidden state, so at every layer you can project the running activation onto the reward direction \(w_r\) and watch the chosen-minus-rejected margin form. The instrument is `LensCrystallization`.

!!! warning "Needs a GPU"
    This block loads the 8B grader. On an 8 GB card it will not fit; `load_signal` is download-gated by default and names the exact dispatch, so opting in with `allow_download=True` is deliberate.

    ```python
    from reward_lens.signals import load_signal
    from reward_lens.data import make_pair
    from reward_lens.data.schema import DataView
    from reward_lens.measure import base as mb
    from reward_lens.measure.battery import LensCrystallization

    signal = load_signal("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", allow_download=True)

    pair = make_pair(
        "Why is the sky blue?",
        "Sunlight is a mix of wavelengths; air molecules scatter the short (blue) ones most "
        "strongly, so blue reaches your eye from all over the sky. That is Rayleigh scattering.",
        "The sky is blue because it has always been blue. Nobody really knows why.",
        "helpfulness", seed_id="sky-is-blue", builder_id="tutorial",
    )
    view = DataView([pair])

    scores = signal.score(view)                                  # chosen -2.22, rejected -26.25
    ev = mb.run(LensCrystallization(), mb.Context(signal=signal, view=view))
    ev.value["mean_crystal_frac"]                                # 0.931  ->  layer 30 of 32
    ```

Skywork prefers the Rayleigh answer, and by a lot: reward \(-2.22\) for the good answer, \(-26.25\) for the bad one, a margin of \(+24.03\). The interesting part is *when* that margin appears. It stays near zero for two-thirds of the network and then forms in a rush, crossing half its final value only at layer 30 of 32. The model has 32 layers of machinery and does not commit until the last two.

![Two reward curves, one for the chosen answer and one for the rejected, tangled together through the early and middle layers then splitting sharply near the end.](../assets/figures/crystallization-schematic-light.svg#only-light){ .rl-fig .rl-fig--hero }
![Two reward curves, one for the chosen answer and one for the rejected, tangled together through the early and middle layers then splitting sharply near the end.](../assets/figures/crystallization-schematic-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**The preference forms late.** Two running rewards, chosen and rejected, stay tangled for most of the depth and separate only near the output. The layer where the gap reaches half its final size is the crystallization depth. Flat, then late, is the shape to recognize.
///

You will not reproduce a layer-30 result on a 2-layer toy, and the library does not pretend otherwise. What you can do on the CPU is confirm the instrument attaches and build the signal every later step reuses:

```python
from reward_lens.signals import from_tiny

signal = from_tiny(seed=0)
type(signal).__name__     # 'ClassifierRM', a real (small) reward model, no download
signal.caps               # Capability.SCORES|PREFIX_SCORES|ACTIVATIONS|GRADIENTS|HVP|LINEAR_READOUT
```

The full formation-curve story, and the empirical Skywork curve, live on the [crystallization concept page](../concepts/crystallization.md) and the [reward-lens instrument page](../instruments/lens-crystallization.md).

## Which components wrote the score

The margin is a sum of per-component contributions, because the residual stream is a running sum and \(w_r\) is fixed. `DirectLinearAttribution` splits the final differential across embedding, attention, and MLP sublayers and hands back each one's signed share.

On Skywork, the credit piles up in the last MLP layers: `mlp_L31` carries \(+3.99\) of the margin, `mlp_L30` \(+1.32\), `mlp_L29` \(+0.86\). The late layers are where the reward is largest, so the late layers are where attribution looks. That is consistent with the crystallization picture and, taken alone, completely misleading. Hold that thought.

The same instrument runs on the toy model. It comes back as **evidence**, not a bare array, and that evidence knows it is untrusted:

```python
from reward_lens.measure import base as mb
from reward_lens.measure.battery import DirectLinearAttribution
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.data.schema import DataView

view = DataView(list(load_diagnostic_v3()["helpfulness"].items)[:6])
ev = mb.run(DirectLinearAttribution(), mb.Context(signal=signal, view=view))

ev.observable                       # 'DirectLinearAttribution'
ev.trust                            # EXPLORATORY
ev.gauge                            # invariant
ev.value["n_pairs"]                 # 6
type(ev.value["dominant_component"])  # list, one dominant component per pair, not a scalar
ev.value["dominant_component"][0]     # 'attn_L0'
```

`EXPLORATORY` is the honest default: a single unbootstrapped attribution with no calibration behind it. The [next tutorial](measurements-you-can-trust.md) is entirely about what it takes to move that word. The instrument details are on the [component attribution page](../instruments/attribution.md).

## Which components cause it

Reading is not intervening. To find what *causes* the preference, you have to change a component and rerun the model. `PatchGrid` swaps each component's activation between the two responses and measures how far the margin moves.

The answer is not the late MLPs. On Skywork the largest causal effects are early: the strongest single head is `head_L12_H6`, effect \(8.47\) on the safety objective, and the top helpfulness head is `head_L0_H29`, in layer 0. Break an early layer and the whole computation downstream is wrong. Break the last MLP and the model mostly recovers, because by then the work is already done and the late layer is only reporting it.

On the CPU the causal instrument attaches the same way, through the same runner, and returns evidence at the same honest trust level:

```python
from reward_lens.measure.battery import PatchGrid

ev = mb.run(PatchGrid(granularity="component"), mb.Context(signal=signal, view=view))
ev.trust                    # EXPLORATORY
ev.gauge                    # invariant
sorted(ev.value.keys())     # [..., 'mean_effect', 'top_component', 'top_components', ...]
```

Head-level patching on an 8B model is memory-hungry and gated to real hardware; the mechanics, and how to run it without exhausting a GPU, are on the [patch grid page](../instruments/patch-grid.md).

## The two answers disagree

Now line the rankings up. If attribution were a good proxy for causal importance, the components it credits most would be the ones patching finds most necessary, and the two would correlate positively.

They do not. Averaged over twelve objective dimensions on Skywork-v0.2, the Spearman rank correlation between attribution and patch effect is \(\rho = -0.171\). Negative. On individual dimensions it is more negative still: \(-0.441\) on code correctness, \(-0.306\) on helpfulness. The reward becomes *visible* late and is *caused* early, and attribution can only see where it is visible.

![Attribution on one axis and patch effect on the other, for every component of a preference pair. The points hug both axes and leave the diagonal empty.](../assets/figures/attribution-vs-patching-light.svg#only-light){ .rl-fig .rl-fig--hero }
![Attribution on one axis and patch effect on the other, for every component of a preference pair. The points hug both axes and leave the diagonal empty.](../assets/figures/attribution-vs-patching-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**What explains the score is not what causes it.** Each point is one component. Horizontal is its attribution, the reward it appears to carry; vertical is its patch effect, the reward it actually moves. The cloud hugs both axes and avoids the diagonal: a component matters to one method or the other, almost never both.
///

This is not a bug and it is not a reason to distrust either instrument. It is the reason the library labels every instrument observational or causal and refuses to let a reading masquerade as a cause. The result is not universal, which is the honest part: on ArmoRM the same correlation is \(+0.047\), near zero and slightly positive. A number that flips sign between models is exactly the kind of number you must not ship on one model and assume on the next. The full argument is the [observational-vs-causal concept](../concepts/observational-vs-causal.md), and the honest limits, off-distribution patching included, are in [interpreting results honestly](../caveats.md).

## Scan it for bias

The same machinery answers a blunter question: does this model reward something it should not? `BiasBattery` holds the content roughly fixed, varies one surface feature per axis, and reports the reward swing as a standardized effect size with an effective sample size that refuses to count near-duplicate rows as independent.

```python
from reward_lens.measure.battery import BiasBattery

d = load_diagnostic_v3()
axes = ["helpfulness", "verbosity", "sycophancy"]
items = [pair for ax in axes for pair in list(d[ax].items)[:4]]
ev = mb.run(BiasBattery(), mb.Context(signal=signal, view=DataView(items)))

ev.trust                    # EXPLORATORY
ev.value["n_axes"]          # 3
sorted(ev.value.keys())     # ['max_abs_effect_size', 'n_axes', 'per_axis', 'strongest_axis']
```

On the toy model the effect sizes are noise, as they should be for a randomly initialized head. The point is the shape of the answer: one effect size per axis, each carrying its own honest sample count, rather than a single verdict. Two real models can rank the same surface feature very differently, even disagree on its sign, which is precisely why the battery reports per-model effect sizes instead of a universal rule. The worked campaign, and the specific recipe for length bias, are on the [bias battery page](../instruments/bias-battery.md) and in [detect length bias](../how-to/detect-length-bias.md).

## Where this goes

You have taken one model apart and caught two of its instruments disagreeing. The natural next question is what any single number here is actually worth, and that is a discipline of its own: [measurements you can trust](measurements-you-can-trust.md) builds it from the ground up, entirely on the CPU. If instead you want to point these instruments at your own grader, [install](../getting-started/install.md) and [models and signals](../models-and-signals/index.md) are the way in.
