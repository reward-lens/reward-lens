# Getting started

Three ways in. One is a browser tab and installs nothing. One runs on your laptop this minute. The last loads a real 8B reward model on a GPU and shows you the result the rest of the site keeps coming back to. Start with whichever matches the hardware in front of you.

```bash
pip install reward-lens
```

Python 3.10 or newer. The [install page](install.md) has the extras, the Hugging Face access notes for gated models, and the honest word on memory.

## In a browser, nothing installed

The [Colab tour](https://colab.research.google.com/drive/1x5zG07HdsWlNsJmkl2ddJ1yalmwwujfY?usp=sharing) is the whole library in one runnable notebook, and it is the shortest path from curiosity to a number you measured yourself. It starts by building a reward model with a rule planted in it, so your first reading is one you can check, and it carries on through the reward direction, evidence and the trust ladder, the intervention algebra, gauge, training loops, and a preregistered study. It runs on Colab's free tier, and the parts that want a GPU say so and offer the CPU path instead.

[Open the tour in Colab](https://colab.research.google.com/drive/1x5zG07HdsWlNsJmkl2ddJ1yalmwwujfY?usp=sharing){ .md-button .md-button--primary }

## On your laptop, right now

`from_tiny` builds a real, small reward model on CPU with random weights. No download, no GPU, under a minute from a cold start. It is enough to meet every moving part: a signal, a measurement, and the receipt the measurement comes back with.

```python
from reward_lens.signals import from_tiny
from reward_lens.measure import base as mb
from reward_lens.measure.battery import DirectLinearAttribution
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.data.schema import DataView

signal = from_tiny(seed=0)
view = DataView(list(load_diagnostic_v3()["helpfulness"].items)[:8])

ev = mb.run(DirectLinearAttribution(), mb.Context(signal=signal, view=view))

print(ev.observable)                    # DirectLinearAttribution
print(ev.value["dominant_component"])   # the component that wrote each pair's reward gap
print(ev.gauge)                         # invariant
print(ev.trust)                         # TrustLevel.EXPLORATORY
```

That last line is the point of the rebuild. The measurement did not return a bare number. It returned a value wrapped in its own credentials, and the trust level reads `EXPLORATORY` because nothing has yet checked this tool against a case with a known answer. It is not the tool's job to promise it is right. It is the tool's job to say how far it has earned your belief. The [trust story](../concepts/measurement-you-can-trust.md) is where that number starts to climb.

Everything in the epistemics layer runs like this, on CPU. You can go a long way before you ever need a GPU.

## On a GPU, on a real model

The same shape, pointed at an 8B classifier reward model. Load a signal, pick a measurement, run it. Here is the reward lens on the canonical pair carried through the whole site: a good and a bad answer to "why is the sky blue?"

```python
from reward_lens.signals import load_signal
from reward_lens.measure import base as mb
from reward_lens.measure.battery import LensCrystallization
from reward_lens.data import make_pair
from reward_lens.data.schema import DataView

signal = load_signal("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", allow_download=True)

prompt = "Why is the sky blue?"
chosen = ("Sunlight is a mix of wavelengths, and air molecules scatter the short, blue ones far "
          "more than the long, red ones. That scattered blue light reaches your eyes from every "
          "direction, so the sky looks blue. It is Rayleigh scattering.")
rejected = "The sky is blue because blue is the color of the sky. It has always been that way."

pair = make_pair(prompt=prompt, chosen=chosen, rejected=rejected,
                 axis="helpfulness", seed_id="sky-blue", builder_id="docs")
ev = mb.run(LensCrystallization(), mb.Context(signal=signal, view=DataView([pair])))

ev.value["mean_crystal_frac"]     # 0.93  ->  the margin is half-formed only near layer 30 of 32
```

On Skywork the preferred answer scores about \(-2.22\) and the rejected one about \(-26.25\), a margin of \(+24.03\). The reward lens shows the two curves staying tangled and flat for most of the network, then splitting late. A crystallization fraction of 0.93 is what that shape means: the model waits until it has nearly finished building its representations before it commits.

!!! warning "This step needs a GPU"
    `load_signal` on a hub model is gated behind `allow_download=True`, and an 8B model in `bfloat16` wants roughly 16 GB of GPU memory. Without the flag the loader refuses rather than pretend, and points you at `wrap_hf_model` for a model you have already loaded, or `from_tiny` for the CPU path above. The numbers here were measured on that model. The library does not fabricate them on hardware that cannot hold it.

## Next

The [concepts](../concepts/index.md) pages are the mental model: one direction, a pair as a controlled experiment, and where the preference forms. If you would rather stay hands-on, the [tutorials](../tutorials/index.md) run two full arcs end to end, and [models and signals](../models-and-signals/index.md) answers the first question half of readers bring, which is whether any of this attaches to the grader you actually have.
