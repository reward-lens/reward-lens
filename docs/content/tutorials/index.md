# Tutorials

Pick one preference pair and never let go of it. That is the whole curriculum. A student asks why the sky is blue; one answer explains Rayleigh scattering, the other says the sky is blue because it has always been blue and nobody really knows why. The arc below points six tools at that single pair, one after another, so they stop being a list of techniques and become six views of one example.

## The arc

Each step is one tool, one question, and one number you can check against your own run.

1. **Trace it.** Project every layer onto the reward direction and watch the margin form. Skywork prefers the good answer by \(\Delta = +24.03\), and the preference does not crystallize until layer 30 of 32. Late. [Reward Lens](../tools/reward-lens.md).
2. **Attribute it.** Split the final score by component and ask which parts wrote it. The late MLPs take almost all the credit, `mlp_L31` most of all at \(+3.99\). [Component Attribution](../tools/component-attribution.md).
3. **Patch it.** Now intervene instead of read. Swap each component between the two responses and measure how the margin moves. The *early* layers turn out to be the necessary ones, `mlp_L0` most of all. On this single pair the two rankings anti-correlate at Spearman \(\rho = -0.230\): the reward is visible late but computed early, and that gap is the most important lesson in the library. [Activation Patching](../tools/activation-patching.md).
4. **Break it.** Hold the content fixed, vary a surface feature, and read the reward swing as an effect size. Skywork and ArmoRM even disagree on sign for overconfident phrasing (Cohen's \(d\) of \(-2.19\) versus \(+2.94\)): one penalizes it, the other rewards it. [Hacking Detector](../tools/hacking-detector.md).
5. **Probe its concepts.** Extract a direction for a human-legible concept and measure its cosine with \(w_r\). "Agreement" aligns at \(+0.343\), and pushing that direction into this response's activations moves the reward almost one-for-one (dose-response slope \(+0.965\)). The concept report puts the model's overall hacking risk at 47.5%. [Concept vectors](../tools/concept-vectors.md).
6. **Compare it.** Run the same pair through a second reward model. Skywork and ArmoRM trace nearly the same formation-curve *shape* (correlation around 0.85), yet ArmoRM commits earlier and noisier than Skywork's near-90%-of-depth crystallization. Same answer, different mechanism. [Compare two models](../how-to/compare-two-models.md).

## Run the whole arc

All six steps live in one notebook, run end to end on the sky-is-blue pair, with the outputs already rendered so you can read it through before running a line.

[Intro demo (notebook)](intro-demo.ipynb)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suhailnadaf509/reward-lens/blob/main/Reward_Lens_Intro_Demo.ipynb)

Open it in Colab for a free GPU, enough to load an 8B reward model in `bfloat16` and reproduce every number above yourself.

!!! note "Coming soon"
    The intro demo is the first tutorial, not the last. On the roadmap next:

    - **A written walkthrough per tool,** the notebook's arc expanded into standalone pages you can read without a GPU in front of you.
    - **A short video** tracing the sky-is-blue pair from model load to cross-model comparison.
    - **An interactive activation viewer,** for scrubbing the per-layer projections instead of reading them off a static plot.

    Until each of these lands, the [tools](../tools/index.md) pages and the intro notebook cover the same ground.
