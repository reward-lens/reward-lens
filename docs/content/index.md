# reward-lens

Every model trained with RLHF was shaped by a reward model. That reward model is the thing that sat in the loop and decided, on pair after pair, which of two answers was better. It is the closest thing the whole pipeline has to a written-down definition of what we asked for. And almost nobody has looked inside one.

That is strange, because the reward model is exactly where alignment gets decided. A policy does not optimize your intentions. It optimizes the number this model hands back. Whatever the reward model fails to measure becomes the precise thing the policy is free to exploit. If you want to know why a model learned to pad its answers, or agree with whatever you said, or wrap everything in confident-sounding structure, the honest place to look is not the policy. It is the function that rewarded it.

`reward-lens` is the instrument for looking. It is built on one observation that turns out to organize the entire subject.

## A reward model has exactly one output direction

A reward model is a language model with its vocabulary head removed and a single linear layer bolted on. The score it produces is a dot product:

\[
r = w_r^{\top} h + b
\]

The final hidden state \(h\) is read out along one fixed vector \(w_r\). That vector is the **reward direction**. It is not something you probe for or approximate. It is sitting in the model's weights, known exactly, the same for every input. A generative model spreads its answer across fifty thousand logits. A reward model concentrates it into one number along one line.

![A residual-stream state projected onto the reward direction.](assets/figures/reward-projection.svg){ .rl-fig .rl-fig--hero }

/// caption
The reward is how far the hidden state reaches along \(w_r\). Everything downstream is a way of asking where that reach comes from.
///

Once you see the reward as a projection, every tool in this library becomes a variation on one move. Project each layer's activation onto \(w_r\) and you can watch the preference form (the [Reward Lens](tools/reward-lens.md)). Split the final state into its parts and project each onto \(w_r\) and you get a per-component ledger ([Component Attribution](tools/component-attribution.md)). Intervene on a component and remeasure the projection and you get a causal test ([Activation Patching](tools/activation-patching.md)). Same direction, different questions.

## The one result to know before you trust anything

Here is the finding that should make you take the rest seriously, because it is the kind of thing a library selling itself would bury.

On this library's own models, the cheap observational tools and the expensive causal ones **disagree about which components matter**. Rank the components by how much reward attribution assigns them, rank them again by how much causal patching says they carry, and the two rankings correlate at Spearman \(\rho = -0.256\) on Skywork. Negative. The last MLP layers dominate attribution. The early layers dominate patching. The place the reward visibly accumulates is not the place that causes it.

That is not a bug to hide. It is the most useful thing these docs have to tell you, and it sets the rule the whole library runs on: **read with the observational tools, confirm with the causal ones, and never quote one as if it were the other.** The [honesty section](caveats.md) is where this lives, and it is a section, not a footnote.

## The tools

Every tool wears a tier. Observational tools read activations and stay quiet about cause. Causal tools intervene and earn the right to a causal claim. Vulnerability tools ask what breaks and whether you could have seen it coming.

<span class="rl-key"><span class="rl-dot rl-dot--observational"></span> Observational &nbsp; <span class="rl-dot rl-dot--causal"></span> Causal &nbsp; <span class="rl-dot rl-dot--vulnerability"></span> Vulnerability</span>

<div class="grid cards rl-obs" markdown>

-   __Reward Lens__

    Which layers have already decided the winner? Project every layer onto \(w_r\) and watch the margin form.

    [:octicons-arrow-right-24: Reward Lens](tools/reward-lens.md)

-   __Component Attribution__

    Which heads and MLPs wrote the reward? A signed, per-component decomposition of the final score.

    [:octicons-arrow-right-24: Component Attribution](tools/component-attribution.md)

-   __SAE feature attribution__

    What interpretable features does the reward decompose into? Split the score through a sparse dictionary.

    [:octicons-arrow-right-24: SAE features](tools/sae-features.md)

-   __Concept vectors__

    Does a surface concept line up with \(w_r\)? Extract a direction and measure its alignment with reward.

    [:octicons-arrow-right-24: Concept vectors](tools/concept-vectors.md)

</div>

<div class="grid cards rl-cau" markdown>

-   __Activation Patching__

    Which components are causally necessary? Swap one between chosen and rejected and measure the change.

    [:octicons-arrow-right-24: Activation Patching](tools/activation-patching.md)

-   __Path Patching__

    Does one head reach the reward through one later component? A two-hop causal path test.

    [:octicons-arrow-right-24: Path Patching](tools/path-patching.md)

-   __Divergence-aware Patching__

    Is your intervention still on-distribution? Patching with a reliability score attached.

    [:octicons-arrow-right-24: Divergence-aware Patching](tools/divergence-patching.md)

</div>

<div class="grid cards rl-vul" markdown>

-   __Hacking Detector__

    Does the reward reward the wrong thing? An A/B suite over length, confidence, formatting, sycophancy, repetition.

    [:octicons-arrow-right-24: Hacking Detector](tools/hacking-detector.md)

-   __Distortion Index__

    Which quality dimensions are under-measured, and therefore next to be gamed? Prediction before the fact.

    [:octicons-arrow-right-24: Distortion Index](tools/distortion-index.md)

-   __Misalignment Cascade__

    Do failures across dimensions move together into systemic risk?

    [:octicons-arrow-right-24: Misalignment Cascade](tools/misalignment-cascade.md)

-   __Reward-Term Conflict__

    Are two reward terms aligned, orthogonal, or pulling against each other?

    [:octicons-arrow-right-24: Reward-Term Conflict](tools/reward-conflict.md)

</div>

## Start here

If you have a GPU and fifteen minutes, [install it and trace your first pair](getting-started/index.md). You will load Skywork, run one preference pair through the Reward Lens, and see where the model made up its mind.

If you would rather understand the idea before the API, read [the reward-direction picture](concepts/index.md) first. It is five short pages and everything else assumes them.

The intro notebook runs the whole arc on one example, end to end, on a free GPU:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suhailnadaf509/reward-lens/blob/main/Reward_Lens_Intro_Demo.ipynb)

```bash
pip install reward-lens
```

!!! note "What you need"
    Model weights, and enough GPU memory to hold them. An 8B reward model in `bfloat16` wants roughly 16 GB for the observational tools. `reward-lens` wraps any HuggingFace reward model directly, so if `transformers` can load it, you can open it up. API-only models are out, because there is nothing to hook.
