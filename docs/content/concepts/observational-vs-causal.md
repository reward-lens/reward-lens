# Observational vs causal

Two questions sound the same and are not:

- Where does the reward *accumulate*?
- What *causes* the reward?

The first is answered by reading activations. Project each component onto \(w_r\), see who has the largest projection, and you know where the reward is visible. The second is answered by intervening. Change a component's activation, rerun the model, and see whether the reward moves. One is a correlation. The other is a cause. On real reward models, they give different answers, and the whole library is organized around not confusing them.

=== "Observational"

    Read an activation and project it onto \(w_r\). Cheap: one or two forward passes for a whole model. Answers "where is the reward, along this decomposition?"

    Tools: [Reward Lens](../tools/reward-lens.md), [Component Attribution](../tools/component-attribution.md), [SAE features](../tools/sae-features.md), [Concept vectors](../tools/concept-vectors.md).

    Claim you may make: "this component's output has a large projection onto the reward direction."

    <span class="rl-badge rl-badge--observational">Observational</span>

=== "Causal"

    Intervene on an activation and measure the change in margin. Expensive: on the order of two forward passes per component. Answers "does this component *cause* the preference?"

    Tools: [Activation Patching](../tools/activation-patching.md), [Path Patching](../tools/path-patching.md), [Divergence-aware Patching](../tools/divergence-patching.md).

    Claim you may make: "ablating this component changes the margin by this much."

    <span class="rl-badge rl-badge--causal">Causal</span>

Every tool page in this site wears one of those badges. It is not decoration. It tells you exactly which of the two claims the tool licenses, and the library refuses to let an observational tool make a causal one.

## The result that forces the doctrine

Here is why this matters, on this library's own models, with numbers.

Take a preference pair. Run [Component Attribution](../tools/component-attribution.md): every component's signed share of the margin. Run [Activation Patching](../tools/activation-patching.md): every component's causal effect on the margin. Now line the two rankings up. If attribution were a good proxy for causal importance, the components attribution ranks highest would be the ones patching finds most necessary, and the two would correlate positively.

They anti-correlate.

![Attribution against patch effect for every component of a helpfulness pair. Points hug the two axes: a component matters to one method or the other, almost never both.](../assets/figures/attribution-vs-patching.svg){ .rl-fig .rl-fig--hero }

/// caption
Each point is one component. Horizontal: its attribution, the reward it appears to carry. Vertical: its patch effect, the reward it actually causes. The cloud hugs both axes and leaves the diagonal empty. `mlp_L0` (early, top left) causes a large swing in the margin but is credited almost nothing by attribution. `mlp_L31` (late, bottom right) is credited the most and causes almost nothing. Spearman \(\rho = -0.45\) on this dimension.
///

Averaged over helpfulness, correctness, and safety, the rank correlation between attribution and patch effect is \(\rho = -0.256\) on Skywork and \(-0.027\) on ArmoRM. Negative to zero. Never positive. The single canonical sky-is-blue pair shows the same thing on its own: \(\rho = -0.230\).

The mechanism behind the number is legible. Attribution credits the **last MLP layers**, because that is where the margin is largest, that is literally what [crystallization](crystallization.md) measures. Patching credits the **early layers**, because that is where the computation the late layers merely report on actually happens. Break an early layer and the whole chain downstream is wrong. Break the last MLP and the model mostly recovers. The reward becomes *visible* late and is *caused* early, and attribution can only see where it is visible.

## What to do about it

This is not a reason to distrust the tools. It is a reason to use them for what each is for.

!!! tip "The workflow the doctrine implies"
    **Explore with the observational tools. Confirm with the causal ones.**

    The Reward Lens and attribution are fast and give you a map: where the reward lives, which components carry it, what concepts it aligns with. Use them freely to generate hypotheses. But the moment a claim becomes load-bearing, "this head is responsible for the length bias," "this circuit implements the preference," you have crossed into a causal statement, and only patching can support it. Run the patch. Do not ship the attribution bar as if it were the cause.

Most overclaiming in interpretability is exactly this substitution: a clean observational picture, presented as a causal one, because the picture was clean and the intervention was never run. Reward models make the substitution unusually tempting, because the scalar target makes attribution so crisp. These docs badge every tool so you always know which claim you are entitled to. The full set of honest limits, off-distribution patching included, is the [interpreting-results-honestly](../caveats.md) section, which is where a careful reader should go next.
