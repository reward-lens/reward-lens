# Tools

Eleven tools, one idea. Every one of them is a projection onto the reward direction \(w_r\) or a decomposition along it. What separates them is the *question* they answer, and the strength of *claim* they let you make.

That last part is why every tool wears a tier badge. It is the single most important thing to know about a tool before you use its output in an argument.

<span class="rl-key"><span class="rl-dot rl-dot--observational"></span> **Observational** &mdash; reads activations, no intervention. Claims about where the reward *is*.</span>
<br>
<span class="rl-key"><span class="rl-dot rl-dot--causal"></span> **Causal** &mdash; intervenes and remeasures. Claims about what the reward *is caused by*.</span>
<br>
<span class="rl-key"><span class="rl-dot rl-dot--vulnerability"></span> **Vulnerability** &mdash; probes what breaks, and whether you could predict it.</span>

## Which one do you want?

```mermaid
flowchart TD
    Q["What do you want to know?"] --> W{"Where the reward<br/>lives, or what<br/>causes it?"}
    W -->|"where it lives"| OBS["Observational"]
    W -->|"what causes it"| CAU["Causal"]
    Q --> BRK["What breaks it?"] --> VUL["Vulnerability"]

    OBS --> RL["Reward Lens<br/><i>which layers decide</i>"]
    OBS --> CA["Attribution<br/><i>which components</i>"]
    OBS --> SF["SAE features<br/><i>which features</i>"]
    OBS --> CV["Concept vectors<br/><i>which concepts align</i>"]

    CAU --> AP["Activation Patching<br/><i>which components are necessary</i>"]
    CAU --> PP["Path Patching<br/><i>which path carries it</i>"]
    CAU --> DP["Divergence-aware<br/><i>is the patch trustworthy</i>"]

    VUL --> HD["Hacking Detector<br/><i>what it rewards wrongly</i>"]
    VUL --> DI["Distortion Index<br/><i>what it will reward wrongly next</i>"]
    VUL --> MC["Misalignment Cascade<br/><i>do failures correlate</i>"]
    VUL --> RC["Reward-Term Conflict<br/><i>do objectives fight</i>"]

    OBS -.->|"confirm a hypothesis"| CAU

    classDef obs fill:#0d948815,stroke:#0d9488,color:#0d9488;
    classDef cau fill:#b4530915,stroke:#b45309,color:#b45309;
    classDef vul fill:#be123c15,stroke:#be123c,color:#be123c;
    class OBS,RL,CA,SF,CV obs;
    class CAU,AP,PP,DP cau;
    class VUL,HD,DI,MC,RC vul;
```

The dotted edge is the workflow the whole library is built around: explore cheaply with the observational tools, then confirm anything load-bearing with the causal ones. They can disagree, and [when they do](../concepts/observational-vs-causal.md), the causal answer wins.

## Observational

<span class="rl-badge rl-badge--observational">Observational</span> &nbsp; Read an activation's projection onto \(w_r\). Cheap, and the right first look. Answers *where*, never *why*.

<div class="grid cards rl-obs" markdown>

-   __[Reward Lens](reward-lens.md)__

    Project every layer onto \(w_r\) and watch the margin form. Where does the preference crystallize?

-   __[Component Attribution](component-attribution.md)__

    Split the final state per component and project each. Which heads and MLPs carry the score?

-   __[SAE feature attribution](sae-features.md)__

    Decompose the reward through a sparse dictionary. Which interpretable features push it up or down?

-   __[Concept vectors](concept-vectors.md)__

    Extract a concept direction and measure its cosine with \(w_r\). Which surface concepts is the reward aligned with, and therefore hackable?

</div>

## Causal

<span class="rl-badge rl-badge--causal">Causal</span> &nbsp; Intervene on an activation and measure how the margin moves. Expensive, and the only tools that earn a causal claim.

<div class="grid cards rl-cau" markdown>

-   __[Activation Patching](activation-patching.md)__

    Swap a component between chosen and rejected. Which components are causally necessary, or sufficient?

-   __[Path Patching](path-patching.md)__

    Restrict the swap to one sender-head to receiver path. Does the effect travel that specific route?

-   __[Divergence-aware Patching](divergence-patching.md)__

    Patching with an off-distribution check. Is this causal claim built on an activation the model actually reaches?

</div>

## Vulnerability

<span class="rl-badge rl-badge--vulnerability">Vulnerability</span> &nbsp; Ask what the reward model gets wrong, and whether the failure was predictable. Each connects to a specific recent result.

<div class="grid cards rl-vul" markdown>

-   __[Hacking Detector](hacking-detector.md)__

    A/B a surface feature, hold content fixed, measure the reward swing as an effect size. Does the model reward length, confidence, formatting, flattery?

-   __[Distortion Index](distortion-index.md)__

    Predict which quality dimensions your evaluation under-covers, and therefore which get gamed. Prediction, not detection.

-   __[Misalignment Cascade](misalignment-cascade.md)__

    Test whether failures across misalignment dimensions move together into systemic risk.

-   __[Reward-Term Conflict](reward-conflict.md)__

    Measure the geometry between reward-term directions: aligned, orthogonal, or in conflict.

</div>
