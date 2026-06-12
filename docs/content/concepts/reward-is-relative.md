# Why reward is relative

A number came out of the reward model. Skywork scores our good sky-is-blue answer at \(-2.22\). Is that good? The question has no answer, and understanding why is the difference between reading these tools correctly and fooling yourself.

## The score has a free constant

Reward models are trained under the Bradley-Terry model of pairwise preference. The probability that a human prefers response \(y_1\) over \(y_2\) is modeled as a sigmoid of the reward gap:

\[
P(y_1 \succ y_2) = \sigma\bigl(r(y_1) - r(y_2)\bigr)
\]

Look at what the training signal can and cannot pin down. It only ever sees *differences* of rewards. Add the same constant to every reward the model produces and every gap \(r(y_1) - r(y_2)\) is unchanged, so every preference probability is unchanged, so the training loss is identical. The model has no way to prefer one overall offset over another. The absolute level of the reward is not identified. It is an arbitrary zero the training left floating.

So \(-2.22\) means nothing on its own. Skywork emits raw logits that happen to sit in the negatives; ArmoRM emits a gated score in a bounded range near zero. Neither number is "the reward." The only thing either model actually commits to is how two responses compare.

!!! danger "The mistake this prevents"
    Never compare raw reward scores across models, and never read meaning into the sign or size of a single score. ArmoRM's mean reward deltas on the hacking probes are around \(0.01\) to \(0.07\); Skywork's are around \(\pm 30\). That is not because ArmoRM is thirty times more robust. It is because the two models live on different, arbitrary scales. Compare them with a scale-free statistic (an effect size, a rank correlation), never with the raw numbers. The [Hacking Detector](../tools/hacking-detector.md) reports Cohen's \(d\) for exactly this reason.

## So we plot differences, and only differences

If the absolute reward is arbitrary, plotting it is worse than useless, it is misleading, because the eye reads a level as if it meant something. Every figure in this site shows a margin instead.

The [Reward Lens](../tools/reward-lens.md) plots \(w_r^{\top}(h_{\text{chosen}} - h_{\text{rejected}})\) per layer, not the two scores side by side. [Attribution](../tools/component-attribution.md) reports each component's contribution to the *difference*. [Patching](../tools/activation-patching.md) measures the *change* in margin. When you see a curve rise from zero to its final value across depth, that zero is meaningful (no preference yet) and that final value is meaningful (the decided margin), because both are differences. A curve of raw reward would have an arbitrary zero and tell you nothing.

This is one place the reward-model view genuinely departs from generative interpretability, which has no reason to make the point. There, a logit is a logit. Here, a reward is only ever half of a comparison. Internalize it once and a lot of confusing plots become clear: they were always showing you a gap, because a gap is the only thing there is.

The longer version, including where the Bradley-Terry assumption itself starts to leak, is in [the theory section](../theory/bradley-terry.md).

Next: given that we track the margin across depth, the natural question is *when* it forms. → [Crystallization depth](crystallization.md)
