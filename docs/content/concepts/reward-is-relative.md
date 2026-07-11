# Why reward is relative

**A number came out of the reward model. Skywork scores the good "why is the sky blue" answer at \(-2.22\). Is that good?** The question has no answer, and seeing why is the difference between reading these instruments correctly and quietly fooling yourself.

## The score has a free constant

Reward models are trained under the Bradley-Terry model, where the probability a human prefers response \(y_1\) over \(y_2\) is a sigmoid of the reward gap:

\[
P(y_1 \succ y_2) = \sigma\bigl(r(y_1) - r(y_2)\bigr)
\]

Look at what the training signal can and cannot pin down. It only ever sees *differences* of rewards. Add the same constant to every reward the model produces and every gap \(r(y_1) - r(y_2)\) is unchanged, so every preference probability is unchanged, so the loss is identical. The model has no way to prefer one overall offset over another. The absolute level is not identified. It is an arbitrary zero the training left floating.

So \(-2.22\) means nothing on its own. Skywork emits raw logits that happen to land in the negatives. A different reward model emits a gated score in a bounded range near zero. Neither number is "the reward." The only thing either model actually commits to is how two responses compare. The margin of \(+24.03\) between the good answer and the bad one is real and reproducible; the \(-2.22\) is a coordinate you should never read on its own.

## Across models, the scale floats too

Within one model the scale is fixed. Multiply every reward by a constant and the sigmoid changes, so the probabilities change, so the fit to human labels changes. Training does hold the model's own scale in place.

Across two models it does not. Each was fit independently, each found its own effective temperature, and there is no shared unit between them. So a raw reward from model A and a raw reward from model B differ by an unknown shift *and* an unknown scale. Comparing them directly is a category error.

!!! danger "The mistake this prevents"
    Never compare raw reward scores across models, and never read meaning into the sign or size of a single score. Two models with very different raw ranges are not thereby more or less robust; they live on different, arbitrary scales. Compare them with a scale-free statistic instead, an effect size or a rank correlation, which is why the [bias battery](../instruments/bias-battery.md) reports Cohen's \(d\) rather than a raw reward delta.

## So we plot differences, and only differences

If the absolute reward is arbitrary, plotting it is worse than useless, because the eye reads a level as if it meant something. Every figure on this site shows a margin instead.

The [reward lens](../instruments/lens-crystallization.md) plots \(w_r^{\top}(h_{\text{chosen}} - h_{\text{rejected}})\) per layer, not two scores side by side. [Attribution](../instruments/attribution.md) reports each component's contribution to the *difference*. A patch measures the *change* in margin. When a curve rises from zero to its final value across depth, that zero is meaningful (no preference formed yet) and that final value is meaningful (the decided margin), because both are differences. A curve of raw reward would have an arbitrary zero and tell you nothing.

This is one place the reward-model view genuinely departs from generative interpretability, which has no reason to make the point. There, a logit is a logit. Here, a reward is only ever half of a comparison. Internalize it once and a lot of confusing plots resolve: they were always showing you a gap, because a gap is the only thing there is.

## The same problem, one level deeper

Here is the part that motivates the whole measurement discipline. Even the difference vectors live in each model's own coordinates. Take two reward models trained to do nearly the same job, read off their reward directions, and measure the angle between them *in raw coordinates*. The cosine comes out around \(0.005\). Essentially orthogonal. Two models that agree about which answers are good look like they share almost nothing.

That number is absurd, and it is absurd on purpose. It is not a fact about the models. It is an artifact of comparing vectors expressed in two unrelated bases, the way two maps drawn with different north poles will disagree about every heading until you align them. To compare anything across models honestly, you first put both into a shared frame, a common set of coordinates fit to the activation geometry. Do that and the near-orthogonality collapses into the real, much smaller angle.

That shared-frame move is the **gauge** idea, and it is enforced, not suggested: a quantity that depends on the basis refuses to be compared across models until you supply the frame, or the library raises. The full treatment, including how a frame is fit and what it certifies, is in [gauge and frames](../discipline/gauge-and-frames.md). The formal statement of what is and is not identifiable is in [the theory section](../theory/identifiability.md).

Next: given that we track the margin across depth, the natural question is *when* it forms. → [Crystallization depth](crystallization.md)
