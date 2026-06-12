# The reward-direction picture

Interpretability tools built for generative models assume the output is a distribution over tokens. A reward model breaks that assumption. It outputs one number. That single difference changes what a lens is, what attribution decomposes, and what a "circuit" even means here, and it is worth installing the right mental model before you touch the API.

The model is small. Six terms carry the whole library, and the rest of the site reuses them exactly as defined here.

## The six terms

**Reward direction** (\(w_r\)). The weight vector of the reward head. A reward model ends in a single linear layer, so its score is \(r = w_r^{\top} h + b\): the final hidden state \(h\) read out along \(w_r\). This vector is fixed by the trained model and known exactly. You do not learn it or probe for it. It is the one direction the score reads along. → [The reward direction](reward-direction.md)

**Projection.** The quantity \(w_r^{\top} h\): how far an activation reaches along the reward direction. Applying this one operation to different activations is what every tool does.

**Margin** (\(\Delta\)). The reward of the chosen completion minus the reward of the rejected one. This is the only quantity that carries meaning, because absolute reward is arbitrary. Every plot in these docs shows a difference, never a level. → [Why reward is relative](reward-is-relative.md)

**Component contribution.** One component's output (an attention layer, an MLP) projected onto \(w_r\): its signed share of the reward. Because the readout is linear, these shares sum to the score. → [Preference geometry](preference-geometry.md)

**Observational vs causal.** Reading an activation's projection is observational. Intervening on the activation and measuring how the reward moves is causal. They answer different questions, and on real models they can disagree, hard. → [Observational vs causal](observational-vs-causal.md)

**Crystallization depth.** The layer where the margin first reaches half its final value: where the model has, in effect, half made up its mind. A reward-model-native measurement with no generative analog. → [Crystallization depth](crystallization.md)

## One example, everywhere

Every page in this site traces the same preference pair. A student asks why the sky is blue. One answer explains Rayleigh scattering. The other says the sky is blue because it has always been blue. Skywork scores the good answer at \(-2.22\) and the bad one at \(-26.25\), a margin of about \(+24\). You will watch that margin form across layers, see which components wrote it, patch to find which ones cause it, and probe the concepts it aligns with. Six tools, one pair, until the mental model is yours.

<div class="grid cards" markdown>

-   :material-arrow-projectile:{ .lg } &nbsp; __[The reward direction](reward-direction.md)__

    What \(w_r\) is, and why a known answer direction makes reward models a privileged target.

-   :material-vector-difference:{ .lg } &nbsp; __[Preference geometry](preference-geometry.md)__

    Chosen and rejected as two points. The margin is their difference, projected.

-   :material-scale-balance:{ .lg } &nbsp; __[Why reward is relative](reward-is-relative.md)__

    Bradley-Terry says only margins mean anything. So we only ever plot differences.

-   :material-chart-line-variant:{ .lg } &nbsp; __[Crystallization depth](crystallization.md)__

    Where the preference forms. On Skywork, around 90% of the way through.

-   :material-call-split:{ .lg } &nbsp; __[Observational vs causal](observational-vs-causal.md)__

    The doctrine of this library, and the result that forces it.

</div>
