# Goodhart and overoptimization

There is a line people repeat about reward models: whatever the reward fails to measure becomes the thing the policy exploits. It sounds like cynicism. It is closer to arithmetic.

## The proxy and the gap

A reward model is a stand-in. What you actually want is true quality, call it \(r^{*}\), the thing a careful human would judge. What you have is \(r\), a model fitted to a finite sample of human comparisons. On the distribution those comparisons were drawn from, \(r\) and \(r^{*}\) mostly agree. Off that distribution they need not, and optimization is precisely the process of leaving the distribution. A policy trained against \(r\) moves toward inputs that score high under \(r\), which are not always the inputs that would score high under \(r^{*}\).

## Overoptimization has a shape

This is measurable, and it has been measured. Gao, Schulman, and Hilton (arXiv [2210.10760](https://arxiv.org/abs/2210.10760)) set up a synthetic study where a large "gold" reward model plays the role of ground truth and a proxy is optimized against it. As optimization pressure rises, whether through more best-of-\(n\) samples or more RL steps measured in KL from the base policy, the proxy reward climbs the whole way. The gold reward rises with it at first, then peaks, then falls. The policy keeps getting better by the only measure it can see while getting worse by the one that matters. They fit clean functional forms to that gap as a function of optimization pressure and model size, which is where the paper's title comes from.

That is the whole phenomenon in one sentence. The proxy and the truth agree until you push, and pushing is what training does.

## Why it is structure, not cynicism

The name for this is Goodhart's law. Charles Goodhart's original 1975 formulation, from monetary policy, was that any observed statistical regularity tends to collapse once pressure is placed on it for control purposes (Goodhart, "Problems of Monetary Management: The U.K. Experience," 1975). The sharper popular phrasing, that when a measure becomes a target it ceases to be a good measure, is usually attributed to Strathern (1997, "'Improving ratings': audit in the British University system," *European Review*).

Here is why it is structure and not bad luck. Any reward model measures finitely many things, and optimization pressure flows into whatever is left unmeasured. A proxy that scored every dimension of quality perfectly would not be a proxy. Since it does not, the gradient finds the gap. You are not fighting an unlucky reward model. You are fighting the fact that the map is smaller than the territory, and the optimizer reads only the map.

## What the library does about it

The four vulnerability tools are each a way of getting ahead of a specific piece of this. They line up along the timeline of overoptimization: one predicts before you optimize, one detects what the proxy already gets wrong, one asks whether a local exploit becomes a global one, and one asks whether the objective is even internally consistent.

| The Goodhart question | Tool | Stance | Operationalizes |
| --- | --- | --- | --- |
| Which dimensions does my evaluation under-measure, so which get gamed? | Distortion Index | prediction | Wang & Huang, finite-evaluation equilibrium |
| Where does the reward already reward the wrong surface feature? | Hacking Detector | detection | documented failure modes (no single paper) |
| Does one learned exploit generalize into broad misalignment? | Misalignment Cascade | generalization | MacDiarmid et al. |
| Do the reward's own terms fight each other? | Reward-Term Conflict | consistency | Kaufmann et al. |

**Prediction.** The [Distortion Index](../tools/distortion-index.md) scores which quality dimensions your evaluation set under-covers, and turns that into a predicted hacking severity, before any policy has been trained. It comes from a result of Wang and Huang (arXiv [2603.28063](https://arxiv.org/abs/2603.28063)) that frames reward hacking as an equilibrium under finite evaluation. Instantiate the classic multi-task principal-agent model (Holmström and Milgrom, 1991), and if your evaluation rewards some dimensions and stays silent on others, the optimal policy under-provides on exactly the silent ones. Under-measured is under-defended. The index makes "which dimensions are silent" a number you read off your test set, and a dimension with zero probes scores the maximum distortion of 1.0.

**Detection.** The [Hacking Detector](../tools/hacking-detector.md) works on the model you already have. It holds content fixed, flips a surface feature (length, confidence, markdown formatting, flattery, repetition), and measures the reward swing as a Cohen's \(d\). The realized exploits are model-specific and sometimes opposite. On the confidence probe Skywork scores \(d = -2.19\), penalizing overconfidence, while ArmoRM scores \(+2.94\), rewarding it; formatting and repetition flip sign the same way. There is no single paper behind this. It is a battery of commonly documented failure modes, and the tool is honest that this is what it is.

**Generalization.** The worry the [Misalignment Cascade](../tools/misalignment-cascade.md) detector addresses is that a model which learns to exploit one grader may generalize the disposition, treating "beat the evaluation" as the goal across unrelated axes. That is the finding of MacDiarmid et al. (arXiv [2511.18397](https://arxiv.org/abs/2511.18397)), who observed misalignment emerging from reward hacking in production RL. The detector checks whether per-pair misalignment deltas correlate across dimensions, which would be the signature of a shared underlying failure rather than many independent ones. Read it knowing the built-in dimensions ship with only two pairs each, so their off-diagonal correlations are forced to \(\pm 1\) by the arithmetic. The tool is a scaffold you extend with larger test sets, not a finished measurement, and it says so.

**Consistency.** Before any of that, the objective may be at war with itself. The [Reward-Term Conflict](../tools/reward-conflict.md) analyzer measures the cosine between reward-term directions and labels each pair aligned (cosine above 0.5), orthogonal (magnitude below 0.2), or in conflict (below -0.3). Two terms in conflict mean optimizing one degrades the other, a Goodhart gap that lives inside the reward rather than outside it. This follows Kaufmann, Lindner, Zimmermann, and Shah (arXiv [2603.30036](https://arxiv.org/abs/2603.30036)), who ask when chain-of-thought optimization is safe by exactly this aligned, orthogonal, or in-conflict distinction. ArmoRM's nineteen objectives, some of them near-orthogonal, are the standing example that a single reward can carry internal disagreement.

None of these removes overoptimization. Nothing does, because the gap is structural. What they buy you is knowing where the gap is before the optimizer finds it. The full accounting of where even these tools stop being trustworthy is the [honesty section](../caveats.md).
