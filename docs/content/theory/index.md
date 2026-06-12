# Background & theory

The rest of these docs take the reward model as given. Here is \(w_r\), here is the score, here is how to read it layer by layer. This section asks the question underneath that one. Why is there a \(w_r\) to read along at all, what did fitting it assume, and where do those assumptions stop being true?

Three threads run through everything the tools do, and each gets a page here.

The first is that the reward is a fitted probabilistic object, not a measurement. It comes from the Bradley-Terry model of pairwise preference, and that model assumes a single scalar quality and a consistent ordering that real human judgments do not always have. Knowing where the fit leaks tells you which results to trust.

The second is that any reward model is a proxy, and optimizing a proxy hard enough eventually moves it away from the thing it stood in for. That is not pessimism. It is Goodhart's law, and it has a measured shape.

The third is that four of the library's tools are each a direct computation of a specific research result. The Distortion Index, the Misalignment Cascade detector, and the Reward-Term Conflict analyzer operationalize recent papers; the Hacking Detector operationalizes a set of documented failure modes. These pages explain the ideas. The tool pages hold the citations and the code.

<div class="grid cards" markdown>

-   :material-compare:{ .lg } &nbsp; __[Bradley-Terry in depth](bradley-terry.md)__

    The preference model the reward is fitted under, the reason only margins mean anything, and the assumptions that leak: one scalar quality, a consistent ordering, one latent judge.

-   :material-target:{ .lg } &nbsp; __[Goodhart & overoptimization](goodhart.md)__

    Why the reward's blind spots become the policy's exploits, as structure and not cynicism, and where each vulnerability tool sits on the overoptimization timeline.

-   :material-family-tree:{ .lg } &nbsp; __[The lens lineage](lens-lineage.md)__

    Where the reward lens sits among methods that read a hidden state along a meaningful direction, and how `reward-lens` positions itself honestly against TransformerLens and nnsight.

</div>

## Where each result is operationalized

Each vulnerability tool folds its citation into its own page. The map from idea to tool:

- [Distortion Index](../tools/distortion-index.md): Wang and Huang, reward hacking as an equilibrium under finite evaluation.
- [Misalignment Cascade](../tools/misalignment-cascade.md): MacDiarmid et al., emergent misalignment from reward hacking in production RL.
- [Reward-Term Conflict](../tools/reward-conflict.md): Kaufmann et al., when reward terms are aligned, orthogonal, or in conflict.
- [Hacking Detector](../tools/hacking-detector.md): a battery of commonly documented surface-feature failures, with no single paper behind it, and honest about that.
