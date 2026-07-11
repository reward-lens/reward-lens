# Background and theory

**Why is there a \(w_r\) to read along at all?** The rest of these docs take the reward model as given: here is the direction, here is the score, here is how to watch the preference form layer by layer. This section asks the question underneath that one. Where does the number come from, what did fitting it assume, and where do those assumptions stop being true?

Five threads run through everything the instruments do, and each gets a page.

The reward is a fitted probabilistic object, not a measurement. It comes from the Bradley-Terry model of pairwise preference, which pins the reward down only up to an additive constant and assumes a single scalar quality that real judgments do not always have. Optimizing any such proxy hard enough moves it away from what it stood in for, which is Goodhart's law with a measured shape rather than a slogan. The instruments themselves belong to a family, the lenses that read a hidden state along a meaningful direction, and the reward lens is the one case where the direction is handed to you exactly. Underneath the additive constant sits a larger identifiability question, the gauge freedom that lets raw cross-model coordinates lie. And underneath the scalar assumption sits the possibility that preference is not rank-one at all, that it cycles in ways no single quality number can express.

<div class="grid cards" markdown>

-   :material-compare:{ .lg } &nbsp; __[Bradley-Terry and preference](bradley-terry.md)__

    The model the reward is fitted under, why only margins carry information, and the assumptions that leak: one scalar quality, a consistent ordering, one latent judge.

-   :material-target:{ .lg } &nbsp; __[Goodhart and overoptimization](goodhart.md)__

    Why the reward's blind spots become the policy's exploits, as structure and not cynicism, and how you can price optimization pressure in nats before you spend it.

-   :material-family-tree:{ .lg } &nbsp; __[The lens lineage](lens-lineage.md)__

    Where the reward lens sits among methods that read a hidden state along a direction, how it positions against TransformerLens and nnsight, and the one thing it adds.

-   :material-axis-arrow:{ .lg } &nbsp; __[Identifiability and gauge](identifiability.md)__

    The transformations that leave preferences unchanged but make raw coordinates lie, why two reward directions can look orthogonal and mean the same thing, and how a frame sees through it.

-   :material-vector-triangle:{ .lg } &nbsp; __[When preference is not rank-one](preference-rank.md)__

    The intransitive preferences a scalar head provably cannot express, the skew-symmetric test that measures them, and why a positive recovery means the scalar summary discarded real structure.

</div>

These pages explain the ideas. Where an idea becomes a preregistered experiment you can run or refute, it turns up again in [the sixteen sciences](../sciences.md), and where the instrument that computes it lives, the [reference](../reference/index.md) holds the exact call.
