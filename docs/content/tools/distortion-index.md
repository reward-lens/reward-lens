<span class="rl-badge rl-badge--vulnerability">Vulnerability</span>

# Distortion Index

**Which quality dimensions are under-measured, and therefore next to be gamed?**

The Hacking Detector tells you a lever moved after you built a probe for it. The Distortion Index asks the earlier question: given the evaluation you actually have, which quality dimension is least defended, and so first in line to be sacrificed? It is a prediction made before any policy trains, read off the shape of your evaluation rather than from observed behavior.

The intuition comes from economics. If a reward barely measures a dimension, an agent optimizing that reward pays almost no penalty for letting the dimension slide, so effort drains out of it and into whatever the reward does measure. A dimension your probes never touch is a dimension optimization gets to spend for free. The Distortion Index scores that exposure, one number per dimension, so you can see the gap before it is exploited.

This is the mechanism Wang and Huang formalize in [Reward Hacking as Equilibrium under Finite Evaluation](https://arxiv.org/abs/2603.28063), which instantiates the Holmström and Milgrom (1991) multi-task principal-agent model: when a principal pays on a finite, imperfect measure, the agent's best response is to reallocate effort toward what is measured and away from what is not. Reward hacking is that reallocation, and it is predictable from coverage.

## The math

For each dimension the analyzer computes an effective coverage \(C_{\text{eff}}(d)\), which grows with how many probes target the dimension and how sharply the reward model separates their preferred and dispreferred sides. The distortion index is the normalized shortfall:

\[
D(d) = 1 - \frac{C_{\text{eff}}(d)}{\max_{d'} C_{\text{eff}}(d')}
\]

A dimension with no probes has \(C_{\text{eff}} = 0\) and therefore \(D = 1.0\), the maximum exposure. The best-covered dimension is the denominator, so it sits at \(D = 0\) by construction. The index ranks dimensions against each other; it is a relative map of where your evaluation is thin, not an absolute score of coverage.

## A worked run

Probe helpfulness and safety with real diagnostic pairs, then leave a third dimension empty on purpose and watch it get flagged.

```python
from reward_lens import RewardModel, DistortionAnalyzer
from reward_lens.diagnostic_data import get_diagnostic_pairs

rm = RewardModel.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")

dimensions = ["helpfulness", "safety", "verbosity"]
probes = {
    "helpfulness": get_diagnostic_pairs(["helpfulness"]),
    "safety": get_diagnostic_pairs(["safety"]),
    "verbosity": [],     # deliberately left unprobed
}

report = DistortionAnalyzer(rm).compute_distortion_index(dimensions, probes)
report.print_summary()

report.per_dimension_distortion     # verbosity comes back at 1.0
report.under_covered_dimensions     # includes "verbosity"
report.predicted_hacking_severity
```

`verbosity`, with an empty probe list, comes back at distortion 1.0 and lands in `under_covered_dimensions`. Helpfulness and safety, actually probed, score lower and carry a smaller predicted severity. The dimension you forgot to measure is the one the report points straight at.

## How to read it

- **`per_dimension_distortion` near 1.0** is a dimension your evaluation barely constrains. Under optimization it is the cheapest thing to trade away, so it is where to expect hacking first.
- **`under_covered_dimensions`** is the shortlist: every dimension past the `distortion_threshold`. Read it as a to-do list for writing more probes.
- **`predicted_hacking_severity`** ranks the exposure by how much it is likely to cost, not just whether coverage is thin.
- **`coverage_matrix` and `effective_coverage`** are the raw material behind the index, for when you want to see which probes landed on which dimension.

## When to reach for it, and when not

Reach for it before you train, as a pre-registration of what your reward actually watches. It turns "we think our eval covers everything" into a ranked list of the dimensions it does not, while there is still time to add probes.

Hold two limits in view as you read it. First, it predicts from coverage; it does not measure realized hacking. A high distortion score warns that a dimension is exposed, not that any policy has exploited it. For that you need behavioral tests like the [Hacking Detector](hacking-detector.md). Second, the index is only as good as the probe set and the dimension list you bring. A dimension you never named cannot be scored, and a weak probe can make a dimension look covered when it is not. The index sees exactly the evaluation you hand it, which is the point, and also the catch.

## Reference

Full signatures and return types: [`DistortionAnalyzer`](../reference/vulnerability.md#reward_lens.distortion.DistortionAnalyzer).
