# Effective sample size of an eval set

**Your eval set has thirty rows. How many independent data points is that actually?**

Rarely thirty. Eval sets are built by sampling a handful of prompts across several seeds, paraphrasing a few templates, or replaying the same scenario with cosmetic edits. Every such row correlates with its siblings, and a confidence interval that treats them as independent is too tight. The honest count is the effective sample size: the number of genuinely independent observations the set is worth.

## Count the independent observations

`effective_sample_size` takes the cluster label of each row (which seed, prompt, or template it came from) and returns the Kish effective size. Thirty rows drawn from six seeds, five rows each, are worth six:

```python
from reward_lens.stats import effective_sample_size

seed_labels = [s for s in range(6) for _ in range(5)]   # 6 seeds, 5 rows each -> 30 rows
print(len(seed_labels), "rows")
print("ESS", effective_sample_size(seed_labels))
# 30 rows
# ESS 6.0
```

Six, not thirty. The five rows sharing a seed move together, so each seed contributes one independent draw, not five. Feed that six into a cluster bootstrap and the interval widens to the width the data earns.

## Detect the clones directly

When the duplication is in the content rather than a seed label, `detect_clones` finds it by hashing each row. It reports how many rows collapse to how many unique items and what fraction are duplicates:

```python
from reward_lens.stats import detect_clones

content = [f"row_from_seed_{s}" for s in range(6) for _ in range(5)]
print(detect_clones(content))
# {'n_rows': 30, 'n_unique': 6, 'weights': {'row_from_seed_0': 5, ...}, 'duplicate_fraction': 0.8}
```

Thirty rows, six unique, `duplicate_fraction` \(0.8\): four of every five rows are a copy of one already counted. The `weights` field names how many times each unique item recurs, which is exactly the cluster structure `effective_sample_size` needs.

![Thirty near-identical rows collapsing to six effective observations.](../assets/figures/clone-collapse-light.svg#only-light){ .rl-fig .rl-fig--hero }
![Thirty near-identical rows collapsing to six effective observations.](../assets/figures/clone-collapse-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**Thirty clones are not thirty data points.** Rows that share a seed or a template collapse into one another; the effective size is the count of genuinely independent draws, here six.
///

The figure shows the thirty rows on the left folding into six distinct observations on the right. The area that disappears is the illusion of precision: a naive standard error shrinks with \(\sqrt{30}\), while the real one shrinks with \(\sqrt{6}\). That gap is roughly a factor of two on every interval you report, which is the difference between a result that replicates and one that does not.

This is why every measurement in reward-lens carries its effective sample size on its uncertainty, not just a raw \(n\). A number computed on thirty clones and a number computed on thirty independent rows are not the same evidence, and the receipt says so.

See also: [the evidence store](../discipline/evidence-store.md), [the anatomy of evidence](../discipline/anatomy-of-evidence.md), [`effective_sample_size`](../reference/stats.md#reward_lens.stats.ess.effective_sample_size), [`detect_clones`](../reference/stats.md#reward_lens.stats.ess.detect_clones).
