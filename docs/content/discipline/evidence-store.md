# The evidence store

**Where does a number live after you measure it, and what stops it from drifting?** In most workflows a result lives in a notebook cell, gets copied into a slide, gets paraphrased into a paper, and by the third hop nobody can say which run produced it or what it cost. The measurement discipline closes that gap with a single rule: every measurement is written once, into an append-only store, with its full provenance, and everything downstream is a view that cites the store rather than a computation that invents a new number.

## Append-only, typed, queryable

The [`EvidenceStore`](../reference/core.md#reward_lens.core.store.EvidenceStore) is a JSONL log of [`Evidence`](anatomy-of-evidence.md) objects. It supports `append`, `get`, `find`, containment, and length. It does not support delete or update. A result goes in once and stays, which means the store is an audit trail, not a scratchpad: you can always ask what was measured, on what subject, at what trust level, and by which run.

```python
from reward_lens.core import make_evidence, SubjectRef
from reward_lens.core.provenance import Provenance, Cost
from reward_lens.core.store import EvidenceStore

store = EvidenceStore(path="runs/evidence")
subj  = SubjectRef(signals=("tiny:seed0",))

margin = make_evidence(observable="reward.margin", observable_version="1", subject=subj,
                       value=24.03, provenance=Provenance(cost=Cost(tokens=128)))
mid = store.append(margin)

# a derived measurement cites its parent by id
chi = make_evidence(observable="index.chi", observable_version="1", subject=subj,
                    value=0.31, provenance=Provenance(parents=(mid,), cost=Cost(tokens=64)))
cid = store.append(chi)

print(len(store), mid.split(":")[0])
# -> 2 ev
print([e.observable for e in store.parents(chi)])
# -> ['reward.margin']
print([e.observable for e in store.ancestors(chi)])
# -> ['reward.margin']
print(cid in store, "evidence:absent" in store)
# -> True False
print(chi.provenance.cost.tokens)
# -> 64
```

Two things in that snippet carry the discipline. First, the derived measurement `chi` records its `parents` by id, and the store can walk the full `ancestors` chain. A number produced from other numbers knows where it came from, and the store refuses to accept derived evidence whose parents it cannot resolve. You cannot cite a source that does not exist. Second, every measurement carries a `Cost` in its provenance, tokens here, gpu-seconds and wall-time in general, so the store meters what each result cost to produce. A result is never just a value; it is a value with a lineage and a price.

## Lineage is metered, so clones do not inflate n

Provenance is not bookkeeping for its own sake. It is what keeps the statistics honest. The most common way a reward-model evaluation lies to you is by counting near-duplicates as independent observations, and the store's lineage is how the library refuses to.

![Thirty rows generated from six seeds collapse to an effective sample size of six; a bootstrap resamples the six seeds, not the thirty rows.](../assets/figures/clone-collapse-light.svg#only-light){ .rl-fig .rl-fig--hero }
![Thirty rows generated from six seeds collapse to an effective sample size of six; a bootstrap resamples the six seeds, not the thirty rows.](../assets/figures/clone-collapse-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**Thirty clones are not thirty data points.** Six seeds, each expanded into five near-duplicate rows, look like \(n = 30\). But \(24\) of those \(30\) rows are copies (a duplicate fraction of \(0.8\)), so the effective sample size is \(6\). A cluster bootstrap resamples the six seeds, not the thirty rows, and the confidence interval comes out honestly wide.
///

The numbers under the figure are computed, not asserted:

```python
from reward_lens.stats import effective_sample_size, detect_clones

seed_labels = [s for s in range(6) for _ in range(5)]        # 30 rows, 6 seeds x 5 each
print(effective_sample_size(seed_labels))
# -> 6.0
print(detect_clones([f"h{s}" for s in range(6) for _ in range(5)])["duplicate_fraction"])
# -> 0.8
```

Because the store records the lineage of every row, an `Evidence` object's uncertainty reports a genuine effective sample size, `n_effective`, alongside the raw `n`. A measurement built on thirty clones does not get to advertise the confidence of thirty independent draws. The [effective sample size](../how-to/effective-sample-size.md) guide walks the full path from a clone-inflated eval set to a lineage-honest interval.

## Artifacts are views, not computations

Everything a reader eventually sees, a model card, an atlas, a safety case, a claim in a manuscript, is a **view** over the store. A view can only cite evidence that is already there. It never mints a number of its own.

![A card, an atlas, and a safety case all draw lines down into an append-only evidence store; a manuscript that cites an id the store cannot show fails the claims check with a nonzero exit.](../assets/figures/artifacts-as-views-light.svg#only-light){ .rl-fig .rl-fig--hero }
![A card, an atlas, and a safety case all draw lines down into an append-only evidence store; a manuscript that cites an id the store cannot show fails the claims check with a nonzero exit.](../assets/figures/artifacts-as-views-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**An artifact can only say what the store can back.** A card, an atlas, and a safety case are all read-only views. Cite an evidence id the store cannot produce and the claims check exits nonzero, so an unbacked number fails the build rather than reaching the page.
///

Two of those views enforce the rule hard enough to fail continuous integration. `check_text` scans a document for claim tags and verifies each against the store; a claim that cites a missing evidence id, or whose value drifts past its tolerance, marks the report not-ok, and the `claims` CLI turns that into a nonzero exit. `assemble_safety_case` goes further: it refuses to assemble at all unless every supporting component is both calibrated and registered, raising rather than producing a case built on exploratory evidence.

```python
from reward_lens.artifacts import check_text, assemble_safety_case
from reward_lens.artifacts.safety_case import SafetyCaseRefusal

# a manuscript claims a number and cites an evidence id the store never saw
report = check_text("margin is [[claim value=24.03 ev=evidence:absent field=value tol=0.01]]", store=store)
print(report.ok, len(report.results))
# -> False 1

# a safety case refuses to assemble from exploratory evidence
ids = {r: store.append(make_evidence(observable=f"safety.{r}", observable_version="1", subject=subj, value=1.0))
       for r in ("k", "m", "e", "h")}
try:
    assemble_safety_case(signal="tiny", k_nats_evidence=ids["k"], monitor_evidence=ids["m"],
                         erasure_evidence=ids["e"], honesty_evidence=ids["h"], store=store)
except SafetyCaseRefusal as e:
    print(type(e).__name__)
# -> SafetyCaseRefusal
```

The unbacked claim fails the check, and the safety case refuses because its four components are only exploratory, not the full trust an operational claim demands. That is the whole architecture in two calls: the store is the single source of truth, and a number that is not in it cannot appear in a card, cannot pass a claims check, and cannot hold up a safety case. The [cards and claims](../how-to/cards-and-claims.md) guide builds these views end to end, and [the anatomy of evidence](anatomy-of-evidence.md) opens up the object every one of them cites.
