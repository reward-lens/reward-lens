# Studies and preregistration

**What stops you from finding the result you wanted?** In most analysis, nothing. You run the tools, you see a pattern, you write it up, and the write-up is graded by the person who most wants it to be true. A preregistered study removes that person from the loop. You state the prediction, freeze it with a hash, and hand a runner you do not control the job of checking the data against the frozen prediction. The author never grades their own exam.

## A study is a spec, not a narrative

A [`StudySpec`](../reference/studies.md) is a typed object. It names an id, a title, the science it belongs to, the subjects it runs on, and a tuple of hypotheses. Each hypothesis carries a **prediction**, and a prediction is machine-checkable, not prose: a metric, a comparator, and a threshold. "Crystallization is late" is a vibe. `Prediction(metric="mean_crystal_frac", comparator=">=", threshold=0.9)` is a claim a runner can adjudicate without asking you what you meant.

```python
from reward_lens.studies import Prediction

p = Prediction(metric="mean_crystal_frac", comparator=">=", threshold=0.9)
print(p.check(0.931), p.check(0.803))   # Skywork clears it, ArmoRM does not
# -> True False
```

The predictions that would *sink* the study are first-class too. A [`KillCriterion`](../reference/studies.md) is the same shape as a prediction, a metric and a comparator and a threshold, but it encodes a condition under which the study is abandoned rather than confirmed. Pre-committing to what would kill a study is how you stop yourself from quietly relaxing the bar when the data comes in soft.

## Freezing puts the prediction out of reach

Once the spec is written, `freeze` seals it. The frozen study carries a content hash of the spec and the git sha of the code that produced it, and after the seal there are no edits:

```python
from reward_lens.studies import StudySpec, Hypothesis, Prediction, KillCriterion, SubjectQuery, freeze

spec = StudySpec(
    id="crystallizes-late",
    title="Skywork crystallizes in the last tenth of depth",
    science="embryology",
    hypotheses=(Hypothesis(
        id="H1",
        statement="Mean crystallization fraction is at least 0.9 on Skywork-v0.2.",
        prediction=Prediction(metric="mean_crystal_frac", comparator=">=", threshold=0.9),
        scoreboard_row="T7",
    ),),
    analysis="studies.crystallizes_late.analyze",
    subjects=SubjectQuery(signals=("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",)),
    kill_criteria=(KillCriterion(
        id="K1", metric="n_pairs", comparator="<", threshold=8,
        description="too few pairs to trust the fraction"),),
)

frozen = freeze(spec)
print(frozen.study_id)
# -> study:crystallizes-late@v1#95e7b9ac
print(frozen.spec_hash)
# -> spec:95e7b9ac182d4ec5045aca72d36fa2cc
print(freeze(spec).spec_hash == frozen.spec_hash)   # re-freezing the same spec is byte-stable
# -> True
```

The `study_id` embeds the first eight characters of the spec hash, so the identity of a study is a function of its content. Change the threshold from \(0.9\) to \(0.8\) and you get a different hash and a different study, on the record, not a silent edit to the old one. The `FrozenStudy` also stamps the git sha of the working tree (with a `+dirty` marker if it had uncommitted changes), so the exact code state is pinned alongside the exact claim. Freeze predates the run; that is the entire content of the word **registered**, and it is worth saying plainly: registered means the prediction came before the data, nothing more.

![The study lifecycle: an editable draft is frozen to a content hash and git sha, then run, with the runner checking each prediction against the frozen spec and exiting confirmed to the scoreboard or refuted when a kill criterion fires.](../assets/figures/study-lifecycle-light.svg#only-light){ .rl-fig .rl-fig--hero }
![The study lifecycle: an editable draft is frozen to a content hash and git sha, then run, with the runner checking each prediction against the frozen spec and exiting confirmed to the scoreboard or refuted when a kill criterion fires.](../assets/figures/study-lifecycle-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**The seal comes before the run.** A draft `StudySpec` is editable. Freezing turns it into a content hash plus a git sha, and nothing changes after that. The runner then checks each result against the *frozen* prediction, not against the author's later reading, and a fired kill criterion exits cleanly as refuted, never as a program error.
///

## The runner adjudicates, not the author

`run_study` takes the frozen study, computes the metrics, and compares them to the predictions that were sealed before any data existed. Its verdict is one of `confirmed`, `refuted`, or `inconclusive`, and it reads that verdict off the frozen thresholds, not off whatever the analysis code would like to conclude now. If a kill criterion fires, the `StudyResult` records `killed` and `killed_by`, and the study exits refuted. A refutation is a clean, expected exit, not a crash. That distinction is deliberate: a discipline where the null result throws an exception is a discipline that punishes honesty.

## The scoreboard keeps refutations in view

Confirmed and refuted studies both land on a **scoreboard** of standing claims, the rows `T1` through `T14`. The first eight are theorems, results that follow from the construction and are treated as closed. The last six are candidate laws, open empirical claims the studies accumulate evidence for and against. Each row carries a status, `open`, `confirmed`, `refuted`, or `mixed`, and a hypothesis names the row it speaks to so its verdict updates the right claim.

The point of the scoreboard is that a refutation is exactly as visible as a confirmation. A candidate law that a study kills flips to `refuted` on the same board, in the same font, as one a study supports. Nothing gets to be quietly forgotten because it did not work.

The repository ships sixteen preregistered specs, one per science plus two meta-studies, frozen in the tree. Several ran and confirmed during the build against real registered evidence: the susceptibility index tracks best-of-\(n\) drift at Spearman \(0.958\), the contested-probe direction comes out orthogonal to \(w_r\), the rollout recorder names a hack three steps before it lands. The population-scale sweep across many models is designed, wired with cost accounting, and pending the hardware. The specs are on the record either way, which is the whole idea: you can read what was predicted before you read what happened. The runnable end-to-end path is in [freeze and run a study](../how-to/freeze-and-run-a-study.md); the standing claims live on [the sciences](../sciences.md).
