# Freeze and run a study

**How do you make a claim that provably predates the data it rests on?**

You freeze the prediction before you look. A `StudySpec` states a hypothesis, a checkable prediction, and a kill criterion; `freeze` stamps it with a content-derived id and the current git sha so the prediction cannot be edited after the fact; `run_study` executes the analysis and then adjudicates the frozen prediction against the measured metric. The analysis does not get to grade itself. That separation is gate 3, registration, and it is what turns a confirmed result into a REGISTERED one.

## Write the analysis and the spec

The analysis measures under the study, so its Evidence is stamped REGISTERED, and returns the metric its prediction named. Here it reads crystallization depth on the tiny CPU model:

```python
from reward_lens.signals import from_tiny
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.data.schema import DataView
from reward_lens.measure.battery import LensCrystallization
from reward_lens.studies import (StudySpec, Hypothesis, Prediction, KillCriterion,
                                 SubjectQuery, StudyResult, freeze, run_study)

signal = from_tiny(seed=0)
view = DataView(list(load_diagnostic_v3()["helpfulness"].items)[:6])

def analysis(run):
    ev = run.measure(LensCrystallization(), run.signal("primary"), view=view)
    return StudyResult(outcomes={}, metrics={"crystal_frac": float(ev.value["mean_crystal_frac"])},
                       summary="tiny lens run")

def spec(threshold, comparator):
    return StudySpec(
        id="tiny-crystal",
        title="Does the margin crystallize before the final layer?",
        science="S07-embryology",
        hypotheses=(Hypothesis(
            id="H1",
            statement="crystallization fraction meets the registered bound",
            prediction=Prediction(metric="crystal_frac", comparator=comparator, threshold=threshold),
            scoreboard_row="T9"),),
        analysis="__main__.analysis",
        subjects=SubjectQuery(signals=("mfp:tiny",)),
        kill_criteria=(KillCriterion(id="K1", metric="crystal_frac", comparator=">", threshold=0.99,
                                     description="frac at the ceiling implies a scoring bug"),))
```

The analysis returns metrics only. It deliberately sets no outcomes: adjudication belongs to the runner, against the prediction that was frozen.

## Freeze, then run

Freezing yields a stable, content-addressed id. Two freezes of the same spec produce the same id; change the registered threshold and the id changes, so a moved goalpost is visible in the identifier itself:

```python
frozen = freeze(spec(1.0, "<"))
print(frozen.study_id)
# study:tiny-crystal@v1#2ee75a88
```

The `#2ee75a88` is the hash of the spec's content, and the frozen study also records the git sha of the checkout it was frozen at. Now run it. A prediction the data meets confirms; one it does not meet refutes; both are returned as plainly as each other:

```python
from reward_lens.core.store import EvidenceStore
import tempfile
store = EvidenceStore(tempfile.mkdtemp())

_, confirmed = run_study(spec(1.0, "<"), subjects={"primary": signal}, store=store, analysis_fn=analysis)
_, refuted   = run_study(spec(0.5, ">"), subjects={"primary": signal}, store=store, analysis_fn=analysis)

print("crystal_frac", round(confirmed.metrics["crystal_frac"], 4))
print("frac < 1.0 ->", confirmed.outcomes["H1"], "| killed", confirmed.killed)
print("frac > 0.5 ->", refuted.outcomes["H1"], "| killed", refuted.killed)
print("evidence trust", store.get(confirmed.evidence[0]).trust)
# crystal_frac 0.1667
# frac < 1.0 -> confirmed | killed False
# frac > 0.5 -> refuted | killed False
# evidence trust REGISTERED
```

On the shallow tiny model the margin crystallizes at \(0.17\) of depth, so "crystallizes before the last layer" confirms and "crystallizes deep" refutes. Passing `analysis_fn` runs the function directly; a committed study instead resolves the spec's `analysis` dotted path, which is how the same run reproduces from the frozen record alone. The kill criterion did not fire, so no scoring-bug alarm was raised.

![The frozen spec, not the author, decides the outcome.](../assets/figures/study-lifecycle-light.svg#only-light){ .rl-fig .rl-fig--hero }
![The frozen spec, not the author, decides the outcome.](../assets/figures/study-lifecycle-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**The spec adjudicates, not the author.** A prediction is frozen with a content hash and git sha before the data is seen; the runner checks the measured metric against that frozen prediction and records confirmed or refuted with equal weight.
///

The lifecycle in the figure has one irreversible step: the freeze. Everything before it is editable, everything after it is a matter of record. The measured metric flows into the adjudication from the right, and the prediction it is checked against was fixed on the left before any number existed. That is the entire content of the word "registered": the claim predated its evidence.

!!! note "What REGISTERED does and does not say"
    REGISTERED means the prediction was frozen before the data, nothing more. It is orthogonal to whether the result is calibrated: a registered refutation on an uncalibrated instrument is still a registered refutation. Registration stops one specific move, editing the hypothesis after seeing the answer, and makes no other promise.

See also: [studies and preregistration](../discipline/studies-and-preregistration.md), [`run_study`](../reference/studies.md#reward_lens.studies.runner.run_study), [`freeze`](../reference/studies.md#reward_lens.studies.freeze.freeze).
