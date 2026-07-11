# Measurements you can trust

**When a reward-model instrument hands you a number, how much of it should you believe?** In 2.0 you do not have to guess. Every measurement comes back as an evidence object that carries its own credentials: a value, an uncertainty with an honest sample count, and a trust level it computed for itself from what you could actually back up. Untrusted by default. It climbs only when you give it something to climb on.

This whole page runs on the CPU in about a minute. No download, no GPU, no flagship model. Six steps, and each one's output is pasted straight from the run. By the end you will have moved one measurement from untrusted to calibrated to registered, and watched a frozen prediction judge itself.

![Four rungs, exploratory at the bottom then calibrated, registered, and adjudicated, each rung labelled with the fact that earns it.](../assets/figures/trust-ladder-light.svg#only-light){ .rl-fig .rl-fig--hero }
![Four rungs, exploratory at the bottom then calibrated, registered, and adjudicated, each rung labelled with the fact that earns it.](../assets/figures/trust-ladder-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**Trust is computed, not claimed.** Four rungs: `EXPLORATORY`, `CALIBRATED`, `REGISTERED`, `ADJUDICATED`. You do not set the level. You supply a fact, a calibration reference or a frozen study, and a gate decides which rung the evidence is allowed to stand on. This tutorial climbs the first three.
///

## Step 1: A signal you can hold in memory

`from_tiny` builds a real reward model, small, randomly initialized, on the CPU, with no network call. It implements the same protocol an 8B grader does, so every instrument that works on Skywork attaches to it unchanged.

```python
from reward_lens.signals import from_tiny

signal = from_tiny(seed=0)
type(signal).__name__     # 'ClassifierRM'
signal.caps               # Capability.SCORES|PREFIX_SCORES|ACTIVATIONS|GRADIENTS|HVP|LINEAR_READOUT
```

The capabilities matter later: an instrument declares what it needs, and the runner refuses to run one whose requirements the signal cannot meet. For now, you have a model.

## Step 2: Every measurement arrives with a receipt

Run an instrument. `DirectLinearAttribution` splits a preference margin across components. What comes back is not an array; it is evidence, and the first thing to read is not its value but its credentials.

```python
from reward_lens.measure import base as mb
from reward_lens.measure.battery import DirectLinearAttribution
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.data.schema import DataView

view = DataView(list(load_diagnostic_v3()["helpfulness"].items)[:6])
ev = mb.run(DirectLinearAttribution(), mb.Context(signal=signal, view=view))

ev.trust           # EXPLORATORY
ev.gauge           # invariant
ev.is_calibrated   # False
ev.calibration     # None
ev.uncertainty     # Uncertainty(ci_low=None, ci_high=None, ci_level=None,
                   #             n=None, n_effective=None, seed_spread=None, method='none')
ev.provenance.git_sha   # '5495f1d7...+dirty' (identifies the exact code; a dirty tree is flagged)
ev.provenance.cost      # Cost(gpu_seconds=0.0, tokens=0, wall_seconds=0.0)
```

![One evidence object opened up: a central value ringed by uncertainty, gauge status, calibration reference, trust level, and provenance.](../assets/figures/anatomy-of-evidence-light.svg#only-light){ .rl-fig .rl-fig--hero }
![One evidence object opened up: a central value ringed by uncertainty, gauge status, calibration reference, trust level, and provenance.](../assets/figures/anatomy-of-evidence-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**A number with its papers attached.** The value sits at the center; around it are the uncertainty, the gauge status (here `invariant`, safe to compare), the calibration reference (here empty), the computed trust level, and the provenance stamp naming the code that produced it. Nothing here was asserted by the author.
///

Now, *why* `EXPLORATORY`? Because `calibration` is `None`. The trust level is not a field you fill in; it is computed by a gate that asks one question: is there a calibration reference behind this number? There is not, so the gate caps the result at the bottom rung. That is the honest reading of a single unbootstrapped attribution: interesting, unaudited, not yet a claim.

The receipt is also honest about sample size, which is where most eval numbers quietly lie. Thirty rows drawn from six seeds are not thirty independent observations, and the effective-sample-size helper will tell you so:

```python
from reward_lens.stats import effective_sample_size, detect_clones

seed_labels = [s for s in range(6) for _ in range(5)]        # 30 rows, 6 seeds
effective_sample_size(seed_labels)                            # 6.0

content_hashes = [f"h{s}" for s in range(6) for _ in range(5)]  # each content 5x
detect_clones(content_hashes)["duplicate_fraction"]           # 0.8
```

Six, not thirty. When an instrument fills the `n_effective` field of its uncertainty, this is the arithmetic it uses. The full anatomy is on the [anatomy of evidence page](../discipline/anatomy-of-evidence.md); the sample-size machinery has its own [how-to](../how-to/effective-sample-size.md).

## Step 3: Plant a rule you know the answer to

To calibrate an instrument you need ground truth, and real reward models do not come with an answer key. So you build one. An **organism** is a synthetic setup where you plant a known rule and hand back the data together with the answer. `spurious_correlation_organism` plants a true rule, prefer the factual answer, and lets a second feature ride along with the label at a controlled strength, the kind of confound that fools a naive detector.

```python
from reward_lens.organisms import spurious_correlation_organism

view, key = spurious_correlation_organism(rho=0.85, n=200, seed=0)
key.family          # 'spurious-cites-rho0.85'
key.channels[0]     # PlantedChannel(kind='spurious', rho=0.85,
                    #                 detail={'spurious_feature': 'cites', 'true_feature': 'factual'})
len(list(view.items))   # 200
```

The `key` is the answer key. It names the family the calibration will certify and records exactly what was planted. Because you built it, you know what a correct detector should say, which is the whole point.

## Step 4: Grade the detector, and trust climbs

Now grade an instrument against that answer key across a sweep of planted strengths. A `MethodScorecard` consumes a detector's scores at each dose and asks the answer-key question: as the planted signal gets stronger, does the detector's ability to recover it rise monotonically, and how high does it get?

```python
from reward_lens.organisms import MethodScorecard, synthetic_dose_detector

doses = [0.0, 0.3, 0.6, 0.85, 0.95]
readouts = {rho: synthetic_dose_detector(rho, n=400, seed=0) for rho in doses}
entry = MethodScorecard("spurious.detector").evaluate(readouts, key, doses=doses)

entry.summary.aucs               # [0.017, 0.217, 0.675, 0.940, 0.974]
entry.summary.is_monotone        # True
entry.summary.monotone_spearman  # 0.9999999999999999
entry.calibration_ref            # CalibrationRef(scorecard_entry='ev:e5ee91ee...',
                                 #   organism_family='spurious-cites-rho0.85', regime_match='exact', ...)
```

The detector's answer-key AUC climbs from near zero at no planted signal to \(0.94\) at the planted dose and \(0.97\) beyond it, and it does so monotonically (rank correlation with dose \(\approx 1.0\)). That behavior is what earns a calibration reference. Attach it to a fresh measurement and the trust level moves on its own:

```python
from reward_lens.core import make_evidence, GaugeStatus, SubjectRef

subj = SubjectRef(signals=("organism:spurious",), dataset="spurious", readout="reward")
before = make_evidence(observable="spurious.detector", observable_version="1", subject=subj,
                       value=0.9, uncertainty=None, gauge=GaugeStatus.INVARIANT)
after  = make_evidence(observable="spurious.detector", observable_version="1", subject=subj,
                       value=0.9, uncertainty=None, gauge=GaugeStatus.INVARIANT,
                       calibration=entry.calibration_ref)

before.trust    # EXPLORATORY
after.trust     # CALIBRATED
```

![A loop: plant a rule, run the detector, grade it against the answer key, and feed the grade back as the calibration reference.](../assets/figures/calibration-loop-light.svg#only-light){ .rl-fig .rl-fig--hero }
![A loop: plant a rule, run the detector, grade it against the answer key, and feed the grade back as the calibration reference.](../assets/figures/calibration-loop-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**No instrument without an answer key.** Plant a known rule, run the detector on it, grade the detector against the truth, and the grade becomes a calibration reference. That reference is what lets a measurement on a real model claim the calibrated rung: it says the instrument was checked where the answer was known.
///

The same measurement, the same value, two different trust levels, and the only thing that changed was whether there was a scorecard behind it. Calibration is a real technique with real edges, covered on the [calibration and organisms page](../discipline/calibration-and-organisms.md), with a hands-on recipe in [calibrate a detector on an organism](../how-to/calibrate-on-an-organism.md).

## Step 5: Freeze the question before you look

Calibration tells you an instrument works in a regime. It does not stop you from running twenty analyses and reporting the one that looked good. The guard against that is registration: you write the prediction down, freeze it, and only then run.

A `StudySpec` states a hypothesis as a checkable prediction: a metric, a comparator, a threshold. `freeze` turns it into an immutable study with a content-derived identifier stamped with the git sha, so the record cannot be quietly edited after the fact.

```python
from reward_lens.core.types import Capability, GaugeStatus
from reward_lens.measure.base import BaseObservable
from reward_lens.studies import (
    StudySpec, Hypothesis, Prediction, SubjectQuery, StudyResult, freeze, run_study,
)

class DemoSignal:                       # stand-in for a real grader
    caps = Capability.SCORES | Capability.ACTIVATIONS
    meta = type("M", (), {"fingerprint": "mfp:tiny-demo"})()

class MeanMargin(BaseObservable):
    name = "MeanMargin"
    gauge_status = GaugeStatus.INVARIANT
    def measure(self, ctx):
        return ctx.emit(0.62, subject_extra={"note": "demo"})

def analysis(run) -> StudyResult:
    ev = run.measure(MeanMargin(), run.signal("primary"))
    return StudyResult(outcomes={}, metrics={"mean_margin": float(ev.value)}, summary="demo")

spec = StudySpec(
    id="demo-margin",
    title="Mean margin clears the registered threshold",
    science="S03-thermo",
    hypotheses=(Hypothesis(id="H1",
        statement="mean margin exceeds the registered threshold",
        prediction=Prediction(metric="mean_margin", comparator=">", threshold=0.3),
        scoreboard_row="T9"),),
    analysis="tutorial.analysis",
    subjects=SubjectQuery(signals=("mfp:tiny-demo",)),
)

freeze(spec).study_id      # 'study:demo-margin@v1#2f7a11a0'
```

The prediction, mean margin above \(0.3\), is now part of the frozen record. Note that the analysis function deliberately does not decide the outcome; it only reports the metric. The verdict is not its job.

## Step 6: Run it, and let the spec adjudicate

Running the study measures the metric and hands the number to the *frozen* prediction, not to whatever the analysis wanted to conclude. The measurement lands in the evidence store at the `REGISTERED` rung, and the runner returns the outcome.

```python
import tempfile
from reward_lens.core.store import EvidenceStore

with tempfile.TemporaryDirectory() as d:
    store = EvidenceStore(d)
    frozen, result = run_study(spec, subjects={"primary": DemoSignal()},
                               store=store, analysis_fn=analysis)
    result.metrics                          # {'mean_margin': 0.62}
    result.outcomes["H1"]                   # 'confirmed'   (0.62 > 0.3)
    result.killed                           # False
    store.get(result.evidence[0]).trust     # REGISTERED
```

It confirms, because \(0.62 > 0.3\) and that was the registered bar. The adjudication is mechanical, and it cuts the other way just as readily. Freeze a prediction the data will not meet and the same run refutes it:

```python
spec_hard = StudySpec(id="demo-margin", title=spec.title, science=spec.science,
    hypotheses=(Hypothesis(id="H1", statement="mean margin exceeds 0.8",
        prediction=Prediction(metric="mean_margin", comparator=">", threshold=0.8),
        scoreboard_row="T9"),),
    analysis="tutorial.analysis", subjects=spec.subjects)

with tempfile.TemporaryDirectory() as d:
    _, result = run_study(spec_hard, subjects={"primary": DemoSignal()},
                          store=EvidenceStore(d), analysis_fn=analysis)
    result.outcomes["H1"]                   # 'refuted'   (0.62 > 0.8 is false)
```

Same code, same data, opposite verdict, decided entirely by a threshold that was fixed before the run. That is the point of freezing it first. The full lifecycle, kill criteria and the scoreboard included, is on [studies and preregistration](../discipline/studies-and-preregistration.md), with a shorter recipe in [freeze and run a study](../how-to/freeze-and-run-a-study.md). Where the registered evidence goes to live is the [evidence store](../discipline/evidence-store.md).

## What the ladder does and does not do

You climbed three rungs on a laptop: untrusted, then calibrated against a planted answer key, then registered behind a frozen prediction. Be precise about what each rung is actually worth, because the ladder caps overclaiming, it does not manufacture insight.

- **`EXPLORATORY` means unaudited, not wrong.** Most real interpretability lives here, and that is fine. It only becomes a problem when an exploratory number is dressed up as a confirmed one.
- **`CALIBRATED` speaks to a regime, not to every model.** The scorecard certifies the instrument where the answer was known, on this organism family. It says nothing about a model that behaves unlike the organism.
- **`REGISTERED` means the prediction predated the data, and nothing more.** It does not make the hypothesis true. It only removes the freedom to have chosen the hypothesis after seeing the result.
- **The top rung, `ADJUDICATED`, is not something this tutorial reaches.** It requires a study to be run and judged against organisms at population scale. For the anti-correlation result from the [previous tutorial](inside-one-reward-model.md), that adjudication is designed and partially run on organisms; the population-scale pass waits on hardware. The library says so rather than claiming the rung.

The plain-words version of this whole idea is the concept page, [a measurement you can trust](../concepts/measurement-you-can-trust.md). The honest limits of every instrument, gathered in one place, are in [interpreting results honestly](../caveats.md).
