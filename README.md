# reward-lens

[![PyPI version](https://badge.fury.io/py/reward-lens.svg)](https://pypi.org/project/reward-lens/)
[![Python](https://img.shields.io/pypi/pyversions/reward-lens.svg)](https://pypi.org/project/reward-lens/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

White-box tools and built-in epistemic discipline for reward models, the objects that define what RLHF optimizes.

Every model trained with RLHF was shaped by a reward model. That reward model sat in the loop and decided, on pair after pair, which of two answers was better. It is the closest thing the pipeline has to a written-down definition of what we asked for, and almost nobody looks inside one. That is strange, because the reward model is where alignment actually gets decided. A policy does not optimize your intentions. It optimizes the number this model hands back, and whatever the reward model fails to measure becomes the exact thing the policy is free to exploit. If you want to know why a model learned to pad its answers, agree with whatever you said, or wrap everything in confident structure, the honest place to look is not the policy. It is the function that rewarded it.

reward-lens is the instrument for looking.

## The one output direction

A reward model is a language model with its vocabulary head removed and a single linear layer bolted on. The score is a dot product:

```
r(x) = w_r · h + b
```

The final hidden state `h` is read out along one fixed vector `w_r`, the reward direction. It is not something you probe for or approximate. It sits in the weights, known exactly, the same for every input. A generative model spreads its answer across fifty thousand logits; a reward model concentrates it into one number along one line. Once you see the reward as a projection, most of the tools here become variations on a single move. Project each layer's activation onto `w_r` and you watch the preference form. Split the final state into its parts and project each, and you get a per-component ledger. Intervene on a component and remeasure, and you get a causal test. Same direction, different questions.

## Why the measurements need discipline, not just tools

The first version of this library was a bag of those primitives. Running them at scale taught the lesson that reshaped everything since: naive reward-model measurement produces confident, wrong numbers, and it does it in ways that look fine on the page.

A concrete one. Rank a model's components by how much attribution assigns them, rank them again by how much causal patching says they carry, and on Skywork the two rankings correlate at Spearman ρ = -0.256. Negative. The last MLP layers dominate attribution; the early layers dominate patching. The place the reward visibly accumulates is not the place that causes it. Quote the cheap observational tool as if it were the causal one and you have published a plausible, backwards result.

There were more of these. Confidence intervals computed over five hand-written stimuli that had been cloned into "thirty pairs," so the interval described the cloning and not the model. Cross-model reward directions compared in raw coordinates, where a change of basis reads as a change of function. Instruments with no answer key, reporting numbers nobody had ever checked against a case with known ground truth. None of these are exotic. They are the predictable output of tools that carry no notion of evidence, provenance, or calibration. So the rebuild does not add more tools. It puts the discipline underneath them.

## One kernel, sixteen sciences, three gates

reward-lens 2.0 is a small kernel of subsystems, a layer of studies that consume it, and three gates that hold the whole thing honest.

Every measurement returns an `Evidence` object, never a bare float. Evidence carries the value, its uncertainty (including an effective sample size that counts unique content, not cloned rows), its gauge status, its calibration reference, its provenance back to the inputs it came from, and a trust level. The trust level is never set by the caller. It is computed by the gates.

- **Calibration gate.** An instrument with no scorecard, meaning no measured performance against a case whose ground truth is known, cannot claim more than exploratory trust. You earn calibration by grading the instrument on model organisms with structure planted by construction, and then the same measurement on a real model cites that scorecard.
- **Gauge gate.** A quantity that only means something in a fixed basis (a direction, an angle, a subspace overlap) cannot be compared across models without a shared frame. Ask for that comparison without fixing the gauge and the library raises, rather than handing back a number that confuses a coordinate change for a real one.
- **Registration gate.** A confirmatory claim requires a frozen preregistration. A study is a spec plus a thin analysis function; freezing it stamps the git sha and locks the predictions before the run, so the result is adjudicated against a prediction made in advance, not a story fit afterward.

The trust ladder runs exploratory, calibrated, registered, adjudicated, and an Evidence sits on the highest rung the facts actually support. This is the part worth being direct about. The honesty is in the type system, not in a disclaimer. You do not have to remember to be careful. The instrument will not let you quote a number as more than it is.

## Install

```bash
pip install reward-lens
```

Python 3.10 or newer. The base install brings torch and transformers, because most of the library eventually touches a model. If all you want is the epistemics layer it is still one line, and `import reward_lens.core` and `import reward_lens.stats` will not import torch.

```bash
pip install "reward-lens[sae]"    # SAE training support
pip install "reward-lens[dev]"    # tests, ruff, mypy
```

From source:

```bash
git clone https://github.com/suhailnadaf509/reward-lens.git
cd reward-lens
pip install -e ".[dev]"
```

## A first look

Start with the part that needs no GPU, because it is the part that explains the rest. Trust is not a label you write down. It is computed from what you actually did to earn it.

```python
from reward_lens.core import make_evidence, CalibrationRef, SubjectRef, ModelFP

subject = SubjectRef(signals=(ModelFP("mfp:demo"),), dataset="ds:demo", readout="reward")

# A bare measurement is exploratory. Nothing has earned it more than that.
ev = make_evidence(observable="BiasBattery", observable_version="1",
                   subject=subject, value=-0.05)
print(ev.trust)          # TrustLevel.EXPLORATORY

# Calibrate it against a scorecard graded on planted ground truth and it climbs a rung.
cal = CalibrationRef(scorecard_entry="ev:...", organism_family="spurious-correlation")
ev = make_evidence(observable="BiasBattery", observable_version="1",
                   subject=subject, value=-0.05, calibration=cal)
print(ev.trust)          # TrustLevel.CALIBRATED
```

When you do reach for a model, a measurement is the same three steps every time: load a signal, pick an observable, run it through the gated runner. Here it is on a small model that runs on CPU, so nothing downloads.

```python
from reward_lens.signals import from_tiny
from reward_lens.measure import base as mb
from reward_lens.measure.battery import DirectLinearAttribution
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.data.schema import DataView

signal = from_tiny(seed=0)
view = DataView(list(load_diagnostic_v3()["helpfulness"].items)[:8])

ev = mb.run(DirectLinearAttribution(), mb.Context(signal=signal, view=view))
print(ev.value["dominant_component"])   # which head or MLP wrote the reward difference
print(ev.trust)                         # EXPLORATORY, until this observable earns a scorecard
```

The same call runs on an 8B classifier reward model, a generative judge, or a process reward model, because they all satisfy one signal protocol. Only the readout changes.

To make a confirmatory claim, you freeze the prediction before the run. Freezing stamps the git sha and locks the predictions, and the runner adjudicates the result against them.

```python
from reward_lens.studies import StudySpec, Hypothesis, Prediction, SubjectQuery, freeze, run_study

spec = StudySpec(
    id="smoke-thermo", title="Mean reward rises under mild optimization", science="S03-thermo",
    hypotheses=(Hypothesis(id="H1", statement="mean reward exceeds 0.3",
        prediction=Prediction(metric="mean_reward", comparator=">", threshold=0.3),
        scoreboard_row="T9"),),
    analysis="yourpkg.analysis.thermo_smoke",
    subjects=SubjectQuery(signals=("mfp:study-test",)),
)

frozen = freeze(spec)                                    # study:smoke-thermo@v1#<hash>, git sha stamped
frozen, result = run_study(spec, subjects={"primary": signal}, store=store)
print(result.outcomes["H1"])                             # "confirmed" or "refuted", against the frozen prediction
```

The Evidence this produces is registered, and it lands in an append-only store. Everything downstream (an RM card, the population leaderboard, a safety case) is a view over that store, so a card and a paper are guaranteed to quote the same number. The command line is the operator surface over the same store, and it draws a clean line. Anything that is a view over stored evidence runs here and now with no model. Anything that needs a reward model names the exact kernel call it would make and refuses rather than printing a number it did not compute.

```bash
reward-lens card mfp:...        # an RM card: every stored Evidence about one model
reward-lens scoreboard          # standing theorems and candidate laws
reward-lens claims paper.md     # nonzero exit if the manuscript cites a number the store cannot back
reward-lens atlas export        # the population leaderboard, as JSON and HTML

reward-lens score <signal>      # GPU-gated: dispatches to signals.load_signal(...).score(...)
```

That `claims` command is worth pausing on. It reads a manuscript, finds every number tagged with an evidence id, and fails if the store does not hold that exact value. A paper cannot claim a figure the evidence does not support, and you find out in CI rather than in review.

## What is in the kernel

The kernel is the set of subsystems every study stands on. You rarely touch all of them; you reach for the ones a question needs. Imports stay lazy and layered, so the epistemics layer pulls only numpy and scipy, and nothing loads torch until you touch a model.

**Signals** give one interface over the different things people call a reward. The `RewardSignal` protocol carries first-class readouts and positions, so eight substrates look the same to every tool downstream: classifier reward models, generative judges, process reward models, implicit (DPO log-ratio) rewards, rubric graders, trajectory models, dense per-token rewards, and ensembles. A new kind of grader becomes a new adapter that passes the conformance suite, and the whole battery works on it unchanged.

**Data** is the plane instruments read from, and never construct themselves. Pairs, quadruples, tournaments, and trajectories are typed and lineage-tracked. A `DataView` reports its effective sample size and a content checksum, which is where the cloned-stimulus problem dies: the statistics count unique content, not duplicated rows.

**Measure** is the library of things you can measure. A battery of eleven observables ports the interpretability primitives (the reward lens across depth, per-component attribution, activation and path patching, the bias battery, concept dose-response, SAE feature alignment, multi-objective geometry, cross-model circuit overlap). On top sit eighteen scalar indices, each one a named theory object with a formal definition it must stay faithful to: the knowledge-utilization gap that predicts which dimension gets hacked, reward susceptibility from fluctuation-dissipation, the tail exponent that sets the critical optimization pressure, a verification score that separates checking work from reading style, and more. Every observable returns gated Evidence. There is no path that returns an unguarded number.

**Interventions** are the causal side: patch, steer, ablate, edit the reward head in weight space, and erase a concept with a closed-form affine map (LEACE) that you can then certify by training a fresh probe and reporting how much it recovers. An erasure that cannot be certified stays exploratory.

**Geometry** is what makes cross-model comparison mean something. It fixes the gauge, whitens to a canonical frame, and reports the STARC-invariant angle whose cosine is the on-distribution correlation of two reward readouts. It also carries Hessian spectroscopy and a skew-symmetric test for the intransitive preferences a scalar head cannot express.

**Dynamics** watches a reward model form across training. Checkpoints link into a hash-verified chain, the battery sweeps over them, and the curves show when bias enters and when the reward direction stops rotating and merely rescales.

**Organisms** are the ground truth. An organism is a reward model with a rule planted by construction, so you know the answer. Grading an instrument against the planted structure is how it earns a calibration scorecard, and the scorecard has to be monotone in the planted signal strength before it counts. This is the floor the calibration gate stands on.

**Loops** wire the instrument into the training run. A framework-agnostic reward function, geometry logging on fixed probes every few steps, best-of-N and tilt analysis for reading optimization pressure, and a rollout recorder that watches reward-feature drift and can name an exploited direction with a lead time before the true reward diverges. Bindings for TRL, veRL, and OpenRLHF share the same reward and logging shapes.

**Studies** are preregistered experiments: a frozen spec, a thin analysis function, kill criteria, and a runner that adjudicates the result against predictions made before the run. Standing results accumulate on a theorem scoreboard.

**Artifacts** are views over the evidence store, never fresh computation. Build an RM card for a model, a population leaderboard, or a safety case, which is the strictest artifact of all: it assembles a claim about what is safe to optimize and refuses unless every component it rests on is both calibrated and registered.

Underneath all of it, **core**, **stats**, and **runtime** hold the floor: the Evidence atom, provenance, the append-only store, and the gates in core; a real numpy statistics engine (effect sizes, bootstrap and cluster-bootstrap CIs, multiplicity control, ROC and calibration, changepoint detection, mutual information) in stats; and the HuggingFace execution layer (hooks, per-family numerics policies with an fp32 reward head, model fingerprinting, an activation cache) in runtime.

## The sixteen sciences

The kernel exists so the research on top can be thin. Each science is a family of preregistered studies that add only a hypothesis and a short analysis function, never new infrastructure, and every result is adjudicated against a prediction frozen before the run. A theorem scoreboard tracks which claims have held up, and refutations show up as plainly as confirmations. Here is what they ask.

| Science | The question it puts to a reward model |
|---|---|
| Gauge | Can two reward directions be compared at all, when a head is fixed only up to shift and scale? |
| Thermodynamics | Which features will optimization exploit, read off base-policy statistics before any RL? |
| Capacity | How much bias does a scalar head force just by routing many criteria through few dimensions? |
| Topology | What share of reward error is topologically obligatory, beyond any scalar reward's reach? |
| Embryology | Does the reward direction form gradually or in jumps, and which features enter first? |
| Factorization | How much does the reward know but fail to use, and is its error epistemic or about values? |
| Verification | Does the reward check the work, or just read the style around it? |
| Decompiling | How much of the decision function can be put into words, and what stays tacit? |
| Values | Does the model encode "this pair is contested," and is a judge's verdict set before its critique? |
| Hackability | Can a number read off the weights name the dimension that gets hacked, before training starts? |
| Coupling | Watching policy and grader as one loop, does representational divergence come before hacking? |
| Phase | Is the hacking transition reversible, or does hysteresis mean a hacked policy cannot anneal back? |
| Forensics | How does the grader weigh evidence: does it rank a caught fabrication below saying nothing? |
| Robustness | Does the model know it is being tested, and does that recognition inflate the score? |
| Universality | Do two reward models converge on values beyond what shared world-modeling forces? |
| Performative | How fast does a metric decay once developers start optimizing against it? |

The last two run across a whole population of reward models rather than a single one.

## Coming from version 1

The 1.0 API still works. Every v1 name lives on under `reward_lens.legacy` and stays importable from the top level, so `from reward_lens import RewardModel, RewardLens, ComponentAttribution` keeps running while you migrate. As each primitive settles into its new home behind the protocols, its legacy entry is repointed at a thin adapter with no change on your side. The pure layers never import any of it, so the torch-free promise holds regardless.

## Status

This is alpha, and honest about which parts are load-bearing today. The epistemics layer (core, stats, the data plane, the index math) is pure, tested, and usable now without a GPU. The measurement, intervention, and study machinery is wired end to end and proven on small synthetic models and organisms in the test suite. The paths that need an 8B model or a flagship GPU, real dataset downloads, or an external judge are gated: they name the call and refuse rather than fabricate a result. Where a dependency is still landing, the code says so in place instead of returning a plausible number. Interfaces in the science layer may still move.

## Documentation

Full documentation, including the theory behind each index and a candid account of what the observational tools can and cannot tell you, is at <https://suhailnadaf509.github.io/reward-lens/>.

## Citation

```bibtex
@software{nadaf2026rewardlens,
    title  = {reward-lens: An Instrument for the Science of Reward Misspecification},
    author = {Nadaf, Mohammed Suhail B},
    year   = {2026},
    url    = {https://github.com/suhailnadaf509/reward-lens},
}
```

## License

MIT
