# Calibrate a detector on an organism

**A detector prints a number. What earns it the right to be believed?**

Not the number's size, and not the confidence of whoever ran it. In reward-lens a measurement is EXPLORATORY until the instrument that produced it has been graded against a system whose ground truth is known by construction. That system is an organism: a model or dataset with a rule planted on purpose, so there is an answer key to grade against. Grade the instrument, register the result, and the same measurement climbs to CALIBRATED and cites the grading.

## Plant an organism with a known answer

`spurious_correlation_organism` plants a confound: a surface feature (here, whether a response cites) correlated with the preference label at a chosen dose \(\rho\). It returns a `DataView` and the `AnswerKey` that names the planted rule.

```python
from reward_lens.organisms.foundry import spurious_correlation_organism

data, key = spurious_correlation_organism(rho=0.8, n=10, seed=0)
print(key.family)
# spurious-cites-rho0.80
```

The answer key knows exactly which rows carry the confound, which is what makes honest grading possible: a detector's output can be scored against truth rather than against a guess.

## Grade a detector across a dose sweep

`MethodScorecard` grades an instrument by computing its answer-key ROC. Run it across a sweep of doses and a sound instrument's AUC climbs from chance at \(\rho = 0.5\) toward one as the planted signal strengthens. The detector graded here is a synthetic stand-in whose separability rises with the dose; grading it is what proves the scorecard machinery is monotone before a real instrument is attached, and a real instrument plugs into the identical `evaluate` path.

```python
from reward_lens.organisms.scorecard import MethodScorecard, synthetic_dose_detector

doses = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
readouts = {rho: synthetic_dose_detector(rho, n=800, seed=1, slope=6.0) for rho in doses}
entry = MethodScorecard("BiasBattery").evaluate(readouts, key)

print([round(a, 3) for a in entry.summary.aucs])
print("monotone", entry.summary.is_monotone, "| spearman", round(entry.summary.monotone_spearman, 3))
# [0.484, 0.685, 0.791, 0.901, 0.941, 0.98]
# monotone True | spearman 1.0
```

At the lowest dose the AUC sits at chance, \(0.48\); at full dose it reaches \(0.98\); the sequence is monotone with a rank correlation of one. An instrument that tracks the planted structure this cleanly has earned a calibration reference for this organism family.

## Register the scorecard, watch trust climb

Registering the entry and installing the gate makes the calibration visible to the measurement runner. Before registration, a `BiasBattery` measurement on a production signal is EXPLORATORY. After, the identical call is CALIBRATED and its Evidence cites the scorecard:

```python
from reward_lens.measure import base as mb
from reward_lens.measure.battery import BiasBattery
from reward_lens.organisms import gate
from reward_lens.signals import from_tiny
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.data.schema import DataView

signal = from_tiny(seed=0)
v = load_diagnostic_v3()
view = DataView(list(v["helpfulness"].items)[:4] + list(v["verbosity"].items)[:4])

before = mb.run(BiasBattery(), mb.Context(signal=signal, view=view))
print("before", before.trust, before.calibration)
# before EXPLORATORY None

gate.register_scorecard("BiasBattery", entry)
gate.install()
after = mb.run(BiasBattery(), mb.Context(signal=signal, view=view))
print("after ", after.trust, "cites", after.calibration.organism_family)
# after  CALIBRATED cites spurious-cites-rho0.80
```

Nothing about the measurement changed except its provenance. The trust rose because the instrument now carries a receipt that it was graded against planted ground truth, and the receipt names the family it was graded on.

![A detector graded against a planted answer key earns a calibration reference.](../assets/figures/calibration-loop-light.svg#only-light){ .rl-fig .rl-fig--hero }
![A detector graded against a planted answer key earns a calibration reference.](../assets/figures/calibration-loop-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**No instrument without an answer key.** An organism supplies ground truth; the scorecard grades the instrument against it; a passing grade is the calibration reference that lifts a measurement out of EXPLORATORY.
///

The loop in the figure runs organism, then instrument, then scorecard, then back. The organism is the only place truth enters, because it was planted. Everything downstream inherits its trust from that grading and from nowhere else, which is why an ungraded number stays EXPLORATORY no matter how it was computed.

!!! note "What CALIBRATED does and does not say"
    A scorecard certifies a regime, not a universe. CALIBRATED here means the instrument recovered a planted rule in the `spurious-cites` family; it is not a promise the instrument behaves on a different failure mode or a different model. EXPLORATORY is not a verdict of wrong, only of unaudited. The gate caps overclaiming; it does not manufacture generality.

See also: [calibration and organisms](../discipline/calibration-and-organisms.md), [the trust ladder](../discipline/trust-ladder.md), [`spurious_correlation_organism`](../reference/organisms.md#reward_lens.organisms.foundry.spurious_correlation_organism), [`MethodScorecard`](../reference/organisms.md#reward_lens.organisms.scorecard.MethodScorecard).
