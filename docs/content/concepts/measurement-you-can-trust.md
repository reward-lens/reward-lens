# A measurement you can trust

Should you believe this number? Every previous page has handed you results, a margin, a crystallization depth, an anti-correlation, and quietly assumed the answer was yes. This page is where that assumption gets examined, because the honest answer is usually "it depends," and the library makes the answer part of the result.

## The receipt

A measurement here does not return a bare number. It returns the number wrapped in the things you would need to decide whether to trust it: the value, an uncertainty that counts how much *independent* data went into it, whether it depends on an arbitrary choice of coordinates, where it came from, what it cost, and one summary of all of that, a **trust level**.

Here is the smallest possible version, with no model at all.

```python
from reward_lens.core import make_evidence, CalibrationRef, SubjectRef, ModelFP

subject = SubjectRef(signals=(ModelFP("mfp:demo"),), dataset="ds:demo", readout="reward")

# A bare measurement. Nothing has earned it more than a starting position.
ev = make_evidence(observable="BiasBattery", observable_version="1", subject=subject, value=-0.05)
print(ev.trust)          # EXPLORATORY

# The same value, now carrying a reference to a scorecard from a case with a known answer.
cal = CalibrationRef(scorecard_entry="ev:...", organism_family="spurious-correlation")
ev = make_evidence(observable="BiasBattery", observable_version="1",
                   subject=subject, value=-0.05, calibration=cal)
print(ev.trust)          # CALIBRATED
```

The value did not change. The trust did, and it changed because of something real that happened: the measurement was tied to a scorecard. You never write the trust level yourself. You supply the facts, and the trust level is computed from them.

![One measurement drawn as a card: the value in the center, its credentials around it.](../assets/figures/anatomy-of-evidence-light.svg#only-light){ .rl-fig }
![One measurement drawn as a card: the value in the center, its credentials around it.](../assets/figures/anatomy-of-evidence-dark.svg#only-dark){ .rl-fig }

/// caption
**A number that carries its own credentials.** The value sits in the middle. Around it: the uncertainty, with the honest effective sample size next to the raw row count; whether the number depends on a basis; where it came from; and the trust level, in the corner, computed from the rest.
///

## The ladder

Trust is not a slider you set to feel confident. It is one of four rungs, and a measurement sits on the highest one the facts actually support.

- **Exploratory.** A number nobody has checked against a known answer. Most measurements start here. Exploratory does not mean wrong. It means unaudited, and you should quote it as a hypothesis, not a result.
- **Calibrated.** The tool that produced it has a scorecard: it was graded on a case where the answer was planted by construction, and it recovered that answer.
- **Registered.** The claim was written down and frozen before the run, so the result is a genuine prediction, not a story fit to the data afterward.
- **Adjudicated.** Both of the above, and the prediction survived.

![Four rungs, each reached only by passing a gate.](../assets/figures/trust-ladder-light.svg#only-light){ .rl-fig .rl-fig--wide }
![Four rungs, each reached only by passing a gate.](../assets/figures/trust-ladder-dark.svg#only-dark){ .rl-fig .rl-fig--wide }

/// caption
**Trust is computed, not claimed.** Each rung is reached only by passing a gate: a scorecard to be calibrated, a frozen prediction to be registered, both plus survival to be adjudicated. No argument to the measurement sets the rung directly.
///

That is the whole idea, in plain words. The anti-correlation from the last page is exactly why it exists: two tools disagreed, both were exploratory, and nothing in the old library could say which to believe, because neither had ever been scored against a case with a known answer.

## Through the door

You now have the altitude-1 version of the trust story, which is enough to read the rest of the site without being misled. The machinery underneath is worth seeing in full, and it is genuinely the most original part of the library:

- [The anatomy of evidence](../discipline/anatomy-of-evidence.md) walks every field on that card and why a bare float was the bug.
- [The trust ladder](../discipline/trust-ladder.md) shows what computes each rung and what each one licenses you to say.
- [Calibration and organisms](../discipline/calibration-and-organisms.md) is how a tool earns a scorecard: plant a rule, train it in, and score every method against it.

Those pages are two clicks deep and do not hold back. This one was the airlock.
