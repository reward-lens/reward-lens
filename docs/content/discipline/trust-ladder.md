# The trust ladder

Who decides how much to trust a number? In most tools, you do, implicitly, by how you write it up. That is exactly the freedom that lets an uncalibrated result get quoted as a finding. Here, the decision is taken out of your hands and made a computation. This page is that computation.

## Four rungs

Trust is a single field on every [evidence](anatomy-of-evidence.md), and it takes one of four values.

<div class="rl-chips">
  <span class="rl-chip rl-chip--fill rl-chip--exploratory">exploratory</span>
  <span class="rl-chip rl-chip--fill rl-chip--calibrated">calibrated</span>
  <span class="rl-chip rl-chip--fill rl-chip--registered">registered</span>
  <span class="rl-chip rl-chip--fill rl-chip--adjudicated">adjudicated</span>
</div>

- **Exploratory.** The starting position. A number produced by a tool that has never been scored against a case with a known answer. Most measurements live here, and that is fine, as long as you quote them as hypotheses.
- **Calibrated.** The tool has a scorecard. It was graded on a model organism whose structure was planted by construction, it recovered that structure, and the score is monotone in the strength of the planted signal. That scorecard is cited by the evidence.
- **Registered.** The claim was frozen before the run. A study spec, with its predictions, was content-hashed and git-stamped, so the result is adjudicated against a prediction that provably predates the data.
- **Adjudicated.** Calibrated and registered, and the frozen prediction held. The top rung, and the rarest.

![Four rungs, each reached only by passing a gate the caller cannot skip.](../assets/figures/trust-ladder-light.svg#only-light){ .rl-fig .rl-fig--wide }
![Four rungs, each reached only by passing a gate the caller cannot skip.](../assets/figures/trust-ladder-dark.svg#only-dark){ .rl-fig .rl-fig--wide }

/// caption
**Trust is computed, not claimed.** Exploratory grey rises to the earned color of adjudicated, and each promotion is a gate: a scorecard, a frozen spec, a surviving prediction. The caller supplies the facts. The rung is computed from them, and no shortcut reaches the top.
///

## What computes it

The rung is not stored. It is derived, every time, from three facts the evidence carries: whether it has a calibration reference, whether it was produced under a frozen study, and whether that study's prediction survived.

```python
from reward_lens.core.gates import compute_trust
from reward_lens.core import TrustLevel

compute_trust(calibration=None, registered=False)                     # EXPLORATORY
compute_trust(calibration="ev:scorecard-1", registered=False)         # CALIBRATED
compute_trust(calibration=None, registered=True)                      # REGISTERED
compute_trust(calibration="ev:scorecard-1", registered=True,
              adjudicated=True)                                        # ADJUDICATED
```

Two things about that function matter more than the logic inside it.

First, it is the only way a rung is set. `make_evidence` has no `trust` parameter. You cannot pass one in. You supply the facts, calibration and registration and adjudication, and the constructor calls `compute_trust` itself and writes the result into a frozen record you then cannot edit. There is no setter and no back door. If you want a higher rung, you have to make a truer fact.

Second, the ordering is deliberate and occasionally surprising. Registered outranks calibrated. A frozen, surviving prediction on an uncalibrated tool is worth more than a calibrated tool run with no prior commitment, because preregistration defends against the subtler failure: fitting a story to the data after seeing it. The top rung demands both, because calibration and preregistration defend against different lies and you want both defended.

## What each rung licenses

The rung is not a vibe. It is a specific permission to make a specific kind of statement.

- Exploratory licenses "we observed." You may report the number and what you saw. You may not call it validated.
- Calibrated licenses "this tool, on cases like this, recovers the truth at this rate." The scorecard is the warrant, and it is quotable.
- Registered licenses "we predicted this in advance, and here is the outcome." The freeze is the warrant.
- Adjudicated licenses the strongest claim the library will let you make about a single measurement: a preregistered prediction, from a calibrated instrument, that held.

## What it does not do

The ladder is a ceiling on overclaiming, not a source of truth, and it is worth being exact about its limits, because a discipline that oversells itself is just a new way to be wrong.

Calibrated does not mean correct on your model. A scorecard speaks to the regime it was earned in. A detector calibrated on a family of planted-rule organisms tells you how that detector behaves on organisms like those. It is evidence about the instrument, not a warranty on your reward model, and the [calibration page](calibration-and-organisms.md) is blunt about how far it transfers.

Registered does not mean true. It means the prediction predated the data. A frozen prediction can be frozen and wrong, and the honest outcome of a registered study is often "refuted," which the library records as prominently as a confirmation.

Exploratory does not mean wrong. It means unaudited. Most of what these docs show you, the anti-correlation included, is exploratory, and it is still the most useful thing here. The ladder does not manufacture insight. It only stops you from dressing insight up as more settled than it is.

Next: [how a tool earns the calibrated rung](calibration-and-organisms.md), which is the one piece of machinery the whole floor rests on.
