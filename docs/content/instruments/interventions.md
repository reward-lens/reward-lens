<div class="rl-chips">
  <span class="rl-chip rl-chip--causal">Causal</span>
  <span class="rl-chip rl-chip--gauge"><span class="rl-chip__k">gauge</span> invariant</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">works on</span> activations + weights</span>
</div>

# The intervention algebra

**What can you actually do to a running reward model, and how do you keep the result honest?**

There are five operations, and they are a small algebra rather than five one-off scripts. Every one implements the same `Intervention` protocol: it compiles against a signal, mounts a hook at a site, and carries a fingerprint. Because they share that shape, they compose, and any measurement taken through an intervened signal records which interventions produced it, so an edited reward can never be mistaken for a clean one. Four of the five change the model and hand you a number. The fifth, erasure, changes the model and hands you a *certificate*, a receipt that says whether the change actually did what you claimed.

## The five operations

| Operation | What it does to an activation or weight |
|---|---|
| **patch** | Splice a source activation into a target run. `ComponentPatch`, `HeadPatch`, `ResidualAddPatch`. |
| **steer** | Add a fixed displacement along a direction: \(h \mapsto h + \alpha\,\hat{u}\). `SteeringIntervention`. |
| **ablate** | Remove a direction, a mean, or a head: directionally, \(h \mapsto h - (\hat{u}^{\top} h)\,\hat{u}\). `AblationIntervention`. |
| **edit** | Rewrite the reward head itself: \(w_r' = w_r - \alpha\,(w_r^{\top}\hat{u})\,\hat{u}\). `EditIntervention`. |
| **erase** | Project out *all* linear information about a concept, then certify it held. `LeaceErasure`, `fit_leace`. |

Patch, steer, and ablate act on activations at a site. Edit is different: it changes the reward head's weight vector directly, so the model is permanently a slightly different model, with one concept's pull surgically removed from the readout. Erase is different again: it removes not one direction but the whole subspace that carries linear information about a concept, and it is the only operation whose success is checkable after the fact.

![The five operations as one algebra; they compose into a single intervention, and erasure alone returns a certificate that gates its own trust.](../assets/figures/intervention-algebra-light.svg#only-light){ .rl-fig .rl-fig--hero }
![The five operations as one algebra; they compose into a single intervention, and erasure alone returns a certificate that gates its own trust.](../assets/figures/intervention-algebra-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**Causal operations compose, and the dangerous one comes with a receipt.** Patch, steer, ablate, and edit each transform the model and return a score. Erase returns an `ErasureCertificate`: a held-out probe recovery that decides, on its own, whether the erased evidence is allowed to climb the trust ladder.
///

Read the figure as a closure property. Any two operations chain into a single `ComposedIntervention` that mounts both hooks in order, so "steer along this concept, then ablate that head" is one object with one fingerprint, not a fragile two-step script. The erasure box is drawn apart on purpose. It is the only operation that carries its own answer key, and that is what lets its output be trusted rather than merely asserted.

## Steering, and why strength zero is exactly the baseline

The most basic intervention is a nudge. Steering adds `strength * unit(direction)` at a site, so at strength zero it is a bit-exact no-op, and increasing the strength moves the reward monotonically. Run it through the patched-score runner:

```python
import numpy as np
from reward_lens.signals import from_tiny
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.interventions import SteeringIntervention, run_patched_scores

signal = from_tiny(seed=0)
pairs = list(load_diagnostic_v3()["helpfulness"].items)[:4]
items = [(p.prompt_text, p.chosen.text) for p in pairs]
read = signal.readouts()[0]                       # the reward readout: direction + site

base = signal.score(items, "reward").value.values
s0 = run_patched_scores(signal, SteeringIntervention(read.vector, read.site, 0.0).compile(signal), items)
print(np.array_equal(base, s0))                    # True  (strength 0 is bit-exact)

for a in (-2.0, 0.0, 1.0, 2.0):
    sc = run_patched_scores(signal, SteeringIntervention(read.vector, read.site, a).compile(signal), items)
    print(f"strength {a:>5}: mean reward {float(np.mean(sc)):+.4f}")
# strength  -2.0: mean reward -0.6142
# strength   0.0: mean reward -0.1154
# strength   1.0: mean reward +0.6115
# strength   2.0: mean reward +0.6142
```

Strength zero returns the clean scores to the bit, not merely close. Steer along the reward direction and the reward climbs; steer against it and the reward falls. The flattening between strength one and two is the model's final normalization saturating, a real property of the tiny model, not a bug.

## Composition

Two interventions become one with `compose`, which mounts their hooks in order at the shared site. The composed object has its own id and fingerprint, so a study can name exactly the compound operation it ran.

```python
from reward_lens.interventions import compose, AblationIntervention, unit_direction

steer  = SteeringIntervention(read.vector, read.site, 2.0)
ablate = AblationIntervention(direction=unit_direction(read.vector), site=read.site, mode="directional")
combo  = compose([steer, ablate])
print(type(combo).__name__, combo.id)             # ComposedIntervention compose(steer,ablate)

sc = run_patched_scores(signal, combo.compile(signal), items)
print(round(float(np.mean(sc)), 4))               # 0.0
```

Order is visible in the result. Steering pushes the activation along the reward direction, then directional ablation removes that direction entirely, so the compound cancels the reward down to zero. The ergonomic wrapper `signal.with_interventions(*ivs)` returns a signal that carries the interventions into every Evidence subject it produces; its full scoring wiring lands in a later milestone, and `run_patched_scores` is the proven path today.

## Erase, and the certificate that gives it teeth

Erasure removes all linear information about a concept from a set of features. Fitting is closed-form from second moments: LEACE builds the projection \(P\) that kills every direction in \(\operatorname{span}(\operatorname{Cov}(X, Z))\), then applies \(h \mapsto P(h-\mu)+\mu\). The question is always whether it *worked*, and the honest way to answer is to try to recover the concept from the erased features with a fresh probe on held-out data. `certify_erasure` does exactly that and returns the recovery as Evidence:

\[
\text{recovery\_auc} = \max_j \operatorname{AUC}\big(\text{probe}_j \mid \text{erased held-out}\big), \qquad \text{passed} \iff \text{recovery\_auc} \le 0.5 + \epsilon.
\]

A passing certificate is itself the calibration: a held-out probe that cannot beat chance is precisely the answer-key check that gate 1 asks for, so the Evidence carries a `CalibrationRef` and lifts to `CALIBRATED`. A failing one confers none and stays `EXPLORATORY`. That is what makes the certificate discriminate a real erasure from a sham: both go through the identical held-out check, and only the one that actually removed the concept passes.

```python
import numpy as np
from reward_lens.interventions import fit_leace, certify_erasure

def planted(seed, n, d, strength):                # a concept planted along one direction
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, d)) @ (rng.standard_normal((d, d)) / np.sqrt(d)).T
    U, _ = np.linalg.qr(rng.standard_normal((d, 1)))
    z = (rng.random((n, 1)) < 0.5).astype(float)
    return base + (strength * (2 * z - 1)) @ U.T, z[:, 0]

X, z = planted(20, 6000, 16, 1.5)
X_tr, z_tr, X_ho, z_ho = X[:4000], z[:4000], X[4000:], z[4000:]

real = fit_leace(X_tr, z_tr, concept_id="concept:planted")          # erases the real concept
sham = fit_leace(X_tr, np.random.default_rng(7).permutation(z_tr))  # erases a random direction

cr = certify_erasure(real, X_ho, z_ho, concept_id="concept:planted")
cs = certify_erasure(sham, X_ho, z_ho, concept_id="concept:planted")
print(f"real: auc={cr.value.recovery_auc:.4f} passed={cr.value.passed} trust={cr.trust}")
print(f"sham: auc={cs.value.recovery_auc:.4f} passed={cs.value.passed} trust={cs.trust}")
# real: auc=0.4981 passed=True trust=CALIBRATED
# sham: auc=1.0000 passed=False trust=EXPLORATORY
```

The real eraser drives held-out recovery into the chance band at \(0.50\) and earns `CALIBRATED`. The sham, which went through the identical motions but targeted a random direction, leaves the concept perfectly recoverable at \(1.0\) and is refused, staying `EXPLORATORY`. Nothing about the certificate can be talked past: it is computed the same way for both, and it lands where the erasure actually did.

!!! note "The robustness arm is deliberately incomplete in the exports"
    There is a second, sensitive certificate, `certify_robustness`, which measures the attack budget needed to re-break an erasure. It depends on a gradient-ascent probe that is dual-use: the same optimizer that proves an erasure is robust is also the tool for defeating one. That primitive is kept out of the public exports on purpose, and the robustness certificate reports honestly that it was skipped when the primitive is absent rather than silently returning a pass.

## When not to reach for it

Every activation-level operation runs off the model's natural distribution, so a large steered or ablated effect can be an artifact of a state the model would never produce on its own; keep displacements modest and confirm surprising results with a second, on-distribution intervention. An eraser is only certified against the concept and the held-out data you gave it: a passing certificate says the concept is linearly gone on that distribution, not that it is gone everywhere or that a nonlinear probe could not find a trace. And weight editing changes the model permanently, so measurements taken before and after an edit are measurements of two different models, which is exactly why the subject fingerprint records the edit.

## How much to trust it

The operations themselves are exact and mechanical: strength-zero steering is bit-exact, directional ablation is idempotent to roughly \(10^{-8}\), and the LEACE fit is closed-form. What is *trusted*, in the ladder sense, is the erasure certificate, and only when it passes: [`certify_erasure`](../reference/interventions.md#reward_lens.interventions.certify.certify_erasure) returns `CALIBRATED` Evidence on a held-out recovery at chance and `EXPLORATORY` otherwise, with an `INVARIANT` gauge because the recovery AUC is a scalar, not a cross-model geometric quantity. The rest of the algebra hands you precise causal numbers that are honest within a model and uncalibrated across models. For the observational side these interventions were built to confirm, start at [observational vs causal](../concepts/observational-vs-causal.md); for how a passing certificate climbs the ladder, [the trust ladder](../discipline/trust-ladder.md).
