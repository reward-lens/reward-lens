# Interventions

**What can you change about a reward model, and which change is dangerous enough to need a receipt?** `reward_lens.interventions` is the causal half of the library: patch an activation, steer along a direction, ablate a subspace, edit the readout weights, or erase a concept. Every operation compiles to a hook on the signal, so they compose in order, and the one that erases returns a certificate that grades whether it worked. The narrated version is [the intervention algebra](../instruments/interventions.md).

## Patching

A patch compiles a fixed activation into a named site and holds it there while the signal scores. `ComponentPatch` replaces a whole component's output, `HeadPatch` a single attention head, and `ResidualAddPatch` adds a vector into the residual stream.

::: reward_lens.interventions.patch.ComponentPatch
    options:
      heading_level: 3

::: reward_lens.interventions.patch.HeadPatch
    options:
      heading_level: 3

::: reward_lens.interventions.patch.ResidualAddPatch
    options:
      heading_level: 3

## Steering and ablation

`SteeringIntervention` pushes the activation along a unit direction by a chosen strength. At strength zero it is bit-exact with the unmodified run, so the null case costs nothing and cannot leak an artifact. `AblationIntervention` removes a direction instead, in directional, mean, or head mode.

::: reward_lens.interventions.steer.SteeringIntervention
    options:
      heading_level: 3

::: reward_lens.interventions.ablate.AblationIntervention
    options:
      heading_level: 3

## Editing the readout

`EditIntervention` works in weight space rather than activation space, projecting a direction \(u\) out of the reward head: \(w_r' = w_r - \alpha\,(w_r \cdot u)\,u\). It changes what the model rewards, not just one forward pass.

::: reward_lens.interventions.edit.EditIntervention
    options:
      heading_level: 3

## Erasure and its certificate

`Eraser` is a fitted LEACE transform that removes a concept from the activations, `fit_leace` fits one from labelled data, and `LeaceErasure` wraps it as a composable intervention. Erasure is the operation that can quietly fail, so it does not get to claim success on its own word.

::: reward_lens.interventions.erase.Eraser
    options:
      heading_level: 3

::: reward_lens.interventions.erase.fit_leace
    options:
      heading_level: 3

::: reward_lens.interventions.erase.LeaceErasure
    options:
      heading_level: 3

`certify_erasure` trains a fresh probe on held-out data and reports the worst recovery AUC it can find. A real erase drops that AUC to chance (1.0 to 0.5056 in the test suite) and the certificate binds a [`CalibrationRef`](core.md#reward_lens.core.gates.CalibrationRef), so the [`Evidence`](core.md#reward_lens.core.evidence.Evidence) comes out `CALIBRATED`. A sham erase leaves the AUC at 1.0 and the evidence stays `EXPLORATORY`. The certificate grades the erasure; the erasure does not grade itself.

::: reward_lens.interventions.certify.certify_erasure
    options:
      heading_level: 3

## Composition

`compose` chains interventions that share a site into one, applied in order. The algebra is closed: a composed intervention is itself an intervention, so a signal can wear several at once through `signal.with_interventions(...)`.

::: reward_lens.interventions.base.compose
    options:
      heading_level: 3
