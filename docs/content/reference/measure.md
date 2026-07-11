# Measure and indices

**What turns a signal into a measurement you can file?** One runner. `reward_lens.measure` holds the observable protocol, the eleven white-box instruments of the battery, and the eighteen scalar indices, and every one of them returns [`Evidence`](core.md#reward_lens.core.evidence.Evidence) through the same gate-enforcing runner. The narrated tour is under [instruments](../instruments/index.md).

## The runner

An `Observable` declares the capability it needs and the gauge status of what it returns. `run` enforces the capability and the frame requirement before the observable is allowed to touch a signal, and threads the calibration and study facts through so the resulting evidence comes out with its trust level already computed. A `Context` carries the signal, the data view, and any comparison target.

::: reward_lens.measure.base.run
    options:
      heading_level: 3

::: reward_lens.measure.base.Context
    options:
      heading_level: 3

::: reward_lens.measure.base.Observable
    options:
      heading_level: 3

## The battery

Eleven observables, imported from `reward_lens.measure.battery`. The first few read the reward geometry directly off activations and the linear readout, and are gauge-invariant. The last few compare readouts or SAE features and are `RAW_ONLY` or need a frame, so they refuse a cross-model claim without one.

`LensCrystallization` finds the depth where the chosen-versus-rejected margin reaches half its final value: the layer the model made up its mind. See [reward lens and crystallization](../instruments/lens-crystallization.md).

::: reward_lens.measure.battery.lens.LensCrystallization
    options:
      heading_level: 3

`DirectLinearAttribution` splits the reward differential into a signed share per component. It reads where the reward is visible, which is not the same as what causes it; that gap is the subject of [observational versus causal](../concepts/observational-vs-causal.md).

::: reward_lens.measure.battery.dla.DirectLinearAttribution
    options:
      heading_level: 3

`PatchGrid` is the causal counterpart: the effect on the reward of patching a component or head. `PathEffect` measures a two-hop sender-to-receiver path. Both are gated to a real model at head granularity.

::: reward_lens.measure.battery.patch.PatchGrid
    options:
      heading_level: 3

::: reward_lens.measure.battery.path.PathEffect
    options:
      heading_level: 3

`ConceptDoseResponse` measures a concept direction's reward alignment and its causal dose-response slope: push the activation along the concept, watch the reward move.

::: reward_lens.measure.battery.concept.ConceptDoseResponse
    options:
      heading_level: 3

`BiasBattery` reports a standardized reward bias per axis with a lineage-honest sample size, and `PromptSNR` the power signal-to-noise of the reward delta.

::: reward_lens.measure.battery.bias.BiasBattery
    options:
      heading_level: 3

::: reward_lens.measure.battery.snr.PromptSNR
    options:
      heading_level: 3

`ConflictMatrix` reads the cosine geometry between per-axis reward terms, and `CircuitJaccard` the top-component overlap between two models.

::: reward_lens.measure.battery.conflict.ConflictMatrix
    options:
      heading_level: 3

::: reward_lens.measure.battery.circuit.CircuitJaccard
    options:
      heading_level: 3

`FeatureRewardAlignment` asks which SAE features drive the reward, and `MultiObjectiveGeometry` reads the per-objective readout geometry of a multi-objective head like ArmoRM.

::: reward_lens.measure.battery.feature.FeatureRewardAlignment
    options:
      heading_level: 3

::: reward_lens.measure.battery.geometry.MultiObjectiveGeometry
    options:
      heading_level: 3

## The index library

Eighteen scalar diagnostics, imported from `reward_lens.measure.indices`. Each pairs a definition with a pure-numpy function and returns `Evidence`. All of them default to `EXPLORATORY`: there is no calibration provider wired for the index library yet, so they name a quantity honestly without claiming a validated regime. Sixteen are gauge-invariant; `VCE` and `Contested` are covariant and need a frame to compare. The narrated version is [the index library](../instruments/index-library.md).

::: reward_lens.measure.indices.kui.KUI
    options:
      heading_level: 3

::: reward_lens.measure.indices.distortion.Distortion
    options:
      heading_level: 3

::: reward_lens.measure.indices.coverage_disparity.CoverageDisparity
    options:
      heading_level: 3

::: reward_lens.measure.indices.teacher_compatibility.TeacherCompatibility
    options:
      heading_level: 3

::: reward_lens.measure.indices.tail.TailIndex
    options:
      heading_level: 3

::: reward_lens.measure.indices.verification_score.VerificationScore
    options:
      heading_level: 3

::: reward_lens.measure.indices.style_share.StyleShare
    options:
      heading_level: 3

::: reward_lens.measure.indices.receipt_reliance.ReceiptReliance
    options:
      heading_level: 3

::: reward_lens.measure.indices.skepticism.Skepticism
    options:
      heading_level: 3

::: reward_lens.measure.indices.coherence.Coherence
    options:
      heading_level: 3

::: reward_lens.measure.indices.dark_reward.DarkReward
    options:
      heading_level: 3

::: reward_lens.measure.indices.interp_coverage.InterpCoverage
    options:
      heading_level: 3

::: reward_lens.measure.indices.chi.Chi
    options:
      heading_level: 3

::: reward_lens.measure.indices.vce.VCE
    options:
      heading_level: 3

::: reward_lens.measure.indices.legibility.Legibility
    options:
      heading_level: 3

::: reward_lens.measure.indices.eval_awareness.EvalAwareness
    options:
      heading_level: 3

::: reward_lens.measure.indices.snr.RobustnessSNR
    options:
      heading_level: 3

::: reward_lens.measure.indices.contested.Contested
    options:
      heading_level: 3
