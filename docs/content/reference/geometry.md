# Geometry

**When is a number computed from two different models a fact, and when is it only a coordinate?** `reward_lens.geometry` answers that with a frame: a per-site whitening artifact that puts two reward directions in a shared basis before anything compares them. Without one, a covariant quantity refuses to report. The plain-words version is [gauge and frames](../discipline/gauge-and-frames.md).

## The frame

A `Frame` is the frozen whitening artifact for one site and corpus: the mean, the square-root covariance, and the null basis that together fix the gauge. `fit_frame` estimates it with Ledoit-Wolf shrinkage and refuses fp16 activations, because a gauge you cannot trust is worse than no gauge at all.

::: reward_lens.geometry.frame.Frame
    options:
      heading_level: 3

::: reward_lens.geometry.frame.fit_frame
    options:
      heading_level: 3

## Canonicalization and the cross-model angle

`canonicalize` projects a direction out of the null space and whitens it into the frame. `effective_angle` then compares two directions inside that frame and returns [`Evidence`](core.md#reward_lens.core.evidence.Evidence) carrying the canonical cosine, the raw cosine, a STARC distance, and a regret bound. It is covariant, so it calls the gauge gate and raises without a frame. Two versions of one reward model can read as near-orthogonal in raw coordinates (raw cosine near \(0.005\)), which is exactly the coordinate artifact a frame removes.

::: reward_lens.geometry.canonical.canonicalize
    options:
      heading_level: 3

::: reward_lens.geometry.canonical.effective_angle
    options:
      heading_level: 3

## Curvature and skew

`hessian_spectrum` estimates the eigenvalue density of the reward Hessian by Lanczos, and `participation_ratio` reduces that spectrum to how many directions the curvature actually spreads across. `PreferenceRankTest` looks for intransitivity: a planted \(A > B > C > A\) cycle that a single scalar reward cannot represent.

::: reward_lens.geometry.hessian.hessian_spectrum
    options:
      heading_level: 3

::: reward_lens.geometry.hessian.participation_ratio
    options:
      heading_level: 3

::: reward_lens.geometry.skew.PreferenceRankTest
    options:
      heading_level: 3
