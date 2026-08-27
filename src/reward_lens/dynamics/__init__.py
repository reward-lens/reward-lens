"""``reward_lens.dynamics`` — formation and time, the developmental science (DESIGN 2.12, M9).

v1 had nothing here. This subsystem is the RM-Pythia / D1 programme: how a reward model's internals
form across training time, measured by sweeping the battery and the index library over a chain of
checkpoints with full provenance.

Four modules:

  - `checkpoints`: `CheckpointSequence`, a fingerprint-chained sequence of ``(step, ModelFP, loader)``
    checkpoints (DESIGN 2.2.5) that verifies against tampering, plus the CPU-provable
    `synthetic_planted_sequence` and the download-gated `from_hf_revisions`.
  - `sweep`: `sweep_over_checkpoints`, running an Observable or index across the sequence, cached and
    resumable so the few-hundred-GPU-hour run is paid for once (DESIGN 4.5). Returns a `Trajectory`.
  - `curves`: the developmental readings — `bias_entry_curve` (per-probe effect size over training),
    `stabilization_report` (when w-tilde stops rotating vs merely rescaling), the
    `second_epoch_collapse_autopsy` skeleton, and the `faithfulness_rho_trajectory` (per-checkpoint
    E04 rho), plus the built-in `LayerwiseProjection` crystallization observable.
  - `devinterp`: the optional bridge to LLC / essential-dynamics tooling: whether that package is
    importable, and the trajectory type its estimators' output lands in.

Importing this package pulls no torch. Every torch-touching path (materializing a checkpoint's signal,
capturing activations, building the synthetic trunk) imports torch lazily at call time, so the pure
epistemics of a developmental analysis (the curves, the chain verification) are usable and testable
without a model, consistent with the rest of the kernel.
"""

from __future__ import annotations

from reward_lens.core.extras import require_extra

require_extra("white-box", subsystem="reward_lens.dynamics")

from reward_lens.dynamics.checkpoints import (
    GENESIS_LINK,
    RM_PYTHIA_RELEASE_INTENT,
    ChainVerification,
    Checkpoint,
    CheckpointSequence,
    SyntheticSequence,
    from_hf_revisions,
    synthetic_planted_sequence,
)
from reward_lens.dynamics.curves import (
    BiasEntryCurves,
    CollapseAutopsy,
    FaithfulnessRhoTrajectory,
    LayerwiseProjection,
    Probe,
    ReadoutVector,
    StabilizationReport,
    bias_entry_curve,
    default_sweep_observable,
    faithfulness_rho_trajectory,
    second_epoch_collapse_autopsy,
    stabilization_report,
)
from reward_lens.dynamics.sweep import (
    SweepManifest,
    SweepPoint,
    Trajectory,
    compute_sweep_id,
    sweep_over_checkpoints,
)

__all__ = [
    # checkpoints
    "GENESIS_LINK",
    "Checkpoint",
    "ChainVerification",
    "CheckpointSequence",
    "SyntheticSequence",
    "synthetic_planted_sequence",
    "from_hf_revisions",
    "RM_PYTHIA_RELEASE_INTENT",
    # sweep
    "sweep_over_checkpoints",
    "Trajectory",
    "SweepPoint",
    "SweepManifest",
    "compute_sweep_id",
    # curves
    "Probe",
    "BiasEntryCurves",
    "bias_entry_curve",
    "StabilizationReport",
    "stabilization_report",
    "ReadoutVector",
    "LayerwiseProjection",
    "default_sweep_observable",
    "CollapseAutopsy",
    "second_epoch_collapse_autopsy",
    "FaithfulnessRhoTrajectory",
    "faithfulness_rho_trajectory",
]
