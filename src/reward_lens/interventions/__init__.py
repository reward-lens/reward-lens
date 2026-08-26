"""``reward_lens.interventions`` — the causal algebra (section 2.6).

An Intervention modifies a forward pass: a patch, a steer, an ablation, an erasure, a head or
weight edit. Interventions and captures share the runtime's single mounting path, so any Observable
runs under any Intervention without either knowing about the other, and interventions compose
(``compose(steer, erase)`` returns one intervention that mounts both in order). Every intervened
Evidence records the intervention fingerprints in its subject, so an erased-model number can never
masquerade as a base-model number.

The surface exported here is the defensive surgery the design calls for. The white-box attack search
that the robustness certificate consumes (``geometry.hessian.gradient_ascent_probe``) is dual-use;
it is deliberately not re-exported from this package, is reached only by its full path, and is marked
sensitive at its source.
"""

from __future__ import annotations

from reward_lens.core.extras import require_extra

require_extra("white-box", subsystem="reward_lens.interventions")

from reward_lens.interventions.ablate import AblationIntervention
from reward_lens.interventions.base import (
    CompiledIntervention,
    ComposedIntervention,
    Intervention,
    MountHook,
    compose,
)
from reward_lens.interventions.certify import (
    ErasureCertificate,
    RobustnessCertificate,
    certify_erasure,
    certify_robustness,
    eraser_evidence,
    probe_recovery_auc,
)
from reward_lens.interventions.edit import EditIntervention, run_edited_scores
from reward_lens.interventions.erase import (
    Eraser,
    LeaceErasure,
    fit_leace,
    leace_matrix,
)
from reward_lens.interventions.patch import (
    ComponentPatch,
    HeadPatch,
    ResidualAddPatch,
    run_patched_scores,
)
from reward_lens.interventions.rescue import (
    Mounted,
    RecordRemoved,
    Reinject,
    RemovedCoordinate,
    RescueError,
    RescueSpec,
    SubspaceDraw,
    knockout_and_rescue,
    mountable,
    norm_matched_random,
    subspace_matched_random,
    target_orthogonal_random,
)
from reward_lens.interventions.steer import SteeringIntervention, unit_direction

__all__ = [
    # protocol and composition (base)
    "Intervention",
    "CompiledIntervention",
    "ComposedIntervention",
    "MountHook",
    "compose",
    # patching (patch)
    "ComponentPatch",
    "HeadPatch",
    "ResidualAddPatch",
    "run_patched_scores",
    # steering (steer)
    "SteeringIntervention",
    "unit_direction",
    # ablation (ablate)
    "AblationIntervention",
    # weight-space edit (edit)
    "EditIntervention",
    "run_edited_scores",
    # LEACE erasure (erase)
    "Eraser",
    "LeaceErasure",
    "fit_leace",
    "leace_matrix",
    # post-hoc certificates (certify)
    "certify_erasure",
    "ErasureCertificate",
    "eraser_evidence",
    "probe_recovery_auc",
    "certify_robustness",
    "RobustnessCertificate",
    # knockout and rescue (rescue)
    "RecordRemoved",  # observer: reads the coordinate an ablation is about to remove
    "Reinject",  # puts a recorded coordinate back, along its own direction or a substitute
    "RemovedCoordinate",  # what RecordRemoved stored, and where it read it
    "RescueSpec",  # which rescue was run, same site or a later one
    "RescueError",  # the recorded coordinate does not fit the site it is replayed into
    "Mounted",  # one hook bound to one site, as the runtime mounts it
    "mountable",  # the hooks an intervention would mount, without mounting them
    "knockout_and_rescue",  # ablate, restore, and report the rescue fraction
    # the control families the rescue is read against (rescue)
    "SubspaceDraw",  # a control family, with the subspace it was drawn from
    "norm_matched_random",  # the AMBIENT control: a unit direction over the full residual space
    "subspace_matched_random",  # C15: norm-matched draws from inside a supplied subspace
    "target_orthogonal_random",  # C16: draws inside the subspace and orthogonal to the target
    # The modules themselves, so that a full path such as
    # ``reward_lens.interventions.rescue`` is a public path rather than a reach into a private one.
    # ``certify`` is named here as a module; the dual-use attack search it can reach is still not
    # re-exported and is still absent from ``certify.__all__``.
    "ablate",
    "base",
    "certify",
    "edit",
    "erase",
    "patch",
    "rescue",
    "steer",
]
