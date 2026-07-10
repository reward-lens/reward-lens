"""Atlas meta — Universality: value convergence excess (VCE, scoreboard T13).

Two reward models built on the same base language model inevitably share world-modeling structure:
their capability subspaces align because they inherit the same pretrained features. The universality
question is sharper than "do they align" - it is whether their *values* converge beyond what shared
world-modeling already forces. The value convergence excess answers it as a difference of alignments,

    VCE = align(canonicalized reward subspaces) - align(matched capability subspaces),

each alignment read against the RUM-identifiability null (`reward_lens.stats.nulls`), which is the
subspace overlap two independently estimated utilities share for free because a random utility model
pins its reward subspace's dimension but not its orientation. A positive VCE is convergence of values
above world-modeling; a VCE at or below zero is reward alignment no greater than the shared base
already explains.

The calibration arm makes the sign of VCE knowable by construction and reads it with the real index
`reward_lens.measure.indices.value_convergence_excess`. It builds two synthetic model pairs in a
shared frame. In the convergent pair the reward subspaces share a planted value subspace tighter than
the capability subspaces do, so VCE must be positive. In the null pair the reward subspaces are
independent while the capability subspaces still share the base structure, so VCE must be at or below
zero. Recovering positive VCE (above the RUM-identifiability null) on the convergent pair and a
clearly lower VCE on the null pair calibrates the sign of the index before it is turned on a real
reward-model pair.

The real arm computes VCE for a production reward-model pair (for example two RewardBench models, or
Skywork v0.1/v0.2) with matched capability probes, with the same `value_convergence_excess` index. It
needs two real reward populations, so it is recorded here as an explicitly gated follow-on.
"""

from __future__ import annotations

import numpy as np

from reward_lens.core.evidence import make_evidence
from reward_lens.core.provenance import Provenance
from reward_lens.core.types import GaugeStatus, SubjectRef
from reward_lens.geometry import canonicalize, fit_frame, subspace_alignment
from reward_lens.measure.indices import value_convergence_excess
from reward_lens.studies.spec import (
    Hypothesis,
    KillCriterion,
    Prediction,
    StudyResult,
    StudySpec,
    SubjectQuery,
)

_VERSION = "1.0"

_D_AMBIENT = 32
_K_REWARD = 4
_K_CAPABILITY = 4
# Perturbation scales. Convergent reward subspaces sit close to a shared value subspace; capability
# subspaces sit a moderate distance from a shared base subspace (world-modeling they share for being
# built on the same base LM but not identically); null reward subspaces are fully independent.
_EPS_REWARD_CONVERGENT = 0.10
_EPS_CAPABILITY = 0.30


def build_spec() -> StudySpec:
    """The frozen universality spec: the sign of VCE is calibrated, the real RM pair is gated."""
    return StudySpec(
        id="atlas-universality-vce",
        title="Universality: value convergence excess is positive when values converge beyond "
        "world-modeling",
        science="AT-universality",
        hypotheses=(
            Hypothesis(
                id="H1-vce-positive",
                statement="on a model pair with a planted shared value subspace, VCE is positive: "
                "reward subspaces align more than the shared capability subspaces do",
                prediction=Prediction(metric="vce_convergent", comparator=">", threshold=0.05),
                scoreboard_row="T13",
            ),
            Hypothesis(
                id="H2-vce-sign-separates",
                statement="the convergent pair's VCE exceeds the independent pair's by a clear "
                "margin, so the sign of the index tracks genuine value convergence",
                prediction=Prediction(metric="vce_sign_gap", comparator=">", threshold=0.1),
            ),
            Hypothesis(
                id="H3-beats-rum-null",
                statement="the convergent reward-subspace alignment beats the RUM-identifiability "
                "null: convergence exceeds what identifiability freedom alone forces",
                prediction=Prediction(
                    metric="reward_convergent_p_value", comparator="<", threshold=0.05
                ),
            ),
            Hypothesis(
                id="H4-real-rm-pair",
                statement="VCE is positive for a production reward-model pair with matched "
                "capability probes",
                prediction=Prediction(metric="real_vce", comparator=">", threshold=0.0),
            ),
        ),
        analysis="studies.atlas_meta.universality.analyze",
        subjects=SubjectQuery(
            organisms=("synthetic-model-pairs",),
            extra={
                "note": "synthetic model pairs with a known shared vs independent value subspace; "
                "the real reward-model-pair VCE is the gated follow-on"
            },
        ),
        kill_criteria=(
            KillCriterion(
                id="K1-vce-negative-on-convergent",
                metric="vce_convergent",
                comparator="<",
                threshold=0.0,
                description="VCE is negative on a pair with a planted shared value subspace, so the "
                "index cannot detect value convergence above world-modeling and the T13 "
                "construction is invalid",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Synthetic model pairs with known shared vs independent value subspaces
# ---------------------------------------------------------------------------


def _orthonormal(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """A ``d x k`` orthonormal basis of a uniformly random ``k``-subspace of R^d."""
    q, _ = np.linalg.qr(rng.standard_normal((d, k)))
    return q[:, :k]


def _perturbed(target: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    """A basis near ``target``: the target columns plus ``eps``-scaled isotropic noise.

    Larger ``eps`` moves the spanned subspace further from ``target``, so two bases perturbed from a
    common target with a small ``eps`` are strongly aligned and with a large ``eps`` only weakly so.
    """
    return target + eps * rng.standard_normal(target.shape)


def _canonical_basis(basis: np.ndarray, frame) -> np.ndarray:
    """Canonicalize each column of a ``d x k`` basis in the shared frame (columns stay a basis)."""
    return np.stack([canonicalize(basis[:, j], frame) for j in range(basis.shape[1])], axis=1)


def _vce_scenario(seed: int = 0) -> dict:
    """Build the convergent and null model pairs and compute their VCE in a shared frame.

    A frame is fit on shared reference activations so the reward subspaces are canonicalized before
    they are compared (the alignment is COVARIANT and frame-gated). The convergent pair's reward
    subspaces are both perturbed from one planted value subspace; the null pair's are independent. The
    capability subspaces of both pairs are perturbed from one shared base subspace, standing in for
    the world-modeling two models on the same base LM share. VCE is the reward alignment minus the
    capability alignment.
    """
    rng = np.random.default_rng(seed)
    reference = rng.standard_normal((2000, _D_AMBIENT)).astype(np.float32)
    frame = fit_frame(reference)

    value_subspace = _orthonormal(_D_AMBIENT, _K_REWARD, rng)
    base_subspace = _orthonormal(_D_AMBIENT, _K_CAPABILITY, rng)

    reward_a_conv = _perturbed(value_subspace, _EPS_REWARD_CONVERGENT, rng)
    reward_b_conv = _perturbed(value_subspace, _EPS_REWARD_CONVERGENT, rng)
    reward_a_null = _orthonormal(_D_AMBIENT, _K_REWARD, rng)
    reward_b_null = _orthonormal(_D_AMBIENT, _K_REWARD, rng)
    cap_a = _perturbed(base_subspace, _EPS_CAPABILITY, rng)
    cap_b = _perturbed(base_subspace, _EPS_CAPABILITY, rng)

    align_reward_conv = subspace_alignment(
        _canonical_basis(reward_a_conv, frame), _canonical_basis(reward_b_conv, frame), frame
    )
    align_reward_null = subspace_alignment(
        _canonical_basis(reward_a_null, frame), _canonical_basis(reward_b_null, frame), frame
    )
    align_capability = subspace_alignment(
        _canonical_basis(cap_a, frame), _canonical_basis(cap_b, frame), frame
    )

    # The VCE arithmetic and its RUM-identifiability-null reading are the real T13 index. The alignment
    # scalars fed in are the frame-canonicalized mean-cos^2 overlaps computed above, the same statistic
    # the null samples, so the excess and the null are directly comparable.
    vce_conv = value_convergence_excess(
        align_reward_conv.alignment,
        align_capability.alignment,
        d=_D_AMBIENT,
        k=_K_REWARD,
        seed=seed,
    )
    vce_null = value_convergence_excess(
        align_reward_null.alignment,
        align_capability.alignment,
        d=_D_AMBIENT,
        k=_K_REWARD,
        seed=seed,
    )
    return {
        "vce_convergent": float(vce_conv["vce"]),
        "vce_null": float(vce_null["vce"]),
        "vce_sign_gap": float(vce_conv["vce"] - vce_null["vce"]),
        "reward_convergent_alignment": float(align_reward_conv.alignment),
        "reward_null_alignment": float(align_reward_null.alignment),
        "capability_alignment": float(align_capability.alignment),
        "reward_convergent_p_value": float(align_reward_conv.p_value),
        "rum_null_mean": float(vce_conv["null_mean"]),
        "convergent_exceeds_rum_null": float(vce_conv["exceeds_identifiability_null"]),
        "null_pair_exceeds_rum_null": float(vce_null["exceeds_identifiability_null"]),
    }


def analyze(run) -> StudyResult:
    """Calibrate the sign of VCE on synthetic model pairs; gate the real reward-model pair."""
    study_id = run.study.study_id
    subject = SubjectRef(extra={"study": study_id})

    s = _vce_scenario()

    ev_align = make_evidence(
        observable="AT.SubspaceAlignments",
        observable_version=_VERSION,
        subject=subject,
        value={
            "reward_convergent_alignment": s["reward_convergent_alignment"],
            "reward_null_alignment": s["reward_null_alignment"],
            "capability_alignment": s["capability_alignment"],
            "rum_null_mean": s["rum_null_mean"],
            "reward_convergent_p_value": s["reward_convergent_p_value"],
        },
        gauge=GaugeStatus.COVARIANT,
        provenance=Provenance(study=study_id),
    )
    run.record(ev_align)

    ev_vce = make_evidence(
        observable="AT.ValueConvergenceExcess",
        observable_version=_VERSION,
        subject=subject,
        value={
            "vce_convergent": s["vce_convergent"],
            "vce_null": s["vce_null"],
            "vce_sign_gap": s["vce_sign_gap"],
            "convergent_exceeds_rum_null": s["convergent_exceeds_rum_null"],
            "null_pair_exceeds_rum_null": s["null_pair_exceeds_rum_null"],
        },
        gauge=GaugeStatus.COVARIANT,
        provenance=Provenance(study=study_id, parents=(ev_align.id,)),
        registered=True,
    )
    run.record(ev_vce)

    ev_gate = make_evidence(
        observable="AT.RealVceGate",
        observable_version=_VERSION,
        subject=subject,
        value={
            "status": "gated",
            "need": "two real reward-model score heads and matched capability probes in a shared "
            "frame (real population / GPU); the reward_lens.measure.indices value_convergence_excess "
            "index calibrated here is then read on the production pair",
            "blocks_metric": "real_vce",
        },
        gauge=GaugeStatus.COVARIANT,
        provenance=Provenance(study=study_id, parents=(ev_vce.id,)),
        registered=True,
    )
    run.record(ev_gate)

    return StudyResult(
        outcomes={},
        metrics={
            "vce_convergent": s["vce_convergent"],
            "vce_null": s["vce_null"],
            "vce_sign_gap": s["vce_sign_gap"],
            "reward_convergent_p_value": s["reward_convergent_p_value"],
        },
        summary=(
            f"On the convergent pair the reward subspaces aligned at "
            f"{s['reward_convergent_alignment']:.3f} against a capability alignment of "
            f"{s['capability_alignment']:.3f}, a VCE of {s['vce_convergent']:+.3f} (p="
            f"{s['reward_convergent_p_value']:.3f} against the RUM null); the independent pair's VCE "
            f"was {s['vce_null']:+.3f}. The sign of value convergence excess is recovered: values "
            f"converge beyond world-modeling only when they genuinely share a value subspace. The "
            f"real reward-model-pair VCE is gated on two real reward populations."
        ),
    )


__all__ = ["build_spec", "analyze"]
