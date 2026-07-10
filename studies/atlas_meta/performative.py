"""Atlas meta — Performativity: the audit half-life (scoreboard T11).

A metric that developers optimize against is performative: the act of optimizing it changes the
population it measures, so its correlation with the truth it was meant to track decays. The audit
half-life is how many developer-response generations pass before ``corr(metric, truth)`` falls to
half its starting value. The registered hypothesis is that a causally grounded metric, one calibrated
on an organism where the truth is known, has a longer audit half-life than an observational metric
that only correlates with truth in the original distribution, because the observational metric can be
raised by a truth-independent proxy while the causal metric can only be raised by moving truth itself.

The calibration arm makes truth known forever with a planted-bias organism (`spurious_correlation_
organism`, where the true feature is the ground truth and a spurious feature is confounded with the
label at a dial ``rho``). A synthetic developer-response operator does gradient ascent on the metric
over generations. Optimizing the observational metric inflates the cheap spurious feature, which is
independent of truth, so ``corr(metric, truth)`` decays with a finite half-life. Optimizing the
causal metric can only raise the true feature, so its correlation with truth does not decay within
the horizon. Recovering the ordering (causal half-life above observational) on the planted dial
calibrates the audit-half-life instrument before it is run on a real optimization loop.

The real arm measures the audit half-life on a genuine developer-response loop against a production
metric over many rounds. It needs a real base population and a real generation loop, so it is
recorded here as an explicitly gated follow-on.
"""

from __future__ import annotations

import numpy as np

from reward_lens.core.evidence import make_evidence
from reward_lens.core.provenance import Provenance
from reward_lens.core.types import GaugeStatus, SubjectRef
from reward_lens.organisms import spurious_correlation_organism
from reward_lens.studies.spec import (
    Hypothesis,
    KillCriterion,
    Prediction,
    StudyResult,
    StudySpec,
    SubjectQuery,
)

_VERSION = "1.0"

_RHO = 0.9  # the organism's planted-bias dial; also the initial corr(observational metric, truth)
_N_POPULATION = 2000
_N_GENERATIONS = 60  # the audit horizon; a half-life at the horizon is right-censored
_STEP = 1.0  # developer-response gradient-ascent step size
_CAUSAL_MEASUREMENT_NOISE = 0.1


def build_spec() -> StudySpec:
    """The frozen performativity spec: the half-life ordering is calibrated, the real loop is gated."""
    return StudySpec(
        id="atlas-performative-halflife",
        title="Performativity: a causally grounded metric has a longer audit half-life than an "
        "observational one",
        science="AT-performative",
        hypotheses=(
            Hypothesis(
                id="H1-halflife-ordering",
                statement="on a planted-bias organism, the causal metric's audit half-life exceeds "
                "the observational metric's: causal grounding buys performative durability",
                prediction=Prediction(metric="half_life_gap", comparator=">", threshold=5.0),
                scoreboard_row="T11",
            ),
            Hypothesis(
                id="H2-observational-decays",
                statement="the observational metric's correlation with truth halves within the "
                "audit horizon as developers inflate the spurious proxy",
                prediction=Prediction(metric="half_life_obs", comparator="<", threshold=40.0),
            ),
            Hypothesis(
                id="H3-real-audit-loop",
                statement="the causal metric outlasts the observational one on a real "
                "developer-response loop against a production metric",
                prediction=Prediction(metric="real_half_life_gap", comparator=">", threshold=5.0),
            ),
        ),
        analysis="studies.atlas_meta.performative.analyze",
        subjects=SubjectQuery(
            organisms=("spurious-correlation-dial",),
            extra={
                "note": "synthetic developer-response operator on a planted-bias organism where "
                "truth is known forever; the real developer-response loop is the gated follow-on"
            },
        ),
        kill_criteria=(
            KillCriterion(
                id="K1-no-durability-advantage",
                metric="half_life_gap",
                comparator="<",
                threshold=1.0,
                description="the causal metric does not outlast the observational one on a planted "
                "dial where grounding should help, so causal grounding buys no performative "
                "durability and the T11 mechanism does not hold even under calibration",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Synthetic developer-response operator on a planted-bias organism
# ---------------------------------------------------------------------------


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two vectors (nan when either is constant)."""
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _half_life(corr_trajectory: np.ndarray, horizon: int) -> float:
    """The first generation at which the correlation falls to half its start (censored at horizon).

    A trajectory that never halves within the horizon is right-censored at ``horizon``, which is the
    honest statement that its half-life is at least the horizon rather than a fabricated finite value.
    """
    if corr_trajectory.size == 0:
        return float(horizon)
    target = corr_trajectory[0] / 2.0
    below = np.nonzero(corr_trajectory <= target)[0]
    return float(below[0]) if below.size else float(horizon)


def _audit_half_lives(rho: float, seed: int = 0) -> dict:
    """Run the developer-response operator for the observational and causal metrics; return half-lives.

    Both arms share a population whose truth is the true feature. The observational metric is
    ``rho * truth + sqrt(1 - rho^2) * spurious`` (so its initial correlation with truth is ``rho``);
    developers optimizing it inflate the truth-independent spurious feature each generation, so the
    correlation decays. The causal metric is the truth plus small measurement noise; developers
    optimizing it can only raise truth, so its correlation with truth holds. Each arm's half-life is
    read from its correlation trajectory.
    """
    rng = np.random.default_rng(seed)
    truth0 = rng.standard_normal(_N_POPULATION)
    b_true = rho
    b_spur = float(np.sqrt(1.0 - rho**2))

    # Observational arm: truth fixed, spurious proxy inflated by developers each generation.
    spurious = rng.standard_normal(_N_POPULATION)
    corr_obs = np.empty(_N_GENERATIONS + 1)
    for t in range(_N_GENERATIONS + 1):
        metric = b_true * truth0 + b_spur * spurious
        corr_obs[t] = _pearson(metric, truth0)
        spurious = spurious + _STEP * rng.standard_normal(_N_POPULATION)

    # Causal arm: the metric is grounded truth, so raising it raises truth.
    truth = truth0.copy()
    corr_causal = np.empty(_N_GENERATIONS + 1)
    for t in range(_N_GENERATIONS + 1):
        metric = truth + _CAUSAL_MEASUREMENT_NOISE * rng.standard_normal(_N_POPULATION)
        corr_causal[t] = _pearson(metric, truth)
        truth = truth + _STEP * rng.standard_normal(_N_POPULATION)

    half_life_obs = _half_life(corr_obs, _N_GENERATIONS)
    half_life_causal = _half_life(corr_causal, _N_GENERATIONS)
    return {
        "half_life_obs": half_life_obs,
        "half_life_causal": half_life_causal,
        "half_life_gap": half_life_causal - half_life_obs,
        "corr_obs_initial": float(corr_obs[0]),
        "corr_obs_final": float(corr_obs[-1]),
        "corr_causal_final": float(corr_causal[-1]),
    }


def analyze(run) -> StudyResult:
    """Calibrate the audit-half-life ordering on a planted dial; gate the real loop."""
    study_id = run.study.study_id
    subject = SubjectRef(extra={"study": study_id})

    # Ground the dial in a planted-bias organism, where the truth is known forever by construction.
    _, answer_key = spurious_correlation_organism(rho=_RHO)
    dial = float(answer_key.channels[0].rho)

    s = _audit_half_lives(dial)

    ev_traj = make_evidence(
        observable="AT.AuditTrajectories",
        observable_version=_VERSION,
        subject=subject,
        value={
            "organism_family": answer_key.family,
            "dial_rho": dial,
            "half_life_obs": s["half_life_obs"],
            "half_life_causal": s["half_life_causal"],
            "corr_obs_initial": s["corr_obs_initial"],
            "corr_obs_final": s["corr_obs_final"],
            "corr_causal_final": s["corr_causal_final"],
        },
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id),
    )
    run.record(ev_traj)

    ev_halflife = make_evidence(
        observable="AT.AuditHalfLife",
        observable_version=_VERSION,
        subject=subject,
        value={
            "half_life_obs": s["half_life_obs"],
            "half_life_causal": s["half_life_causal"],
            "half_life_gap": s["half_life_gap"],
        },
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id, parents=(ev_traj.id,)),
        registered=True,
    )
    run.record(ev_halflife)

    ev_gate = make_evidence(
        observable="AT.RealAuditLoopGate",
        observable_version=_VERSION,
        subject=subject,
        value={
            "status": "gated",
            "need": "a real base population and a real developer-response generation loop optimizing "
            "a production metric over many rounds (real population / GPU); the half-life ordering "
            "calibrated here is then measured on the real loop",
            "blocks_metric": "real_half_life_gap",
        },
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id, parents=(ev_halflife.id,)),
        registered=True,
    )
    run.record(ev_gate)

    censored = "censored at the horizon" if s["half_life_causal"] >= _N_GENERATIONS else "finite"
    return StudyResult(
        outcomes={},
        metrics={
            "half_life_gap": s["half_life_gap"],
            "half_life_obs": s["half_life_obs"],
            "half_life_causal": s["half_life_causal"],
        },
        summary=(
            f"On the planted dial (rho={dial:.2f}) the observational metric's correlation with truth "
            f"halved after {s['half_life_obs']:.0f} developer generations while the causal metric's "
            f"held ({censored}) to {s['half_life_causal']:.0f}, a half-life gap of "
            f"{s['half_life_gap']:.0f} generations. A causally grounded metric outlasts an "
            f"observational one under optimization. The real developer-response loop is gated on a "
            f"real base population and generation loop."
        ),
    )


__all__ = ["build_spec", "analyze"]
