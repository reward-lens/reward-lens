"""S14 — Phase structure and the hysteresis protocol (DESIGN Part III, Tier IV, S14; deepens T3/T9).

The question is whether the reward-hacking transition is reversible. If hacking were a gradual drift,
a policy pushed past onset by raising optimization pressure could be annealed back by lowering it,
and KL-annealing would be a legitimate recovery tool. If the transition is first-order, the order
parameter follows a different branch on the way down than on the way up, the two branches enclose a
nonzero area, and a hacked policy cannot be annealed back. That loop area is the signature and its
deployment consequence is immediate.

One registered experiment runs here, on a CPU-provable bistable reward system where the hysteresis is
analytically present, so the loop-area measurement is provable without a GPU. The system is a tilted
double well ``F(m; beta) = (m^2 - 1)^2 - beta * m``: an aligned well at ``m = -1`` and a hacked well
at ``m = +1``, with optimization pressure ``beta`` tilting the landscape toward the hacked well. The
protocol runner (``loops.anneal.run_hysteresis``) sweeps ``beta`` up through onset and back down,
letting the order parameter settle to its local optimum at each step from the previous state so
history is carried, and integrates the area the two branches enclose. Following the local rather than
global optimum is what makes the transition first-order, and the metastable gap between the up-branch
onset (near ``beta ~ 1.5``) and the down-branch onset (near ``beta ~ 0``) is the width of the
hysteresis.

The arm that anneals a real KL or pressure parameter on a trained policy carrying a planted exploit,
measuring gold reward and feature occupations on both branches, is recorded as
inconclusive-because-gated: it needs a real RL loop with live training callbacks and a GPU, and the
feature-occupation order parameter would be raw-coordinate rather than the abstract double-well ``m``.

If ``reward_lens.loops.anneal`` is importable the study uses it; if it were not, an inline double-well
responder and shoelace loop-area integration with the same contract run instead.
"""

from __future__ import annotations

import numpy as np

from reward_lens.core.evidence import Evidence, Uncertainty, make_evidence
from reward_lens.core.provenance import Provenance
from reward_lens.core.types import GaugeStatus, SubjectRef
from reward_lens.studies.spec import (
    Hypothesis,
    KillCriterion,
    Prediction,
    StudyResult,
    StudySpec,
    SubjectQuery,
)

_VERSION = "1.0"


def build_spec() -> StudySpec:
    """The frozen S14 spec: one hypothesis that the annealing loop has nonzero area."""
    return StudySpec(
        id="s14-phase",
        title="Phase structure and hysteresis: reward hacking is first-order, so a hacked policy "
        "cannot be annealed back",
        science="S14-phase",
        hypotheses=(
            Hypothesis(
                id="H1-nonzero-loop-area",
                statement="the anneal-up / anneal-down loop encloses a nonzero area, the signature "
                "of an irreversible first-order transition",
                prediction=Prediction(metric="loop_area", comparator=">", threshold=0.1),
                scoreboard_row="T9",
            ),
        ),
        analysis="studies.s14_phase.analysis.analyze",
        subjects=SubjectQuery(
            extra={
                "note": "a CPU-provable tilted double well; the real-RL anneal of a KL/pressure "
                "parameter on a trained exploited policy is GPU-gated"
            }
        ),
        kill_criteria=(
            KillCriterion(
                id="K1-loop-closes",
                metric="loop_area",
                comparator="<",
                threshold=0.01,
                description="the loop closes (a smooth crossover retraces its path), so hacking is a "
                "gradual reversible drift and KL-annealing is a legitimate recovery tool, which is a "
                "publishable negative result",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Inline fallback (only used if reward_lens.loops.anneal is unavailable)
# ---------------------------------------------------------------------------


def _inline_double_well(beta: float, m: float, *, n_iter: int = 400, lr: float = 0.02) -> float:
    """Gradient relaxation of the order parameter in the tilted double well from state ``m``."""
    x = float(m)
    for _ in range(n_iter):
        x -= lr * (4.0 * x * (x * x - 1.0) - beta)
    return x


def _inline_hysteresis(beta0: float, beta1: float, n: int):
    """Sweep beta up then down through the inline double well; return branches and shoelace area."""
    up = np.linspace(beta0, beta1, n)
    down = up[::-1].copy()
    order_up = np.empty(n)
    state = -1.0
    for i, b in enumerate(up):
        state = _inline_double_well(float(b), state)
        order_up[i] = state
    order_down = np.empty(n)
    for i, b in enumerate(down):
        state = _inline_double_well(float(b), state)
        order_down[i] = state
    bx = np.concatenate([up, down])
    by = np.concatenate([order_up, order_down])
    area = float(0.5 * abs(np.sum(bx * np.roll(by, -1) - np.roll(bx, -1) * by)))
    return up, order_up, down, order_down, area


# ---------------------------------------------------------------------------
# Gated-arm evidence
# ---------------------------------------------------------------------------


def _gated_arm(
    study_id: str, subject: SubjectRef, *, arm: str, needs: str, produces: str
) -> Evidence:
    """A REGISTERED record that an arm is inconclusive because a subsystem or hardware is missing."""
    return make_evidence(
        observable="S14.GatedArm",
        observable_version=_VERSION,
        subject=subject,
        value={
            "arm": arm,
            "status": "inconclusive-because-gated",
            "needs": needs,
            "produces": produces,
        },
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id),
        registered=True,
    )


def analyze(run) -> StudyResult:
    """Run the hysteresis protocol on the bistable stand-in, record the gated real-RL design."""
    study_id = run.study.study_id
    subject = SubjectRef(extra={"study": study_id})

    try:
        from reward_lens.loops.anneal import (
            double_well_responder,
            run_hysteresis,
            up_down_schedule,
        )

        responder = double_well_responder()
        up, down = up_down_schedule(0.0, 3.0, 40)
        hyst_ev = run_hysteresis(responder, up, down, init_state=-1.0)
        run.record(hyst_ev)
        loop = hyst_ev.value
        loop_area = float(loop.loop_area)
        up_transition = (
            float(loop.up_transition) if loop.up_transition is not None else float("nan")
        )
        down_transition = (
            float(loop.down_transition) if loop.down_transition is not None else float("nan")
        )
        order_up_final = float(loop.order_up[-1])
        order_down_final = float(loop.order_down[-1])
        parents = (hyst_ev.id,)
    except ImportError:
        up, order_up, down, order_down, loop_area = _inline_hysteresis(0.0, 3.0, 40)
        d_up = np.abs(np.diff(order_up)) / (np.abs(np.diff(up)) + 1e-12)
        d_down = np.abs(np.diff(order_down)) / (np.abs(np.diff(down)) + 1e-12)
        up_transition = float(0.5 * (up[np.argmax(d_up)] + up[np.argmax(d_up) + 1]))
        down_transition = float(0.5 * (down[np.argmax(d_down)] + down[np.argmax(d_down) + 1]))
        order_up_final = float(order_up[-1])
        order_down_final = float(order_down[-1])
        parents = ()

    # The metastable gap between the up-branch and down-branch onsets is the width of the hysteresis.
    hysteresis_width = float(abs(up_transition - down_transition))
    ev_loop = make_evidence(
        observable="S14.HysteresisLoop",
        observable_version=_VERSION,
        subject=subject,
        value={
            "loop_area": loop_area,
            "irreversible": bool(loop_area > 0.01),
            "up_transition": up_transition,
            "down_transition": down_transition,
            "hysteresis_width": hysteresis_width,
            "order_up_final": order_up_final,
            "order_down_final": order_down_final,
        },
        uncertainty=Uncertainty(n=len(up) + len(down), method="none"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id, parents=parents),
        registered=True,
    )
    run.record(ev_loop)

    run.record(
        _gated_arm(
            study_id,
            subject,
            arm="real-rl-anneal",
            needs="a real RL loop with live training callbacks (reward_lens.loops training "
            "integrations) and a GPU, on a trained policy carrying a planted exploit",
            produces="the gold reward and raw-coordinate feature occupations on both anneal branches, "
            "the production form of the loop-area irreversibility test",
        )
    )

    metrics = {"loop_area": loop_area, "hysteresis_width": hysteresis_width}
    summary = (
        f"The anneal-up / anneal-down protocol on the tilted double well enclosed a loop of area "
        f"{loop_area:.3f} (nonzero, so first-order and irreversible): the aligned well destabilizes "
        f"near beta {up_transition:.2f} on the way up but the hacked well persists until beta "
        f"{down_transition:.2f} on the way down, a hysteresis width of {hysteresis_width:.2f}. The "
        f"real-RL anneal on a trained exploited policy is recorded as inconclusive-because-gated on "
        f"a GPU training loop."
    )
    return StudyResult(outcomes={}, metrics=metrics, summary=summary)


__all__ = ["build_spec", "analyze"]
