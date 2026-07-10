"""S14 runs end to end as a frozen study, emitting REGISTERED Evidence and updating T9.

The hysteresis protocol runs on the CPU-provable tilted double well where the loop area is
analytically nonzero: the anneal-up / anneal-down branches enclose a nonzero area, the signature of a
first-order irreversible transition. The real-RL anneal on a trained exploited policy is recorded as
inconclusive-because-gated on a GPU training loop.
"""

from __future__ import annotations

from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import TrustLevel
from reward_lens.studies import Scoreboard, render_report, run_study
from studies.s14_phase.analysis import build_spec


def test_s14_runs_and_registers(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    # The loop area is nonzero on the bistable stand-in, so the transition is first-order.
    assert result.outcomes["H1-nonzero-loop-area"] == "confirmed", result.metrics
    assert not result.killed
    assert result.metrics["loop_area"] > 0.1
    # The metastable gap between the up-branch and down-branch onsets is a real hysteresis width.
    assert result.metrics["hysteresis_width"] > 0.1

    # The headline Evidence is REGISTERED and traces to its anneal parent (a real DAG).
    loop = store.find(observable="S14.HysteresisLoop")
    assert loop and loop[0].trust is TrustLevel.REGISTERED
    parents = store.parents(loop[0])
    assert parents and parents[0].observable == "loops.anneal.hysteresis"


def test_s14_gated_arm_recorded(tmp_path):
    store = EvidenceStore(tmp_path)
    run_study(build_spec(), store=store)

    gated = store.find(observable="S14.GatedArm")
    assert {ev.value["arm"] for ev in gated} == {"real-rl-anneal"}
    for ev in gated:
        assert ev.value["status"] == "inconclusive-because-gated"
        assert ev.value["needs"] and ev.value["produces"]
        assert ev.trust is TrustLevel.REGISTERED


def test_s14_updates_scoreboard(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    board = Scoreboard(tmp_path / "scoreboard.json")
    board.update_from_result(frozen.study_id, frozen.spec.hypotheses, result)
    assert board.rows["T9"].status == "confirmed"
    assert board.rows["T9"].adjudicating_evidence

    report = render_report(frozen, result, store)
    assert "CONFIRMED" in report
    assert frozen.study_id in report
