"""S15 runs end to end as a frozen study, emitting REGISTERED Evidence and updating T14.

The Skepticism score S and the Receipt Reliance Score RRS are calibrated on synthetic planted-receipt
trajectories where the grader's reliance and skepticism are known by construction: the two statistics
recover the planted reliance, and in the 2x2 honesty grid the credulity axis predicts omission while
fabrication depends on the skepticism axis, so the kill criterion does not fire. The GRPO confirmation
on a trained policy is recorded as inconclusive-because-gated on a GPU RL loop.
"""

from __future__ import annotations

from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import TrustLevel
from reward_lens.studies import Scoreboard, render_report, run_study
from studies.s15_forensics.analysis import build_spec


def test_s15_runs_and_registers(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    # Both calibration hypotheses confirmed on the planted-receipt data.
    assert result.outcomes["H1-reliance-recovery"] == "confirmed", result.metrics
    assert result.outcomes["H2-credulity-predicts-omission"] == "confirmed", result.metrics
    assert not result.killed

    # S and RRS recover the planted reliance; the credulity axis drives omission.
    assert result.metrics["reliance_recovery"] > 0.8
    assert result.metrics["omission_credulity_gap"] > 0.2
    # Fabrication depends on the skepticism axis, so the disclosure-game framing is not empty.
    assert result.metrics["fabrication_s_effect"] > 0.1
    # The liar quadrant (reads receipts, forgives silence) fabricates heavily.
    assert result.metrics["liar_quadrant_fabrication"] > 0.5

    recovery = store.find(observable="S15.SkepticismRecovery")
    assert recovery and recovery[0].trust is TrustLevel.REGISTERED

    grid = store.find(observable="S15.HonestyGrid")
    assert grid and grid[0].trust is TrustLevel.REGISTERED
    parents = store.parents(grid[0])
    assert parents and parents[0].observable == "S15.SkepticismRecovery"


def test_s15_gated_arm_recorded(tmp_path):
    store = EvidenceStore(tmp_path)
    run_study(build_spec(), store=store)

    gated = store.find(observable="S15.GatedArm")
    assert {ev.value["arm"] for ev in gated} == {"grpo-confirmation"}
    for ev in gated:
        assert ev.value["status"] == "inconclusive-because-gated"
        assert ev.value["needs"] and ev.value["produces"]
        assert ev.trust is TrustLevel.REGISTERED


def test_s15_updates_scoreboard(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    board = Scoreboard(tmp_path / "scoreboard.json")
    board.update_from_result(frozen.study_id, frozen.spec.hypotheses, result)
    assert board.rows["T14"].status == "confirmed"
    assert board.rows["T14"].adjudicating_evidence

    report = render_report(frozen, result, store)
    assert "CONFIRMED" in report
    assert frozen.study_id in report
