"""S13 runs end to end as a frozen study, emitting REGISTERED Evidence and updating T11.

The recorder-on-organism experiment is exercised on the synthetic planted-hack rollouts where the
exploited direction and the divergence onset are known by construction: the monitor must name the
planted direction with a positive lead time, and the w_r-anchored signal must survive the obfuscation
that kills a free-floating probe. The kinship kappa and weak-to-strong arms are recorded as
inconclusive-because-gated on the GPU sibling base population.
"""

from __future__ import annotations

from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import TrustLevel
from reward_lens.studies import Scoreboard, render_report, run_study
from studies.s13_coupling.analysis import build_spec


def test_s13_runs_and_registers(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    # Both calibration hypotheses confirmed on the synthetic organisms.
    assert result.outcomes["H1-named-lead-time"] == "confirmed", result.metrics
    assert result.outcomes["H2-obfuscation-resistance"] == "confirmed", result.metrics
    assert not result.killed

    # The monitor named the planted direction and saw it before the gold divergence.
    assert result.metrics["names_planted"] == 1.0
    assert result.metrics["lead_time"] > 0
    assert result.metrics["named_lead_time"] > 0
    # The w_r-anchored signal degraded far less than the free-floating probe under pressure.
    assert result.metrics["obfuscation_gap"] > 0.3
    assert result.metrics["anchored_retention"] > result.metrics["free_retention"]

    # The headline Evidence is REGISTERED and traces to its recorder parent (a real DAG).
    naming = store.find(observable="S13.RecorderNaming")
    assert naming and naming[0].trust is TrustLevel.REGISTERED
    parents = store.parents(naming[0])
    assert parents and parents[0].observable == "loops.recorder.drift"

    obf = store.find(observable="S13.Obfuscation")
    assert obf and obf[0].trust is TrustLevel.REGISTERED


def test_s13_gated_arms_recorded(tmp_path):
    store = EvidenceStore(tmp_path)
    run_study(build_spec(), store=store)

    gated = store.find(observable="S13.GatedArm")
    arms = {ev.value["arm"] for ev in gated}
    assert arms == {"kinship-kappa", "weak-to-strong-alpha-gamma"}
    for ev in gated:
        assert ev.value["status"] == "inconclusive-because-gated"
        assert ev.value["needs"] and ev.value["produces"]
        assert ev.trust is TrustLevel.REGISTERED


def test_s13_updates_scoreboard(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    board = Scoreboard(tmp_path / "scoreboard.json")
    board.update_from_result(frozen.study_id, frozen.spec.hypotheses, result)
    assert board.rows["T11"].status == "confirmed"
    assert board.rows["T11"].adjudicating_evidence

    report = render_report(frozen, result, store)
    assert "CONFIRMED" in report
    assert frozen.study_id in report
