"""S12 runs end to end as a frozen study, emitting REGISTERED Evidence and updating T2/T3/T4/T5.

The forecast, accuracy-paradox, teacher-variance, heavy-tail, and prevention arms are exercised on
the planted-hack base-policy draw where the exploited dimension is known by construction. The
weights-derived susceptibility must name the first-hacked dimension and rank-recover the realized
best-of-n drift while a high-accuracy benchmark control does not, and projecting the flagged
direction out of the reward head must remove the predicted hack, inline and (when the certified
modules import) with a passing LEACE probe-recovery certificate. The four-model campaign, the real
GRPO/PPO hump, and the real-model certified radius are recorded as inconclusive-because-gated.
"""

from __future__ import annotations

import numpy as np

from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import TrustLevel
from reward_lens.loops.bon import expected_bon_reward
from reward_lens.studies import Scoreboard, render_report, run_study
from studies.s12_hackability.analysis import (
    _bon_expect,
    _planted_hack_draw,
    build_spec,
)


def test_s12_runs_and_registers(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    # All five registered arms confirmed on the planted vehicle, and no kill criterion fired.
    assert result.outcomes["H1-forecast-first-hacked"] == "confirmed", result.metrics
    assert result.outcomes["H2-beats-accuracy"] == "confirmed", result.metrics
    assert result.outcomes["H3-teacher-variance-speed"] == "confirmed", result.metrics
    assert result.outcomes["H4-heavy-tail-defeats-kl"] == "confirmed", result.metrics
    assert result.outcomes["H5-prevention"] == "confirmed", result.metrics
    assert result.outcomes["H6-overoptimization-hump"] == "confirmed", result.metrics
    assert not result.killed

    # The headline Evidence is REGISTERED and traces to the best-of-n ladder and the planted draw.
    forecast = store.find(observable="S12.Forecast")
    assert forecast and forecast[0].trust is TrustLevel.REGISTERED
    parent_obs = {p.observable for p in store.parents(forecast[0])}
    assert "loops.bon.ladder" in parent_obs
    assert "S12.PlantedHackDraw" in parent_obs


def test_s12_forecast_beats_accuracy(tmp_path):
    store = EvidenceStore(tmp_path)
    _, result = run_study(build_spec(), store=store)
    m = result.metrics

    # Prediction: chi names the planted first-hacked dimension and rank-recovers the realized drift.
    assert m["chi_names_hacked"] == 1.0
    assert m["chi_forecast_spearman"] > 0.8
    assert m["chi_forecast_r2"] > 0.7

    # The accuracy paradox: the reward model's benchmark accuracy is high, yet its per-feature
    # attribution does not recover the hacked dimension the internal index names.
    assert m["benchmark_accuracy"] > 0.8
    assert m["accuracy_forecast_spearman"] < 0.5
    assert m["forecast_margin"] > 0.4

    # The Forecast/AccuracyParadox evidence carries the naming and the paradox explicitly.
    fc = store.find(observable="S12.Forecast")[0].value
    assert fc["realized_hack_dim"] == fc["planted_hack_dim"]
    assert fc["hump_is_interior"] and fc["hump_peak_kl"] > 0.0
    ap = store.find(observable="S12.AccuracyParadox")[0].value
    assert ap["accuracy_names_hacked"] == 0.0
    assert ap["distortion_names_hacked"] == 1.0  # the internal distortion index isolates the hack


def test_s12_teacher_variance_and_heavy_tail(tmp_path):
    store = EvidenceStore(tmp_path)
    _, result = run_study(build_spec(), store=store)
    m = result.metrics

    # T3: teacher variance predicts best-of-n speed (Razin), with a scale-independent proportionality.
    assert m["teacher_speed_spearman"] > 0.9
    speed = store.find(observable="S12.TeacherVarianceSpeed")[0].value
    assert speed["proportionality_cv"] < 0.05  # gain / sqrt(teacher variance) is nearly constant

    # T5: a heavy (polynomial) tail extracts more reward than a light one at matched KL and variance.
    assert m["heavy_tail_excess"] > 0.1
    tail = store.find(observable="S12.HeavyTail")[0].value
    assert tail["heavy_regime"] == "polynomial" and np.isfinite(tail["heavy_alpha"])
    assert not np.isfinite(tail["light_alpha"])  # the light tail has no finite polynomial index


def test_s12_prevention_and_certified(tmp_path):
    store = EvidenceStore(tmp_path)
    _, result = run_study(build_spec(), store=store)
    m = result.metrics

    # Prevention: projecting the flagged direction out of the head collapses the hack drift and
    # flattens the gold overoptimization hump.
    assert m["hack_drift_reduction"] > 0.8
    assert m["gold_overopt_drop_before"] > 0.3
    assert m["gold_overopt_drop_after"] < 0.1 * m["gold_overopt_drop_before"]

    prev = store.find(observable="S12.Prevention")[0].value
    assert abs(prev["edited_head_component_on_flag"]) < 1e-9  # the flagged direction is removed
    assert "EditIntervention" in prev["production_seam"]

    # The certified LEACE arm is the production upgrade: it runs on this numpy vehicle when the
    # modules import and its held-out probe-recovery certificate passes, else it is recorded pending.
    cert_rec = store.find(observable="S12.CertifiedErasure")
    assert cert_rec
    cv = cert_rec[0].value
    assert cv["status"] in {"ran", "pending-certified-modules-absent"}
    if cv["status"] == "ran":
        assert cv["certificate_passed"] is True
        assert cv["certificate_recovery_auc"] <= 0.55  # probe at chance after erasure
        assert cv["hack_drift_reduction"] > 0.8
        assert abs(cv["leace_head_component_on_flag"]) < 0.1  # LEACE also removes the direction
        # The raw certificate Evidence earned CALIBRATED trust from its passing held-out recovery.
        certs = store.find(observable="interventions.certify_erasure")
        assert certs and certs[0].trust is TrustLevel.CALIBRATED


def test_s12_organism_tie_in(tmp_path):
    store = EvidenceStore(tmp_path)
    run_study(build_spec(), store=store)

    # The synthetic vehicle is tied to the foundry's real hack-direction organism, which must carry
    # the hack signature (rewarded by the label, bad for the gold objective).
    org = store.find(observable="S12.OrganismSignature")
    assert org and org[0].trust is TrustLevel.REGISTERED
    val = org[0].value
    if val["organism_family"] is not None:
        assert val["carries_hack_signature"] is True
        assert val["cov_hack_label"] > 0 and val["cov_hack_gold"] <= 0


def test_s12_gated_arms_recorded(tmp_path):
    store = EvidenceStore(tmp_path)
    run_study(build_spec(), store=store)

    gated = store.find(observable="S12.GatedArm")
    arms = {ev.value["arm"] for ev in gated}
    assert {"four-model-campaign", "grpo-ppo-hump", "real-model-certified-radius"} <= arms
    for ev in gated:
        assert ev.value["status"] == "inconclusive-because-gated"
        assert ev.value["needs"] and ev.value["produces"]
        assert ev.trust is TrustLevel.REGISTERED


def test_s12_updates_scoreboard(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    board = Scoreboard(tmp_path / "scoreboard.json")
    board.update_from_result(frozen.study_id, frozen.spec.hypotheses, result)
    for row in ("T2", "T3", "T4", "T5"):
        assert board.rows[row].status == "confirmed", row
        assert board.rows[row].adjudicating_evidence

    report = render_report(frozen, result, store)
    assert "CONFIRMED" in report
    assert frozen.study_id in report


def test_s12_bon_concomitant_matches_frozen_primitive():
    # The concomitant estimator must reduce to loops.bon.expected_bon_reward when the tracked
    # quantity is the proxy itself, which pins the gold/feature frontier to the frozen BoN identity.
    draw = _planted_hack_draw(seed=1)
    bank = draw.proxy()[0]  # one prompt's proxy bank
    for n in (1, 2, 8, 64):
        got = _bon_expect(bank[None, :], bank[None, :], n)
        assert np.isclose(got, expected_bon_reward(bank, n), atol=1e-9)
