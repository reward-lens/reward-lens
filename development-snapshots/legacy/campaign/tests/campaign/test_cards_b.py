"""Tiny-path tests for the loops card analyses: CHI-DRIFT, HUMP, PPE-BON, LADDER.

Each card's planted tiny subjects run through the real ``run_study`` path, so the assertions
cover the full contract: every registry-required metric is emitted, the plant is recovered
rather than merely present, no kill fires on a confirming plant, and the two frozen rules the
staged freezes import (``predict_hump_kl``, ``fit_ladder_intervals``) reproduce closed-form
answers. Resolver round-trips exercise the store reads with a wrinkle: the meta-borne
correctness labels and votes (PPE), the frozen json records (HUMP, LADDER), and the
prompt-alignment checks (CHI-DRIFT).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import campaign.config
from campaign import chidrift, hump, ladder, ppe
from campaign import evidence_keys as ek
from campaign.config import LADDER_HELD_OUT, LADDER_RUNGS, ROSTER, task_seed
from campaign.payloads import BankedFeatures, ScoreBank, TablePayload
from campaign.registry import CARDS_BY_ID, required_metrics_for
from campaign.specs import build_spec
from reward_lens.core.store import EvidenceStore
from reward_lens.loops.bon import bon_kl
from reward_lens.studies.runner import run_study

_MODULES = {
    "CHI-DRIFT": chidrift,
    "HUMP": hump,
    "PPE-BON": ppe,
    "LADDER": ladder,
}


def _run_tiny(card_id: str, tmp_path):
    module = _MODULES[card_id]
    rng = np.random.default_rng(task_seed("test-cards-b", card_id))
    store = EvidenceStore(tmp_path / "store")
    spec = build_spec(CARDS_BY_ID[card_id])
    frozen, result = run_study(spec, subjects=module.tiny_subjects(rng), store=store)
    return frozen, result


@pytest.mark.parametrize("card_id", sorted(_MODULES))
def test_tiny_path_emits_required_metrics_and_confirms(card_id, tmp_path):
    card = CARDS_BY_ID[card_id]
    _, result = _run_tiny(card_id, tmp_path)
    for metric in required_metrics_for(card):
        assert metric in result.metrics, f"{card_id} omitted {metric}"
    assert not result.killed, f"{card_id} kill fired on a confirming plant: {result.killed_by}"
    for hyp_id, outcome in result.outcomes.items():
        assert outcome == "confirmed", f"{card_id}/{hyp_id} was {outcome} on the plant"


# ---------------------------------------------------------------------------
# Planted recoveries.
# ---------------------------------------------------------------------------


def test_chidrift_recovers_the_gibbs_plant(tmp_path):
    _, result = _run_tiny("CHI-DRIFT", tmp_path)
    m = result.metrics
    # The Gibbs-tilted plant makes chi genuinely predictive and the variance ordering
    # anti-predictive, so the correlation is high and the margin over the baseline is large.
    assert m["chi_bon_spearman"] > 0.7
    assert m["chi_bon_spearman_ci_low"] > 0.0
    assert m["chi_minus_variance_baseline_spearman"] > 0.5
    assert m["chi_minus_variance_baseline_spearman_ci_low"] > 0.0
    # The headline is the per-model minimum; the second model's value can only sit at or
    # above it, and the CI bounds stay ordered around the point.
    assert m["chi_bon_spearman"] <= m["chi_bon_spearman_second_model"]
    assert (
        m["chi_bon_spearman_ci_low"]
        <= m["chi_bon_spearman"]
        <= m["chi_bon_spearman_ci_high"]
    )


def test_hump_locates_the_planted_interior_hump(tmp_path):
    _, result = _run_tiny("HUMP", tmp_path)
    m = result.metrics
    assert m["hump_interior_present"] == 1.0
    # The plant peaks at n = 4, whose KL bound equals the frozen map's value at tail index 2.
    assert m["hump_kl_predicted"] == pytest.approx(hump.predict_hump_kl(2.0))
    assert m["hump_kl_abs_error"] < 0.5
    assert 0.0 < m["hump_kl_observed"] < float(bon_kl(32))


def test_hump_boundary_case_keeps_the_kill_adjudicable(tmp_path):
    # Gold equal to proxy makes selection gain monotone in n: no interior hump exists, the
    # observed location is reported at the boundary the fitted form prefers, and the location
    # error stays a real number the kill criterion can fire on.
    rng = np.random.default_rng(task_seed("test-cards-b", "hump-boundary"))
    proxy = rng.exponential(1.0, size=(40, 32))
    subjects = {
        "proxy_bank": proxy,
        "gold_bank": proxy.copy(),
        "hump_kl_predicted": hump.predict_hump_kl(2.0),
        "tail_record": {"tail_index": 2.0, "regime": "planted"},
    }
    store = EvidenceStore(tmp_path / "store")
    _, result = run_study(build_spec(CARDS_BY_ID["HUMP"]), subjects=subjects, store=store)
    m = result.metrics
    assert m["hump_interior_present"] == 0.0
    assert m["hump_kl_observed"] == pytest.approx(float(bon_kl(32)))
    assert np.isfinite(m["hump_kl_abs_error"])
    assert result.killed and result.killed_by == ["K-hump"]


def test_hump_tiny_recorder_replay_carries_a_leading_onset():
    rng = np.random.default_rng(task_seed("test-cards-b", "HUMP"))
    report = hump.tiny_subjects(rng)["drift_report"]
    assert report.exploited_direction == "hack"
    assert report.feature_onset is not None
    assert report.gold_onset is not None
    assert report.lead_time is not None and report.lead_time > 0


def test_ppe_recovers_the_planted_tail_plateau_ordering(tmp_path):
    _, result = _run_tiny("PPE-BON", tmp_path)
    m = result.metrics
    assert m["spearman_tail_vs_bon_plateau"] > 0.5
    assert m["spearman_tail_vs_bon_plateau_ci_low"] > 0.0
    assert m["bon_lowerquantile_beats_mean"] == 1.0


def test_ppe_gold_curve_is_a_probability_and_saturates_on_a_perfect_rm():
    # When correctness is exactly "top half by score", the best-of-k accuracy must rise
    # from the base rate at k = 1 toward 1.0, monotonically: the concomitant weights put
    # all mass on the top score rank as k grows.
    scores = np.tile(np.arange(8.0), (10, 1))
    correct = (scores >= 4.0).astype(np.float64)
    ks, acc = ppe.gold_best_of_k_curve(scores, correct)
    assert ks[0] == 1 and acc[0] == pytest.approx(0.5)
    assert np.all(np.diff(acc) >= -1e-12)
    assert acc[-1] > 0.99
    assert np.all((acc >= 0.0) & (acc <= 1.0))


def test_ladder_recovers_the_planted_trend(tmp_path):
    _, result = _run_tiny("LADDER", tmp_path)
    m = result.metrics
    assert m["ladder_8b_metrics_in_interval"] == 3.0
    assert m["ladder_8b_crystallization_in_interval"] == 1.0
    assert m["ladder_8b_chi_teacher_in_interval"] == 1.0
    assert m["ladder_8b_snr_in_interval"] == 1.0
    # The 8B sits on the trend, so extrapolation beats the flat baseline on every quantity
    # and the coarse three-value bootstrap stays entirely below zero.
    assert m["ladder_extrap_minus_flat_error"] < 0.0
    assert m["ladder_extrap_minus_flat_error_ci_high"] < 0.0


# ---------------------------------------------------------------------------
# The two frozen rules, on closed-form cases.
# ---------------------------------------------------------------------------


def test_predict_hump_kl_closed_forms():
    # KL* = alpha log 2 - 1 + 2^(-alpha), directly.
    assert hump.predict_hump_kl(1.0) == pytest.approx(math.log(2.0) - 0.5)
    assert hump.predict_hump_kl(2.0) == pytest.approx(2.0 * math.log(2.0) - 0.75)
    # The map is bon_kl evaluated at the hedge point n* = 2^alpha.
    for alpha in (0.5, 1.0, 2.0, 3.7):
        assert hump.predict_hump_kl(alpha) == pytest.approx(float(bon_kl(2.0**alpha)))


def test_predict_hump_kl_is_monotone_in_the_tail_index():
    grid = np.linspace(0.2, 8.0, 40)
    vals = [hump.predict_hump_kl(a) for a in grid]
    assert np.all(np.diff(vals) > 0)


def test_predict_hump_kl_edge_cases():
    with pytest.raises(ValueError):
        hump.predict_hump_kl(0.0)
    with pytest.raises(ValueError):
        hump.predict_hump_kl(-1.0)
    with pytest.raises(ValueError):
        hump.predict_hump_kl(float("nan"))
    # A light tail predicts no hump at any finite pressure.
    assert hump.predict_hump_kl(float("inf")) == float("inf")


def test_fit_ladder_intervals_exact_line_gives_a_point_interval():
    params = [0.6, 1.7, 4.0]
    a, b = 2.0, 3.0
    values = [a + b * math.log(p) for p in params]
    out = ladder.fit_ladder_intervals(params, {"q": values})
    point = a + b * math.log(ROSTER[LADDER_HELD_OUT].params_b)
    assert out["q"]["point"] == pytest.approx(point)
    # Zero residuals give a zero-width interval: the rule never invents uncertainty.
    assert out["q"]["lo"] == pytest.approx(point)
    assert out["q"]["hi"] == pytest.approx(point)
    assert out["q"]["flat_baseline"] == pytest.approx(values[-1])


def test_fit_ladder_intervals_matches_the_hand_computed_case():
    # Rungs at params (1, e, e^2) put x = (0, 1, 2). With y = (0, 2, 1) the OLS line is
    # y = 0.5 + 0.5 x with residuals (-0.5, 1.0, -0.5), so s^2 = 1.5 on one degree of
    # freedom, where the t quantile is the Cauchy one: t_{0.90, 1} = tan(0.4 pi).
    params = [1.0, math.e, math.e**2]
    out = ladder.fit_ladder_intervals(params, {"q": [0.0, 2.0, 1.0]})["q"]
    x0 = math.log(ROSTER[LADDER_HELD_OUT].params_b)
    point = 0.5 + 0.5 * x0
    s = math.sqrt(1.5)
    half = math.tan(0.4 * math.pi) * s * math.sqrt(1.0 + 1.0 / 3.0 + (x0 - 1.0) ** 2 / 2.0)
    assert out["point"] == pytest.approx(point)
    assert out["lo"] == pytest.approx(point - half)
    assert out["hi"] == pytest.approx(point + half)
    assert out["flat_baseline"] == pytest.approx(1.0)  # the largest rung's value


def test_fit_ladder_intervals_refuses_degenerate_input():
    with pytest.raises(ValueError):
        ladder.fit_ladder_intervals([0.6, 1.7], {"q": [1.0, 2.0]})
    with pytest.raises(ValueError):
        ladder.fit_ladder_intervals([2.0, 2.0, 2.0], {"q": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        ladder.fit_ladder_intervals([0.6, 1.7, 4.0], {"q": [1.0, 2.0]})


# ---------------------------------------------------------------------------
# Resolver round-trips.
# ---------------------------------------------------------------------------


def _seed_chidrift_store(store: EvidenceStore, rng: np.random.Generator) -> dict:
    tiny = chidrift.tiny_subjects(rng)
    n_prompts = tiny["features"].shape[0]
    prompt_ids = [f"prompt-{i}" for i in range(n_prompts)]
    ids = {}
    for key in CARDS_BY_ID["CHI-DRIFT"].signals:
        bank = ScoreBank(
            item_ids=prompt_ids,
            scores=np.asarray(tiny[f"bank::{key}"]),
            layout="bank",
        )
        ids[key] = store.append(
            ek.record_intermediate(
                observable=ek.OBS_SCORES,
                roster_key=key,
                slice_name="ultrafeedback-bank",
                value=bank,
            )
        )
    feats = BankedFeatures(
        prompt_ids=prompt_ids,
        names=list(tiny["feature_names"]),
        tensor=np.asarray(tiny["features"]),
    )
    ids["features"] = store.append(
        ek.record_intermediate(
            observable=ek.OBS_BANKED_FEATURES,
            roster_key=ek.DATA_KEY,
            slice_name="ultrafeedback-bank",
            value=feats,
        )
    )
    return ids


def test_chidrift_resolver_round_trip(tmp_path):
    rng = np.random.default_rng(task_seed("test-cards-b", "chidrift-resolve"))
    store = EvidenceStore(tmp_path / "store")
    ids = _seed_chidrift_store(store, rng)
    subjects = chidrift.resolve_subjects(store)
    assert subjects["features__ev"] == ids["features"]
    for key in CARDS_BY_ID["CHI-DRIFT"].signals:
        assert subjects[f"bank::{key}__ev"] == ids[key]
    _, result = run_study(
        build_spec(CARDS_BY_ID["CHI-DRIFT"]), subjects=subjects, store=store
    )
    assert result.metrics["chi_bon_spearman"] > 0.7


def test_chidrift_resolver_rejects_misaligned_prompts(tmp_path):
    rng = np.random.default_rng(task_seed("test-cards-b", "chidrift-misaligned"))
    store = EvidenceStore(tmp_path / "store")
    tiny = chidrift.tiny_subjects(rng)
    n_prompts = tiny["features"].shape[0]
    for j, key in enumerate(CARDS_BY_ID["CHI-DRIFT"].signals):
        bank = ScoreBank(
            item_ids=[f"model{j}-prompt-{i}" for i in range(n_prompts)],
            scores=np.asarray(tiny[f"bank::{key}"]),
            layout="bank",
        )
        store.append(
            ek.record_intermediate(
                observable=ek.OBS_SCORES,
                roster_key=key,
                slice_name="ultrafeedback-bank",
                value=bank,
            )
        )
    with pytest.raises(KeyError, match="prompt-aligned"):
        chidrift.resolve_subjects(store)


def test_ppe_resolver_round_trip(tmp_path):
    rng = np.random.default_rng(task_seed("test-cards-b", "ppe-resolve"))
    store = EvidenceStore(tmp_path / "store")
    tiny = ppe.tiny_subjects(rng)
    card = CARDS_BY_ID["PPE-BON"]
    for set_name in ppe._SETS:
        for rm in card.signals:
            bank = ScoreBank(
                item_ids=[f"{set_name}-p{i}" for i in range(40)],
                scores=np.asarray(tiny[f"scores::{rm}::{set_name}"]),
                layout="bank",
                meta={"correct": np.asarray(tiny[f"correct::{rm}::{set_name}"])},
            )
            store.append(
                ek.record_intermediate(
                    observable=ek.OBS_SCORES,
                    roster_key=rm,
                    slice_name=f"ppe-best-of-k::{set_name}",
                    value=bank,
                )
            )
    votes = np.asarray(tiny["human_votes"])
    human = ScoreBank(
        item_ids=[f"battle-{i // 2}-side-{'ab'[i % 2]}" for i in range(2 * votes.size)],
        scores=np.asarray(tiny["human_scores"]),
        layout="bank",
        meta={"votes": votes},
    )
    store.append(
        ek.record_intermediate(
            observable=ek.OBS_SCORES,
            roster_key=card.signals[0],
            slice_name="ppe-human",
            value=human,
        )
    )
    subjects = ppe.resolve_subjects(store)
    np.testing.assert_allclose(subjects["human_votes"], votes)
    _, result = run_study(build_spec(card), subjects=subjects, store=store)
    assert result.metrics["bon_lowerquantile_beats_mean"] == 1.0
    assert result.metrics["spearman_tail_vs_bon_plateau"] > 0.5


def test_ppe_resolver_raises_precisely_without_correctness_labels(tmp_path):
    rng = np.random.default_rng(task_seed("test-cards-b", "ppe-nolabels"))
    store = EvidenceStore(tmp_path / "store")
    tiny = ppe.tiny_subjects(rng)
    rm = CARDS_BY_ID["PPE-BON"].signals[0]
    set_name = ppe._SETS[0]
    bank = ScoreBank(
        item_ids=[f"p{i}" for i in range(40)],
        scores=np.asarray(tiny[f"scores::{rm}::{set_name}"]),
        layout="bank",
    )
    store.append(
        ek.record_intermediate(
            observable=ek.OBS_SCORES,
            roster_key=rm,
            slice_name=f"ppe-best-of-k::{set_name}",
            value=bank,
        )
    )
    with pytest.raises(KeyError, match="correct"):
        ppe.resolve_subjects(store)


def test_hump_resolver_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign.config, "FROZEN_DIR", tmp_path / "frozen")
    rng = np.random.default_rng(task_seed("test-cards-b", "hump-resolve"))
    store = EvidenceStore(tmp_path / "store")
    tiny = hump.tiny_subjects(rng)
    item_ids = [f"prompt-{i}" for i in range(tiny["proxy_bank"].shape[0])]
    for name, key in (("proxy_bank", hump._PROXY), ("gold_bank", hump._GOLD)):
        store.append(
            ek.record_intermediate(
                observable=ek.OBS_SCORES,
                roster_key=key,
                slice_name="hump-prompts",
                value=ScoreBank(item_ids=item_ids, scores=np.asarray(tiny[name]), layout="bank"),
            )
        )
    store.append(
        ek.record_intermediate(
            observable=ek.OBS_DRIFT_REPORT,
            roster_key=hump._POLICY,
            slice_name="hump-prompts",
            value=tiny["drift_report"],
        )
    )
    # The staged-freeze record must exist before the resolver will hand out subjects.
    with pytest.raises(KeyError, match="frozen hump prediction"):
        hump.resolve_subjects(store)
    (tmp_path / "frozen").mkdir()
    (tmp_path / "frozen" / "hump_prediction.json").write_text(
        json.dumps(
            {
                "hump_kl_predicted": tiny["hump_kl_predicted"],
                "tail": {"tail_index": 2.0, "regime": "planted"},
            }
        ),
        encoding="utf-8",
    )
    subjects = hump.resolve_subjects(store)
    assert subjects["hump_kl_predicted"] == pytest.approx(tiny["hump_kl_predicted"])
    assert subjects["drift_report"].exploited_direction == "hack"
    _, result = run_study(build_spec(CARDS_BY_ID["HUMP"]), subjects=subjects, store=store)
    assert result.metrics["hump_interior_present"] == 1.0
    assert result.metrics["hump_kl_abs_error"] < 0.5


def test_ladder_resolver_round_trip(tmp_path, monkeypatch):
    intervals_path = tmp_path / "frozen" / "ladder_intervals.json"
    monkeypatch.setattr(campaign.config, "LADDER_INTERVALS_PATH", intervals_path)
    rng = np.random.default_rng(task_seed("test-cards-b", "ladder-resolve"))
    store = EvidenceStore(tmp_path / "store")
    tiny = ladder.tiny_subjects(rng)
    for i, key in enumerate(LADDER_RUNGS):
        table = TablePayload(
            name=f"{key} ladder quantities",
            columns=["metric", "value"],
            rows=[[q, tiny["rung_values"][q][i]] for q in ladder._QUANTITIES],
        )
        store.append(
            ek.record_intermediate(
                observable=ek.OBS_INDEX_TABLE,
                roster_key=key,
                slice_name="rb2-full",
                value=table,
            )
        )
    table_8b = TablePayload(
        name="held-out ladder quantities",
        columns=["metric", "value"],
        rows=[[q, tiny["values_8b"][q]] for q in ladder._QUANTITIES],
    )
    store.append(
        ek.record_intermediate(
            observable=ek.OBS_INDEX_TABLE,
            roster_key=LADDER_HELD_OUT,
            slice_name="rb2-full",
            value=table_8b,
        )
    )
    # The staged-freeze record must exist before the resolver will hand out subjects.
    with pytest.raises(KeyError, match="frozen ladder intervals"):
        ladder.resolve_subjects(store)
    intervals_path.parent.mkdir(parents=True)
    intervals_path.write_text(
        json.dumps(
            {
                "intervals": tiny["intervals"],
                "held_out_params_b": tiny["held_out_params_b"],
            }
        ),
        encoding="utf-8",
    )
    subjects = ladder.resolve_subjects(store)
    assert subjects["values_8b"]["snr"] == pytest.approx(tiny["values_8b"]["snr"])
    _, result = run_study(build_spec(CARDS_BY_ID["LADDER"]), subjects=subjects, store=store)
    assert result.metrics["ladder_8b_metrics_in_interval"] == 3.0
    assert result.metrics["ladder_extrap_minus_flat_error_ci_high"] < 0.0


def test_ladder_resolver_raises_precisely_on_a_gutted_index_table(tmp_path, monkeypatch):
    monkeypatch.setattr(
        campaign.config, "LADDER_INTERVALS_PATH", tmp_path / "ladder_intervals.json"
    )
    store = EvidenceStore(tmp_path / "store")
    table = TablePayload(
        name="incomplete", columns=["metric", "value"], rows=[["crystallization", 0.8]]
    )
    store.append(
        ek.record_intermediate(
            observable=ek.OBS_INDEX_TABLE,
            roster_key=LADDER_RUNGS[0],
            slice_name="rb2-full",
            value=table,
        )
    )
    with pytest.raises(KeyError, match="chi_teacher_variance"):
        ladder.resolve_subjects(store)
