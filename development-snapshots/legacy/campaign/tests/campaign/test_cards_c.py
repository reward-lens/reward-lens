"""Tiny-path tests for the gauge and substrate card analyses.

Every card module here runs its planted tiny subjects through the real ``run_study`` path, so
the assertions cover the full contract: the frozen spec adjudicates against the exact metric
strings the registry requires, the plant is recovered rather than merely emitted, and no kill
criterion fires on a confirming plant. The extra tests pin the operationalizations that carry
the most weight: the gauge gate refusal is recorded as a works-ledger row, the PRM localization
AUC is oriented so that low step scores mark the error, the judge's flip rate stays numeric
without a swapped arm, the contested CI survives the campaign overlay, and the forensic grid
kill fires when the skepticism axis is flat.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.core.store import EvidenceStore
from reward_lens.studies.runner import run_study

from campaign import contest, evalaware, forensic, gauge, gaugexfam, judge, verif
from campaign import evidence_keys as ek
from campaign.adjudicate import apply_ci_overlay
from campaign.config import ENSEMBLE_5, task_seed
from campaign.payloads import (
    FeatureMatrix,
    MatrixPayload,
    PatchGridResult,
    ScoreBank,
    StepScores,
    TablePayload,
    VectorPayload,
)
from campaign.registry import CARDS_BY_ID, required_metrics_for
from campaign.specs import build_spec

_MODULES = {
    "GAUGE-E19": gauge,
    "GAUGE-XFAM": gaugexfam,
    "VERIF-PRM": verif,
    "JUDGE-VBC": judge,
    "VALUES-CONTEST": contest,
    "FORENSIC-RECEIPT": forensic,
    "EVAL-AWARE": evalaware,
}


def _run_tiny(card_id, tmp_path, subjects=None, store=None):
    module = _MODULES[card_id]
    rng = np.random.default_rng(task_seed("test-cards-c", card_id))
    store = store if store is not None else EvidenceStore(tmp_path / "store")
    spec = build_spec(CARDS_BY_ID[card_id])
    subjects = subjects if subjects is not None else module.tiny_subjects(rng)
    frozen, result = run_study(spec, subjects=subjects, store=store)
    return frozen, result, store


@pytest.mark.parametrize("card_id", sorted(_MODULES))
def test_tiny_path_emits_required_metrics_and_confirms(card_id, tmp_path):
    card = CARDS_BY_ID[card_id]
    _, result, _ = _run_tiny(card_id, tmp_path)
    for metric in required_metrics_for(card):
        assert metric in result.metrics, f"{card_id} omitted {metric}"
    assert not result.killed, f"{card_id} kill fired on a confirming plant: {result.killed_by}"
    for hyp_id, outcome in result.outcomes.items():
        assert outcome == "confirmed", f"{card_id}/{hyp_id} was {outcome} on the plant"


# ---------------------------------------------------------------------------
# GAUGE-E19: the planted co-rotation and the gate refusal.
# ---------------------------------------------------------------------------


def test_gauge_recovers_corotation_and_names_the_residual_concept(tmp_path):
    _, result, _ = _run_tiny("GAUGE-E19", tmp_path)
    m = result.metrics
    assert abs(m["raw_cos_v01_v02"]) < 0.02  # planted raw orthogonality
    assert m["canonical_cos"] > 0.9  # the null admixture is quotiented away
    assert m["canonical_minus_raw"] > 0.4
    assert (
        m["canonical_minus_raw_ci_low"]
        <= m["canonical_minus_raw"]
        <= m["canonical_minus_raw_ci_high"]
    )
    assert m["residual_concept_matches_top_behavioral_delta"] == 1.0


def test_gauge_gate_refusal_fires_and_lands_in_the_ledger_row(tmp_path):
    _, _, store = _run_tiny("GAUGE-E19", tmp_path)
    hits = [ev for ev in store if ev.observable == "campaign.result.GAUGE-E19"]
    assert hits, "the summary evidence was not recorded"
    demo = hits[-1].value.meta["gauge_gate_refusal"]
    assert demo["raised"] is True
    assert demo["error"] == "GaugeError"
    assert "frame" in demo["message"].lower()


def test_gauge_ci_overlay_keeps_the_planted_confirmation(tmp_path):
    spec = build_spec(CARDS_BY_ID["GAUGE-E19"])
    rng = np.random.default_rng(task_seed("test-cards-c", "gauge-overlay"))
    store = EvidenceStore(tmp_path / "store")
    _, result = run_study(spec, subjects=gauge.tiny_subjects(rng), store=store)
    adjudicated = apply_ci_overlay(spec, result)
    assert adjudicated.result.outcomes["H-canon"] == "confirmed"
    assert not adjudicated.notes


# ---------------------------------------------------------------------------
# GAUGE-XFAM: the raw cosine must be the uninformative control.
# ---------------------------------------------------------------------------


def test_gaugexfam_raw_control_is_uninformative_on_the_plant(tmp_path):
    _, result, _ = _run_tiny("GAUGE-XFAM", tmp_path)
    m = result.metrics
    assert m["spearman_canonical_angle_vs_disagreement"] > 0.6
    assert m["raw_cos_control_spearman"] < m["spearman_canonical_angle_vs_disagreement"] - 0.3
    assert m["xfam_perm_p"] < 0.05


# ---------------------------------------------------------------------------
# VERIF-PRM: the AUC orientation and the labeled-error restriction.
# ---------------------------------------------------------------------------


def test_verif_auc_orientation_flips_with_the_score_convention(tmp_path):
    rng = np.random.default_rng(task_seed("test-cards-c", "verif-flip"))
    subjects = verif.tiny_subjects(rng)
    # Negating every step score plants the opposite convention: error steps now score HIGH,
    # so the negated-score detector must land far below chance and fire the methodological kill.
    subjects["steps"]["values"] = -subjects["steps"]["values"]
    store = EvidenceStore(tmp_path / "store")
    _, result = run_study(build_spec(CARDS_BY_ID["VERIF-PRM"]), subjects=subjects, store=store)
    assert result.metrics["dense_localization_auc"] < 0.5
    assert result.killed and "K-verif" in result.killed_by


def test_verif_pooling_excludes_correct_and_malformed_items():
    values = np.asarray([1.0, -1.5, 1.0, 2.0, 2.0, 3.0, 3.0], dtype=np.float64)
    offsets = np.asarray([0, 3, 5, 7], dtype=np.int64)
    labels = np.asarray([1, -1, 9], dtype=np.int64)  # labeled, correct, out-of-range
    scores, flags, n_used, n_skipped = verif._pool_step_labels(values, offsets, labels)
    assert n_used == 1 and n_skipped == 1
    assert scores.shape == flags.shape == (3,)
    assert flags.tolist() == [0, 1, 0]


def test_verif_resolver_round_trips_all_five_intermediates(tmp_path):
    rng = np.random.default_rng(task_seed("test-cards-c", "verif-resolve"))
    tiny = verif.tiny_subjects(rng)
    store = EvidenceStore(tmp_path / "store")
    steps = tiny["steps"]
    n_items = len(steps["labels"])
    store.append(
        ek.record_intermediate(
            observable=ek.OBS_STEP_SCORES,
            roster_key="qwen-prm",
            slice_name="processbench-full",
            value=StepScores(
                item_ids=[f"pb-{i}" for i in range(n_items)],
                values=steps["values"],
                offsets=steps["offsets"],
                labels=steps["labels"],
            ),
        )
    )
    grid = tiny["span_grid"]
    store.append(
        ek.record_intermediate(
            observable=ek.OBS_PATCH_GRID,
            roster_key="qwen-prm",
            slice_name="processbench-full",
            value=PatchGridResult(
                pair_ids=[f"pair-{i}" for i in range(grid["deltas"].shape[0])],
                components=grid["components"],
                deltas=grid["deltas"],
                meta={"original_differential": [float(v) for v in grid["original_differential"]]},
            ),
        )
    )
    store.append(
        ek.record_intermediate(
            observable=ek.OBS_INTERNAL_FEATURES,
            roster_key="qwen-prm",
            slice_name="processbench-span-delta",
            value=VectorPayload(name="span-delta", vector=tiny["style_delta"]),
        )
    )
    store.append(
        ek.record_intermediate(
            observable=ek.OBS_SUBSPACE,
            roster_key="qwen-prm",
            slice_name="processbench-style-basis",
            value=MatrixPayload(name="style-basis", matrix=tiny["style_basis"]),
        )
    )
    store.append(
        ek.record_intermediate(
            observable=ek.OBS_REWARD_DIRECTION,
            roster_key="qwen-prm",
            slice_name="processbench-full",
            value=VectorPayload(name="w_r", vector=tiny["w_r"]),
        )
    )
    subjects = verif.resolve_subjects(store)
    _, result = run_study(build_spec(CARDS_BY_ID["VERIF-PRM"]), subjects=subjects, store=store)
    assert result.metrics["dense_localization_auc"] > 0.7
    assert result.metrics["style_share"] < 0.5


# ---------------------------------------------------------------------------
# JUDGE-VBC: decode arithmetic and the numeric flip rate without a swap arm.
# ---------------------------------------------------------------------------


def test_judge_decode_counts_zero_diff_as_non_match(tmp_path):
    from campaign.payloads import JudgeVerdicts

    subjects = {
        "verdicts": JudgeVerdicts(
            item_ids=["a", "b", "c", "d"],
            prefix_logit_diff=np.asarray([1.0, -1.0, 1.0, 0.0]),
            final_verdict=np.asarray([0, 1, 1, 0]),
            swapped_verdict=None,
        )
    }
    store = EvidenceStore(tmp_path / "store")
    _, result = run_study(build_spec(CARDS_BY_ID["JUDGE-VBC"]), subjects=subjects, store=store)
    assert result.metrics["verdict_prefix_match_rate"] == 0.5
    assert result.metrics["flip_rate"] == 0.0


def test_judge_flip_rate_reads_only_the_covered_swap_entries(tmp_path):
    rng = np.random.default_rng(task_seed("test-cards-c", "judge-swap"))
    subjects = judge.tiny_subjects(rng)
    v = subjects["verdicts"]
    swapped = np.full(len(v.item_ids), -1.0)
    swapped[:10] = 1 - np.asarray(v.final_verdict[:10])  # every covered item flips
    v.swapped_verdict = swapped
    store = EvidenceStore(tmp_path / "store")
    _, result = run_study(build_spec(CARDS_BY_ID["JUDGE-VBC"]), subjects=subjects, store=store)
    assert result.metrics["flip_rate"] == 1.0


def test_judge_resolver_round_trip(tmp_path):
    rng = np.random.default_rng(task_seed("test-cards-c", "judge-resolve"))
    payload = judge.tiny_subjects(rng)["verdicts"]
    store = EvidenceStore(tmp_path / "store")
    ev = ek.record_intermediate(
        observable=ek.OBS_JUDGE_VERDICTS,
        roster_key="skywork-critic",
        slice_name="judge-pairs-1000",
        value=payload,
    )
    store.append(ev)
    subjects = judge.resolve_subjects(store)
    assert subjects["verdicts__ev"] == ev.id
    _, result = run_study(build_spec(CARDS_BY_ID["JUDGE-VBC"]), subjects=subjects, store=store)
    assert result.metrics["verdict_prefix_match_rate"] > 0.9


# ---------------------------------------------------------------------------
# VALUES-CONTEST: the overlay, and the id alignment through the resolver.
# ---------------------------------------------------------------------------


def test_contest_ci_overlay_keeps_the_planted_confirmation(tmp_path):
    spec = build_spec(CARDS_BY_ID["VALUES-CONTEST"])
    rng = np.random.default_rng(task_seed("test-cards-c", "contest-overlay"))
    store = EvidenceStore(tmp_path / "store")
    _, result = run_study(spec, subjects=contest.tiny_subjects(rng), store=store)
    adjudicated = apply_ci_overlay(spec, result)
    assert adjudicated.result.outcomes["H-contest"] == "confirmed"
    assert result.metrics["contested_raterspread_spearman_ci_low"] > 0.0


def test_contest_resolver_realigns_a_shuffled_ensemble_bank(tmp_path):
    rng = np.random.default_rng(task_seed("test-cards-c", "contest-resolve"))
    tiny = contest.tiny_subjects(rng)
    store = EvidenceStore(tmp_path / "store")
    ids = list(tiny["loading_ids"])
    store.append(
        ek.record_intermediate(
            observable=ek.OBS_CONTESTED,
            roster_key="skywork-v2-llama31-8b",
            slice_name="helpsteer3-pref-2000",
            value=VectorPayload(name="loading", vector=tiny["loading"], labels=ids),
        )
    )
    store.append(
        ek.record_intermediate(
            observable=ek.OBS_RATER_SPREAD,
            roster_key=ek.DATA_KEY,
            slice_name="helpsteer3-pref-2000",
            value=VectorPayload(name="spread", vector=tiny["spread"], labels=ids),
        )
    )
    for m, model in enumerate(ENSEMBLE_5):
        bank = tiny["ensemble"][model]
        order = np.arange(len(ids))
        if m == 0:  # one bank arrives in a scrambled item order; ids must carry the join
            order = rng.permutation(order)
        store.append(
            ek.record_intermediate(
                observable=ek.OBS_SCORES,
                roster_key=model,
                slice_name="helpsteer3-pref-2000",
                value=ScoreBank(
                    item_ids=[ids[i] for i in order],
                    scores=np.asarray(bank["scores"])[order],
                    layout="flat",
                ),
            )
        )
    subjects = contest.resolve_subjects(store)
    _, result = run_study(
        build_spec(CARDS_BY_ID["VALUES-CONTEST"]), subjects=subjects, store=store
    )
    assert result.metrics["contested_raterspread_spearman"] > 0.2
    assert result.metrics["ensemble_minus_contested_spearman"] < 0.05


# ---------------------------------------------------------------------------
# FORENSIC-RECEIPT: the grid contrasts and their kill.
# ---------------------------------------------------------------------------


def test_forensic_kill_fires_when_the_skepticism_axis_is_flat(tmp_path):
    rng = np.random.default_rng(task_seed("test-cards-c", "forensic-flat"))
    graders = {}
    for g in range(6):
        # Every grader relies on receipts but none treats a caught fabrication differently
        # from silence, so the s-axis carries nothing and the disclosure-game kill must fire.
        reliance = float(rng.uniform(0.8, 1.5))
        valid = reliance + 0.5 * rng.standard_normal(150)
        absent = 0.5 * rng.standard_normal(150)
        failing = 0.5 * rng.standard_normal(150)
        graders[f"flat-{g}"] = np.stack([valid, absent, failing], axis=1)
    store = EvidenceStore(tmp_path / "store")
    _, result = run_study(
        build_spec(CARDS_BY_ID["FORENSIC-RECEIPT"]), subjects={"graders": graders}, store=store
    )
    assert result.metrics["skepticism_effect"] < 0.1
    assert result.killed and "K-forensic" in result.killed_by
    assert result.metrics["reliance_recovery"] > 0.5  # reliance itself is still there


def test_forensic_resolver_scans_triples_and_ignores_other_layouts(tmp_path):
    rng = np.random.default_rng(task_seed("test-cards-c", "forensic-resolve"))
    tiny = forensic.tiny_subjects(rng)
    store = EvidenceStore(tmp_path / "store")
    for key, scores in tiny["graders"].items():
        store.append(
            ek.record_intermediate(
                observable=ek.OBS_SCORES,
                roster_key=key,
                slice_name="receipt-triples-1600",
                value=ScoreBank(
                    item_ids=[f"t-{i}" for i in range(scores.shape[0])],
                    scores=scores,
                    layout="triple",
                ),
            )
        )
    # A flat bank on the same slice must not enter the grader set.
    store.append(
        ek.record_intermediate(
            observable=ek.OBS_SCORES,
            roster_key="not-a-grader",
            slice_name="receipt-triples-1600",
            value=ScoreBank(item_ids=["x"], scores=np.asarray([1.0]), layout="flat"),
        )
    )
    subjects = forensic.resolve_subjects(store)
    assert set(subjects["graders"]) == set(tiny["graders"])
    _, result = run_study(
        build_spec(CARDS_BY_ID["FORENSIC-RECEIPT"]), subjects=subjects, store=store
    )
    assert result.metrics["omission_credulity_gap"] > 0.2
    assert not result.killed


# ---------------------------------------------------------------------------
# EVAL-AWARE: the fixed threshold and the optional freshness arm.
# ---------------------------------------------------------------------------


def test_evalaware_runs_without_the_wildchat_arm(tmp_path):
    rng = np.random.default_rng(task_seed("test-cards-c", "evalaware-nowc"))
    subjects = evalaware.tiny_subjects(rng)
    subjects["wildchat_features"] = None
    store = EvidenceStore(tmp_path / "store")
    _, result = run_study(build_spec(CARDS_BY_ID["EVAL-AWARE"]), subjects=subjects, store=store)
    assert result.metrics["real_probe_balanced_acc"] > 0.6
    assert result.metrics["organic_fpr"] < 0.1


def test_evalaware_resolver_round_trips_and_skips_absent_wildchat(tmp_path):
    rng = np.random.default_rng(task_seed("test-cards-c", "evalaware-resolve"))
    tiny = evalaware.tiny_subjects(rng)
    store = EvidenceStore(tmp_path / "store")
    n = tiny["paired_features"].shape[0]
    store.append(
        ek.record_intermediate(
            observable=ek.OBS_EVAL_PAIRED,
            roster_key="evalaware-rms",
            slice_name="evalaware-paired",
            value=FeatureMatrix(
                item_ids=[f"p-{i}" for i in range(n)],
                names=[f"f{j}" for j in range(tiny["paired_features"].shape[1])],
                matrix=tiny["paired_features"],
                meta={"is_benchmark": [int(v) for v in tiny["paired_labels"]]},
            ),
        )
    )
    store.append(
        ek.record_intermediate(
            observable=ek.OBS_INTERNAL_FEATURES,
            roster_key="evalaware-rms",
            slice_name="arena-organic",
            value=FeatureMatrix(
                item_ids=[f"a-{i}" for i in range(tiny["arena_features"].shape[0])],
                names=[f"f{j}" for j in range(tiny["arena_features"].shape[1])],
                matrix=tiny["arena_features"],
            ),
        )
    )
    store.append(
        ek.record_intermediate(
            observable=ek.OBS_STEER_SWEEP,
            roster_key="evalaware-rms",
            slice_name="evalaware-paired",
            value=TablePayload(
                name="steer-sweep",
                columns=["strength", "delta_r"],
                rows=[
                    [float(s), float(d)]
                    for s, d in zip(tiny["steer_strength"], tiny["steer_delta_r"])
                ],
                meta={"baseline_score_sd": float(tiny["steer_baseline_sd"])},
            ),
        )
    )
    subjects = evalaware.resolve_subjects(store)
    assert subjects["wildchat_features"] is None  # no WildChat intermediate was recorded
    _, result = run_study(build_spec(CARDS_BY_ID["EVAL-AWARE"]), subjects=subjects, store=store)
    assert result.metrics["real_probe_balanced_acc"] > 0.6
    assert result.metrics["delta_r_per_steer"] > 0.05


def test_evalaware_probe_stays_at_chance_on_an_unseparable_plant():
    rng = np.random.default_rng(task_seed("test-cards-c", "evalaware-null"))
    x = rng.standard_normal((160, 8))
    y = np.concatenate([np.ones(80), np.zeros(80)]).astype(np.int64)
    acc = evalaware._cv_balanced_acc(x, y, seed=0)
    assert abs(acc - 0.5) < 0.12  # no direction to find; the probe must not manufacture one
