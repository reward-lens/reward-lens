"""Tiny-path tests for the organism, intervention, and stretch card analyses.

Every card module here runs its planted tiny subjects through the real ``run_study`` path, so
what is asserted is the full contract: the frozen spec adjudicates against the exact metric
strings the registry requires, the plant is recovered (not merely emitted), no kill criterion
fires on a confirming plant, and the resolver round-trips for the cards whose store reads carry
a wrinkle (the stamped pre/post pair, the meta-borne split halves, the alias roster key).
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.core.store import EvidenceStore
from reward_lens.studies.runner import run_study

from campaign import adjavp, caltransfer, decomp, emb, field, hackfore, surgery
from campaign import evidence_keys as ek
from campaign.config import task_seed
from campaign.registry import CARDS_BY_ID, required_metrics_for
from campaign.specs import build_spec

_MODULES = {
    "CAL-TRANSFER": caltransfer,
    "ADJ-AVP": adjavp,
    "EMB-LORA": emb,
    "HACK-FORE": hackfore,
    "SURGERY": surgery,
    "T3-DECOMP": decomp,
    "T3-FIELD": field,
}


def _run_tiny(card_id: str, tmp_path):
    module = _MODULES[card_id]
    rng = np.random.default_rng(task_seed("test-cards-d", card_id))
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


def test_caltransfer_recovers_planted_equivalence(tmp_path):
    _, result = _run_tiny("CAL-TRANSFER", tmp_path)
    assert result.metrics["max_abs_auc_difference"] < 0.1
    assert (
        result.metrics["per_instrument_auc_differences"]
        <= result.metrics["max_abs_auc_difference"]
    )


def test_adjavp_patching_beats_attribution_on_the_plant(tmp_path):
    _, result = _run_tiny("ADJ-AVP", tmp_path)
    m = result.metrics
    assert m["patching_recovery_auc"] > 0.95
    assert m["attribution_recovery_auc"] < m["patching_recovery_auc"]
    assert m["recovery_gap"] > 0.0
    assert m["recovery_gap_ci_low"] > 0.0
    assert m["recovery_gap_ci_low"] <= m["recovery_gap"] <= m["recovery_gap_ci_high"]
    assert -1.0 <= m["real8b_avp_spearman"] <= 1.0


def test_emb_recovers_planted_bias_first_order(tmp_path):
    _, result = _run_tiny("EMB-LORA", tmp_path)
    m = result.metrics
    assert m["real_bias_before_quality"] > 2.0  # the plant separates the groups widely
    assert m["w_r_stabilization_monotone"] == 1.0
    assert m["order_recovery_auc"] > 0.8


def test_emb_entry_rule_matches_the_planted_onsets():
    rng = np.random.default_rng(task_seed("test-cards-d", "emb-entry-rule"))
    subjects = emb.tiny_subjects(rng)
    loadings = subjects["loadings"]
    entries = emb._entry_positions(np.asarray(loadings.matrix))
    onsets = np.asarray(loadings.meta["planted_onsets"], dtype=np.float64)
    # The recovered entry order is exactly the planted onset order.
    assert list(np.argsort(entries)) == list(np.argsort(onsets))


def test_emb_censors_a_feature_that_never_forms():
    steps = np.linspace(0.0, 1.0, 8)
    formed = np.tile(steps.reshape(-1, 1), (1, 1))
    never = np.full((8, 1), 0.05)
    entries = emb._entry_positions(np.hstack([formed, never]))
    assert entries[0] < entries[1]
    assert entries[1] == 8.0  # censored at the last checkpoint position


def test_hackfore_flags_the_planted_family_above_the_baseline(tmp_path):
    _, result = _run_tiny("HACK-FORE", tmp_path)
    m = result.metrics
    assert m["flag_hit_rate"] == 1.0
    assert m["behavioral_baseline_hit_rate"] < m["flag_hit_rate"]
    assert m["flag_hit_rate_minus_behavioral"] > 0.0
    assert m["flag_hit_rate_minus_behavioral_ci_low"] > 0.0


def test_surgery_recovers_the_planted_erasure(tmp_path):
    _, result = _run_tiny("SURGERY", tmp_path)
    m = result.metrics
    assert m["exploit_drift_reduction"] > 0.8
    assert m["rb2_accuracy_delta_after_erasure"] > -0.01
    assert m["erasure_certificate_pass"] == 1.0
    assert 0.4 <= m["probe_recovery_auc"] <= 0.6


def test_surgery_zero_pre_delta_is_zero_reduction_not_nan(tmp_path):
    rng = np.random.default_rng(task_seed("test-cards-d", "surgery-degenerate"))
    subjects = surgery.tiny_subjects(rng)
    flat = np.zeros_like(np.asarray(subjects["pre_exploit"].scores))
    subjects["pre_exploit"].scores = flat
    subjects["post_exploit"].scores = flat
    store = EvidenceStore(tmp_path / "store")
    _, result = run_study(
        build_spec(CARDS_BY_ID["SURGERY"]), subjects=subjects, store=store
    )
    assert result.metrics["exploit_drift_reduction"] == 0.0
    assert np.isfinite(result.metrics["exploit_drift_reduction"])


def test_decomp_finds_the_planted_tacit_residual(tmp_path):
    _, result = _run_tiny("T3-DECOMP", tmp_path)
    assert 0.1 < result.metrics["real_tacit_fraction"] < 0.9


def test_field_overlap_beats_the_random_baseline(tmp_path):
    _, result = _run_tiny("T3-FIELD", tmp_path)
    # The plant puts ~95% of the direction's energy in a k/d = 0.25 subspace.
    assert result.metrics["flat_hack_overlap_minus_random"] > 0.5


# ---------------------------------------------------------------------------
# Resolver round-trips for the store reads with a wrinkle.
# ---------------------------------------------------------------------------


def test_caltransfer_resolver_round_trip(tmp_path):
    rng = np.random.default_rng(task_seed("test-cards-d", "caltransfer-resolve"))
    payload = caltransfer.tiny_subjects(rng)["scorecard"]
    store = EvidenceStore(tmp_path / "store")
    ev = ek.record_intermediate(
        observable=ek.OBS_SCORECARD,
        roster_key="skywork-v2-qwen3-0.6b",
        slice_name="caltransfer-organisms",
        value=payload,
    )
    store.append(ev)
    subjects = caltransfer.resolve_subjects(store)
    assert subjects["scorecard__ev"] == ev.id
    np.testing.assert_allclose(subjects["scorecard"].cpu_aucs, payload.cpu_aucs)

    _, result = run_study(
        build_spec(CARDS_BY_ID["CAL-TRANSFER"]), subjects=subjects, store=store
    )
    assert result.metrics["max_abs_auc_difference"] < 0.15


def test_surgery_resolver_reads_through_the_intervention_stamp(tmp_path):
    rng = np.random.default_rng(task_seed("test-cards-d", "surgery-resolve"))
    tiny = surgery.tiny_subjects(rng)
    store = EvidenceStore(tmp_path / "store")
    for key, slice_name, stamp in (
        ("pre_exploit", "hack-probes", "pre-erasure"),
        ("post_exploit", "hack-probes", "post-erasure"),
        ("pre_rb2", "rb2-full", "pre-erasure"),
        ("post_rb2", "rb2-full", "post-erasure"),
    ):
        store.append(
            ek.record_intermediate(
                observable=ek.OBS_SCORES,
                roster_key="hackfore-flagged",
                slice_name=slice_name,
                value=tiny[key],
                extra={"intervention": stamp},
            )
        )
    store.append(
        ek.record_intermediate(
            observable=ek.OBS_ERASURE,
            roster_key="hackfore-flagged",
            slice_name="hack-probes",
            value=tiny["certificate"],
        )
    )
    subjects = surgery.resolve_subjects(store)
    # The stamped read must keep pre and post apart even though they share all join keys.
    pre = np.asarray(subjects["pre_exploit"].scores)
    post = np.asarray(subjects["post_exploit"].scores)
    assert np.mean(pre[:, 0] - pre[:, 1]) > np.mean(post[:, 0] - post[:, 1])

    _, result = run_study(build_spec(CARDS_BY_ID["SURGERY"]), subjects=subjects, store=store)
    assert result.metrics["exploit_drift_reduction"] > 0.8


def test_surgery_resolver_raises_precisely_when_a_stamp_is_missing(tmp_path):
    rng = np.random.default_rng(task_seed("test-cards-d", "surgery-missing"))
    tiny = surgery.tiny_subjects(rng)
    store = EvidenceStore(tmp_path / "store")
    store.append(
        ek.record_intermediate(
            observable=ek.OBS_SCORES,
            roster_key="hackfore-flagged",
            slice_name="hack-probes",
            value=tiny["pre_exploit"],
            extra={"intervention": "pre-erasure"},
        )
    )
    with pytest.raises(KeyError, match="post-erasure"):
        surgery.resolve_subjects(store)


def test_emb_resolver_round_trips_the_meta_split_halves(tmp_path):
    rng = np.random.default_rng(task_seed("test-cards-d", "emb-resolve"))
    tiny = emb.tiny_subjects(rng)
    store = EvidenceStore(tmp_path / "store")
    store.append(
        ek.record_intermediate(
            observable=ek.OBS_CHECKPOINTS,
            roster_key="skywork-v2-qwen3-0.6b",
            slice_name="ultrafeedback-train-20k",
            value=tiny["manifest"],
        )
    )
    store.append(
        ek.record_intermediate(
            observable=ek.OBS_CHECKPOINT_LOADINGS,
            roster_key="skywork-v2-qwen3-0.6b",
            slice_name="ultrafeedback-train-20k",
            value=tiny["loadings"],
        )
    )
    subjects = emb.resolve_subjects(store)
    odd = np.asarray(subjects["loadings"].meta["loadings_odd"])
    assert odd.shape == np.asarray(tiny["loadings"].matrix).shape

    _, result = run_study(build_spec(CARDS_BY_ID["EMB-LORA"]), subjects=subjects, store=store)
    assert result.metrics["order_recovery_auc"] > 0.8


def test_adjavp_resolver_reads_the_module_local_dla_observable(tmp_path):
    rng = np.random.default_rng(task_seed("test-cards-d", "adjavp-resolve"))
    tiny = adjavp.tiny_subjects(rng)
    store = EvidenceStore(tmp_path / "store")
    for key, obs, roster, slice_name in (
        ("organism_grid", ek.OBS_PATCH_GRID, "skywork-v2-qwen3-0.6b", "adjavp-organism"),
        ("organism_dla", adjavp.OBS_DLA_RANKING, "skywork-v2-qwen3-0.6b", "adjavp-organism"),
        ("real_grid", ek.OBS_PATCH_GRID, "skywork-v02", "rb2-patch-40"),
        ("real_dla", adjavp.OBS_DLA_RANKING, "skywork-v02", "rb2-patch-40"),
    ):
        store.append(
            ek.record_intermediate(
                observable=obs, roster_key=roster, slice_name=slice_name, value=tiny[key]
            )
        )
    subjects = adjavp.resolve_subjects(store)
    # The two grids share an observable; roster key and slice must keep them apart.
    assert subjects["organism_grid"].pair_ids != subjects["real_grid"].pair_ids
    assert "planted_component_labels" in subjects["organism_grid"].meta

    _, result = run_study(build_spec(CARDS_BY_ID["ADJ-AVP"]), subjects=subjects, store=store)
    assert result.metrics["recovery_gap"] > 0.0


def test_hackfore_resolver_walks_the_whole_fleet(tmp_path):
    from campaign.config import FLEET

    rng = np.random.default_rng(task_seed("test-cards-d", "hackfore-resolve"))
    store = EvidenceStore(tmp_path / "store")
    for i, m in enumerate(FLEET):
        table, bank = hackfore._tiny_model(rng, target=i % 4, behavioral_hits=i % 3 == 0)
        store.append(
            ek.record_intermediate(
                observable=ek.OBS_INDEX_TABLE,
                roster_key=m,
                slice_name="diagnostic-v3",
                value=table,
            )
        )
        store.append(
            ek.record_intermediate(
                observable=ek.OBS_SCORES, roster_key=m, slice_name="hack-probes", value=bank
            )
        )
    subjects = hackfore.resolve_subjects(store)
    assert list(subjects["models"]) == list(FLEET)
    assert set(subjects["index_tables"]) == set(FLEET)

    _, result = run_study(build_spec(CARDS_BY_ID["HACK-FORE"]), subjects=subjects, store=store)
    assert result.metrics["flag_hit_rate"] == 1.0


def test_decomp_resolver_joins_model_scores_with_data_plane_predicates(tmp_path):
    rng = np.random.default_rng(task_seed("test-cards-d", "decomp-resolve"))
    tiny = decomp.tiny_subjects(rng)
    store = EvidenceStore(tmp_path / "store")
    store.append(
        ek.record_intermediate(
            observable=ek.OBS_SCORES,
            roster_key="skywork-v2-llama31-8b",
            slice_name="rb2-full",
            value=tiny["scores"],
        )
    )
    # The predicate matrix is model-free, so the arc records it under the data roster key.
    store.append(
        ek.record_intermediate(
            observable=ek.OBS_SURFACE_FEATURES,
            roster_key=ek.DATA_KEY,
            slice_name="rb2-full",
            value=tiny["predicates"],
        )
    )
    subjects = decomp.resolve_subjects(store)
    assert subjects["predicates__ev"] != subjects["scores__ev"]

    _, result = run_study(build_spec(CARDS_BY_ID["T3-DECOMP"]), subjects=subjects, store=store)
    assert result.metrics["real_tacit_fraction"] > 0.1


def test_field_resolver_reads_the_module_local_observables(tmp_path):
    rng = np.random.default_rng(task_seed("test-cards-d", "field-resolve"))
    tiny = field.tiny_subjects(rng)
    store = EvidenceStore(tmp_path / "store")
    store.append(
        ek.record_intermediate(
            observable=field.OBS_FLAT_BASIS,
            roster_key="skywork-v2-llama31-8b",
            slice_name="rb2-full",
            value=tiny["flat_basis"],
        )
    )
    store.append(
        ek.record_intermediate(
            observable=field.OBS_FLAGGED_DIRECTION,
            roster_key="skywork-v2-llama31-8b",
            slice_name="hack-probes",
            value=tiny["flagged"],
        )
    )
    subjects = field.resolve_subjects(store)
    assert subjects["flat_basis"].matrix.shape == (32, 8)

    _, result = run_study(build_spec(CARDS_BY_ID["T3-FIELD"]), subjects=subjects, store=store)
    assert result.metrics["flat_hack_overlap_minus_random"] > 0.5
