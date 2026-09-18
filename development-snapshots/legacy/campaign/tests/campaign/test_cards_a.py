"""Tiny-path tests for the fleet and observational card analyses.

Every card runs its planted tiny subjects through the real ``run_study`` path, so the assertion
is the full contract: the frozen spec adjudicates against the exact metric strings the registry
requires, the plant is recovered rather than merely emitted, and no kill criterion fires on a
confirming plant. The shared feature extraction gets its own shape and determinism checks, and
the resolvers with slice-name wrinkles (the two ERR families, the VCE shared-space records) are
round-tripped through a real store.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.core.store import EvidenceStore
from reward_lens.studies.runner import run_study

from campaign import capacity, confpartial, err, factkui, features, forecast, style, vce
from campaign import evidence_keys as ek
from campaign.config import task_seed
from campaign.registry import CARDS_BY_ID, required_metrics_for
from campaign.specs import build_spec

_MODULES = {
    "ERR-RB2": err,
    "FORECAST-CAT": forecast,
    "STYLE-RMB": style,
    "FACT-KUI": factkui,
    "CAPACITY-WELCH": capacity,
    "CONF-PARTIAL": confpartial,
    "ATLAS-VCE": vce,
}


def _run_tiny(card_id: str, tmp_path):
    module = _MODULES[card_id]
    rng = np.random.default_rng(task_seed("test-cards-a", card_id))
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
# Per-card planted-recovery assertions.
# ---------------------------------------------------------------------------


def test_err_internals_beat_the_capture_margin_on_both_families(tmp_path):
    _, result = _run_tiny("ERR-RB2", tmp_path)
    m = result.metrics
    for fam in ("qwen8b", "llama8b"):
        assert m[f"err_auroc_internal_{fam}"] > m[f"err_auroc_margin_{fam}"]
        assert m[f"err_auroc_delta_{fam}"] > 0.03
        assert m[f"err_auroc_delta_{fam}_ci_low"] > 0.0
        assert (
            m[f"err_auroc_delta_{fam}_ci_low"]
            <= m[f"err_auroc_delta_{fam}"]
            <= m[f"err_auroc_delta_{fam}_ci_high"]
        )
    assert m["err_auroc_delta_min_family"] == min(
        m["err_auroc_delta_qwen8b"], m["err_auroc_delta_llama8b"]
    )


def test_err_excludes_the_ties_subset_from_labels():
    rng = np.random.default_rng(task_seed("test-cards-a", "err-ties"))
    subjects = err.tiny_subjects(rng)
    bank = subjects["banks"]["qwen8b"]
    subsets = bank.meta["subsets"]
    n_ties = sum(1 for s in subsets if s == "Ties")
    assert n_ties == 12
    res = err._family_analysis("qwen8b", bank, subjects["features"]["qwen8b"])
    assert res["n_ties_excluded"] == n_ties
    assert res["n_items"] == len(bank.item_ids) - n_ties


def test_err_rank_auc_matches_roc_pr():
    from reward_lens.stats import roc_pr

    rng = np.random.default_rng(task_seed("test-cards-a", "err-auc"))
    labels = (rng.random(200) < 0.4).astype(np.int64)
    scores = labels + rng.normal(0.0, 1.0, 200)
    assert err._rank_auc(scores, labels) == pytest.approx(roc_pr(scores, labels).auc)


def test_forecast_beats_both_baselines_on_the_plant(tmp_path):
    _, result = _run_tiny("FORECAST-CAT", tmp_path)
    m = result.metrics
    assert m["forecast_mae"] < 0.06
    assert m["forecast_mae"] < m["size_baseline_mae"]
    assert m["forecast_mae"] < m["slice_baseline_mae"]
    assert m["forecast_mae_minus_size_baseline"] < 0.0
    assert m["forecast_mae_minus_size_baseline_ci_high"] < 0.0


def test_style_internal_battery_tracks_hard_ranking(tmp_path):
    _, result = _run_tiny("STYLE-RMB", tmp_path)
    m = result.metrics
    assert m["spearman_biasbattery_vs_rmbench_hard"] > 0.6
    assert m["biasbattery_rmbench_perm_p"] < 0.05
    assert m["spearman_minus_behavioral_baseline"] >= 0.0


def test_factkui_star_gap_clears_the_null_and_the_alias_holds(tmp_path):
    _, result = _run_tiny("FACT-KUI", tmp_path)
    m = result.metrics
    assert m["kui_gap_fleet_median"] > 0.2
    assert m["kui_gap_perm_p"] < 0.05
    assert m["kui_control_fleet_median"] < 0.1
    assert m["kui_gap"] == m["kui_gap_fleet_median"]


def test_capacity_welch_slack_matches_the_shipped_primitives(tmp_path):
    from reward_lens.measure.indices.coherence import (
        coherence_matrix,
        max_offdiagonal_coherence,
        welch_bound,
    )

    rng = np.random.default_rng(task_seed("test-cards-a", "capacity-slack"))
    subjects = capacity.tiny_subjects(rng)
    dirs = np.asarray(subjects["armorm_directions"].matrix)
    expected = max_offdiagonal_coherence(coherence_matrix(dirs)) - welch_bound(*dirs.shape)
    got = capacity._welch_slack(dirs)
    assert got["slack"] == pytest.approx(expected)
    # 19 criteria in 12 dimensions is the over-packed regime: the bound is a real floor.
    assert got["welch_bound"] > 0.0
    assert got["slack"] >= 0.0

    _, result = _run_tiny("CAPACITY-WELCH", tmp_path)
    assert result.metrics["dark_reward_kdeff_spearman"] > 0.6
    assert result.metrics["capacity_perm_p"] < 0.05


def test_capacity_rejects_a_precomputed_coherence_matrix():
    mu = np.eye(5)
    mu[0, 1] = mu[1, 0] = 0.3
    with pytest.raises(ValueError, match="precomputed coherence matrix"):
        capacity._welch_slack(mu)


def test_confpartial_signal_survives_and_confounds_collapse(tmp_path):
    rng = np.random.default_rng(task_seed("test-cards-a", "conf-partials"))
    subjects = confpartial.tiny_subjects(rng)
    models = subjects["models"]
    log_size = np.log(np.array([subjects["sizes"][m] for m in models]))
    rb2 = np.array([float(row[4]) for row in subjects["outcomes"].rows])
    controls = np.column_stack([log_size, rb2])

    def _index(name):
        return np.array(
            [
                dict((str(r[0]), float(r[1])) for r in subjects["index_tables"][m].rows)[name]
                for m in models
            ]
        )

    def _outcome(col):
        return np.array([float(row[col]) for row in subjects["outcomes"].rows])

    rho_chi, _, _ = confpartial._partial_correlation(
        _index("chi_teacher_variance"), _outcome(1), controls
    )
    rho_deff, _, _ = confpartial._partial_correlation(
        _index("coherence_d_eff"), _outcome(2), controls
    )
    assert abs(rho_chi) > 0.5
    assert abs(rho_deff) < abs(rho_chi)

    _, result = _run_tiny("CONF-PARTIAL", tmp_path)
    m = result.metrics
    assert m["max_abs_partial_corr"] > 0.3
    assert m["best_index_perm_p"] < 0.05
    assert m["lineage_check_pass"] == 1.0


def test_vce_convergent_fleet_clears_the_rum_null(tmp_path):
    _, result = _run_tiny("ATLAS-VCE", tmp_path)
    m = result.metrics
    assert m["real_vce"] > 0.0
    assert m["real_vce_ci_low"] > 0.0
    assert m["real_vce_ci_low"] <= m["real_vce"] <= m["real_vce_ci_high"]
    assert m["reward_convergent_p_value"] < 0.05


def test_vce_fails_loudly_on_a_nan_null():
    class _FakeResult:
        null_mean = float("nan")

    with pytest.raises(RuntimeError, match="non-finite RUM null"):
        vce._require_finite_null(_FakeResult(), "reward pair (a, b)")


# ---------------------------------------------------------------------------
# The shared feature extraction: shapes, semantics, determinism.
# ---------------------------------------------------------------------------


def _synthetic_capture(rng, n=40, d=16, n_layers=3):
    site_order = ["embed"] + [f"resid_post_L{layer}" for layer in range(n_layers)]
    chosen = {k: rng.standard_normal((n, d)).astype(np.float32) for k in site_order}
    rejected = {k: rng.standard_normal((n, d)).astype(np.float32) for k in site_order}
    w_r = rng.standard_normal(d)
    rejected_scores = rng.normal(0.0, 0.3, size=(n, 3))
    chosen_scores = rejected_scores.max(axis=1) + rng.normal(0.2, 0.5, n)
    scores = np.column_stack([chosen_scores, rejected_scores])
    return site_order, chosen, rejected, w_r, scores


def test_features_margins_and_labels_agree():
    rng = np.random.default_rng(task_seed("test-cards-a", "features-labels"))
    _, _, _, _, scores = _synthetic_capture(rng)
    margins = features.bestof4_margins(scores)
    labels = features.bestof4_wrong_labels(scores)
    np.testing.assert_array_equal(labels, (margins <= 0.0).astype(np.int8))
    top = features.top_rejected_index(scores)
    assert top.min() >= 1 and top.max() <= 3
    picked = scores[np.arange(scores.shape[0]), top]
    np.testing.assert_allclose(picked, scores[:, 1:].max(axis=1))


def test_features_differential_curves_project_onto_w_r():
    rng = np.random.default_rng(task_seed("test-cards-a", "features-curves"))
    site_order, chosen, rejected, w_r, _ = _synthetic_capture(rng)
    curves = features.reward_differential_curves(chosen, rejected, w_r, site_order)
    assert curves.shape == (40, len(site_order))
    key = site_order[-1]
    expected = (
        chosen[key].astype(np.float64) - rejected[key].astype(np.float64)
    ) @ w_r.astype(np.float64)
    np.testing.assert_allclose(curves[:, -1], expected, rtol=1e-4, atol=1e-4)


def test_features_crystallization_matches_a_handcrafted_curve():
    # Half of the final differential (1.0) is first reached at layer 1 of 3.
    curve = np.array([[0.0, 0.2, 0.6, 1.0]])
    fracs = features.crystallization_fractions(curve)
    assert fracs.shape == (1,)
    assert fracs[0] == pytest.approx(1.0 / 3.0)


def test_features_assemble_shapes_names_and_determinism():
    rng = np.random.default_rng(task_seed("test-cards-a", "features-assemble"))
    site_order, chosen, rejected, w_r, scores = _synthetic_capture(rng)
    n, d = 40, 16
    item_ids = [f"item-{i}" for i in range(n)]
    prop_names = ["star", "priced", "noise"]
    prop_dec = np.array([0.9, 0.85, 0.5])
    prop_dirs = rng.standard_normal((3, d))

    def _assemble():
        return features.assemble_internal_features(
            item_ids, scores, chosen, rejected, w_r, site_order,
            prop_names, prop_dec, prop_dirs,
        )

    fm = _assemble()
    assert fm.names == list(features.FEATURE_NAMES)
    assert np.asarray(fm.matrix).shape == (n, len(features.FEATURE_NAMES))
    assert np.all(np.isfinite(np.asarray(fm.matrix)))
    assert fm.meta["kui_star"] in prop_names
    assert fm.meta["n_layers"] == len(site_order) - 1
    again = _assemble()
    np.testing.assert_array_equal(np.asarray(fm.matrix), np.asarray(again.matrix))


def test_features_prompt_snr_is_finite_even_when_degenerate():
    scores = np.array([[1.0, 0.5, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]])
    out = features.prompt_snr(scores)
    assert out.shape == (2,)
    assert np.all(np.isfinite(out))
    assert out[1] == 0.0  # a fully degenerate item carries no signal


# ---------------------------------------------------------------------------
# Resolver round-trips through a real store.
# ---------------------------------------------------------------------------


def test_err_resolver_round_trips_both_families(tmp_path):
    rng = np.random.default_rng(task_seed("test-cards-a", "err-resolve"))
    tiny = err.tiny_subjects(rng)
    store = EvidenceStore(tmp_path / "store")
    for family, roster_key in err._FAMILY_ROSTER.items():
        store.append(
            ek.record_intermediate(
                observable=ek.OBS_SCORES,
                roster_key=roster_key,
                slice_name="rb2-full",
                value=tiny["banks"][family],
            )
        )
        store.append(
            ek.record_intermediate(
                observable=ek.OBS_INTERNAL_FEATURES,
                roster_key=roster_key,
                slice_name=err._FEATURE_SLICE,
                value=tiny["features"][family],
            )
        )
    subjects = err.resolve_subjects(store)
    assert set(subjects["banks"]) == {"qwen8b", "llama8b"}
    assert subjects["bank.qwen8b__ev"]
    _, result = run_study(build_spec(CARDS_BY_ID["ERR-RB2"]), subjects=subjects, store=store)
    assert result.metrics["err_auroc_delta_min_family"] > 0.03


def test_vce_resolver_reads_the_shared_space_records(tmp_path):
    from campaign import config

    rng = np.random.default_rng(task_seed("test-cards-a", "vce-resolve"))
    tiny = vce.tiny_subjects(rng)
    store = EvidenceStore(tmp_path / "store")
    store.append(
        ek.record_intermediate(
            observable=ek.OBS_SUBSPACE,
            roster_key=ek.DATA_KEY,
            slice_name="rb2-helpfulness-512-frame",
            value=vce.MatrixPayload(name="frame reference", matrix=tiny["reference"]),
        )
    )
    tiny_models = tiny["models"]
    for m, fleet_key in zip(tiny_models, config.FLEET):
        store.append(
            ek.record_intermediate(
                observable=ek.OBS_SUBSPACE,
                roster_key=fleet_key,
                slice_name="rb2-helpfulness-512-reward",
                value=tiny["reward_bases"][m],
            )
        )
        store.append(
            ek.record_intermediate(
                observable=ek.OBS_SUBSPACE,
                roster_key=fleet_key,
                slice_name="rb2-helpfulness-512-capability",
                value=tiny["capability_bases"][m],
            )
        )
    with pytest.raises(KeyError, match="rb2-helpfulness-512-reward"):
        # Only six of the ten fleet seats were planted; the resolver names what is missing.
        vce.resolve_subjects(store)


def test_factkui_resolver_raises_precisely_when_a_table_is_missing(tmp_path):
    store = EvidenceStore(tmp_path / "store")
    with pytest.raises(KeyError, match="rb2-full-kui"):
        factkui.resolve_subjects(store)
