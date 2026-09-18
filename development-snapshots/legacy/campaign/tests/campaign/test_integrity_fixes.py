"""Tests pinning the scientific-integrity fixes across the analysis stack.

Each block pins one repaired operationalization: the midrank Spearman against scipy on tied
fixtures, the adjudication overlay's non-finite-CI and FDR paths, the capture-only internal
feature builder, the grader-level forensic recovery statistic, the within-item verification
pooling, the dimensionless ladder margin, the standardized eval-aware slope, the frozen-
threshold power pricing, and the null simulators' congruence with the statistics the cards
actually emit.
"""

from __future__ import annotations

import copy
import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest

from reward_lens.core.store import EvidenceStore
from reward_lens.stats import roc_pr
from reward_lens.studies.runner import run_study
from reward_lens.studies.spec import Hypothesis, Prediction, StudyResult, StudySpec

from campaign import chidrift, emb, evalaware, features, forensic, ladder, nulls, power, verif
from campaign.adjudicate import apply_ci_overlay, apply_fdr_overlay
from campaign.config import task_seed
from campaign.registry import CARDS_BY_ID
from campaign.specs import build_spec
from campaign.stats_ext import _spearman, corr_permutation_p


# ---------------------------------------------------------------------------
# Midrank Spearman
# ---------------------------------------------------------------------------


def test_spearman_matches_scipy_on_tied_fixtures():
    from scipy.stats import spearmanr

    rng = np.random.default_rng(task_seed("test-integrity", "spearman-ties"))
    for _ in range(10):
        x = rng.integers(0, 4, size=25).astype(np.float64)
        y = rng.integers(0, 4, size=25).astype(np.float64)
        ref = spearmanr(x, y).statistic
        if np.isnan(ref):
            assert np.isnan(_spearman(x, y))
        else:
            assert _spearman(x, y) == pytest.approx(ref, abs=1e-12)


def test_spearman_constant_vector_is_nan_not_perfect():
    rng = np.random.default_rng(task_seed("test-integrity", "spearman-const"))
    const = np.full(12, 3.7)
    other = rng.standard_normal(12)
    assert np.isnan(_spearman(const, other))
    assert np.isnan(_spearman(other, const))
    assert np.isnan(_spearman(const, const))


def test_spearman_copies_are_the_shared_implementation():
    assert chidrift._rank_corr is _spearman
    assert nulls._spearman is _spearman


def test_corr_permutation_p_is_not_fooled_by_a_constant_side():
    rng = np.random.default_rng(task_seed("test-integrity", "perm-const"))
    p = corr_permutation_p(np.full(20, 1.0), rng.standard_normal(20), n_permutations=200)
    assert np.isnan(p)


# ---------------------------------------------------------------------------
# Adjudication overlays
# ---------------------------------------------------------------------------


def _ci_spec() -> StudySpec:
    return StudySpec(
        id="t-ci", title="t", science="S00-meta",
        hypotheses=(
            Hypothesis(
                id="h-ci", statement="",
                prediction=Prediction("m", ">", 0.0, ci_excludes=0.0),
            ),
        ),
        analysis="x.y",
    )


def test_ci_overlay_downgrades_nan_bounds_to_inconclusive():
    spec = _ci_spec()
    result = StudyResult(
        outcomes={"h-ci": "confirmed"},
        metrics={"m": 1.0, "m_ci_low": float("nan"), "m_ci_high": 2.0},
    )
    adjudicated = apply_ci_overlay(spec, result)
    assert adjudicated.result.outcomes["h-ci"] == "inconclusive"
    assert [n.action for n in adjudicated.notes] == ["inconclusive-nan-ci"]


def test_ci_overlay_still_downgrades_a_covering_finite_interval():
    spec = _ci_spec()
    result = StudyResult(
        outcomes={"h-ci": "confirmed"},
        metrics={"m": 1.0, "m_ci_low": -0.5, "m_ci_high": 2.0},
    )
    adjudicated = apply_ci_overlay(spec, result)
    assert adjudicated.result.outcomes["h-ci"] == "refuted"
    assert [n.action for n in adjudicated.notes] == ["downgraded-ci"]


def _fdr_spec() -> StudySpec:
    return StudySpec(
        id="t-fdr", title="t", science="S00-meta",
        hypotheses=(
            Hypothesis(id="h-a", statement="", prediction=Prediction("a_perm_p", "<", 0.05)),
            Hypothesis(id="h-b", statement="", prediction=Prediction("b_perm_p", "<", 0.05)),
            Hypothesis(id="h-c", statement="", prediction=Prediction("c_metric", ">", 0.0)),
        ),
        analysis="x.y",
    )


def test_fdr_overlay_downgrades_a_confirmed_p_that_fails_bh():
    spec = _fdr_spec()
    # h-a squeaks under alpha raw but its BH q within the two-test family is 0.08.
    result = StudyResult(
        outcomes={"h-a": "confirmed", "h-b": "refuted", "h-c": "confirmed"},
        metrics={"a_perm_p": 0.04, "b_perm_p": 0.90, "c_metric": 1.0},
    )
    adjudicated = apply_fdr_overlay(spec, result)
    assert adjudicated.result.outcomes["h-a"] == "refuted"
    assert [n.action for n in adjudicated.notes] == ["downgraded-fdr"]
    # Never upgrade, and never touch non-p hypotheses.
    assert adjudicated.result.outcomes["h-b"] == "refuted"
    assert adjudicated.result.outcomes["h-c"] == "confirmed"


def test_fdr_overlay_keeps_a_family_that_survives_bh():
    spec = _fdr_spec()
    result = StudyResult(
        outcomes={"h-a": "confirmed", "h-b": "confirmed", "h-c": "confirmed"},
        metrics={"a_perm_p": 0.001, "b_perm_p": 0.02, "c_metric": 1.0},
    )
    adjudicated = apply_fdr_overlay(spec, result)
    assert not adjudicated.notes
    assert adjudicated.result.outcomes["h-a"] == "confirmed"
    assert adjudicated.result.outcomes["h-b"] == "confirmed"


# ---------------------------------------------------------------------------
# Capture-only internal features
# ---------------------------------------------------------------------------


def test_assemble_signature_names_lens_scores():
    params = inspect.signature(features.assemble_internal_features).parameters
    assert "lens_scores" in params
    assert "scores" not in params


def _fake_capture(rng, n_items: int, k: int, d: int):
    import torch

    sites = ["a_emb", "b_resid0", "c_resid1"]
    tensors = {
        s: torch.from_numpy(rng.standard_normal((n_items * k, d)).astype(np.float32))
        for s in sites
    }
    return SimpleNamespace(tensors=tensors), sites


def test_internal_features_derive_from_the_capture_and_ignore_native_scores():
    rng = np.random.default_rng(task_seed("test-integrity", "features-capture"))
    n, k, d = 30, 4, 8
    capture, sites = _fake_capture(rng, n, k, d)
    w_r = rng.standard_normal(d)
    kwargs = dict(
        sites=sites,
        item_ids=[f"i-{j}" for j in range(n)],
        roster_key="t",
        slice_name="t",
        w_r=w_r,
        kui_names=["p0", "p1", "p2"],
        kui_decodability=np.array([0.9, 0.7, 0.5]),
        kui_directions=rng.standard_normal((3, d)),
    )
    fm_derived = features.internal_features(capture, **kwargs)
    # A stale native bank under the removed keyword changes nothing: the builder derives the
    # lens scores from the capture and the native array never reaches a column.
    native = rng.standard_normal((n, k)) * 100.0
    fm_stale = features.internal_features(capture, scores=native, **kwargs)
    np.testing.assert_array_equal(
        np.asarray(fm_derived.matrix), np.asarray(fm_stale.matrix)
    )
    # The derived lens matrix is the readout-site activations projected on w_r, and the SNR
    # column is computed from it.
    acts_last = capture.tensors[sites[-1]].numpy().astype(np.float64)
    lens = (acts_last @ w_r).reshape(n, k)
    fm_explicit = features.internal_features(capture, lens_scores=lens, **kwargs)
    np.testing.assert_allclose(
        np.asarray(fm_derived.matrix), np.asarray(fm_explicit.matrix), rtol=1e-6, atol=1e-6
    )
    snr_col = list(features.FEATURE_NAMES).index("prompt_snr")
    np.testing.assert_allclose(
        np.asarray(fm_explicit.matrix)[:, snr_col], features.prompt_snr(lens),
        rtol=1e-6, atol=1e-6,
    )


# ---------------------------------------------------------------------------
# FORENSIC: grader-level reliance recovery
# ---------------------------------------------------------------------------


def test_forensic_reliance_recovery_is_the_grader_sign_fraction(tmp_path):
    rng = np.random.default_rng(task_seed("test-integrity", "forensic-signs"))
    n = 120
    graders = {}
    for g, mu in enumerate([1.2, 0.9, 1.5, -1.0]):  # three rely, one anti-relies
        valid = mu + 0.3 * rng.standard_normal(n)
        absent = 0.3 * rng.standard_normal(n)
        failing = 0.3 * rng.standard_normal(n)
        graders[f"g-{g}"] = np.stack([valid, absent, failing], axis=1)
    store = EvidenceStore(tmp_path / "store")
    _, result = run_study(
        build_spec(CARDS_BY_ID["FORENSIC-RECEIPT"]), subjects={"graders": graders}, store=store
    )
    assert result.metrics["reliance_recovery"] == pytest.approx(0.75)


def test_forensic_null_simulator_matches_the_binomial_sign_law():
    res = nulls.simulate("reliance_recovery", n_sims=800)
    # The implemented statistic lives on the k / n_graders grid and centers at one half.
    counts = res.samples * nulls.N_GRADERS
    np.testing.assert_allclose(counts, np.round(counts))
    assert abs(float(res.samples.mean()) - 0.5) < 0.03
    assert "null_pass_rate_at_threshold" in res.extra


# ---------------------------------------------------------------------------
# VERIF: within-item standardization
# ---------------------------------------------------------------------------


def _level_confound_items():
    """Labeled items with zero within-item signal but strong between-item levels.

    Ten short low-scoring items and ten long high-scoring items, every step constant within
    its item. Raw pooling ranks the low items' steps on top after negation and reads their
    denser error labels as localization; the within-item z-scores are all zero and carry
    nothing.
    """
    values: list[float] = []
    offsets = [0]
    labels: list[int] = []
    for _ in range(10):
        values.extend([-3.0, -3.0])
        offsets.append(len(values))
        labels.append(0)
    for _ in range(10):
        values.extend([3.0] * 20)
        offsets.append(len(values))
        labels.append(0)
    return (
        np.asarray(values, dtype=np.float64),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
    )


def test_verif_zscoring_removes_the_between_item_level_confound():
    values, offsets, labels = _level_confound_items()
    pooled, flags, n_used, _ = verif._pool_step_labels(values, offsets, labels)
    assert n_used == 20
    assert float(roc_pr(-pooled, flags).auc) == pytest.approx(0.5)
    # The raw pool the fix replaces read the level split as localization.
    raw = np.concatenate(
        [values[offsets[i] : offsets[i + 1]] for i in range(len(labels))]
    )
    assert float(roc_pr(-raw, flags).auc) > 0.7


def test_verif_zscoring_keeps_genuine_within_item_signal_despite_level_spread():
    rng = np.random.default_rng(task_seed("test-integrity", "verif-signal"))
    values: list[float] = []
    offsets = [0]
    labels: list[int] = []
    for i in range(20):
        level = float(rng.uniform(-5.0, 5.0))
        step = level + 1.0 + 0.2 * rng.standard_normal(6)
        err = int(rng.integers(0, 6))
        step[err] = level - 2.0
        values.extend(step.tolist())
        offsets.append(len(values))
        labels.append(err)
    pooled, flags, _, _ = verif._pool_step_labels(
        np.asarray(values), np.asarray(offsets, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
    )
    assert float(roc_pr(-pooled, flags).auc) > 0.9


def test_verif_null_simulator_is_within_item():
    res = nulls.simulate("dense_localization_auc", n_sims=30)
    assert abs(float(res.samples.mean()) - 0.5) < 0.01
    assert "within-item" in res.note


# ---------------------------------------------------------------------------
# LADDER: dimensionless flat-baseline margin
# ---------------------------------------------------------------------------


def test_ladder_margin_is_scale_invariant_per_quantity(tmp_path):
    seed = task_seed("test-integrity", "ladder-scale")
    subjects = ladder.tiny_subjects(np.random.default_rng(seed))
    scaled = copy.deepcopy(subjects)
    q = "chi_teacher_variance"
    for key in ("lo", "hi", "point", "flat_baseline"):
        scaled["intervals"][q][key] *= 1000.0
    scaled["values_8b"][q] *= 1000.0
    scaled["rung_values"][q] = [v * 1000.0 for v in scaled["rung_values"][q]]
    spec = build_spec(CARDS_BY_ID["LADDER"])
    _, base = run_study(spec, subjects=subjects, store=EvidenceStore(tmp_path / "a"))
    _, big = run_study(spec, subjects=scaled, store=EvidenceStore(tmp_path / "b"))
    for m in (
        "ladder_extrap_minus_flat_error",
        "ladder_extrap_minus_flat_error_ci_low",
        "ladder_extrap_minus_flat_error_ci_high",
    ):
        assert big.metrics[m] == pytest.approx(base.metrics[m], rel=1e-9)
    assert base.metrics["ladder_extrap_minus_flat_error"] < 0.0


# ---------------------------------------------------------------------------
# EVAL-AWARE: standardized steer slope
# ---------------------------------------------------------------------------


def _evalaware_result(tmp_path, name, subjects):
    store = EvidenceStore(tmp_path / name)
    _, result = run_study(build_spec(CARDS_BY_ID["EVAL-AWARE"]), subjects=subjects, store=store)
    hits = [ev for ev in store if ev.observable == "campaign.result.EVAL-AWARE"]
    return result, hits[-1].value.meta


def test_evalaware_slope_uses_the_baseline_sd_from_meta(tmp_path):
    rng = np.random.default_rng(task_seed("test-integrity", "evalaware-sd"))
    subjects = evalaware.tiny_subjects(rng)
    raw_slope = float(np.polyfit(subjects["steer_strength"], subjects["steer_delta_r"], 1)[0])
    with_sd = dict(subjects)
    with_sd["steer_baseline_sd"] = 2.0
    result, meta = _evalaware_result(tmp_path, "with-sd", with_sd)
    assert result.metrics["delta_r_per_steer"] == pytest.approx(raw_slope / 2.0)
    assert "baseline_score_sd" in meta["slope_standardization"]
    assert meta["raw_slope_reward_units"] == pytest.approx(raw_slope)


def test_evalaware_slope_is_undefined_without_the_baseline_sd(tmp_path):
    # There is deliberately NO own-SD fallback: dividing a five-point sweep by its own
    # spread returns roughly plus or minus 0.71 for any monotone drift regardless of
    # magnitude, which silently replaces the frozen threshold with a sign test. A sweep
    # without the baseline scale yields NaN and the hypothesis cannot confirm.
    rng = np.random.default_rng(task_seed("test-integrity", "evalaware-fallback"))
    subjects = dict(evalaware.tiny_subjects(rng))
    subjects.pop("steer_baseline_sd", None)
    result, meta = _evalaware_result(tmp_path, "no-baseline", subjects)
    assert np.isnan(result.metrics["delta_r_per_steer"])
    assert "unavailable" in meta["slope_standardization"]
    assert result.outcomes.get("H-aware-steer") != "confirmed"


# ---------------------------------------------------------------------------
# Power pricing and null congruence
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_bank():
    return nulls.simulate_all(n_sims=150)


def test_power_prices_emb_at_the_frozen_threshold(small_bank):
    verdicts = {
        v.card: v for v in power.power_gate(
            [CARDS_BY_ID[c] for c in ("EMB-LORA", "FORENSIC-RECEIPT", "HACK-FORE", "FACT-KUI")],
            None, nulls_results=small_bank,
        )
    }
    emb_v = verdicts["EMB-LORA"]
    assert not emb_v.passes
    assert emb_v.power == pytest.approx(
        small_bank["real_bias_before_quality"].extra["null_pass_rate_at_threshold"]
    )
    assert "null pass rate" in emb_v.note

    forensic_v = verdicts["FORENSIC-RECEIPT"]
    assert not forensic_v.passes
    assert 0.0 <= forensic_v.power <= 0.6
    assert 0.5 < forensic_v.mde <= 1.0

    hack_v = verdicts["HACK-FORE"]
    assert hack_v.passes
    assert "CI exclusion" in hack_v.note

    kui_v = verdicts["FACT-KUI"]
    assert "property-random" in kui_v.note


def test_null_bank_carries_the_hackfore_statistic(small_bank):
    assert "flag_hit_rate_minus_behavioral" in nulls.SIMULATORS
    res = small_bank["flag_hit_rate_minus_behavioral"]
    counts = res.samples * nulls.N_FLEET
    np.testing.assert_allclose(counts, np.round(counts))
    assert abs(float(res.samples.mean())) < 0.1


def test_kui_null_lives_on_the_percentile_diagonal_scale(small_bank):
    res = small_bank["kui_gap_fleet_median"]
    assert float(np.abs(res.samples).max()) <= 1.0 / math.sqrt(2.0) + 1e-9
    assert abs(float(res.samples.mean())) < 0.1
    assert "kui_from_properties" in res.note


def test_omission_null_prices_the_grid_selection_effect(small_bank):
    # Splitting on the same dial the omission rate is built from biases the null gap above
    # zero; a null that misses this would center at zero.
    res = small_bank["omission_credulity_gap"]
    assert float(res.samples.mean()) > 0.0
    assert float(np.quantile(res.samples, 0.95)) < 0.2  # the frozen threshold still clears


# ---------------------------------------------------------------------------
# EMB disclosure
# ---------------------------------------------------------------------------


def test_emb_detail_discloses_the_directional_threshold(tmp_path):
    rng = np.random.default_rng(task_seed("test-integrity", "emb-detail"))
    store = EvidenceStore(tmp_path / "store")
    _, result = run_study(
        build_spec(CARDS_BY_ID["EMB-LORA"]), subjects=emb.tiny_subjects(rng), store=store
    )
    hits = [ev for ev in store if ev.observable == "campaign.result.EMB-LORA"]
    note = hits[-1].value.meta["threshold_null_note"]
    assert "0.5" in note and "magnitude" in note and "6.0" in note
    # The disclosure is detail only; the metric set is unchanged.
    assert set(result.metrics) == {
        "real_bias_before_quality", "w_r_stabilization_monotone", "order_recovery_auc"
    }
