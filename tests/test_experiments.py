"""Tests for the two decisive experiments.

The point of these is that a reader can tell the machinery is right *independently of what the real
data says*. Every estimator is exercised against a case whose answer is known in closed form: a
planted ``C`` and ``S`` where the reordering is constructed by hand, a planted linear model where
``S = C beta`` holds by construction, and a planted run where the variance derivative peaks at a
step chosen in advance.

Nothing here touches the campaign store. The two experiments' store readers are tested against
synthetic row records instead, so the suite stays offline and the assertions stay about logic.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.c1_kill_test import (
    RankComparison,
    basis_r_squared,
    compare_rankings,
    effective_dimension,
    population_covariance,
    ridge_sweep,
    selection_gradient,
    shrinkage_that_removes_sign_flips,
    spearman,
)
from experiments.i5_variance_derivative import (
    arl0,
    cadence_resolution_in_widths,
    classify_rows,
    cusum_threshold,
    derivative,
    envelope_field_names,
    fit_transition,
    lead_time_in_widths,
    peak_index,
    planted_run,
    score_detectors,
)

# ---------------------------------------------------------------------------
# C1: the selection gradient
# ---------------------------------------------------------------------------


def test_selection_gradient_is_the_plain_solve_when_unregularized():
    C = np.array([[2.0, 0.3], [0.3, 1.0]])
    S = np.array([1.0, -0.5])
    assert np.allclose(selection_gradient(C, S), np.linalg.solve(C, S), rtol=0, atol=0)


def test_planted_suppressor_flips_a_sign_and_reorders():
    """The textbook suppressor: a trait with zero marginal association and a large direct effect.

    ``f0`` has ``S_0 = 0`` exactly, so ``chi`` ranks it as carrying no reward at all. Because it
    correlates 0.8 with ``f1``, which selection does act on, its *direct* coefficient is large and
    negative. Closed form: with ``C = [[1, .8], [.8, 1]]`` and ``S = (0, 0.5)``,
    ``beta = (1/0.36) (0 - 0.4, 0.5) = (-10/9, 25/18)``.
    """
    C = np.array([[1.0, 0.8], [0.8, 1.0]])
    S = np.array([0.0, 0.5])
    beta = selection_gradient(C, S)
    assert np.allclose(beta, [-10.0 / 9.0, 25.0 / 18.0])
    # S says f1 is the strongly selected one and f0 is inert; beta says f0 is pushed hard the other
    # way. Signed ranks therefore disagree on which feature sits at the bottom.
    cmp = compare_rankings(S, beta)
    assert cmp.sign_flips == 1
    assert cmp.ranks_differential == (1, 2)
    assert cmp.ranks_gradient == (1, 2)
    # With k = 2 the ordering cannot reverse, so the sign flip is the whole of the disagreement.
    assert cmp.spearman == pytest.approx(1.0)


def test_planted_reordering_is_recovered_exactly():
    """A hand-built case where the top of the ``S`` ranking is the bottom of the ``beta`` ranking.

    ``beta = (-1, 2, 0.5)`` against a ``C`` in which feature 0 and feature 1 correlate at 0.9.
    ``S = C beta`` is computed from it, so the differential is exact rather than estimated, and the
    solve must return the planted ``beta`` to machine precision.
    """
    C = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.0], [0.1, 0.0, 1.0]])
    beta_true = np.array([-1.0, 2.0, 0.5])
    S = C @ beta_true
    assert np.allclose(selection_gradient(C, S), beta_true)
    cmp = compare_rankings(S, selection_gradient(C, S))
    # S = (0.85, 1.1, 0.4): every entry positive and feature 0 second. beta has feature 0 negative
    # and last. That is a sign flip and a two-rank move on k = 3.
    assert cmp.sign_flips == 1
    assert cmp.ranks_differential == (2, 3, 1)
    assert cmp.ranks_gradient == (1, 3, 2)
    assert cmp.max_rank_change == 1


def test_identity_covariance_leaves_the_ordering_alone():
    """With ``C = I`` the gradient is the differential, so nothing can reorder."""
    C = np.eye(4)
    S = np.array([3.0, -1.0, 0.5, -2.0])
    cmp = compare_rankings(S, selection_gradient(C, S))
    assert cmp.spearman == pytest.approx(1.0)
    assert cmp.sign_flips == 0
    assert cmp.max_rank_change == 0


def test_gradient_is_recovered_from_sampled_data():
    """Draw ``f ~ N(0, C)`` and ``r = f . beta + noise``; the sample solve must find ``beta``.

    This is the end-to-end check that ``population_covariance`` and the differential estimator
    agree on their normalisation. If one used ddof = 1 and the other ddof = 0 the recovered beta
    would be off by ``n / (n - 1)`` and this would catch it.
    """
    rng = np.random.default_rng(0)
    C_true = np.array([[1.0, 0.85, 0.0], [0.85, 1.0, 0.2], [0.0, 0.2, 1.0]])
    beta_true = np.array([2.0, -1.5, 0.75])
    L = np.linalg.cholesky(C_true)
    f = rng.standard_normal((200_000, 3)) @ L.T
    r = f @ beta_true + rng.normal(0.0, 0.5, f.shape[0])
    fc = f - f.mean(axis=0)
    S = (fc * (r - r.mean())[:, None]).mean(axis=0)
    beta_hat = selection_gradient(population_covariance(f), S)
    assert np.allclose(beta_hat, beta_true, atol=0.02)
    # and the differential is genuinely a different vector, so the test is not vacuous
    assert not np.allclose(S, beta_true, atol=0.1)


def test_effective_dimension_closed_forms():
    assert effective_dimension(np.eye(4)) == pytest.approx(4.0)
    assert effective_dimension(np.diag([4.0, 1.0, 1.0])) == pytest.approx(1.5)
    # a rank-one covariance has collapsed onto one direction
    v = np.array([[1.0], [2.0], [3.0]])
    assert effective_dimension(v @ v.T) == pytest.approx(1.0)


def test_ridge_shrinks_beta_toward_the_differential():
    """As ``delta`` grows, ``beta -> S / (delta tr C / k)``, so the ordering must converge to S's.

    This is why a large shrinkage showing "no reordering" is not evidence of no reordering: at that
    point the estimator has stopped computing a gradient. The test pins the degeneracy so nobody
    reads the tail of a sweep as a result.
    """
    C = np.array([[1.0, 0.95], [0.95, 1.0]])
    S = np.array([0.1, 0.9])
    rows = ridge_sweep(C, S, deltas=(0.0, 1.0, 1e6))
    assert rows[0]["sign_flips"] == 1  # unregularised: f0's direct effect is negative
    assert rows[-1]["sign_flips"] == 0
    assert rows[-1]["spearman"] == pytest.approx(1.0)
    big = selection_gradient(C, S, ridge=1e6)
    assert np.allclose(big / np.linalg.norm(big), S / np.linalg.norm(S), atol=1e-5)


def test_shrinkage_bisection_brackets_the_sign_flip():
    C = np.array([[1.0, 0.95], [0.95, 1.0]])
    S = np.array([0.1, 0.9])
    d = shrinkage_that_removes_sign_flips(C, S)
    assert 0.0 < d < 1e3
    assert np.all(np.sign(selection_gradient(C, S, ridge=d)) == np.sign(S))
    assert np.any(np.sign(selection_gradient(C, S, ridge=d * 0.9)) != np.sign(S))
    # and it returns exactly zero when there is nothing to remove
    assert shrinkage_that_removes_sign_flips(np.eye(2), np.array([1.0, 2.0])) == 0.0


def test_basis_r_squared_is_one_for_a_noiseless_linear_reward():
    rng = np.random.default_rng(1)
    f = rng.standard_normal((50_000, 3))
    beta = np.array([1.0, -2.0, 0.5])
    r = f @ beta
    fc = f - f.mean(axis=0)
    S = (fc * (r - r.mean())[:, None]).mean(axis=0)
    b = selection_gradient(population_covariance(f), S)
    assert basis_r_squared(S, b, float(np.var(r))) == pytest.approx(1.0, abs=1e-9)


def test_spearman_matches_the_hand_computed_value():
    # one adjacent transposition on k = 7 gives sum d^2 = 2, rho = 1 - 12/336 = 0.9643
    assert spearman([1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 6, 5, 7]) == pytest.approx(
        0.9642857, abs=1e-6
    )
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_selection_gradient_rejects_mismatched_shapes_and_negative_ridge():
    with pytest.raises(ValueError):
        selection_gradient(np.eye(3), np.ones(2))
    with pytest.raises(ValueError):
        selection_gradient(np.eye(2), np.ones(2), ridge=-1.0)


def test_rank_comparison_threshold_counts_what_it_says():
    S = np.array([1.0, 2.0, 3.0, 4.0])
    beta = np.array([4.0, 3.0, 2.0, 1.0])
    cmp = compare_rankings(S, beta, threshold=3)
    assert isinstance(cmp, RankComparison)
    assert cmp.spearman == pytest.approx(-1.0)
    assert cmp.max_rank_change == 3
    assert cmp.n_rank_changes_at_least == 2  # the two outermost features move 3 ranks each


# ---------------------------------------------------------------------------
# I5: the variance derivative
# ---------------------------------------------------------------------------


def test_transition_width_is_the_ten_to_ninety_rise_of_the_planted_logistic():
    """A noiseless logistic of scale ``s`` has a 10-to-90 width of ``2 s ln 9`` exactly."""
    t = np.arange(400, dtype=float)
    s = 12.0
    y = 1.0 / (1.0 + np.exp(-(t - 200.0) / s))
    fit = fit_transition(y, t)
    assert fit.valid
    assert fit.midpoint == pytest.approx(200.0, abs=0.1)
    assert fit.width == pytest.approx(2.0 * np.log(9.0) * s, rel=1e-3)
    assert fit.rmse < 1e-6


def test_transition_fit_reports_failure_rather_than_a_number():
    fit = fit_transition([0.0, 1.0])
    assert not fit.valid
    assert np.isnan(fit.width)


def test_planted_variance_derivative_peaks_where_it_was_planted():
    """A Gaussian bump of width ``sigma`` peaking at ``t_p`` has ``d/dt`` peaking at ``t_p - sigma``."""
    run = planted_run(noise=0.0, variance_sigma=10.0, onset=140.0)
    assert run["planted_derivative_peak"] == 130.0
    assert peak_index(run["variance"]) == int(run["planted_variance_peak"])
    d = derivative(run["variance"], run["steps"])
    assert peak_index(d) == int(run["planted_derivative_peak"])


def test_planted_variance_derivative_peak_survives_noise():
    run = planted_run(noise=0.02, seed=3)
    d = derivative(run["variance"], run["steps"])
    assert abs(peak_index(d) - run["planted_derivative_peak"]) <= 4


def test_cusum_threshold_inverts_the_arl_approximation():
    for target in (100.0, 370.0, 1000.0, 10_000.0):
        h = cusum_threshold(target, 0.5)
        assert arl0(h, 0.5) == pytest.approx(target, rel=1e-6)
    # the spec quotes h = 5.71 for one false alarm per 1000 steps at k = 0.5; this form gives 5.75,
    # and 5.71 scores an ARL of 960 under it, which is the same design to the precision available
    assert cusum_threshold(1000.0, 0.5) == pytest.approx(5.7496, abs=1e-3)
    assert arl0(5.71, 0.5) == pytest.approx(960.0, rel=0.01)
    # the shipped stats.changepoint default of h = 5.0 is a design nobody made
    assert arl0(5.0, 0.5) == pytest.approx(469.0, rel=0.01)
    with pytest.raises(ValueError):
        cusum_threshold(0.5)


def test_lead_time_in_widths_is_the_spec_arithmetic():
    # "a fitted width of 58 steps. A 40-step lead is 0.69 of a window"
    assert lead_time_in_widths(100.0, 140.0, 58.0) == pytest.approx(0.6897, abs=1e-4)
    assert lead_time_in_widths(140.0, 100.0, 58.0) < 0  # a late alarm reads negative
    assert np.isnan(lead_time_in_widths(1.0, 2.0, 0.0))


def test_cadence_resolution_is_one_sample_in_width_units():
    assert cadence_resolution_in_widths(148.0, 592.0) == pytest.approx(0.25)
    assert np.isnan(cadence_resolution_in_widths(148.0, float("nan")))


def test_score_detectors_returns_a_lead_for_every_detector_on_the_plant():
    run = planted_run(seed=11)
    fit, results = score_detectors(
        variance=run["variance"],
        grad_norm=run["grad_norm"],
        outcome=run["outcome"],
        steps=run["steps"],
        baseline=80,
    )
    assert fit.width == pytest.approx(run["planted_width"], rel=0.05)
    names = {r.name for r in results}
    assert names == {
        "variance derivative, CUSUM",
        "variance level, CUSUM",
        "gradient norm, CUSUM",
        "gradient norm, peak",
    }
    for r in results:
        assert r.alarm_step is not None, f"{r.name} did not alarm on the plant"
        assert r.lead_in_widths == pytest.approx((fit.midpoint - r.alarm_step) / fit.width)
    # The gradient-norm peak was planted at the onset, so it leads by nothing. The tolerance is one
    # sampling interval, which on a 35-step width is 0.03 widths: an argmax on a noisy series cannot
    # be pinned tighter than the grid it lives on, and that is the point of reporting the cadence
    # resolution alongside every lead.
    peak = next(r for r in results if r.name == "gradient norm, peak")
    assert abs(peak.alarm_step - run["planted_onset"]) <= 2.0
    assert abs(peak.lead_in_widths) <= 2.0 / fit.width


def test_score_detectors_without_a_gradient_norm_drops_only_those_rows():
    run = planted_run(seed=5)
    _, results = score_detectors(
        variance=run["variance"],
        grad_norm=None,
        outcome=run["outcome"],
        steps=run["steps"],
        baseline=80,
    )
    assert {r.name for r in results} == {"variance derivative, CUSUM", "variance level, CUSUM"}


# ---------------------------------------------------------------------------
# I5: reading a store's shape
# ---------------------------------------------------------------------------


def test_field_names_come_from_names_and_never_from_data():
    """The absence claim depends on this: prose and base64 must not answer a field-name query."""
    value = {
        "__type__": "campaign.payloads.TablePayload",
        "fields": {
            "name": "hump generation bank",
            "columns": {"__seq__": ["prompt_id", "sample", "text"]},
            "rows": {
                "__seq__": [
                    {"__seq__": ["uf:27144", 0, "use technology to your advantage while hiding"]}
                ]
            },
            "meta": {"__map__": {"grad_norm": 1.25}},
            "vector": {"__ndarray__": {"b64": "ea2advantage", "dtype": "float32", "shape": [2]}},
        },
    }
    names = envelope_field_names(value)
    assert {
        "name",
        "columns",
        "rows",
        "meta",
        "grad_norm",
        "prompt_id",
        "text",
        "uf:27144",
    } <= names
    assert not any("advantage" in n for n in names)  # the prose row body is data, not a name
    assert not any("ea2" in n for n in names)  # neither is the encoded array


def test_classify_rows_finds_a_step_ladder_and_the_three_inputs():
    rows = [
        {
            "observable": "run.telemetry",
            "slice": "toy-run::step000000",
            "roster_key": "policy",
            "names": {"group_std", "grad_norm", "hack_rate"},
        },
        {
            "observable": "run.telemetry",
            "slice": "toy-run::step000010",
            "roster_key": "policy",
            "names": {"group_std", "grad_norm", "hack_rate"},
        },
        {
            "observable": "run.telemetry",
            "slice": "toy-run::step000020",
            "roster_key": "policy",
            "names": {"group_std", "grad_norm", "hack_rate"},
        },
    ]
    inv = classify_rows(rows)
    assert inv.n_rows == 3
    assert len(inv.series) == 1
    series = inv.series[0]
    assert series.base_slice == "toy-run"
    assert series.steps == [0, 10, 20]
    assert series.cadence == pytest.approx(10.0)
    assert inv.within_group_hits and inv.grad_norm_hits and inv.outcome_hits
    assert inv.runnable


def test_classify_rows_reports_the_campaign_shape_as_not_runnable():
    """A checkpoint ladder that carries loadings and no reward telemetry is not an I5 series."""
    rows = [
        {
            "observable": "campaign.checkpoints.loadings",
            "slice": f"ultrafeedback-train-20k::step{step:06d}",
            "roster_key": "skywork-v2-qwen3-0.6b",
            "names": {"name", "columns", "rows", "meta", "w_r", "step", "len_chars", "cohens_d"},
        }
        for step in (0, 148, 296)
    ]
    inv = classify_rows(rows)
    assert len(inv.series) == 1
    assert inv.series[0].cadence == pytest.approx(148.0)
    assert inv.within_group_hits == {}
    assert inv.grad_norm_hits == {}
    assert inv.outcome_hits == {}
    assert not inv.runnable


def test_classify_rows_ignores_a_step_word_that_is_not_a_training_step():
    """ProcessBench reasoning steps must not be mistaken for a training-step axis."""
    rows = [
        {
            "observable": "campaign.prm.steps",
            "slice": "processbench-full::part0000",
            "roster_key": "qwen-prm",
            "names": {"item_ids", "values", "offsets", "labels", "partial", "part"},
        }
    ]
    inv = classify_rows(rows)
    assert inv.series == []
    assert not inv.runnable
