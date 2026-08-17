"""The reconstruction machinery, on tables whose true composition is known by construction.

Every test here builds a rollout table from a reward the test itself chose, then asks the module to
recover it. That is the only arrangement in which a failure is informative: a closure test that only
ever runs against the real artifact cannot tell you whether the solver is wrong or the artifact is,
and finding out which of those two you are looking at is the entire job of a preflight.

The synthetic tables are deliberately small and deliberately awkward. Uneven groups, a column that
is a near-copy of a real reward term, a series with a genuine off-by-one, a step where every group
is degenerate. Each of those is a property the real series is suspected of having, tested here where
the answer is not in doubt.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.ledger.reconstruct import (
    closure_residuals,
    compose,
    fit_composition,
    group_statistics,
    informative_group_counts,
    search_subsets,
    shift_residual_curve,
    snap_to_simple_ratio,
    steps_and_means,
)
from reward_lens.measure.ledger.trainer_log import (
    quantisation_step,
    read_trainer_state,
    rounding_rms,
)

# ---------------------------------------------------------------------------
# A synthetic run whose reward this file chose
# ---------------------------------------------------------------------------

#: The composition the fixture below is built from. Three components at round weights, one of which
#: takes quarter values, which is the shape section 0 of the gate document argues for on the real
#: series. The solver is never told these.
TRUE_WEIGHTS = {"format_ok": 1.0, "passed": 4.0, "partial_credit": 1.0}
TRUE_INTERCEPT = 0.0


def synthetic_run(
    n_steps: int = 60,
    n_groups: int = 4,
    n_per_group: int = 16,
    seed: int = 7,
) -> dict[str, np.ndarray]:
    """A rollout table with a rising pass rate, in the shape a published one comes in.

    `hacked` is the adjudication column: it is strongly associated with `passed`, because on a real
    run the policy learns to pass by hacking, and it carries no reward weight at all. Recovering a
    zero on it in the presence of that association is the check that matters.
    """
    rng = np.random.default_rng(seed)
    rows: dict[str, list[float]] = {
        "step": [],
        "group": [],
        "format_ok": [],
        "passed": [],
        "partial_credit": [],
        "hacked": [],
    }
    texts: list[str] = []
    for t in range(n_steps):
        p_pass = 0.02 + 0.95 / (1.0 + math.exp(-(t - n_steps / 2) / 4.0))
        for g in range(n_groups):
            for k in range(n_per_group):
                passed = float(rng.random() < p_pass)
                rows["step"].append(float(t))
                rows["group"].append(float(g))
                rows["format_ok"].append(float(rng.random() < 0.9))
                rows["passed"].append(passed)
                rows["partial_credit"].append(float(rng.integers(0, 5)) / 4.0)
                rows["hacked"].append(passed * float(rng.random() < 0.8))
                texts.append(f"solution-{t % 3}-{g}-{k % 5}")
    table = {k: np.asarray(v, dtype=np.float64) for k, v in rows.items()}
    table["text"] = np.asarray(texts, dtype=object)
    table["reward"] = compose(table, TRUE_WEIGHTS, TRUE_INTERCEPT)
    return table


@pytest.fixture(scope="module")
def run() -> dict[str, np.ndarray]:
    return synthetic_run()


@pytest.fixture(scope="module")
def logged(run):
    """What a trainer would have logged for this run: the per-step mean of the true reward."""
    steps, means = steps_and_means(run["step"], {"reward": run["reward"]})
    return steps, means["reward"]


# ---------------------------------------------------------------------------
# 1. The fit recovers the composition it was never told
# ---------------------------------------------------------------------------


def test_the_fit_recovers_the_true_weights_and_they_snap_to_the_configured_constants(run, logged):
    steps, target = logged
    _, means = steps_and_means(
        run["step"],
        {k: run[k] for k in ("format_ok", "passed", "partial_credit", "hacked")},
    )
    fit = fit_composition(means, target, n_bootstrap=300, seed=1)

    for name, truth in TRUE_WEIGHTS.items():
        assert fit.weight(name) == pytest.approx(truth, abs=1e-8)
    assert fit.intercept == pytest.approx(TRUE_INTERCEPT, abs=1e-8)
    assert fit.all_snapped()
    snaps = dict(zip(fit.names, fit.snapped))
    assert snaps["passed"] == "4"
    assert snaps["format_ok"] == "1"
    assert snaps["partial_credit"] == "1"


def test_the_adjudication_column_fits_zero_despite_being_correlated_with_a_reward_term(run, logged):
    """The falsification test, on a table where the correct answer is known to be zero.

    `hacked` is only ever 1 where `passed` is 1, so a solver that reads association as composition
    will hand it weight. The assertion is both halves: the point estimate is zero and the interval
    says so, because a point estimate of zero with an interval spanning two reward units would not
    license the sentence this check exists to license.
    """
    steps, target = logged
    _, means = steps_and_means(
        run["step"],
        {k: run[k] for k in ("format_ok", "passed", "partial_credit", "hacked")},
    )
    fit = fit_composition(means, target, n_bootstrap=500, seed=2)
    assert fit.weight("hacked") == pytest.approx(0.0, abs=1e-8)
    assert fit.contains_zero("hacked")
    lo, hi = fit.interval("hacked")
    assert hi - lo < 0.5


def test_a_perfectly_collinear_column_is_visible_in_the_condition_number(run, logged):
    """A duplicated column is not a discovery, and the fit has to make that inspectable."""
    steps, target = logged
    columns = {k: run[k] for k in ("format_ok", "passed", "partial_credit")}
    columns["passed_copy"] = run["passed"].copy()
    _, means = steps_and_means(run["step"], columns)
    fit = fit_composition(means, target, n_bootstrap=0)
    assert fit.condition_number > 1e6
    assert fit.weight("passed") + fit.weight("passed_copy") == pytest.approx(4.0, abs=1e-6)


def test_the_subset_search_prefers_the_configuration_over_the_larger_collinear_fit(run, logged):
    """Ranking on the snapped residual is what makes this work, and this is the test of that.

    A spurious extra term always lowers the fitted residual and does not land on a round number, so
    a search ranked on the fitted residual would return the four-term answer every time.
    """
    steps, target = logged
    _, means = steps_and_means(
        run["step"],
        {k: run[k] for k in ("format_ok", "passed", "partial_credit", "hacked")},
    )
    results = search_subsets(
        means,
        target,
        candidates=("format_ok", "passed", "partial_credit", "hacked"),
        max_terms=4,
    )
    best = results[0]
    assert set(best.names) == set(TRUE_WEIGHTS)
    assert best.all_snapped
    assert best.residual_snapped < 1e-9


def test_a_composition_the_columns_cannot_express_leaves_a_residual_that_does_not_snap(run):
    """The `OPEN` shape: a reward with a term nobody published stays open, visibly."""
    steps, means = steps_and_means(
        run["step"], {k: run[k] for k in ("format_ok", "passed", "partial_credit")}
    )
    hidden = 2.5 * np.sin(np.asarray(run["step"], dtype=float) / 3.0)
    target_rows = compose(run, TRUE_WEIGHTS, TRUE_INTERCEPT) + hidden
    _, target = steps_and_means(run["step"], {"r": target_rows})
    fit = fit_composition(means, target["r"], n_bootstrap=0)
    predicted = sum(fit.weight(n) * means[n] for n in fit.names) + fit.intercept
    residual = closure_residuals(predicted, target["r"], target="reward", steps=steps)
    assert residual.relative_rms > 0.01
    assert not fit.all_snapped()


# ---------------------------------------------------------------------------
# 2. Residuals are read as a pattern, not only as a size
# ---------------------------------------------------------------------------


def test_a_constant_offset_shows_up_as_a_biased_mean_and_not_only_as_an_rms():
    obs = np.linspace(1.0, 2.0, 80)
    resid = closure_residuals(obs + 0.05, obs, target="reward")
    assert resid.mean_residual == pytest.approx(0.05)
    assert abs(resid.mean_z) > 10.0


def test_independent_noise_is_unbiased_and_uncorrelated_and_the_statistics_say_so():
    rng = np.random.default_rng(3)
    obs = np.linspace(1.0, 2.0, 400)
    resid = closure_residuals(obs + rng.normal(0, 0.01, obs.size), obs, target="reward")
    assert abs(resid.mean_z) < 3.0
    assert abs(resid.lag1_autocorrelation) < 0.15


def test_a_systematic_drift_shows_up_in_the_autocorrelation():
    obs = np.linspace(1.0, 2.0, 200)
    drift = np.linspace(-0.05, 0.05, 200)
    resid = closure_residuals(obs + drift, obs, target="reward")
    assert resid.lag1_autocorrelation > 0.9


def test_a_residual_no_larger_than_the_logging_grid_is_reported_as_at_floor():
    rng = np.random.default_rng(4)
    q = 1.0 / 256.0
    truth = rng.uniform(0.5, 6.0, 300)
    obs = np.round(truth / q) * q
    resid = closure_residuals(truth, obs, target="reward")
    assert resid.at_floor(q)
    assert 0.5 < resid.floor_ratio(q) < 2.0
    assert not resid.at_floor(q / 100.0)


def test_the_quantisation_grid_is_measured_rather_than_assumed():
    values = np.arange(160, 1530, 7) / 256.0
    assert quantisation_step(values) == pytest.approx(1.0 / 256.0)
    assert rounding_rms(1.0 / 256.0) == pytest.approx((1.0 / 256.0) / math.sqrt(12.0))
    assert quantisation_step(np.array([0.1, 0.20000000003, 0.31234567])) == 0.0


# ---------------------------------------------------------------------------
# 3. The shift-residual curve finds an alignment it was not told
# ---------------------------------------------------------------------------


def test_the_shift_curve_finds_a_planted_offset_and_calls_it_clear():
    rng = np.random.default_rng(5)
    logged = rng.normal(3.0, 1.0, 120)
    # The reconstruction is the logged series read two steps late, which is the failure mode P5 is
    # looking for: file order taken as step order when the first file is a pre-training evaluation.
    reconstructed = logged[2:]
    curve = shift_residual_curve(reconstructed, logged)
    assert curve.best_shift == 2
    assert curve.has_clear_minimum_at(2)
    assert not curve.has_clear_minimum_at(0)
    assert curve.residual_at(2) < 1e-12


def test_a_curve_with_no_real_minimum_is_not_called_clear():
    rng = np.random.default_rng(6)
    curve = shift_residual_curve(rng.normal(0, 1, 200), rng.normal(0, 1, 200))
    assert not curve.has_clear_minimum_at(curve.best_shift)


def test_the_overlap_count_travels_with_each_shift():
    curve = shift_residual_curve(np.arange(50.0), np.arange(50.0), shifts=range(-5, 6))
    assert curve.n_overlap[curve.shifts.index(0)] == 50
    assert curve.n_overlap[curve.shifts.index(5)] == 45


# ---------------------------------------------------------------------------
# 4. Grouping, including the two conventions that agree until they do not
# ---------------------------------------------------------------------------


def test_groups_are_formed_within_a_step_and_never_across_steps(run):
    stats = group_statistics(run["step"], run["group"], run["reward"])
    assert len(stats) == 60
    assert all(g.n_groups == 4 for g in stats)
    assert all(g.sizes == (16, 16, 16, 16) for g in stats)
    assert all(g.n_rollouts == 64 for g in stats)


def test_a_problem_recurring_across_steps_is_counted_as_separate_groups():
    steps = np.array([0.0, 0.0, 1.0, 1.0])
    groups = np.array(["p", "p", "p", "p"], dtype=object)
    rewards = np.array([0.0, 1.0, 5.0, 5.0])
    stats = group_statistics(steps, groups, rewards)
    assert [g.n_groups for g in stats] == [1, 1]
    assert stats[0].n_informative() == 1
    assert stats[1].n_informative() == 0


def test_the_two_zero_std_conventions_agree_on_even_batches_and_diverge_on_uneven_ones():
    """TRL has logged `frac_reward_zero_std` over groups and over rollouts across versions.

    On an even batch the two are identical, which is why the difference goes unnoticed. This builds
    the uneven case where they cannot both be right, so the real series decides which it is instead
    of a remembered version number.
    """
    even = group_statistics(
        np.array([0.0] * 8),
        np.array(["a", "a", "a", "a", "b", "b", "b", "b"], dtype=object),
        np.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0]),
    )[0]
    assert even.frac_zero_std_by_group() == pytest.approx(0.5)
    assert even.frac_zero_std_by_rollout() == pytest.approx(0.5)

    uneven = group_statistics(
        np.array([0.0] * 8),
        np.array(["a", "a", "a", "a", "a", "a", "b", "b"], dtype=object),
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 3.0]),
    )[0]
    assert uneven.frac_zero_std_by_group() == pytest.approx(0.5)
    assert uneven.frac_zero_std_by_rollout() == pytest.approx(0.75)


def test_a_step_where_every_group_is_degenerate_carries_no_informative_groups():
    steps = np.array([0.0] * 8)
    groups = np.array(["a"] * 4 + ["b"] * 4, dtype=object)
    stats = group_statistics(steps, groups, np.array([5.0] * 8))[0]
    assert stats.n_informative() == 0
    assert stats.frac_zero_std_by_group() == 1.0


# ---------------------------------------------------------------------------
# 5. Clone accounting, where the point is that the raw count lies
# ---------------------------------------------------------------------------


def test_eight_identical_groups_are_worth_one_independent_group():
    steps = np.array([0.0] * 16)
    groups = np.asarray([chr(ord("a") + i // 2) for i in range(16)], dtype=object)
    rewards = np.asarray([0.0, 1.0] * 8)
    hashes = np.asarray(["h0", "h1"] * 8, dtype=object)
    counts = informative_group_counts(steps, groups, rewards, hashes)[0]
    assert counts.n_groups == 8
    assert counts.n_informative == 8
    assert counts.n_informative_clone_adjusted == pytest.approx(1.0)
    assert counts.duplicate_fraction == pytest.approx(7 / 8)


def test_distinct_groups_are_worth_their_raw_count():
    steps = np.array([0.0] * 16)
    groups = np.asarray([chr(ord("a") + i // 2) for i in range(16)], dtype=object)
    rewards = np.asarray([0.0, 1.0] * 8)
    hashes = np.asarray([f"h{i}" for i in range(16)], dtype=object)
    counts = informative_group_counts(steps, groups, rewards, hashes)[0]
    assert counts.n_informative_clone_adjusted == pytest.approx(8.0)
    assert counts.duplicate_fraction == 0.0


def test_with_no_hashes_the_adjusted_count_equals_the_raw_one(run):
    counts = informative_group_counts(run["step"], run["group"], run["reward"])
    assert all(
        c.n_informative_clone_adjusted == pytest.approx(float(c.n_informative)) for c in counts
    )


def test_a_degenerate_group_is_excluded_before_the_clone_adjustment_not_after():
    steps = np.array([0.0] * 8)
    groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"], dtype=object)
    rewards = np.asarray([0.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    hashes = np.asarray([f"h{i}" for i in range(8)], dtype=object)
    counts = informative_group_counts(steps, groups, rewards, hashes)[0]
    assert counts.n_groups == 4
    assert counts.n_informative == 1
    assert counts.n_informative_clone_adjusted == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 6. Snapping, and the line between a constant and a coefficient
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (4.0, "4"),
        (0.5, "1/2"),
        (0.25, "1/4"),
        (-1.0, "-1"),
        (1.0000004, "1"),
        (0.0, "0"),
        (2.0000001, "2"),
    ],
)
def test_a_configured_constant_snaps(value, expected):
    got = snap_to_simple_ratio(value)
    assert got is not None and got[1] == expected


@pytest.mark.parametrize("value", [0.7137, 1.4142135, 3.14159, 0.0731])
def test_a_regression_coefficient_does_not_snap(value):
    assert snap_to_simple_ratio(value) is None


def test_an_equal_ratio_snaps_to_the_form_a_configuration_would_have_written():
    got = snap_to_simple_ratio(2.0)
    assert got is not None and got[1] == "2"


# ---------------------------------------------------------------------------
# 7. The trainer log is read by verified key, never by a plausible one
# ---------------------------------------------------------------------------


def _write_state(tmp_path, entries, **state):
    path = tmp_path / "trainer_state.json"
    path.write_text(json.dumps({"log_history": entries, **state}), encoding="utf-8")
    return path


def test_the_key_census_reports_names_verbatim_with_their_counts(tmp_path):
    path = _write_state(
        tmp_path,
        [
            {"step": 1, "reward": 0.5, "reward_std": 0.1},
            {"step": 2, "reward": 0.6, "reward_std": 0.2},
            {"step": 2, "train_runtime": 12.0},
        ],
        global_step=2,
    )
    census = read_trainer_state(path).census
    assert census.n_entries == 3
    assert census.counts == {
        "step": 3,
        "reward": 2,
        "reward_std": 2,
        "train_runtime": 1,
    }
    assert census.carried_by_all() == ("step",)
    assert "train_runtime" in census.rare()


def test_an_absent_key_refuses_with_the_available_names_rather_than_substituting(tmp_path):
    path = _write_state(tmp_path, [{"step": 1, "rewards/total": 0.5}])
    got = read_trainer_state(path).series("reward")
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.RECORD_INCOMPLETE
    assert "rewards/total" in got.detail
    assert got.remedy


def test_a_present_key_comes_back_in_step_order_with_the_key_attached(tmp_path):
    path = _write_state(
        tmp_path,
        [{"step": 3, "reward": 0.3}, {"step": 1, "reward": 0.1}, {"step": 2, "reward": 0.2}],
    )
    series = read_trainer_state(path).series("reward")
    assert not isinstance(series, Refusal)
    assert series.key == "reward"
    assert list(series.steps) == [1.0, 2.0, 3.0]
    assert list(series.values) == pytest.approx([0.1, 0.2, 0.3])


def test_a_summary_row_without_a_step_does_not_silently_join_the_series(tmp_path):
    path = _write_state(
        tmp_path,
        [{"step": 1, "reward": 0.1}, {"step": 2, "reward": 0.2}, {"train_loss": 0.9}],
    )
    series = read_trainer_state(path).series("reward")
    assert not isinstance(series, Refusal)
    assert series.n == 2


def test_log_history_zero_is_returned_verbatim(tmp_path):
    path = _write_state(tmp_path, [{"step": 1, "reward": 0.1, "epoch": 0.01}])
    assert read_trainer_state(path).first() == {"step": 1, "reward": 0.1, "epoch": 0.01}
