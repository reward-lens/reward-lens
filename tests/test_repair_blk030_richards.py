"""BLK-030 — the shipped transition fitter is symmetric by construction, so `C20` had no estimator.

`C20` registers "likelihood-ratio test of a four-parameter Richards curve against the symmetric
logistic, `p < 0.05`, **and** the shape parameter's sign matches the prediction, on at least 3 of 5
seeds". The library ships `_logistic` at `measure/threshold/variance.py:82`, which is symmetric by
construction and cannot express the asymmetry the row is about. A grep for "richards" over the whole
package returns nothing.

**This row is OPEN, not repaired.** BLK-120 owns the Richards parameterisation, the sign convention
and the fitting method, is in packet `P1-ROWS`, and has not landed. What is built here is a fitter
against a **clearly stated assumed parameterisation**, recorded in
`chain/repair/proofs/BLK-030/ASSUMED-PARAMETERISATION.md`. If BLK-120 registers a different one, the
tests below that name `nu` change meaning and this file has to be re-read, not merely re-run:
BLK-120's own evidence is that "the mapping from the shape parameter to 'left-skewed' flips between
the common parameterisations".
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.measure.threshold.richards import (
    RichardsFit,
    fit_richards,
    richards_curve,
)
from reward_lens.measure.threshold.variance import fit_transition

STEPS = np.linspace(0.0, 240.0, 241)


def _series(
    nu: float, *, rate: float = 0.08, midpoint: float = 120.0, noise: float = 0.0, seed: int = 0
):
    y = richards_curve(STEPS, 0.05, 0.85, rate, midpoint, nu)
    if noise:
        y = y + np.random.default_rng(seed).normal(0.0, noise, y.size)
    return y


# ---------------------------------------------------------------------------
# The closure proof, first half: recover a known shape parameter to two decimals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nu", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_the_fitter_recovers_a_known_shape_parameter_on_a_noiseless_series(nu: float) -> None:
    """BLK-030's closure proof, first half, verbatim: to two decimal places."""
    fit = fit_richards(_series(nu), STEPS, outcome_bounds=(0.0, 1.0))
    assert fit.converged, fit.method
    assert fit.nu == pytest.approx(nu, abs=0.01), f"planted {nu}, recovered {fit.nu}"


def test_the_other_three_parameters_come_back_too() -> None:
    fit = fit_richards(_series(3.0, rate=0.06, midpoint=90.0), STEPS, outcome_bounds=(0.0, 1.0))
    assert fit.nu == pytest.approx(3.0, abs=0.01)
    assert fit.rate == pytest.approx(0.06, abs=1e-3)
    assert fit.midpoint == pytest.approx(90.0, abs=0.5)
    assert fit.baseline == pytest.approx(0.05, abs=1e-3)
    assert fit.asymptote == pytest.approx(0.85, abs=1e-3)


def test_at_shape_one_the_richards_curve_is_the_shipped_logistic_exactly() -> None:
    """The property that makes `C20`'s likelihood-ratio test well posed.

    The logistic has to be the `nu = 1` special case of the Richards family, or the two models are
    not nested and a likelihood-ratio test between them has no null distribution. Checked against
    the shipped `_logistic` rather than against a second copy written here.
    """
    from reward_lens.measure.threshold.variance import _logistic

    ours = richards_curve(STEPS, 0.05, 0.90, 0.08, 120.0, 1.0)
    theirs = _logistic(STEPS, 0.05, 0.85, 120.0, 0.08)
    assert np.max(np.abs(ours - theirs)) < 1e-12


# ---------------------------------------------------------------------------
# The closure proof, second half: refuse when the rise holds too few points
# ---------------------------------------------------------------------------


def test_it_refuses_when_the_rise_holds_fewer_points_than_the_fit_has_parameters() -> None:
    """BLK-030's closure proof, second half, verbatim.

    A transition read at a checkpoint stride of 11 over a rise of 20 steps has two points inside it.
    Five free parameters fitted through two points is not an underdetermined fit, it is an arbitrary
    one, and `curve_fit` returns from it without complaint.
    """
    steps = np.arange(0.0, 240.0, 11.0)
    y = richards_curve(steps, 0.05, 0.85, 1.2, 120.0, 2.0)  # a rise about 8 steps wide
    fit = fit_richards(y, steps, outcome_bounds=(0.0, 1.0))
    assert not fit.converged
    assert "in-rise" in fit.method, fit.method
    assert fit.n_in_rise < fit.n_parameters
    assert np.isnan(fit.nu)


def test_it_does_not_refuse_when_the_rise_is_read_densely_enough() -> None:
    """A refusal that fires on everything closes the row and removes the estimator."""
    fit = fit_richards(_series(2.0), STEPS, outcome_bounds=(0.0, 1.0))
    assert fit.converged
    assert fit.n_in_rise >= fit.n_parameters


def test_the_refusal_reports_what_it_counted() -> None:
    steps = np.arange(0.0, 240.0, 11.0)
    fit = fit_richards(
        richards_curve(steps, 0.05, 0.85, 1.2, 120.0, 2.0), steps, outcome_bounds=(0.0, 1.0)
    )
    assert fit.n_parameters == 5
    assert 0 <= fit.n_in_rise < 5
    assert "in-rise" in fit.render()


def test_a_series_too_short_to_fit_at_all_refuses() -> None:
    fit = fit_richards(np.array([0.1, 0.2, 0.3, 0.4]), np.arange(4.0), outcome_bounds=(0.0, 1.0))
    assert not fit.converged
    assert "too-short" in fit.method


def test_a_flat_series_does_not_come_back_as_an_asymmetric_transition() -> None:
    """The failure the shipped fitter's own docstring records, in the asymmetric family.

    A five-parameter curve fitted to noise converges even more happily than a four-parameter one, so
    the likelihood-ratio test against the logistic is the number that has to say no here.
    """
    rng = np.random.default_rng(30)
    flat = rng.normal(0.5, 0.10, STEPS.size)
    fit = fit_richards(flat, STEPS, outcome_bounds=(0.0, 1.0))
    if fit.converged:
        assert fit.lrt_p > 0.05, f"a flat series was called asymmetric at p = {fit.lrt_p:.4f}"


# ---------------------------------------------------------------------------
# C20's own test: the likelihood ratio against the symmetric logistic
# ---------------------------------------------------------------------------


def test_the_likelihood_ratio_rejects_the_logistic_on_a_genuinely_asymmetric_series() -> None:
    """`C20`'s statistic. One degree of freedom, because the logistic is the `nu = 1` restriction."""
    fit = fit_richards(_series(4.0, noise=0.01, seed=1), STEPS, outcome_bounds=(0.0, 1.0))
    assert fit.converged
    assert fit.lrt_dof == 1
    assert fit.lrt_p < 0.05, f"lrt_p {fit.lrt_p:.4g} on a series with nu = 4"
    assert fit.loglik_richards > fit.loglik_logistic


def test_the_likelihood_ratio_does_not_reject_on_a_symmetric_series() -> None:
    """The half that can fail. A test that rejects everything is not evidence about shape."""
    fit = fit_richards(_series(1.0, noise=0.01, seed=2), STEPS, outcome_bounds=(0.0, 1.0))
    assert fit.converged
    assert fit.lrt_p > 0.05, f"a logistic series was called asymmetric at p = {fit.lrt_p:.4g}"


def test_the_logistic_half_of_the_comparison_agrees_with_the_shipped_fitter() -> None:
    """The null model is the shipped symmetric fit, not a second logistic written here."""
    y = _series(3.0, noise=0.01, seed=3)
    fit = fit_richards(y, STEPS, outcome_bounds=(0.0, 1.0))
    shipped = fit_transition(y, STEPS)
    assert fit.logistic_rmse == pytest.approx(shipped.rmse, rel=1e-3)


# ---------------------------------------------------------------------------
# BLK-120: the branch is an inequality, and the parameterisation is an assumption
# ---------------------------------------------------------------------------


def test_the_predicted_branch_is_an_inequality_and_not_a_sign() -> None:
    """BLK-120: "the predicted branch stated as `nu > 1` rather than as a sign".

    The Richards shape parameter in this parameterisation is strictly positive and has no sign, so
    `C20`'s registered rule that "the shape parameter's sign matches the prediction" cannot be
    evaluated as written. What the fitter reports is the branch.
    """
    slow_then_sharp = fit_richards(_series(4.0), STEPS, outcome_bounds=(0.0, 1.0))
    sharp_then_slow = fit_richards(_series(0.25), STEPS, outcome_bounds=(0.0, 1.0))
    assert slow_then_sharp.nu > 0 and sharp_then_slow.nu > 0, "nu is strictly positive, so no sign"
    assert slow_then_sharp.branch == "nu>1"
    assert sharp_then_slow.branch == "nu<1"
    assert fit_richards(_series(1.0), outcome_bounds=(0.0, 1.0)).branch == "nu=1"


def test_the_branch_matches_the_shape_the_curve_actually_has() -> None:
    """`nu > 1` has to mean slow rise and sharp finish in this parameterisation, and it does.

    A2_ARITHMETIC's F3.4 states the prediction lives on the branch where the rise is slow and the
    finish sharp, and calls that `nu > 1`. This measures the two half-widths off the fitted curve
    rather than trusting the label, because BLK-120's own evidence is that the mapping from the
    shape parameter to a skew direction flips between the common parameterisations.
    """
    for nu, slower_first in ((4.0, True), (0.25, False)):
        y = richards_curve(STEPS, 0.0, 1.0, 0.08, 120.0, nu)
        t10 = STEPS[np.argmax(y >= 0.1)]
        t50 = STEPS[np.argmax(y >= 0.5)]
        t90 = STEPS[np.argmax(y >= 0.9)]
        rise, finish = t50 - t10, t90 - t50
        assert bool(rise > finish) is slower_first, f"nu = {nu}: rise {rise}, finish {finish}"


def test_the_parameterisation_is_recorded_on_every_fit() -> None:
    """The row is OPEN on BLK-120, so a fit that does not say which family it used is unreadable."""
    fit = fit_richards(_series(2.0), STEPS, outcome_bounds=(0.0, 1.0))
    assert isinstance(fit, RichardsFit)
    assert fit.parameterisation == "A + (K-A) / (1 + nu*exp(-r(t-tm)))^(1/nu)"
    assert "BLK-120" in RichardsFit.__doc__
    assert "assumed" in fit.render().lower() or "BLK-120" in fit.render()


def test_the_shipped_symmetric_fitter_is_left_alone() -> None:
    """`fit_transition` is the null model of `C20`'s own test, so it is not touched."""
    fit = fit_transition(_series(1.0, noise=0.01, seed=4), STEPS)
    assert fit.converged
    assert fit.method == "logistic-10-90"
    assert fit.midpoint == pytest.approx(120.0, abs=2.0)
