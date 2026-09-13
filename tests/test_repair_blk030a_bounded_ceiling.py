"""BLK-030-a — `fit_richards` ran the ceiling off the top of a truncated series.

`C20`'s series is a rate. A 240-step budget ends before the transition finishes, so what the run
produces is a truncated curve, and with the asymptote unbounded a Richards curve and a symmetric
logistic both degenerate to the same locally exponential shape over the observed window. On the
design's own predicted curve, noiseless, read every step over its own budget:

    fitted ceiling K   169366        on a rate
    fitted shape nu    1.1621
    richards rmse      0.0058197     logistic rmse 0.00581971
    likelihood ratio   p = 0.973

`C20` returning `p = 0.97` on noiseless data that is its own prediction. Not noisy data, not few
points: 241 reads with 30 inside the rise.

BLK-030's own fixture cannot see this, because a planted Richards curve fitted by a Richards model is
never truncated in the sense that matters. What makes the ceiling unidentifiable is misspecification
plus truncation together, and the design's predicted curve is not a member of the family it is tested
against (BLK-120-a).

**What this file does not claim.** A low `p` here is not `C20` confirmed. BLK-120-a stands: the
Richards family's rise-to-finish ratio is bounded above at 2.7381 and the design's own curve needs
3.81, so the test discriminates against the logistic without testing the prediction. This row is
about whether the estimator can be computed; that one is about whether the alternative contains the
prediction. Fixing either leaves the other.

Raw before-and-after at `chain/repair/proofs/BLK-030-a/`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from reward_lens.measure.threshold.richards import fit_richards, richards_curve

RATE_BOUNDS = (0.0, 1.0)
R_CORPUS = 0.1927  # chain/parts/part_02.md section 2.4
S_DESMOOTHED = 0.2432  # errata E-74, the de-smoothed selection coefficient
BUDGET_STEPS = 240


def _integrate(steps: np.ndarray, lam0: float) -> np.ndarray:
    a = 1.0 - R_CORPUS

    def rhs(_t, y):
        p = 1.0 / (1.0 + math.exp(-y[0]))
        return [S_DESMOOTHED * (R_CORPUS + a * p) / (1.0 - a * p)]

    sol = solve_ivp(
        rhs,
        (0.0, float(steps[-1])),
        [lam0],
        method="RK45",
        rtol=1e-12,
        atol=1e-14,
        dense_output=True,
        max_step=1.0,
    )
    assert sol.success, sol.message
    return 1.0 / (1.0 + np.exp(-np.asarray([sol.sol(t)[0] for t in steps])))


def design_curve(steps: np.ndarray, level_at_end: float) -> np.ndarray:
    """Part 2.4's ODE, placed so the curve reaches `level_at_end` at the last read.

    `dlambda/dt = s (r + (1-r) p) / (1 - (1-r) p)` with `p = sigmoid(lambda)`. This is the design's
    own prediction, and the level at the last read is what "the run ends before the curve finishes"
    means as a number.
    """
    ends = np.asarray([0.0, float(steps[-1])])
    lam0 = brentq(lambda lam: _integrate(ends, lam)[-1] - level_at_end, -60.0, 10.0)
    return _integrate(steps, lam0)


BUDGET = np.arange(0.0, BUDGET_STEPS + 1.0, 1.0)


# ---------------------------------------------------------------------------
# The closure proof the child blocker states
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level_at_end", [0.4, 0.5, 0.8])
def test_the_ceiling_stays_inside_the_declared_range_on_a_truncated_curve(
    level_at_end: float,
) -> None:
    """Before the repair these fitted `K` between 1.4e5 and 2.6e5 on a rate."""
    y = design_curve(BUDGET, level_at_end)
    fit = fit_richards(y, BUDGET, outcome_bounds=RATE_BOUNDS)

    assert fit.converged
    assert 0.0 <= fit.asymptote <= 1.0, f"ceiling {fit.asymptote:g} is not a rate"
    assert 0.0 <= fit.baseline <= 1.0
    # The curve is still rising at the budget's end, which is the condition under test.
    assert fit.asymptote > y.max()


@pytest.mark.parametrize("level_at_end", [0.4, 0.5, 0.8])
def test_the_shape_parameter_is_identifiable_on_a_truncated_curve(level_at_end: float) -> None:
    """Before the repair `nu` collapsed to 1.16 for every truncation, whatever the shape.

    The design's curve is left-leaning and far from symmetric, so a `nu` pinned near 1 is the
    signature of the degeneracy rather than a measurement. After the repair it lands well above 1
    and it moves with the input.
    """
    y = design_curve(BUDGET, level_at_end)
    fit = fit_richards(y, BUDGET, outcome_bounds=RATE_BOUNDS)
    assert fit.converged
    assert fit.nu > 5.0, f"nu came back at {fit.nu:g}, which is the collapsed value"
    assert fit.branch == "nu>1"


def test_the_two_models_stop_having_the_same_residual() -> None:
    """Before the repair the Richards and logistic rmse agreed to five decimals on this input.

    That agreement is the mechanism: with the ceiling free both models degenerate to the same
    locally exponential shape, so the extra parameter buys nothing and the likelihood ratio
    collapses. Refitting the null inside the same bounds is what separates them.
    """
    y = design_curve(BUDGET, 0.5)
    fit = fit_richards(y, BUDGET, outcome_bounds=RATE_BOUNDS)
    assert fit.logistic_rmse > 1.4 * fit.rmse, (
        f"richards {fit.rmse:g} against logistic {fit.logistic_rmse:g}"
    )
    assert fit.lrt_p < 1e-10


def test_span_observed_says_how_much_of_the_rise_was_actually_seen() -> None:
    """The truncation is on the record as a number rather than left to be inferred."""
    truncated = fit_richards(design_curve(BUDGET, 0.5), BUDGET, outcome_bounds=RATE_BOUNDS)
    long_steps = np.arange(0.0, 1200.0, 1.0)
    complete = fit_richards(design_curve(long_steps, 0.999), long_steps, outcome_bounds=RATE_BOUNDS)
    assert truncated.span_observed < 0.7
    assert complete.span_observed > 0.95
    assert truncated.span_observed < complete.span_observed


# ---------------------------------------------------------------------------
# The bound is the repair, so the absence of one has to be visible
# ---------------------------------------------------------------------------


def test_the_bound_has_no_default() -> None:
    """A default would be the defect. The caller states the observable's range or gets nothing."""
    y = design_curve(BUDGET, 0.5)
    with pytest.raises(TypeError):
        fit_richards(y, BUDGET)  # type: ignore[call-arg]


def test_declaring_the_observable_unbounded_reproduces_the_runaway() -> None:
    """The old behaviour is still reachable, and now only by asking for it in writing.

    This is the fail case for the repair: it shows the bound is what is doing the work, on the same
    input, in the same call, with one argument changed.
    """
    y = design_curve(BUDGET, 0.5)
    free = fit_richards(y, BUDGET, outcome_bounds=(float("-inf"), float("inf")))
    bounded = fit_richards(y, BUDGET, outcome_bounds=RATE_BOUNDS)

    assert free.asymptote > 1e4, f"unbounded ceiling came back at {free.asymptote:g}"
    assert free.nu < 2.0
    assert free.lrt_p > 0.5
    assert bounded.asymptote <= 1.0
    assert bounded.lrt_p < 1e-10


def test_the_declared_range_is_checked_against_the_data() -> None:
    """A series that leaves the range it was declared to live in is a refusal, not a fit."""
    y = design_curve(BUDGET, 0.5)
    fit = fit_richards(y, BUDGET, outcome_bounds=(0.2, 1.0))
    assert not fit.converged
    assert fit.method == "richards:outside-declared-bounds"
    assert fit.outcome_bounds == (0.2, 1.0)


def test_an_inverted_range_is_rejected_outright() -> None:
    with pytest.raises(ValueError, match="increasing pair"):
        fit_richards(design_curve(BUDGET, 0.5), BUDGET, outcome_bounds=(1.0, 0.0))


def test_the_bounds_travel_on_every_fit() -> None:
    """A ceiling means nothing without the range it was fitted inside."""
    fit = fit_richards(design_curve(BUDGET, 0.5), BUDGET, outcome_bounds=RATE_BOUNDS)
    assert fit.outcome_bounds == RATE_BOUNDS


# ---------------------------------------------------------------------------
# The repair must not have bought the truncated case with the ordinary one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nu_true", [0.4, 1.0, 2.5, 4.0])
def test_a_planted_shape_is_still_recovered_when_the_curve_completes(nu_true: float) -> None:
    """The bound must not move a fit the data already determined."""
    steps = np.arange(0.0, 241.0, 1.0)
    y = richards_curve(steps, 0.01, 0.9, 0.04, 120.0, nu_true)
    fit = fit_richards(y, steps, outcome_bounds=RATE_BOUNDS)
    assert fit.nu == pytest.approx(nu_true, abs=0.01)
    assert fit.asymptote == pytest.approx(0.9, abs=0.01)


@pytest.mark.parametrize("nu_true", [0.4, 3.0])
def test_a_planted_shape_is_recovered_under_truncation_too(nu_true: float) -> None:
    """Where the baseline lost the shape entirely.

    At `nu = 0.4` with the midpoint past the end of the window the baseline returned `K = 0.104`
    and `nu = 1`, reporting a symmetric curve where the planted one is strongly asymmetric.
    """
    steps = np.arange(0.0, 241.0, 1.0)
    y = richards_curve(steps, 0.01, 0.9, 0.04, 300.0, nu_true)
    fit = fit_richards(y, steps, outcome_bounds=RATE_BOUNDS)
    assert fit.nu == pytest.approx(nu_true, abs=0.05)
    assert fit.asymptote == pytest.approx(0.9, abs=0.02)


def test_a_noisy_symmetric_series_is_not_called_asymmetric() -> None:
    """The instrument has to be able to return both answers, or it is not a test."""
    steps = np.arange(0.0, 241.0, 1.0)
    rng = np.random.default_rng(11)
    y = np.clip(
        richards_curve(steps, 0.01, 0.9, 0.04, 120.0, 1.0) + rng.standard_normal(steps.size) * 0.01,
        0.0,
        1.0,
    )
    fit = fit_richards(y, steps, outcome_bounds=RATE_BOUNDS)
    assert fit.converged
    assert not fit.asymmetric_at(0.05), f"p = {fit.lrt_p:g} on a symmetric series"


def test_a_noisy_asymmetric_series_is_called_asymmetric() -> None:
    steps = np.arange(0.0, 241.0, 1.0)
    rng = np.random.default_rng(11)
    y = np.clip(
        richards_curve(steps, 0.01, 0.9, 0.04, 120.0, 3.0) + rng.standard_normal(steps.size) * 0.01,
        0.0,
        1.0,
    )
    fit = fit_richards(y, steps, outcome_bounds=RATE_BOUNDS)
    assert fit.asymmetric_at(0.05), f"p = {fit.lrt_p:g} on a curve of nu = 3"
