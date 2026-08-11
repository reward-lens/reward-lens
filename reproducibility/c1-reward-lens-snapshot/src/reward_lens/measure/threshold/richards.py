"""An asymmetric transition fit alongside the symmetric one, for `C20` (BLK-030, `CHAIN_GAPS` G19d).

`fit_transition` next door fits a four-parameter logistic, which is symmetric by construction: its
rise and its finish are mirror images and no fitted parameter can say otherwise. §2.4 records why
that matters, and it is not a fussy point:

    on the published run the symmetric fit predicts the first step above a 0.05 hack rate at
    106 - ln(19)/0.1838 = 90.0, and the observed value is 90, which is agreement that carries no
    information about shape.

`C20` registers a likelihood-ratio test of a Richards curve against that logistic, and it is the
design's one genuinely falsifiable prediction about the shape of a curve nobody has explained. There
was no estimator: a grep for "richards" over the package returned nothing.

**This module is written against an ASSUMED parameterisation and the row it belongs to is OPEN.**
BLK-120 owns the parameterisation, the sign convention, the error model and the fitting method, and
has not landed. What is assumed, and why, is written down in full in
`chain/repair/proofs/BLK-030/ASSUMED-PARAMETERISATION.md`. The short version is in
`RichardsFit.parameterisation`, on every fit this module returns, because BLK-120's own evidence is
that the mapping from the shape parameter to a skew direction flips between the common
parameterisations, and a fitted shape parameter with no family attached is not a number anybody can
read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2

from reward_lens.core.evidence import register_payload
from reward_lens.measure.threshold.variance import fit_transition

#: The assumed family, written out so it travels with every fit. At `nu = 1` this is exactly the
#: `_logistic` next door, which is the property that makes `C20`'s likelihood-ratio test well posed:
#: the two models are nested and the test has one degree of freedom. Asserted in the tests against
#: the shipped `_logistic` rather than against a second copy.
PARAMETERISATION = "A + (K-A) / (1 + nu*exp(-r(t-tm)))^(1/nu)"

#: Free parameters: baseline, asymptote, rate, midpoint, shape.
N_PARAMETERS = 5


def richards_curve(
    t: np.ndarray,
    baseline: float,
    asymptote: float,
    rate: float,
    midpoint: float,
    nu: float,
) -> np.ndarray:
    """The assumed Richards family. `nu = 1` is the logistic, exactly.

    `nu > 1` is a slow rise with a sharp finish and `nu < 1` is the reverse, **in this
    parameterisation**. That direction is measured off the curve in the tests rather than asserted,
    because it is the thing BLK-120 records as flipping between the common forms.

    `nu` is strictly positive and therefore has no sign, which is why `C20`'s registered rule that
    "the shape parameter's sign matches the prediction" cannot be evaluated as written and why
    BLK-120's remedy is to state the branch as an inequality.
    """
    span = asymptote - baseline
    exponent = np.clip(-rate * (np.asarray(t, dtype=np.float64) - midpoint), -700.0, 700.0)
    inner = 1.0 + nu * np.exp(exponent)
    # `inner ** (1 / nu)` overflows for a small `nu` on a long tail, and the optimiser walks there
    # on a flat series. Taken through logs instead, where the same quantity is representable and the
    # result underflows to the asymptote rather than becoming an inf that poisons the residual.
    return baseline + span * np.exp(-np.clip(np.log(inner) / nu, -700.0, 700.0))


@register_payload
@dataclass(frozen=True)
class RichardsFit:
    """An asymmetric transition fit and its likelihood ratio against the symmetric one.

    ``nu`` is the shape parameter in `PARAMETERISATION` and ``branch`` is which side of 1 it fell,
    as an inequality rather than as a sign. **The parameterisation is assumed, not registered:
    BLK-120 owns it and has not landed**, so ``parameterisation`` travels on every fit and a reader
    who does not check it against the registered one is reading a number whose meaning is open.

    ``lrt_p`` is `C20`'s statistic: `-2 log Lambda` against the logistic, referred to a chi-squared
    on one degree of freedom because the logistic is the `nu = 1` restriction of this family rather
    than a separate model. ``loglik_richards`` and ``loglik_logistic`` are the two Gaussian
    log-likelihoods behind it, which are the two fields `C20` names.

    ``n_in_rise`` is how many read points fell between the 10 percent and 90 percent levels of the
    observed range. When it is below ``n_parameters`` the fit refuses: five parameters through two
    points is not an underdetermined fit but an arbitrary one, and the optimiser returns from it
    without complaint. That refusal is the second half of this row's closure proof and it is a real
    risk here rather than a theoretical one, because this run reads at a checkpoint stride of 11.

    ``span_observed`` is the fraction of the fitted rise the data actually cover,
    `(max(y) - baseline) / (asymptote - baseline)`. It is a diagnostic and it carries no threshold,
    because none is registered. What it exists to make visible is truncation: a 240-step budget ends
    before the transition finishes, and on a truncated series the ceiling is not identifiable from
    the data at all. Read it beside ``nu``: a shape parameter fitted where ``span_observed`` is small
    is a shape parameter fitted mostly on extrapolation (BLK-030-a).

    ``outcome_bounds`` records the admissible range the caller declared for the observable, which is
    the only thing that makes the ceiling identifiable on a truncated series. With it left free, the
    fit ran the asymptote to `1.18e5` on a rate, both models degenerated to the same locally
    exponential shape, and `C20` returned `p = 0.97` on noiseless data that was its own prediction.
    """

    nu: float
    baseline: float
    asymptote: float
    rate: float
    midpoint: float
    rmse: float
    logistic_rmse: float
    loglik_richards: float
    loglik_logistic: float
    lrt_statistic: float
    lrt_p: float
    lrt_dof: int
    n: int
    n_in_rise: int
    n_parameters: int
    method: str
    span_observed: float = float("nan")
    outcome_bounds: tuple[float, float] = (float("-inf"), float("inf"))
    parameterisation: str = PARAMETERISATION

    @property
    def converged(self) -> bool:
        return bool(np.isfinite(self.nu) and self.nu > 0)

    @property
    def branch(self) -> str:
        """Which side of 1 the shape parameter fell, as an inequality.

        BLK-120: "the predicted branch stated as `nu > 1` rather than as a sign". A2_ARITHMETIC's
        F3.4 puts the design's prediction on `nu > 1`, the slow rise with the sharp finish.
        """
        if not self.converged:
            return "undetermined"
        if abs(self.nu - 1.0) <= 1e-6:
            return "nu=1"
        return "nu>1" if self.nu > 1.0 else "nu<1"

    def asymmetric_at(self, alpha: float = 0.05) -> bool:
        """`C20`'s first condition: the likelihood ratio rejects the logistic at `alpha`."""
        return bool(self.converged and np.isfinite(self.lrt_p) and self.lrt_p < alpha)

    def render(self) -> str:
        if not self.converged:
            return (
                f"Richards fit refused ({self.method}); {self.n_in_rise} read points inside the "
                f"rise against {self.n_parameters} free parameters. Parameterisation assumed, "
                f"BLK-120 open"
            )
        return (
            f"Richards shape nu = {self.nu:.4g} ({self.branch}), midpoint {self.midpoint:.1f}, "
            f"rate {self.rate:.4g}, rmse {self.rmse:.5g}; likelihood ratio against the symmetric "
            f"logistic p = {self.lrt_p:.4g} on {self.lrt_dof} d.f. Parameterisation ASSUMED "
            f"({self.parameterisation}); BLK-120 open"
        )


def _refused(
    method: str,
    *,
    n: int,
    n_in_rise: int,
    outcome_bounds: tuple[float, float] = (float("-inf"), float("inf")),
) -> RichardsFit:
    nan = float("nan")
    return RichardsFit(
        nu=nan,
        baseline=nan,
        asymptote=nan,
        rate=nan,
        midpoint=nan,
        rmse=nan,
        logistic_rmse=nan,
        loglik_richards=nan,
        loglik_logistic=nan,
        lrt_statistic=nan,
        lrt_p=nan,
        lrt_dof=1,
        n=n,
        n_in_rise=n_in_rise,
        n_parameters=N_PARAMETERS,
        method=method,
        span_observed=nan,
        outcome_bounds=outcome_bounds,
    )


def _logistic_in_family(
    t: np.ndarray, baseline: float, asymptote: float, rate: float, midpoint: float
) -> np.ndarray:
    """The `nu = 1` member of `PARAMETERISATION`, which is the shipped `_logistic` exactly.

    The null is fitted here rather than taken from `fit_transition` so that both models see the same
    admissible range for the ceiling. Fitting the alternative under a bound and the null without one
    is not a nested comparison: on a truncated series the unbounded null wins on residual, the
    statistic floors at zero, and the test reports `p = 1` whatever the shape is.
    """
    return richards_curve(t, baseline, asymptote, rate, midpoint, 1.0)


def _gaussian_loglik(rss: float, n: int) -> float:
    """The Gaussian log-likelihood at the maximum-likelihood variance, which is `rss / n`."""
    if not (rss > 0) or n <= 0:
        return float("nan")
    return -0.5 * n * (math.log(2.0 * math.pi) + math.log(rss / n) + 1.0)


def fit_richards(
    outcome: Sequence[float] | np.ndarray,
    steps: Sequence[float] | np.ndarray | None = None,
    *,
    outcome_bounds: tuple[float, float],
) -> RichardsFit:
    """Fit the assumed Richards family and test it against the symmetric logistic inside it.

    Started from the symmetric fit `fit_transition` returns, with `nu = 1`, so the optimiser begins
    at the null model rather than at a guess. That is both the best-conditioned start available and
    the conservative one: a search that begins at `nu = 1` and stays there has found no asymmetry,
    where a search begun elsewhere would have to be argued not to have imported one.

    ``outcome_bounds`` is the range the observable can take, `(lo, hi)`, and it is **required with no
    default**. The baseline and the asymptote are both fitted inside it. A default would be the
    defect: with the ceiling free, a 240-step budget produces a truncated transition on which a
    Richards curve and a symmetric logistic both degenerate to the same locally exponential shape,
    the fit runs the asymptote to `1.18e5` on a series bounded by 1, the residuals agree to five
    decimals, and `C20` returns `p = 0.97` on noiseless data that is its own prediction (BLK-030-a).
    For a rate the bounds are `(0.0, 1.0)`. A caller who genuinely has an unbounded observable passes
    `(-inf, inf)` and gets the old behaviour, having chosen it.

    The null is refitted here as the `nu = 1` member of the same family under the same bounds, rather
    than read off `fit_transition`, so the likelihood ratio compares two models with the same
    admissible ceiling. `logistic_rmse` is that fit's. On a series whose logistic fit lands inside the
    bounds anyway the two agree, which is asserted in the tests against the shipped fitter.

    Refuses, with `converged` False and `nu` NaN, when the rise holds fewer read points than the fit
    has free parameters, and when the data fall outside the range they were declared to live in.
    This run's checkpoint stride is 11, so the first is not a corner case.
    """
    y = np.asarray(outcome, dtype=np.float64).ravel()
    t = (
        np.arange(y.size, dtype=np.float64)
        if steps is None
        else np.asarray(steps, dtype=np.float64).ravel()
    )
    keep = np.isfinite(y) & np.isfinite(t)
    y, t = y[keep], t[keep]
    n = int(y.size)

    bound_lo, bound_hi = (float(outcome_bounds[0]), float(outcome_bounds[1]))
    if not (bound_lo < bound_hi):
        raise ValueError(f"outcome_bounds must be an increasing pair; got ({bound_lo}, {bound_hi})")
    declared = (bound_lo, bound_hi)

    if n < N_PARAMETERS + 2:
        return _refused("richards:too-short", n=n, n_in_rise=0, outcome_bounds=declared)

    lo, hi = float(y.min()), float(y.max())
    # A tolerance on the declared range, relative to the observed span, so that a rate recorded as
    # 1.0000000000000002 is not a refusal while a series that genuinely leaves its declared range is.
    slack = 1e-9 * max(hi - lo, 1.0)
    if lo < bound_lo - slack or hi > bound_hi + slack:
        return _refused(
            "richards:outside-declared-bounds", n=n, n_in_rise=0, outcome_bounds=declared
        )

    span = hi - lo
    if span <= 0:
        return _refused("richards:no-range", n=n, n_in_rise=0, outcome_bounds=declared)
    inside = (y > lo + 0.1 * span) & (y < lo + 0.9 * span)
    n_in_rise = int(np.count_nonzero(inside))
    if n_in_rise < N_PARAMETERS:
        return _refused(
            "richards:too-few-in-rise", n=n, n_in_rise=n_in_rise, outcome_bounds=declared
        )

    def _inside(value: float, fallback: float) -> float:
        """Keep a starting point strictly inside the declared range, as `curve_fit` requires."""
        if not np.isfinite(value):
            value = fallback
        pad = 1e-6 * (span or 1.0)
        return float(np.clip(value, bound_lo + pad, bound_hi - pad))

    symmetric = fit_transition(y, t)
    if symmetric.converged:
        p0 = (
            _inside(float(symmetric.baseline), lo),
            _inside(float(symmetric.baseline + symmetric.amplitude), hi),
            float(symmetric.rate),
            float(symmetric.midpoint),
            1.0,
        )
    else:
        gradient = np.gradient(y, t)
        p0 = (
            _inside(lo, lo),
            _inside(hi, hi),
            4.0 / (float(t[-1] - t[0]) or 1.0),
            float(t[int(np.argmax(np.abs(gradient)))]),
            1.0,
        )

    bounds = (
        (bound_lo, bound_lo, -np.inf, -np.inf, 1e-4),
        (bound_hi, bound_hi, np.inf, np.inf, 1e4),
    )
    tolerances = {"xtol": 1e-15, "ftol": 1e-15, "gtol": 1e-15}
    try:
        popt, _ = curve_fit(
            richards_curve, t, y, p0=p0, bounds=bounds, maxfev=200_000, **tolerances
        )
    except (RuntimeError, ValueError):
        return _refused(
            "richards:no-convergence", n=n, n_in_rise=n_in_rise, outcome_bounds=declared
        )

    # The null, refitted inside the same bounds at `nu = 1`. Started twice: from the same place the
    # alternative started, and from where the alternative **finished**. The second restart is not
    # belt and braces. The null is nested inside the alternative, so its optimum cannot honestly be
    # worse; when the alternative comes back at `nu = 1` and the null was left at a slightly poorer
    # convergence point, the difference between two optimiser tolerances is reported as a likelihood
    # ratio, and on a noiseless series that difference is the whole statistic.
    logistic_rss = float("nan")
    for start in (p0[:4], tuple(float(v) for v in popt[:4])):
        try:
            popt_null, _ = curve_fit(
                _logistic_in_family,
                t,
                y,
                p0=start,
                bounds=(bounds[0][:4], bounds[1][:4]),
                maxfev=200_000,
                **tolerances,
            )
        except (RuntimeError, ValueError):
            continue
        null_resid = y - _logistic_in_family(t, *popt_null)
        candidate = float(null_resid @ null_resid)
        if np.isfinite(candidate) and not (np.isfinite(logistic_rss) and logistic_rss <= candidate):
            logistic_rss = candidate

    # The null's likelihood is the maximum over the **admissible** parameters, so where
    # `fit_transition`'s unconstrained logistic happens to land inside the declared range it is an
    # admissible candidate too and the better of the three is the null.
    if symmetric.converged:
        unconstrained_top = float(symmetric.baseline + symmetric.amplitude)
        unconstrained_bottom = float(symmetric.baseline)
        if (
            bound_lo - slack <= unconstrained_bottom <= bound_hi + slack
            and bound_lo - slack <= unconstrained_top <= bound_hi + slack
        ):
            candidate = float(symmetric.rmse**2 * n)
            if np.isfinite(candidate) and not (
                np.isfinite(logistic_rss) and logistic_rss <= candidate
            ):
                logistic_rss = candidate

    baseline, asymptote, rate, midpoint, nu = (float(v) for v in popt)
    resid = y - richards_curve(t, *popt)
    rss = float(resid @ resid)
    fitted_span = asymptote - baseline
    span_observed = float((hi - baseline) / fitted_span) if fitted_span != 0 else float("nan")

    if not np.isfinite(logistic_rss):
        return _refused(
            "richards:no-symmetric-null", n=n, n_in_rise=n_in_rise, outcome_bounds=declared
        )

    loglik_r = _gaussian_loglik(rss, n)
    loglik_l = _gaussian_loglik(logistic_rss, n)
    if np.isfinite(loglik_r) and np.isfinite(loglik_l):
        # The Richards family contains the logistic, so its residual cannot honestly be the larger
        # one. When the optimiser leaves it larger the extra parameter has bought nothing and the
        # statistic is floored at zero rather than allowed to go negative.
        statistic = max(0.0, 2.0 * (loglik_r - loglik_l))
        p_value = float(chi2.sf(statistic, 1))
    else:
        statistic = float("nan")
        p_value = float("nan")

    return RichardsFit(
        nu=nu,
        baseline=baseline,
        asymptote=asymptote,
        rate=rate,
        midpoint=midpoint,
        rmse=float(np.sqrt(rss / n)),
        logistic_rmse=float(np.sqrt(logistic_rss / n)),
        loglik_richards=loglik_r,
        loglik_logistic=loglik_l,
        lrt_statistic=float(statistic),
        lrt_p=p_value,
        lrt_dof=1,
        n=n,
        n_in_rise=n_in_rise,
        n_parameters=N_PARAMETERS,
        method="richards-lrt",
        span_observed=span_observed,
        outcome_bounds=declared,
    )


__all__ = [
    "N_PARAMETERS",
    "PARAMETERISATION",
    "RichardsFit",
    "fit_richards",
    "richards_curve",
]
