"""
Statistics utilities for reward-lens.

The v1 paper's reviewer criticism was that aggregate stats were either
single-pair (Spearman over 64 components on n=1) or produced numerically
degenerate effect sizes (Cohen's d = inf when std == 0 with n=2). This
module is the response: every aggregate stat ships with a bootstrap CI,
multi-test corrections are first-class, and degenerate inputs return NaN
with a documented reason rather than ``inf``.

All routines are pure-numpy and thread-safe. Heavy ops use a default of
10 000 resamples; bump for tighter CIs at the cost of runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

# =============================================================================
# Effect sizes
# =============================================================================


def cohens_d(
    a: Sequence[float] | np.ndarray,
    b: Optional[Sequence[float] | np.ndarray] = None,
    paired: bool = False,
) -> float:
    """Cohen's d effect size.

    Three modes:
      - ``b is None``: one-sample d (mean(a) / std(a, ddof=1)). Useful for
        paired-difference data where you've already taken `a - b` yourself.
      - ``paired=True``: paired-samples d_z = mean(a-b) / std(a-b, ddof=1).
      - ``paired=False, b is not None``: independent-samples d with pooled SD.

    Returns ``nan`` (not ``inf``) when n < 2 or the relevant std is zero.
    The v1 hacking detector returned ``inf`` in those cases — that's the
    bug this function exists to fix.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    if b is None:
        if a.size < 2:
            return float("nan")
        sd = a.std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            return float("nan")
        return float(a.mean() / sd)

    b = np.asarray(b, dtype=np.float64).ravel()
    if paired:
        if a.size != b.size:
            raise ValueError(f"paired=True requires equal lengths; got {a.size} vs {b.size}")
        diff = a - b
        if diff.size < 2:
            return float("nan")
        sd = diff.std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            return float("nan")
        return float(diff.mean() / sd)

    if a.size < 2 or b.size < 2:
        return float("nan")
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    pooled = ((a.size - 1) * var_a + (b.size - 1) * var_b) / (a.size + b.size - 2)
    if not np.isfinite(pooled) or pooled <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(pooled))


# =============================================================================
# Bootstrap
# =============================================================================


@dataclass
class BootstrapResult:
    """A point estimate together with its bootstrap CI.

    Attributes:
        point: The statistic computed on the observed sample.
        ci_low: Lower CI bound at the requested confidence level.
        ci_high: Upper CI bound.
        ci_level: Confidence level used (e.g. 0.95).
        n_resamples: How many bootstrap resamples were drawn.
    """

    point: float
    ci_low: float
    ci_high: float
    ci_level: float
    n_resamples: int

    def as_tuple(self) -> tuple[float, float, float]:
        return self.point, self.ci_low, self.ci_high


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> BootstrapResult:
    """Percentile bootstrap CI for an arbitrary scalar statistic.

    Args:
        values: 1-D array of observations.
        statistic: Callable taking a numpy array and returning a scalar.
        n_resamples: Number of bootstrap resamples.
        ci: Confidence level (e.g. 0.95).
        seed: RNG seed for reproducibility.

    Returns:
        BootstrapResult with point estimate and percentile CI bounds.

    Notes:
        Uses the percentile method, not BCa. The percentile method is biased
        for skewed distributions but is closed-form and adequate for the
        moderate-skew distributions we encounter (per-pair Spearman ρ,
        per-pair patch effects). If you need BCa, refactor here — every
        downstream caller goes through this function.

        Returns NaN bounds (not raise) for n < 2; the caller decides whether
        to drop the cell or surface it.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    n = arr.size
    point = float(statistic(arr)) if n > 0 else float("nan")
    if n < 2:
        return BootstrapResult(point, float("nan"), float("nan"), ci, 0)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    samples = arr[idx]
    # Vectorise where possible; fall back to a Python loop for non-vectorisable stats.
    try:
        replicates = np.asarray(statistic(samples.T)).ravel() if statistic is np.mean else None
    except Exception:
        replicates = None
    if replicates is None or replicates.size != n_resamples:
        replicates = np.empty(n_resamples, dtype=np.float64)
        for i in range(n_resamples):
            replicates[i] = float(statistic(samples[i]))

    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(replicates, alpha))
    hi = float(np.quantile(replicates, 1.0 - alpha))
    return BootstrapResult(point, lo, hi, ci, n_resamples)


def bootstrap_cohens_d(
    a: Sequence[float] | np.ndarray,
    b: Optional[Sequence[float] | np.ndarray] = None,
    paired: bool = False,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> BootstrapResult:
    """Bootstrap CI for Cohen's d.

    Resamples observations (or paired observations) with replacement,
    recomputes d on each resample, and returns the percentile CI.

    For paired data, resamples (a_i, b_i) jointly to preserve pair structure.
    For independent samples, resamples a and b independently.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    point = cohens_d(a, b, paired=paired)
    n_a = a.size

    if (n_a < 2) or (b is not None and np.asarray(b).size < 2):
        return BootstrapResult(point, float("nan"), float("nan"), ci, 0)

    rng = np.random.default_rng(seed)
    replicates = np.empty(n_resamples, dtype=np.float64)

    if b is None:
        idx = rng.integers(0, n_a, size=(n_resamples, n_a))
        for i in range(n_resamples):
            replicates[i] = cohens_d(a[idx[i]], None, paired=False)
    else:
        b = np.asarray(b, dtype=np.float64).ravel()
        if paired:
            if a.size != b.size:
                raise ValueError("paired=True requires equal lengths")
            idx = rng.integers(0, n_a, size=(n_resamples, n_a))
            for i in range(n_resamples):
                replicates[i] = cohens_d(a[idx[i]], b[idx[i]], paired=True)
        else:
            n_b = b.size
            idx_a = rng.integers(0, n_a, size=(n_resamples, n_a))
            idx_b = rng.integers(0, n_b, size=(n_resamples, n_b))
            for i in range(n_resamples):
                replicates[i] = cohens_d(a[idx_a[i]], b[idx_b[i]], paired=False)

    finite = replicates[np.isfinite(replicates)]
    if finite.size < 10:
        return BootstrapResult(point, float("nan"), float("nan"), ci, n_resamples)
    alpha = (1.0 - ci) / 2.0
    return BootstrapResult(
        point,
        float(np.quantile(finite, alpha)),
        float(np.quantile(finite, 1.0 - alpha)),
        ci,
        n_resamples,
    )


# =============================================================================
# Permutation test
# =============================================================================


def paired_permutation_test(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    n_permutations: int = 10_000,
    statistic: str = "mean_diff",
    alternative: str = "two-sided",
    seed: Optional[int] = None,
) -> float:
    """Two-sided paired permutation test.

    For each permutation, randomly flip the sign of (a_i - b_i) and
    recompute the statistic. The p-value is the fraction of permutations
    whose statistic is at least as extreme as the observed one.

    Args:
        a: First set of paired observations.
        b: Second set of paired observations, the same length as ``a``.
        n_permutations: Number of sign-flip permutations.
        statistic: One of {"mean_diff", "median_diff"}.
        alternative: One of {"two-sided", "greater", "less"}.
        seed: RNG seed.

    Returns:
        p-value. Returns 1.0 when n < 2 (no power).
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError(f"paired test requires equal lengths; got {a.size} vs {b.size}")
    n = a.size
    if n < 2:
        return 1.0

    diff = a - b
    if statistic == "mean_diff":
        stat_fn = np.mean
    elif statistic == "median_diff":
        stat_fn = np.median
    else:
        raise ValueError(f"unknown statistic: {statistic}")

    observed = float(stat_fn(diff))
    rng = np.random.default_rng(seed)
    # Sign flips: each permutation multiplies each diff by ±1
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
    permuted = signs * diff[None, :]
    if statistic == "mean_diff":
        replicates = permuted.mean(axis=1)
    else:
        replicates = np.median(permuted, axis=1)

    # +1 in numerator/denominator → unbiased p-value (Phipson & Smyth 2010)
    if alternative == "two-sided":
        count = int(np.sum(np.abs(replicates) >= abs(observed)))
    elif alternative == "greater":
        count = int(np.sum(replicates >= observed))
    elif alternative == "less":
        count = int(np.sum(replicates <= observed))
    else:
        raise ValueError(f"unknown alternative: {alternative}")
    return (count + 1) / (n_permutations + 1)


# =============================================================================
# Multiple-testing correction
# =============================================================================


def bh_fdr(
    p_values: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini–Hochberg FDR correction.

    Args:
        p_values: 1-D array of raw p-values.
        alpha: Target false discovery rate.

    Returns:
        Tuple of (rejected, q_values), both shape (n,):
            - rejected[i] = True iff hypothesis i is rejected at FDR=alpha.
            - q_values[i] = BH-adjusted p-value for hypothesis i (monotone).

    Notes:
        NaN p-values are passed through to NaN q-values and are never
        rejected. This is the right behaviour for "test was numerically
        undefined" — don't claim a discovery for it, don't penalise the
        rest of the family for it.
    """
    p = np.asarray(p_values, dtype=np.float64).ravel()
    n = p.size
    rejected = np.zeros(n, dtype=bool)
    q = np.full(n, np.nan, dtype=np.float64)

    finite_mask = np.isfinite(p)
    finite_idx = np.where(finite_mask)[0]
    if finite_idx.size == 0:
        return rejected, q

    p_finite = p[finite_idx]
    m = p_finite.size
    order = np.argsort(p_finite)
    ranked = p_finite[order]
    # BH q-values: q_(i) = min over k>=i of ( m * p_(k) / k )
    raw_q = ranked * m / np.arange(1, m + 1)
    # Enforce monotonicity from the right
    monotone_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    monotone_q = np.clip(monotone_q, 0.0, 1.0)

    # Map back to original order
    q_finite = np.empty(m, dtype=np.float64)
    q_finite[order] = monotone_q
    q[finite_idx] = q_finite
    rejected[finite_idx] = q_finite <= alpha
    return rejected, q


# =============================================================================
# Spearman with CI
# =============================================================================


def spearman_with_ci(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> BootstrapResult:
    """Spearman rank correlation with bootstrap CI.

    Pair-bootstrap (resample (x_i, y_i) jointly, preserving pairing).
    The point estimate uses scipy if available, else a numpy implementation.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError(f"length mismatch: {x.size} vs {y.size}")
    n = x.size
    if n < 3:
        return BootstrapResult(float("nan"), float("nan"), float("nan"), ci, 0)

    def _spearman(xs: np.ndarray, ys: np.ndarray) -> float:
        # Average-rank Spearman; ties handled by averaging.
        rx = _rankdata(xs)
        ry = _rankdata(ys)
        rx = rx - rx.mean()
        ry = ry - ry.mean()
        denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
        if denom == 0:
            return float("nan")
        return float((rx * ry).sum() / denom)

    point = _spearman(x, y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    replicates = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        replicates[i] = _spearman(x[idx[i]], y[idx[i]])
    finite = replicates[np.isfinite(replicates)]
    if finite.size < 10:
        return BootstrapResult(point, float("nan"), float("nan"), ci, n_resamples)
    alpha = (1.0 - ci) / 2.0
    return BootstrapResult(
        point,
        float(np.quantile(finite, alpha)),
        float(np.quantile(finite, 1.0 - alpha)),
        ci,
        n_resamples,
    )


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank of an array (ties get averaged ranks)."""
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(a.size)
    sorted_a = a[sorter]
    # Find runs of ties
    obs = np.r_[True, sorted_a[1:] != sorted_a[:-1]]
    dense = obs.cumsum()[inv]
    # dense gives ranks for unique values; convert to average ranks
    count = np.r_[np.nonzero(obs)[0], a.size]
    ranks = 0.5 * (count[dense] + count[dense - 1] + 1)
    return ranks


__all__ = [
    "BootstrapResult",
    "cohens_d",
    "bootstrap_ci",
    "bootstrap_cohens_d",
    "paired_permutation_test",
    "bh_fdr",
    "spearman_with_ci",
]
