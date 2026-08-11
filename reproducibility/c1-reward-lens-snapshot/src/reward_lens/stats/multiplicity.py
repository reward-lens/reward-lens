"""Multiple-testing correction.

``bh_fdr`` is the Benjamini-Hochberg procedure ported unchanged from v1's
``statistics.py``. ``hierarchical_fdr`` is the v3 addition the Atlas needs:
when the same battery of tests is run across many models (a battery of
batteries), a flat BH over the pooled p-values either ignores the grouping or
over-corrects. Two-level FDR corrects within each group and across groups so
that a group with one strong effect is not drowned out by a group of nulls.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


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


def _simes_p(p: np.ndarray) -> float:
    """Simes combined p-value for a family of p-values.

    ``min_i m * p_(i) / i`` over the sorted p-values. It is a valid global
    p-value for the intersection null (all hypotheses in the family true) under
    independence or positive dependence, and is the family representative used
    by the Benjamini-Bogomolov selective-inference procedure. NaNs are dropped;
    an empty family returns ``nan``.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    p = p[np.isfinite(p)]
    m = p.size
    if m == 0:
        return float("nan")
    ranked = np.sort(p)
    return float(np.min(ranked * m / np.arange(1, m + 1)))


def hierarchical_fdr(
    groups: dict[str, Sequence[float]],
    alpha: float = 0.05,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Two-level (hierarchical) FDR control across grouped families of tests.

    This is the Benjamini-Bogomolov (2014) selective-inference construction,
    which the Atlas uses when a fixed battery of Observables is run over many
    subjects: each subject (or index family) is a group, and we want to control
    the false-discovery rate both across groups and within each selected group.

    The procedure is:

      1. Represent each group by its Simes combined p-value (``_simes_p``): a
         valid test of the group's intersection null.
      2. Run BH at level ``alpha`` on the group representatives to select the
         groups that show any signal. Let ``R`` of ``m`` groups be selected.
      3. Within each selected group, run BH at the scaled level
         ``alpha * R / m``. Groups that are not selected reject nothing.

    Scaling the within-group level by ``R / m`` is what makes the two levels
    compose: it charges each selected group only its share of the global error
    budget, so the average FDR over the selected groups is controlled at
    ``alpha``. A group examined in isolation (``m == 1``) reduces to a plain BH
    at ``alpha``.

    Args:
        groups: Mapping from group label to that group's raw p-values.
        alpha: Target false discovery rate at both levels.

    Returns:
        Mapping from group label to ``(rejected, q_values)`` for that group,
        with the same NaN-passthrough semantics as ``bh_fdr``. The within-group
        q-values are the group's own BH q-values (independent of selection); the
        ``rejected`` flags additionally require the group to have been selected
        and use the scaled level.
    """
    keys = list(groups)
    m = len(keys)
    if m == 0:
        return {}

    rep = np.array([_simes_p(np.asarray(groups[k], dtype=np.float64)) for k in keys])
    group_rejected, _ = bh_fdr(rep, alpha)
    n_selected = int(np.sum(group_rejected))
    inner_alpha = alpha * n_selected / m if n_selected > 0 else 0.0

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i, k in enumerate(keys):
        p = np.asarray(groups[k], dtype=np.float64).ravel()
        level = inner_alpha if group_rejected[i] else 0.0
        out[k] = bh_fdr(p, level)
    return out


__all__ = [
    "bh_fdr",
    "hierarchical_fdr",
]
