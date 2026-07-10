"""
Thin re-export of the library-level statistics module.

The experiment layout calls for ``experiments/utils/statistics.py``. All
statistical primitives live in :mod:`reward_lens.statistics`; this module
re-exports them so experiment code can do either:

    from experiments.utils.statistics import bootstrap_ci
    from reward_lens.statistics import bootstrap_ci

Both work; the library module is the single source of truth.
"""

from reward_lens.statistics import (  # noqa: F401
    BootstrapResult,
    bootstrap_ci,
    bootstrap_cohens_d,
    bh_fdr,
    cohens_d,
    paired_permutation_test,
    spearman_with_ci,
)

__all__ = [
    "BootstrapResult",
    "bootstrap_ci",
    "bootstrap_cohens_d",
    "bh_fdr",
    "cohens_d",
    "paired_permutation_test",
    "spearman_with_ci",
]
