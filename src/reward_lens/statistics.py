"""Compatibility shim over :mod:`reward_lens.stats` (design rule R15).

v1 shipped its population statistics here, in one flat module. v3 splits them
across ``reward_lens.stats`` (effect sizes and bootstrap in ``stats.effects``,
FDR in ``stats.multiplicity``) so the epistemics engine has one canonical
implementation. This module keeps the v1 import path working by re-exporting
exactly the v1 public names from their new homes, so existing code and the v1
test suite keep passing unchanged while nothing is implemented twice.

Prefer importing from ``reward_lens.stats`` in new code; this shim exists for
source compatibility until the E-parity suite has passed twice, then it is
deprecated.
"""

from __future__ import annotations

from reward_lens.stats.effects import (
    BootstrapResult,
    bootstrap_ci,
    bootstrap_cohens_d,
    cohens_d,
    paired_permutation_test,
    spearman_with_ci,
)
from reward_lens.stats.multiplicity import bh_fdr

__all__ = [
    "BootstrapResult",
    "cohens_d",
    "bootstrap_ci",
    "bootstrap_cohens_d",
    "paired_permutation_test",
    "bh_fdr",
    "spearman_with_ci",
]
