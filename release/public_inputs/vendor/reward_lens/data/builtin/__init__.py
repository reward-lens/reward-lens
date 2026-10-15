"""Versioned builtin datasets that ship in the package wheel (section 2.4.2).

These are the small, human-authored (and, where marked, mechanically authored) seed sets that seed
the data plane: the v1 diagnostic triples imported with honest lineage, plus the two new dimensions
authored for v3. Importing this package registers their dataset cards so ``registry.load_dataset``
can serve them by name. Everything here is torch-free and cheap to import.
"""

from __future__ import annotations

from reward_lens.data.builtin.diagnostic_v3 import (
    ALL_DIMENSIONS_V3,
    all_pairs,
    load_diagnostic_v3,
    matched_prompt_views,
)

__all__ = [
    "ALL_DIMENSIONS_V3",
    "load_diagnostic_v3",
    "all_pairs",
    "matched_prompt_views",
]
