"""Loader for the v2 diagnostic preference set."""
from __future__ import annotations

from typing import Optional

from .datasets import PreferencePair


def load_diagnostic_v2(dimensions: Optional[list[str]] = None,
                       limit_per_dim: Optional[int] = None) -> list[PreferencePair]:
    """Load diagnostic_data_v2 as a list of PreferencePair, optionally
    filtering by dimension and/or capping per-dimension."""
    from reward_lens.diagnostic_data_v2 import ALL_DIMENSIONS_V2, get_pairs_v2
    dims = dimensions or list(ALL_DIMENSIONS_V2.keys())
    out: list[PreferencePair] = []
    for d in dims:
        pairs = get_pairs_v2([d])
        if limit_per_dim is not None:
            pairs = pairs[:limit_per_dim]
        for i, p in enumerate(pairs):
            out.append(PreferencePair(
                prompt=p.prompt,
                preferred=p.preferred,
                dispreferred=p.dispreferred,
                dimension=p.dimension,
                source="diagnostic_v2",
                pair_id=f"dv2-{d}-{i}",
                metadata={"description": p.description},
            ))
    return out
