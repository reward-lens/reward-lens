"""``reward_lens.attribution`` — reward decomposition (Direct Linear Attribution, section 2.8.2).

This package holds two layers. :mod:`reward_lens.attribution.dla` is the canonical, substrate-free
implementation of the head- and component-level reward decomposition that the v3 battery calls.
:mod:`reward_lens.attribution.component` is the v1 ``ComponentAttribution`` primitive, kept alive as
the E-parity reference (R15) and re-exported here so ``from reward_lens.attribution import
ComponentAttribution`` keeps working exactly as before. The v1 head decomposition now delegates to
the canonical implementation, so there is a single source of truth for the head math.
"""

from __future__ import annotations

from reward_lens.attribution.component import (
    ComponentAttribution,
    ComponentResult,
    _batch_head_attribution,
)
from reward_lens.attribution.dla import (
    component_reward_contributions,
    head_reward_contributions,
    project_onto_reward,
)

__all__ = [
    # v1 primitive (E-parity reference)
    "ComponentAttribution",
    "ComponentResult",
    "_batch_head_attribution",
    # canonical DLA (what the battery uses)
    "project_onto_reward",
    "head_reward_contributions",
    "component_reward_contributions",
]
