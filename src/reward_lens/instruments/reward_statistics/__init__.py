"""Reward composition, observed call cost, and observed call latency (section 7.3)."""

from __future__ import annotations

from reward_lens.instruments.base import Panel

from ._recorded import PriceSource, RecordedCompletion, TraceReview
from .latency import LatencyPercentiles
from .price import PriceSourceError, RewardPrice
from .semantics import RewardSemantics, findings_for

__all__ = [
    "LatencyPercentiles",
    "PANEL",
    "PriceSource",
    "PriceSourceError",
    "RecordedCompletion",
    "RewardPrice",
    "RewardSemantics",
    "TraceReview",
    "findings_for",
]

PANEL = Panel(
    section="reward_statistics",
    instruments=(RewardSemantics(), RewardPrice(), LatencyPercentiles()),
)
