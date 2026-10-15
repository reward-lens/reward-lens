"""The six rungs of compare, with lazy public scorecard re-exports.

The package is consulted by the root-help availability probe.  Importing
scorecard there imports contracts and pydantic, so the old eager re-exports
made a source-only capability check violate D-30.  The public names remain
identical; they load only when a consumer actually asks for one.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ROWS",
    "Absence",
    "CapabilityFloor",
    "Estimate",
    "Scorecard",
    "ScorecardRow",
    "Tradeoff",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import scorecard

    value = getattr(scorecard, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
