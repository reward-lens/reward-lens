"""Lift saved Rung-3 replay witnesses into the public case evidence extension.

This module does not run a comparison.  Its caller supplies already-installed
canonical entries after subject settlement, so the entry digests refer to the
same evidence the public writer will retain.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from reward_lens import contracts

from .rung3 import R3_ENTRY_ID

CASE_EXTENSION = "https://reward-lens.github.io/case/v1"
_CANDIDATE_ENTRY_PREFIX = R3_ENTRY_ID + ".candidate."
_REQUIRED_COUNTERS = ("wall_s", "cpu_s", "usd", "api_calls")


def _entry_digest(entry: contracts.Entry) -> str:
    return contracts.digest(entry)


def _candidate_id(entry: contracts.Entry) -> str:
    candidate_id = entry.result.get("candidate_id") if entry.result else None
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError(
            f"candidate entry {entry.entry_id!r} is missing its source-bound candidate_id"
        )
    return candidate_id


def _candidate_row(entry: contracts.Entry) -> tuple[dict[str, str], dict[str, Any]]:
    if entry.entry_id == R3_ENTRY_ID or not entry.entry_id.startswith(_CANDIDATE_ENTRY_PREFIX):
        raise ValueError(f"entry {entry.entry_id!r} is not an R3 candidate replay witness")
    if entry.kind != "witness" or entry.state != "complete":
        raise ValueError(f"candidate entry {entry.entry_id!r} must be a complete witness")
    if entry.method.id != R3_ENTRY_ID or entry.method.version != "1.0.0":
        raise ValueError(f"candidate entry {entry.entry_id!r} does not use the R3 replay method")
    if entry.provenance.counters is None:
        raise ValueError(f"candidate entry {entry.entry_id!r} is missing provenance counters")

    serialized = entry.to_dict()
    counters = serialized["provenance"]["counters"]
    missing = [field for field in _REQUIRED_COUNTERS if counters.get(field) is None]
    if missing:
        raise ValueError(f"candidate entry {entry.entry_id!r} counters omit {', '.join(missing)}")

    candidate_id = _candidate_id(entry)
    reference = {
        "candidate_id": candidate_id,
        "entry_id": entry.entry_id,
        "digest": _entry_digest(entry),
    }
    return reference, {
        **reference,
        "wall_s": counters["wall_s"],
        "cpu_s": counters["cpu_s"],
        "usd": counters["usd"],
        "api_calls": counters["api_calls"],
    }


def build_case_evidence(
    decisive: contracts.Entry, candidates: Sequence[contracts.Entry]
) -> dict[str, Any]:
    """Build the v1 case extension from the decisive R3 entry and its two actual replays.

    The returned values are copied from canonical entry provenance.  This helper
    deliberately refuses aggregate record costs, estimates, missing counters,
    and anything other than the two candidate entries that the decisive replay
    itself names.
    """
    if decisive.entry_id != R3_ENTRY_ID:
        raise ValueError(f"decisive entry must be {R3_ENTRY_ID!r}")
    if decisive.kind != "estimate" or decisive.state != "complete":
        raise ValueError("decisive R3 entry must be a complete estimate")
    if decisive.method.id != R3_ENTRY_ID or decisive.method.version != "1.0.0":
        raise ValueError("decisive R3 entry does not use the R3 replay method")
    if not decisive.result:
        raise ValueError("decisive R3 entry is missing its result")
    expected_ids = [decisive.result.get("version_a"), decisive.result.get("version_b")]
    if any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in expected_ids):
        raise ValueError("decisive R3 result is missing candidate version identities")
    if not isinstance(decisive.result.get("verdict"), str):
        raise ValueError("decisive R3 result is missing its verdict")
    if len(candidates) != 2:
        raise ValueError("case evidence requires exactly two candidate replay witnesses")

    for entry in candidates:
        if entry.subject_ref != decisive.subject_ref:
            raise ValueError(
                "candidate replay witness subject_ref does not match the decisive R3 entry"
            )

    rows = [_candidate_row(entry) for entry in candidates]
    decisive_pair = [reference for reference, _ in rows]
    candidate_costs = [cost for _, cost in rows]
    actual_ids = [reference["candidate_id"] for reference in decisive_pair]
    if actual_ids != expected_ids:
        raise ValueError(
            "candidate witnesses do not match the decisive R3 version identities and order"
        )

    return {
        "scope": "local_replay_execution",
        "decision": {
            "entry_id": decisive.entry_id,
            "digest": _entry_digest(decisive),
            "verdict": decisive.result["verdict"],
        },
        "decisive_pair": decisive_pair,
        "candidate_costs": candidate_costs,
    }


__all__ = ["CASE_EXTENSION", "build_case_evidence"]
