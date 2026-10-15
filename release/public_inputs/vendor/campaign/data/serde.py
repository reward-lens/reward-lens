"""JSONL codec for reward_lens schema items, used by the slice cache.

The cache exists so a re-ingest or a later arc run does not re-download a source. A cache
file must therefore round-trip items exactly: same texts, same spans, same lineage, same
content hash, so a view rebuilt from cache has the same ``DataView.checksum()`` as the view
built from the network. Reads verify the lineage invariant (``lineage.content_hash`` equals
the hash of ``content_of(item)``) on every item, which makes a corrupted or hand-edited
cache file a loud error instead of a silently different dataset.

Only the item types the converters produce are supported: Pair, Tournament, Trajectory.
"""

from __future__ import annotations

from typing import Any

from reward_lens.core.errors import DataError
from reward_lens.core.types import Span, content_hash
from reward_lens.data.lineage import Lineage
from reward_lens.data.schema import (
    EdgeObs,
    Pair,
    Response,
    Tournament,
    TrajStep,
    Trajectory,
    content_of,
)


def _span_to_dict(span: Span) -> dict[str, Any]:
    return {"start": span.start, "end": span.end, "kind": span.kind, "meta": dict(span.meta)}


def _span_from_dict(d: dict[str, Any]) -> Span:
    return Span(start=d["start"], end=d["end"], kind=d["kind"], meta=dict(d.get("meta", {})))


def _response_to_dict(r: Response) -> dict[str, Any]:
    return {
        "text": r.text,
        "spans": [_span_to_dict(s) for s in r.spans],
        "meta": dict(r.meta),
    }


def _response_from_dict(d: dict[str, Any]) -> Response:
    return Response(
        text=d["text"],
        spans=tuple(_span_from_dict(s) for s in d.get("spans", [])),
        meta=dict(d.get("meta", {})),
    )


def _lineage_to_dict(lin: Lineage) -> dict[str, Any]:
    return {
        "seed_id": lin.seed_id,
        "builder_id": lin.builder_id,
        "ops": list(lin.ops),
        "content_hash": lin.content_hash,
    }


def _lineage_from_dict(d: dict[str, Any]) -> Lineage:
    return Lineage(
        seed_id=d["seed_id"],
        builder_id=d["builder_id"],
        ops=tuple(d["ops"]),
        content_hash=d["content_hash"],
    )


def _step_to_dict(s: TrajStep) -> dict[str, Any]:
    return {
        "action": s.action,
        "tool_call": s.tool_call,
        "text": s.text,
        "receipts": [_span_to_dict(sp) for sp in s.receipts],
        "narrative": [_span_to_dict(sp) for sp in s.narrative],
        "meta": dict(s.meta),
    }


def _step_from_dict(d: dict[str, Any]) -> TrajStep:
    return TrajStep(
        action=d["action"],
        tool_call=d.get("tool_call"),
        text=d.get("text", ""),
        receipts=tuple(_span_from_dict(sp) for sp in d.get("receipts", [])),
        narrative=tuple(_span_from_dict(sp) for sp in d.get("narrative", [])),
        meta=dict(d.get("meta", {})),
    )


def item_to_dict(item: Any) -> dict[str, Any]:
    """Serialize a schema item to a JSON-native dict."""
    if isinstance(item, Pair):
        return {
            "t": "pair",
            "prompt": item.prompt_text,
            "chosen": _response_to_dict(item.chosen),
            "rejected": _response_to_dict(item.rejected),
            "axis": item.axis,
            "lineage": _lineage_to_dict(item.lineage),
            "meta": dict(item.meta),
        }
    if isinstance(item, Tournament):
        return {
            "t": "tournament",
            "prompt": item.prompt_text,
            "responses": [_response_to_dict(r) for r in item.responses],
            "edges": [
                [e.i, e.j, e.wins_i, e.wins_j, e.annotator_id, e.judge_id] for e in item.edges
            ],
            "lineage": _lineage_to_dict(item.lineage),
            "meta": dict(item.meta),
        }
    if isinstance(item, Trajectory):
        return {
            "t": "trajectory",
            "prompt": item.prompt_text,
            "steps": [_step_to_dict(s) for s in item.steps],
            "outcome": dict(item.outcome),
            "lineage": _lineage_to_dict(item.lineage),
            "meta": dict(item.meta),
        }
    raise DataError(f"serde: unsupported item type {type(item).__name__}")


def item_from_dict(d: dict[str, Any], *, verify: bool = True) -> Any:
    """Rebuild a schema item from its dict form, verifying the lineage hash invariant."""
    kind = d.get("t")
    lineage = _lineage_from_dict(d["lineage"])
    if kind == "pair":
        item: Any = Pair(
            prompt=d["prompt"],
            chosen=_response_from_dict(d["chosen"]),
            rejected=_response_from_dict(d["rejected"]),
            axis=d["axis"],
            lineage=lineage,
            meta=dict(d.get("meta", {})),
        )
    elif kind == "tournament":
        item = Tournament(
            prompt=d["prompt"],
            responses=tuple(_response_from_dict(r) for r in d["responses"]),
            edges=tuple(
                EdgeObs(i=e[0], j=e[1], wins_i=e[2], wins_j=e[3], annotator_id=e[4], judge_id=e[5])
                for e in d["edges"]
            ),
            lineage=lineage,
            meta=dict(d.get("meta", {})),
        )
    elif kind == "trajectory":
        item = Trajectory(
            steps=tuple(_step_from_dict(s) for s in d["steps"]),
            outcome=dict(d["outcome"]),
            lineage=lineage,
            prompt=d.get("prompt", ""),
            meta=dict(d.get("meta", {})),
        )
    else:
        raise DataError(f"serde: unknown item tag {kind!r}")
    if verify:
        expected = content_hash(content_of(item), "ch")
        if expected != lineage.content_hash:
            raise DataError(
                f"serde: content hash mismatch for seed {lineage.seed_id!r}: cache carries "
                f"{lineage.content_hash} but the rebuilt item hashes to {expected}"
            )
    return item


__all__ = ["item_to_dict", "item_from_dict"]
