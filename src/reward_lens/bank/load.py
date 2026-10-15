"""Load the frozen JSONL campaign banks without rewriting their provenance."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

from reward_lens.contracts import canonical_bytes

from .types import BankProvenance, Response, ResponseBank, TaskId


def _digest(value: Any) -> str:
    # Wave 1's canonicaliser receives an Assay or mapping.  A bank is a JSONL sequence,
    # so bind the whole loaded sequence under one explicit mapping key rather than hash raw
    # line layout or silently switch serialisers.
    payload = value if isinstance(value, dict) else {"loaded_rows": value}
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _recorded_scores(raw: Any) -> dict[str, Decimal] | None:
    if not isinstance(raw, dict):
        return None
    return {str(key): Decimal(str(value)) for key, value in raw.items()}


def _response(task_id: TaskId, raw: Any) -> Response:
    if isinstance(raw, str):
        raw = {"text": raw}
    if not isinstance(raw, dict) or not isinstance(raw.get("text"), str):
        raise ValueError("a response-bank candidate needs string text")
    scores = _recorded_scores(raw.get("recorded_scores", raw.get("scores")))
    digest = _digest(
        {
            "task_id": str(task_id),
            "text": raw["text"],
            "recorded_scores": None
            if scores is None
            else {key: str(value) for key, value in sorted(scores.items())},
        }
    )
    return Response(digest=digest, task_id=task_id, text=raw["text"], recorded_scores=scores)


def _task_id(row: dict[str, Any], index: int) -> TaskId:
    explicit = row.get("task_id")
    if explicit is not None:
        return TaskId(str(explicit))
    lineage = row.get("lineage")
    if isinstance(lineage, dict) and lineage.get("seed_id") is not None:
        return TaskId(str(lineage["seed_id"]))
    return TaskId(f"row-{index}:{_digest(row)}")


def load_bank(path: Path, *, name: str) -> ResponseBank:
    """Load a finite JSONL response bank and bind its parsed rows to SHA-256 digests.

    The campaign banks encode pair candidates as ``chosen``/``rejected``.  The generic
    ``responses`` row form is supported for authored banks with more than two candidates.
    """

    source = Path(path)
    raw_bytes = source.read_bytes()
    rows = [json.loads(line) for line in raw_bytes.decode("utf-8").splitlines() if line.strip()]
    if not rows or not isinstance(rows[0], dict):
        raise ValueError("a response-bank JSONL file needs a metadata row")
    header = rows[0]
    declared_name = header.get("name")
    if declared_name is not None and declared_name != name:
        raise ValueError(f"response-bank name {name!r} does not match source {declared_name!r}")

    grouped: dict[TaskId, list[Response]] = defaultdict(list)
    payloads: dict[TaskId, Mapping[str, object]] = {}
    for index, raw_row in enumerate(rows[1:], start=1):
        if not isinstance(raw_row, dict):
            raise ValueError("a response-bank row must be an object")
        task_id = _task_id(raw_row, index)
        payloads[task_id] = {"id": str(task_id), "prompt": str(raw_row.get("prompt", ""))}
        if isinstance(raw_row.get("responses"), list):
            candidates = raw_row["responses"]
        elif "chosen" in raw_row and "rejected" in raw_row:
            candidates = [raw_row["chosen"], raw_row["rejected"]]
        else:
            raise ValueError(
                "a response-bank row needs responses or chosen and rejected candidates"
            )
        grouped[task_id].extend(_response(task_id, candidate) for candidate in candidates)

    tasks = tuple(sorted(grouped))
    provenance = BankProvenance(
        source_path=str(source),
        generating_command=(
            str(header["generating_command"])
            if header.get("generating_command") is not None
            else None
        ),
        revision=str(header["revision"]) if header.get("revision") is not None else None,
        digest=hashlib.sha256(raw_bytes).hexdigest(),
    )
    return ResponseBank(
        id=name,
        digest=_digest(rows),
        provenance=provenance,
        tasks=tasks,
        responses={task: tuple(grouped[task]) for task in tasks},
        task_payloads=payloads,
    )
