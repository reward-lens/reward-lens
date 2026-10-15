"""A run directory that is still being written, read through `follow` (section 7.4).

The record this produces always says the run was incomplete when it was read, because it was: the
reader tails a directory and has no way to know whether the writer has stopped. Under `permissive`
a torn tail and a sequence gap are carried in the record, and the record names the policy that
allowed them. Under `strict` the same two conditions refuse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .follow import follow
from .integrity import CheckInput, enforce
from .record import SERIES_NAMES, Policy, Series, TornTail, TraceRecord, TraceRow

__all__ = ["READER", "LiveFollowReader"]

_STATEMENT = (
    "RECORD: the values the writer had flushed when the directory was listed, and nothing else"
)


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


class LiveFollowReader:
    """Live following, read-only and non-interfering."""

    name = "live_follow"

    def can_read(self, path: Any) -> bool:
        try:
            return bool(list(Path(path).glob("chunk_*.jsonl")))
        except OSError:  # pragma: no cover
            return False

    def read(self, path: Any, *, policy: Policy = "strict") -> TraceRecord:
        root = Path(path)
        rows: list[TraceRow] = []
        torn: TornTail | None = None
        gaps: list[int] = []
        index = 0
        for chunk in follow(root, policy=policy):
            gaps.extend(chunk.gap_before)
            for position, raw in enumerate(chunk.rows):
                rows.append(
                    TraceRow(
                        index=index,
                        step=int(raw.get("step", 0) or 0),
                        group_id=None if raw.get("group_id") is None else str(raw["group_id"]),
                        row_id=str(raw.get("row_id") or f"{chunk.seq:05d}:{position}"),
                        components={},
                        reward_recorded=_number(raw.get("reward_recorded")),
                        reward_applied=_number(raw.get("reward_applied")),
                        advantage=_number(raw.get("advantage")),
                        selection=_number(raw.get("selection")),
                    )
                )
                index += 1
            if chunk.torn:
                torn = TornTail(
                    path=chunk.path,
                    seq=chunk.seq,
                    bytes_dropped=len(chunk.torn_bytes),
                    text=chunk.torn_bytes,
                    allowed_by=(
                        f'the declared capture policy is "{policy}", which records a torn tail '
                        f"rather than refusing"
                    ),
                )

        series = _series(rows)
        results = enforce(
            CheckInput(rows=tuple(rows), series=series, source=str(root)), policy=policy
        )
        limitations = []
        if gaps:
            limitations.append(
                "chunk(s) " + ", ".join(str(one) for one in gaps) + " were missing from the capture"
            )
        if torn is not None:
            limitations.append(f"{torn.bytes_dropped} bytes of a partial write end the capture")
        return TraceRecord(
            reader=self.name,
            source=str(root),
            run_id=str(root.name),
            rows=tuple(rows),
            series=series,
            components={},
            access_level="RECORD",
            access_statement=_STATEMENT,
            policy=policy,
            complete=False,
            completeness_statement=(
                "the run was incomplete when read: following tails a directory and never attaches "
                "to the training process, so there is no way to know the writer has stopped"
            ),
            subject_ref=_subject_ref(root, rows),
            depends_on=("digest:training_semantics", "digest:samples", "digest:policy"),
            framework="",
            framework_version="",
            integrity=results,
            torn_tail=torn,
            limitations=tuple(limitations),
        )


def _series(rows: list[TraceRow]) -> dict[str, Series]:
    built: dict[str, Series] = {}
    for name in SERIES_NAMES:
        values = tuple(getattr(row, name) for row in rows)
        present = any(value is not None for value in values)
        built[name] = Series(
            name=name,
            values=values,
            present=present,
            absent_reason=(
                "" if present else f"the chunks carry no {name} field for any row read so far"
            ),
            definition=f"{name}, as the followed chunks carry it",
        )
    return built


def _subject_ref(root: Path, rows: list[TraceRow]) -> str:
    from reward_lens.contracts import digest

    return digest({"followed": str(root.name), "rows": len(rows)})


READER = LiveFollowReader()
