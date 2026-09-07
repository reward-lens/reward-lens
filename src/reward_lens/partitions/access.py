"""The access log: one append-only JSONL file per partition, written at the access.

The event is on disk before the bytes reach the caller. A log written after the read is a log that
loses exactly the accesses that mattered, the ones where the read raised or the process died
holding the file open, so the order here is deliberate and the flush is not optional.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

__all__ = ["AccessEvent", "AccessLog", "utc_now"]


def utc_now() -> str:
    """The record's timestamp shape: whole seconds, UTC, `Z`."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class AccessEvent:
    """One attempt on one item, granted or refused, with who asked and why."""

    at: str
    partition_id: str
    item_id: str
    capability: str
    purpose: str
    granted: bool
    reason: str
    use_count: int
    action: str = "read"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class AccessLog:
    """An append-only log of every attempt on a partition, kept beside it and reloaded on open.

    There is no method that removes, rewrites or truncates an event. A caller that wants a shorter
    log has to delete the file, which is a thing a person does on purpose and not a thing this
    class offers.
    """

    def __init__(self, path: Path | str, *, partition_id: str) -> None:
        self.path = Path(path)
        self.partition_id = partition_id
        self._events: list[AccessEvent] = list(self._load())

    def _load(self) -> list[AccessEvent]:
        if not self.path.exists():
            return []
        rows: list[AccessEvent] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("partition_id") != self.partition_id:
                continue
            rows.append(AccessEvent(**row))
        return rows

    @property
    def events(self) -> tuple[AccessEvent, ...]:
        return tuple(self._events)

    def uses(self, item_id: str) -> int:
        """How many times this item has been opened, counting granted reads only."""
        return sum(
            1
            for event in self._events
            if event.item_id == item_id and event.granted and event.action == "read"
        )

    def record(
        self,
        *,
        item_id: str,
        capability: str,
        purpose: str,
        granted: bool,
        reason: str = "",
        action: str = "read",
    ) -> AccessEvent:
        """Append one event, on disk and flushed, and return it."""
        count = self.uses(item_id) + (1 if granted and action == "read" else 0)
        event = AccessEvent(
            at=utc_now(),
            partition_id=self.partition_id,
            item_id=item_id,
            capability=capability,
            purpose=purpose,
            granted=granted,
            reason=reason,
            use_count=count,
            action=action,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._events.append(event)
        return event
