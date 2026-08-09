"""`--format jsonl`: one versioned event per line as each result lands (D-26).

A long run streams `started`, then one event per sample as it is scored, then `result`. It is not a
progress bar: it is the form an agent processes one record at a time without buffering the run.
"""

from __future__ import annotations

import json as _json

SCHEMA = "assay-event/1.0"


def render(event: str, **payload) -> str:
    line = {"schema_version": SCHEMA, "event": event}
    line.update(payload)
    return _json.dumps(line, sort_keys=False)
