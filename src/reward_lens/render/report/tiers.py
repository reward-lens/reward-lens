"""The three size bands of D-15, and the two different objects they are measured on.

Tier A's limits are on the canonical JSON of the record, because an inlined record is the thing
whose size is a property of the measurement rather than of the renderer. Tier B's ceiling is on
the whole HTML file, because that is what a reader receives: the escaping (`<` written as
`\\u003c`, and the two Unicode line separators), the base64 Parquet blocks, the inlined stylesheet
and the inlined script are all part of the file and none of them is part of the record.

Conflating the two was the defect this module was rewritten to close. The ceiling used to be
applied to the canonical bytes, so a record of 25.9 MB of canonical JSON was called tier B and
written out as a file of 27,954,515 bytes against a 26,214,400-byte ceiling, and nothing on the
page said so.

A record never inlines the tables the bundle holds. It refers to each one by digest and row count
and carries at most fifty sample rows under `tables[].inline_rows` (D-16), so those rows are the
only per-row data a record has. Tier B is how they reach the page without a fetch: each table's
rows are written as Parquet, base64-encoded into a block of its own, and read in the browser by
hyparquet. Tier C is when even that does not fit, or when no Parquet writer is installed, and
then the page names the tables it could not carry and says where they are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MB = 1024 * 1024

SOFT_A = 2 * MB
"""On the canonical JSON. Above this the record is no longer comfortably a page, and it says so."""

HARD_A = 5 * MB
"""On the canonical JSON. Above this the record stops being inlined on its own: tier B or C."""

CEILING_B = 25 * MB
"""On the whole HTML file: 26,214,400 bytes. Above it the tables stay in the bundle (tier C)."""


@dataclass(frozen=True)
class Embedding:
    """What `plan_embedding` decided, and why.

    `to_dict()` emits exactly the four keys the frozen schema allows under `embedding`; `reason`,
    `omission_notes` and `past_soft_limit` are the plan's own account of the decision and stay out
    of the record, which `additionalProperties: false` would refuse anyway. The renderer re-derives
    all three from the record and the tables it is handed, so the page states them whether or not
    the plan's result travelled.

    `omission_notes` pairs an omitted table with the reason it is not in the file, which at tier B
    is a fact about this render rather than about the record: `payload not supplied`, or
    `preview only: k of n rows` for a table whose sample rows are all the record itself holds.
    """

    tier: str
    omitted_tables: tuple[str, ...] = ()
    record_bytes: int = 0
    html_bytes: int = 0
    reason: str | None = None
    omission_notes: tuple[tuple[str, str], ...] = ()
    past_soft_limit: bool = False
    passes: int = field(default=1, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "omitted_tables": list(self.omitted_tables),
            "record_bytes": self.record_bytes,
            "html_bytes": self.html_bytes,
        }


def ceiling_of(tier: str) -> tuple[str, int] | None:
    """The limit a tier has to honour, and the object it is measured on."""
    if tier == "A":
        return ("the canonical JSON of the record", HARD_A)
    if tier == "B":
        return ("the whole HTML file", CEILING_B)
    return None
