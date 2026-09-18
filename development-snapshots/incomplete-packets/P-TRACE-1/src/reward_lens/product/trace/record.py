"""The shapes a reader hands back (section 7.4, wave-2 interfaces section 5).

Four series, named separately and always all four present, because the first rule of the run-record
standard is that the native component scores, the displayed aggregate, the optimised composite and
any convenience aggregate are distinct named things. A series that could not be built is still
named; it carries `present=False` and the reason. Naming an absence is the point: E1 was paid for
by a single column whose name said "the total" and whose values were a batch mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Mapping

if TYPE_CHECKING:  # pragma: no cover - typing only
    from reward_lens.contracts import Entry

    from .integrity import RuleResult

__all__ = [
    "Policy",
    "SERIES_NAMES",
    "Series",
    "TornTail",
    "TraceRecord",
    "TraceRow",
]

#: Declared by the caller and recorded in every record a follow produces (section 7.4).
Policy = Literal["strict", "permissive"]

#: The four, in the order every record names them.
SERIES_NAMES: tuple[str, ...] = (
    "reward_recorded",
    "reward_applied",
    "advantage",
    "selection",
)


@dataclass(frozen=True)
class Series:
    """One named series, with the weight vector beside it when it is a composite.

    `values` is one entry per row of the record, in row order, and a `None` is the trainer's own
    missing value preserved rather than normalised to zero.
    """

    name: str
    values: tuple[float | None, ...] = ()
    weights: tuple[float, ...] | None = None
    components: tuple[str, ...] = ()
    definition: str = ""
    present: bool = True
    absent_reason: str = ""


@dataclass(frozen=True)
class TraceRow:
    """One completion as the optimiser saw it.

    `group_id` is the group the source declared. It is `None` when the source declared none, and
    it is never filled in from what the neighbouring rows look like (D-42).
    """

    index: int
    step: int
    group_id: str | None
    row_id: str
    components: Mapping[str, float | None] = field(default_factory=dict)
    reward_recorded: float | None = None
    reward_applied: float | None = None
    advantage: float | None = None
    selection: float | None = None


@dataclass(frozen=True)
class TornTail:
    """A partial write at the end of a chunk, kept visible rather than silently dropped."""

    path: str
    seq: int
    bytes_dropped: int
    text: str
    allowed_by: str


@dataclass(frozen=True)
class TraceRecord:
    """What one reader made of one artifact."""

    reader: str
    source: str
    run_id: str
    rows: tuple[TraceRow, ...]
    series: Mapping[str, Series]
    components: Mapping[str, Series]
    access_level: str
    access_statement: str
    policy: Policy
    complete: bool
    completeness_statement: str
    subject_ref: str
    depends_on: tuple[str, ...]
    framework: str = ""
    framework_version: str = ""
    integrity: tuple["RuleResult", ...] = ()
    torn_tail: TornTail | None = None
    limitations: tuple[str, ...] = ()

    def entries(self) -> tuple["Entry", ...]:
        """The record-level half of the join: one entry per job, each naming its subject."""
        from .entries import trace_entries

        return trace_entries(self)
