"""What an instrument is, and the context every instrument runs in (wave-2 interfaces section 2).

Frozen in wave 1 so that a wave-2 instrument packet lands a `PANEL` in its own package and never
edits `product/audit/`. Five things live here and nothing else: the `Instrument` protocol, the
`Subject` and `Corpus` protocols naming what an instrument may read of what it measures and of the
rollouts it measures on, the `RunContext` that carries the sandbox, the project, the budget, the
offline flag and the clock, and the `Panel` that groups instruments under one section of the record.

`Subject` and `Corpus` are protocols rather than concrete classes on purpose. The objects that
satisfy them are built by `product/audit/`, which an instrument packet may not edit and therefore
may not import from; typing against the protocol is how a packet says what it reads without
reaching into the runner.

`RunContext.provenance()` is the single source of the `EntryProvenance` every entry and every
`absence()` carries (A-004). An instrument that builds its own provenance block is how two entries
in one record come to disagree about which sandbox tier they ran under.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence, runtime_checkable

from reward_lens import contracts

__all__ = ["Corpus", "Instrument", "Panel", "RunContext", "Subject", "utc_now"]


def utc_now() -> str:
    """The record's timestamp format, to the second, always UTC."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@runtime_checkable
class Subject(Protocol):
    """The reward system under audit, as an instrument may read it.

    The runner resolves one of these from the request and the project before any panel runs, and
    passes the same object to every instrument. It is not `contracts.Subject`: that is the record's
    subject block, whose `instrument_method` digest is computed from the methods that ran, so it
    cannot exist until after the instruments have returned.
    """

    grader_path: Path
    entrypoint: str
    project_dir: Path | None
    tasks_path: Path | None
    responses_path: Path | None
    trainer: str | None
    name: str
    source: str


@runtime_checkable
class Corpus(Protocol):
    """The rollouts an instrument measures on, and where they came from.

    `origin` says in one line what was supplied, and is the string the record's context carries; it
    is written even when `rollouts` is empty, because "no task set and no response bank were
    supplied" is the answer an absence needs.
    """

    rollouts: tuple[Any, ...]
    origin: str
    tasks: tuple[dict, ...]
    n_tasks: int

    def as_list(self) -> Any:
        """The corpus as the verifier series takes it."""
        ...

    def __len__(self) -> int:
        """How many rollouts."""
        ...


@runtime_checkable
class Instrument(Protocol):
    """One measurement that fills one section. `id` is a rule id; `section` is a record section."""

    id: str
    section: contracts.Section

    def run(self, subject: Subject, corpus: Corpus, ctx: "RunContext") -> list[contracts.Entry]:
        """Every entry this instrument produced, evidence or absence. The runner keeps all of them;
        returning none is an instrument that measured nothing, and the runner records it as such.
        Exceptions are not caught here: the audit catches them and writes a `COULD_NOT_CHECK` hole
        naming what failed."""
        ...


@dataclass(frozen=True)
class Panel:
    """One section of the record and the instruments that fill it."""

    section: contracts.Section
    instruments: tuple[Instrument, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "instruments", tuple(self.instruments))

    @classmethod
    def of(cls, section: contracts.Section, instruments: Iterable[Instrument] = ()) -> "Panel":
        return cls(section=section, instruments=tuple(instruments))


@dataclass
class RunContext:
    """The sandbox, the project, the budget, the offline flag and the clock, in one object."""

    sandbox: Any = None
    project: Any = None
    budget: str = "0.00"
    offline: bool = True
    clock: Callable[[], float] = time.monotonic
    started: str = field(default_factory=utc_now)

    @property
    def budget_usd(self) -> str:
        """Read-only alias for `budget`, kept for one release. Nothing in this tree reads it; it is
        here for a caller written against the name the first cut shipped."""
        return self.budget

    def tier(self) -> str:
        """The tier actually held, from the sandbox itself; `T0` when there is no sandbox."""
        for attribute in ("tier", "tier_held"):
            value = getattr(self.sandbox, attribute, None)
            if isinstance(value, str) and value:
                return value
        return "T0"

    def provenance(
        self,
        *,
        duration_s: float = 0.0,
        started: str | None = None,
        arm: str | None = None,
        counters: Any = None,
    ) -> contracts.EntryProvenance:
        """The provenance block every entry in this run carries (A-004)."""
        block = contracts.EntryProvenance(
            started=started or self.started,
            duration_s=max(0.0, float(duration_s)),
            sandbox_tier=self.tier(),
            offline=bool(self.offline),
        )
        if arm is not None:
            block.arm = arm
        if counters is not None:
            block.counters = counters
        return block

    def absence(
        self,
        section: str,
        entry_id: str,
        measurand: str,
        missing_access: str,
        remedy: str,
        affected_claims: Sequence[str] = (),
        *,
        subject_ref: str,
        state: str = "NOT_MEASURED",
        depends_on: Sequence[str] = (),
        limitations: Sequence[str] = (),
        duration_s: float = 0.0,
    ) -> contracts.Entry:
        """One honest absence, stamped with this run's provenance (D-18, A-004)."""
        entry = contracts.absence(
            section,
            entry_id,
            measurand,
            missing_access,
            remedy,
            tuple(affected_claims),
            subject_ref=subject_ref,
            provenance=self.provenance(duration_s=duration_s),
            depends_on=tuple(depends_on),
            state=state,
        )
        if limitations:
            entry.limitations = list(limitations)
        return entry
