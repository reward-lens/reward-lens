"""What an audit would attempt before it does it. D-41: the default path prices zero paid calls.

The list of panels is not written down here. It is the two this engine fills itself plus whatever
`registry.discover()` finds under `reward_lens.instruments`, put back in record order, so a wave-2
instrument packet that lands its own `PANEL` appears in the plan with no line changing in
`product/audit/`. The runner reuses the same function for the set it skips when it writes the
absences, which is what keeps a landed panel from carrying both its entries and a `NOT_MEASURED`
absence saying this build ships none of it. A planned instrument can still write an honest absence
when the project lacks its required inputs; dry run does not execute it merely to predict that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from reward_lens import contracts
from reward_lens.api.requests import Plan

from . import registry

__all__ = ["BUILTIN_PANELS", "PANEL_SECONDS", "SECTIONS", "estimate_s", "panels", "plan"]

#: The record's sections, in record order, read from the record model itself so that the plan, the
#: runner and the absences cannot drift apart about either the set or the order.
SECTIONS: tuple[str, ...] = tuple(contracts.models.Measurement.model_fields)

#: The panels this engine fills itself. Every other filled panel arrives through discovery.
BUILTIN_PANELS: tuple[str, ...] = ("validity",)

#: Seconds a built-in panel is expected to take, measured on the shipped example and rounded up to
#: the whole second. Discovered panels carry their own value on `Panel.seconds`, keeping the timing
#: estimate beside the panel that supplies the work rather than in the audit runner. These values
#: are estimates, not measurements: re-measure by summing `provenance.duration_s` by section over a
#: record this build wrote.
PANEL_SECONDS: dict[str, int] = {"validity": 1}

#: What a panel this build has never timed is expected to take. A wave-2 panel that lands its own
#: `PANEL` gets counted at this until someone measures it, which is a guess that says so rather
#: than a zero that would claim the panel is free.
DEFAULT_PANEL_SECONDS = 1


def estimate_s(filled: Iterable[str], discovered: Iterable[Any] | None = None) -> int:
    """How long a run over `filled` is expected to take, in whole seconds.

    The sum over the panels that will actually run, which is why it moves when a wave-2 panel lands
    and why it is not a number written down once for the demo. Zero panels is zero seconds: a run
    that fills nothing takes no measuring time, and the surface prints the estimate beside the panel
    count, so the two agree or the reader can see they do not.
    """
    found = registry.discover() if discovered is None else tuple(discovered)
    discovered_seconds = {
        str(getattr(panel, "section", "")): int(getattr(panel, "seconds", DEFAULT_PANEL_SECONDS))
        for panel in found
    }
    return sum(
        PANEL_SECONDS.get(section, discovered_seconds.get(section, DEFAULT_PANEL_SECONDS))
        for section in filled
    )


def panels(discovered: Iterable[Any] | None = None) -> tuple[str, ...]:
    """The sections this build attempts: `BUILTIN_PANELS` plus every discovered panel, in record order.

    Pass `discovered` (the tuple `registry.discover()` returned) to reuse one discovery pass; the
    runner does, so that what it plans, what it runs and what it writes absences for are the one
    answer computed once.
    """
    found = registry.discover() if discovered is None else tuple(discovered)
    filled = set(BUILTIN_PANELS) | {str(getattr(panel, "section", "")) for panel in found}
    return tuple(section for section in SECTIONS if section in filled)


def plan(req: Any, *, project: Any = None, sandbox: Any = None) -> Plan:
    """The non-spending plan for `audit`; attempted panels may still honestly be absent."""
    seeker = str(getattr(req, "seeker", "off"))
    discovered = registry.discover()
    filled = panels(discovered)
    plugged = tuple(section for section in filled if section not in BUILTIN_PANELS)
    dark = len(SECTIONS) - len(filled)
    notes = [
        "validity runs the four static checks, D10 replay determinism and D1 decision coverage",
    ]
    if plugged:
        notes.append(
            "this build has "
            + ", ".join(plugged)
            + " plugged in, and each one runs the instruments its own package registered"
        )
    if dark:
        notes.append(
            f"the other {dark} panels write an absence naming what they needed"
            if dark > 1
            else "the one remaining panel writes an absence naming what it needed"
        )
    if seeker != "off":
        notes.append(
            f"the seeker arm is set to {seeker}, and this build does not ship it; the plan prices "
            "it at nothing because nothing will run"
        )
    if project is None and Path(getattr(req, "path", ".")).is_file():
        notes.append("no task set and no response bank: the three task-set checks will not run")
    return Plan(
        command="audit",
        panels=filled,
        paid_calls=0,
        estimate_usd="0.00",
        estimate_s=estimate_s(filled, discovered),
        notes=tuple(notes),
    )
