"""The six-rung comparison registry.

The enum is deliberately complete before all engines exist.  A registry entry is
added only by the packet that implements that rung, so an absent implementation
remains an RL0703 capability refusal instead of a callable placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Callable

from reward_lens.contracts import CapabilityUnavailable, Entry, Scope

from .scorecard import Scorecard

if TYPE_CHECKING:
    from reward_lens.bank import ResponseBank


class Rung(IntEnum):
    RE_SCORE = 1
    SELECTION_STRESS = 2
    ADVANTAGE_REPLAY = 3
    SAME_BATCH_SWAP = 4
    SHORT_FORK = 5
    TRANSFER = 6


@dataclass(frozen=True)
class CompareInputs:
    """The shared dispatch input without inventing a per-rung command protocol."""

    bank: ResponseBank | None = None
    version_a: object = ""
    version_b: object = ""
    run_record: object | None = None
    selector: object | None = None
    outcome_check: object | None = None
    ladder: tuple[int, ...] = ()
    bootstrap_resamples: int = 0
    sandbox: object | None = None
    limits: object | None = None


@dataclass(frozen=True)
class CompareResult:
    scope: Scope
    scorecard: Scorecard
    entries: list[Entry]
    establishes: str
    does_not_establish: str
    reading: object | None = None


RungFn = Callable[[CompareInputs], CompareResult]

_REGISTERED: dict[Rung, RungFn] = {}
_SHIPPING_WAVE: dict[Rung, int] = {
    Rung.RE_SCORE: 3,
    Rung.SELECTION_STRESS: 3,
    Rung.ADVANTAGE_REPLAY: 3,
    Rung.SAME_BATCH_SWAP: 4,
    Rung.SHORT_FORK: 4,
    Rung.TRANSFER: 4,
}


def register(rung: Rung, fn: RungFn) -> None:
    """Register one implemented rung; duplicate registrations are a programming error."""
    if not isinstance(rung, Rung):
        raise TypeError("rung must be a Rung")
    if not callable(fn):
        raise TypeError("rung function must be callable")
    previous = _REGISTERED.get(rung)
    if previous is not None and previous is not fn:
        raise ValueError(f"compare rung {rung.value} is already registered")
    _REGISTERED[rung] = fn


def _load(rung: Rung) -> None:
    """Import only an implemented rung module, which registers its real engine."""
    if rung is Rung.RE_SCORE and rung not in _REGISTERED:
        from . import rung1  # noqa: F401 - rung1 registers its real entry point
    elif rung is Rung.SELECTION_STRESS and rung not in _REGISTERED:
        from . import rung2  # noqa: F401 - rung2 registers its real entry point
    elif rung is Rung.ADVANTAGE_REPLAY and rung not in _REGISTERED:
        from . import rung3  # noqa: F401 - rung3 registers its real entry point
    elif rung is Rung.SAME_BATCH_SWAP and rung not in _REGISTERED:
        # Rung 4's public factory owns the actual update and retained evidence.
        # This thin registry adapter accepts only that factory-produced pair; it
        # cannot manufacture an update result from a request mapping.
        from .rung4 import (
            ControlledUpdateEvidence,
            ControlledUpdateResult,
            controlled_update_compare,
        )

        def run_rung_four(inputs: CompareInputs) -> CompareResult:
            produced = inputs.run_record
            if (
                not isinstance(produced, tuple)
                or len(produced) != 2
                or not isinstance(produced[0], ControlledUpdateResult)
                or not isinstance(produced[1], ControlledUpdateEvidence)
            ):
                raise TypeError("compare rung 4 requires the actual factory result and evidence")
            return controlled_update_compare(produced[0], produced[1])

        register(rung, run_rung_four)
    elif rung is Rung.SHORT_FORK and rung not in _REGISTERED:
        from . import rung5  # noqa: F401 - rung5 registers its real entry point
    elif rung is Rung.TRANSFER and rung not in _REGISTERED:
        from . import rung6  # noqa: F401 - rung6 registers its real entry point


def get(rung: Rung) -> RungFn:
    """Return a real rung implementation or the typed absent-engine refusal."""
    if not isinstance(rung, Rung):
        raise TypeError("rung must be a Rung")
    _load(rung)
    fn = _REGISTERED.get(rung)
    if fn is not None:
        return fn
    wave = _SHIPPING_WAVE[rung]
    raise CapabilityUnavailable(
        code="RL0703",
        message=(
            f"compare rung {rung.value} is not in this build; its implementation ships in wave {wave}"
        ),
        remediation="select a registered compare rung or install the wave that ships this rung",
        context={"rung": rung.value, "wave": wave, "command": "reward-lens"},
    )


__all__ = ["CompareInputs", "CompareResult", "Rung", "RungFn", "get", "register"]
