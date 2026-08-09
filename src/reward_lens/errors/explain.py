"""The long form, the two-line form, and the builder that turns a code into a raisable error.

All three read the catalogue and nothing else. `explain` opens no file and no socket: the text it
prints is compiled into the package, so it answers on a machine with no network, which is exactly
the machine a person is on when they hit an error about the network.
"""

from __future__ import annotations

from typing import Any

from reward_lens.contracts import (
    BudgetExceeded,
    CapabilityUnavailable,
    InternalFailure,
    PendingDecision,
    RewardLensError,
    UsageError,
)

from .catalogue import CATALOGUE, EXIT_MEANINGS, ErrorSpec, fill

__all__ = ["explain", "make", "render_two_line"]

_TYPE_FOR_EXIT: dict[int, type[RewardLensError]] = {
    3: PendingDecision,
    4: UsageError,
    5: CapabilityUnavailable,
    6: BudgetExceeded,
    7: InternalFailure,
}

_SURFACE_LINE = {
    "finding": (
        "Nothing stops here. {code} is a finding the record carries, and a decision resting on the "
        "section it lands in stays unresolved until the finding is dealt with or accepted."
    ),
    "state": (
        "Nothing stops here. {code} is a state the record carries: it says what the measurement "
        "could not qualify, rather than that something went wrong."
    ),
}


def _unknown(code: object) -> UsageError:
    shown = code if isinstance(code, str) and code.strip() else repr(code)
    spec = CATALOGUE["RL0001"]
    detail = f"{shown} is not an error code this build knows"
    return UsageError(
        code=spec.code,
        message=fill(spec.cause, {"detail": detail}),
        remediation=fill(spec.remedies[0], {}),
        context={"asked_for": shown},
    )


def _lookup(code: object) -> ErrorSpec:
    if not isinstance(code, str):
        raise _unknown(code)
    spec = CATALOGUE.get(code)
    if spec is None:
        raise _unknown(code)
    return spec


def explain(code: str) -> str:
    """The long form of one code: what it means, what it stops, and what resolves each cause.

    Nothing here is filled from a context, because a reader asking what a code means has not hit
    it yet: the text comes from the entry's long form, which is written to read with none.

    Raises `UsageError` with code RL0001 and exit 4 for a code this build does not hold.
    """
    spec = _lookup(code)

    lines = [f"{spec.code}  {spec.title}", ""]
    lines.append("  " + spec.long_form_cause)
    lines.append("")
    lines.append("What this means for the run")
    surface_line = _SURFACE_LINE.get(spec.surface)
    if surface_line is not None:
        lines.append("  " + surface_line.format(code=spec.code))
    if spec.verdicts:
        lines.append("  " + spec.verdict_rule)
    lines.append("  " + EXIT_MEANINGS[spec.exit_code])
    lines.append("")
    lines.append("What usually causes it, and what resolves each")
    for remedy in spec.long_form_remedies:
        lines.append("  - " + remedy)
    lines.append("")
    lines.append(f"  Every code: reward-lens explain {spec.code} works with the network off.")
    return "\n".join(lines) + "\n"


def make(code: str, **context: Any) -> RewardLensError:
    """Build the error a code stands for, with the context filled into its cause and its remedy.

    The type is the one the exit status calls for, so a caller that catches on type still sees the
    same families. Raises `UsageError` with RL0001 for a code this build does not hold.
    """
    spec = _lookup(code)
    message = fill(spec.cause, context)
    remediation = fill(spec.remedies[0], context) if spec.remedies else ""
    kind = _TYPE_FOR_EXIT.get(spec.exit_code)

    if kind is PendingDecision:
        return PendingDecision(
            code=spec.code,
            message=message,
            remediation=remediation,
            context=dict(context),
            argv=list(context.get("argv", ())),
            command=str(context.get("command", "")),
            decision=str(context.get("decision", "")),
        )
    if kind is not None:
        return kind(
            code=spec.code,
            message=message,
            remediation=remediation,
            context=dict(context),
        )
    return RewardLensError(
        code=spec.code,
        message=message,
        remediation=remediation,
        context=dict(context),
        exit_code=spec.exit_code,
    )


def render_two_line(error: RewardLensError) -> str:
    """The shape a caller sees on stderr: what happened, a command to run, and where to read more."""
    return "\n".join(
        [
            f"error: {error.message}",
            f"help:  {error.remediation}",
            f"       reward-lens explain {error.code}",
        ]
    )
