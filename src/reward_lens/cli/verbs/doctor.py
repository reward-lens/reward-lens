"""`reward-lens doctor`: what can be measured here, and what it would cost."""

from __future__ import annotations

import dataclasses

import click

from .. import registry
from . import _shared


def run(*, out, **opts) -> int:
    from reward_lens import api

    capabilities = api.doctor()
    document = _document(capabilities)
    return out.deliver(document=document, text=_text(capabilities), exit_code=0)


def _document(capabilities) -> dict:
    return {
        "schema_version": "capabilities/1.0",
        "command": "doctor",
        "install": _fields(capabilities.install),
        "sandbox": _fields(capabilities.sandbox),
        "capabilities": [_fields(item) for item in capabilities.capabilities],
        "error": None,
    }


def _fields(value):
    """Serialise by the object's own fields, whatever kind of object it is.

    `Capabilities.sandbox` is the frozen `SandboxProbe`, a dataclass, and the rest of the
    capabilities report is pydantic. Neither is iterable, so a `dict(...)` fallback turns a
    serialisation gap into `TypeError` inside the verb (RL0900) rather than into a report. This
    asks each object for its fields and refuses to guess.
    """
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, dict):
        return dict(value)
    raise _shared.not_in_this_build(
        f"serialising {type(value).__name__} into the doctor report",
        remedy="give the type a to_dict, a model_dump, or make it a dataclass",
    )


def _text(capabilities) -> str:
    """The panel is `reward_lens.product.access.doctor.render_text` and nothing of this verb's.

    Section 5.6's panel is one rendering, and a second copy here is a second surface to keep in
    step with it. The JSON form stays the verb's, because the envelope is the CLI's contract.
    """
    try:
        from reward_lens.product.access.doctor import render_text
    except ImportError as missing:  # pragma: no cover - the product ships in the same wheel
        raise _shared.not_in_this_build(
            "the doctor panel", remedy="install the distribution that carries reward_lens.product"
        ) from missing
    return render_text(capabilities)


@click.command("doctor", short_help=registry.SUMMARIES["doctor"], epilog=_shared.EPILOG)
@_shared.common
@_shared.verb("doctor")
def doctor(**opts) -> int:
    """What can be measured here, and what it would cost."""
    return run(**opts)
