"""`reward-lens explain`: what an error code or a finding means, offline."""

from __future__ import annotations

import click

from .. import ids, registry
from . import _shared


def run(*, out, code, **opts) -> int:
    from reward_lens import errors

    wanted = ids.validate_identifier(code, field="code").upper()
    long_form = errors.explain(wanted)
    spec = errors.CATALOGUE[wanted]
    document = {
        "schema_version": "error-explanation/1.0",
        "command": "explain",
        "code": spec.code,
        "title": spec.title,
        "cause": spec.cause,
        "remedies": list(spec.remedies),
        "exit_code": spec.exit_code,
        "surface": spec.surface,
        "error": None,
    }
    return out.deliver(document=document, text=long_form, exit_code=0)


@click.command("explain", short_help=registry.SUMMARIES["explain"], epilog=_shared.EPILOG)
@click.argument("code")
@_shared.common
@_shared.verb("explain")
def explain(**opts) -> int:
    """What an error code or a finding means."""
    return run(**opts)
