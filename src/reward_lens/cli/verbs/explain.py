"""`reward-lens explain`: what an error code or a finding means, offline."""

from __future__ import annotations

import click

from .. import ids, registry
from . import _shared


def run(*, out, code, **opts) -> int:
    from reward_lens import api, errors

    wanted = ids.validate_identifier(code, field="code").upper()
    long_form = errors.explain(wanted)
    document = api.explain(wanted).model_dump()
    return out.deliver(document=document, text=long_form, exit_code=0)


@click.command("explain", short_help=registry.SUMMARIES["explain"], epilog=_shared.EPILOG)
@click.argument("code")
@_shared.common
@_shared.verb("explain")
def explain(**opts) -> int:
    """What an error code or a finding means."""
    return run(**opts)
