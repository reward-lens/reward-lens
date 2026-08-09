"""`reward-lens improve`: start from a goal and run the loop."""

from __future__ import annotations

import click

from .. import registry
from . import _shared


def run(*, out, **opts) -> int:
    from reward_lens import api

    _shared.budget_guard(out, seeker=opts.get("seeker", "off"), max_budget_usd=opts.get("max_budget_usd"), command="improve")
    out.event("started", command="improve")
    request = api.ImproveRequest(path=_shared.ids.validate_path_argument(opts['path'], field='path'))
    record = api.improve(request)
    return _shared.deliver_record(out, record, command="improve")


@click.command("improve", short_help=registry.SUMMARIES["improve"], epilog=_shared.EPILOG)
@click.argument("path", required=False, default=".")
@_shared.common
@_shared.measuring
@_shared.verb("improve")
def improve(**opts) -> int:
    """start from a goal and run the loop."""
    return run(**opts)
