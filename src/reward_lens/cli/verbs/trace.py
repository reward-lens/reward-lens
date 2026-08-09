"""`reward-lens trace`: what a training run optimised, and what it selected."""

from __future__ import annotations

import click

from .. import registry
from . import _shared


def run(*, out, **opts) -> int:
    from reward_lens import api

    _shared.budget_guard(out, seeker=opts.get("seeker", "off"), max_budget_usd=opts.get("max_budget_usd"), command="trace")
    out.event("started", command="trace")
    request = api.TraceRequest(run=_shared.identifier(opts['run'], field='run'))
    record = api.trace(request)
    # The run is named by id, not by path, so the project this was run from is the only root
    # the verb can name. A trace that wrote nothing there leaves `artifacts` empty.
    artifacts, project = _shared.artifacts_for_record(record, ".")
    return _shared.deliver_record(out, record, command="trace", artifacts=artifacts, project=project)


@click.command("trace", short_help=registry.SUMMARIES["trace"], epilog=_shared.EPILOG)
@click.argument("run")
@_shared.common
@_shared.measuring
@_shared.verb("trace")
def trace(**opts) -> int:
    """what a training run optimised, and what it selected."""
    return run(**opts)
