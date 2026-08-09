"""`reward-lens compare`: what changes if the reward changes."""

from __future__ import annotations

import click

from .. import registry
from . import _shared


def run(*, out, **opts) -> int:
    from reward_lens import api

    _shared.budget_guard(out, seeker=opts.get("seeker", "off"), max_budget_usd=opts.get("max_budget_usd"), command="compare")
    out.event("started", command="compare")
    # CompareRequest takes its two sides as strings, not paths, so the validated path is spelled
    # back out. The validation still runs: a null byte, a control character or a double-encoded
    # argument is refused here, before anything downstream sees it.
    request = api.CompareRequest(
        baseline=str(_shared.ids.validate_path_argument(opts["baseline"], field="baseline")),
        candidate=str(_shared.ids.validate_path_argument(opts["candidate"], field="candidate")),
    )
    record = api.compare(request)
    artifacts, project = _shared.artifacts_for_record(record, request.candidate, request.baseline, ".")
    return _shared.deliver_record(
        out, record, command="compare", artifacts=artifacts, project=project
    )


@click.command("compare", short_help=registry.SUMMARIES["compare"], epilog=_shared.EPILOG)
@click.argument("baseline")
@click.argument("candidate")
@_shared.common
@_shared.measuring
@_shared.verb("compare")
def compare(**opts) -> int:
    """what changes if the reward changes."""
    return run(**opts)
