"""`reward-lens forecast`: freeze a prediction, resolve it, score the ledger."""

from __future__ import annotations

import click

from .. import registry
from . import _shared


def run(*, out, action: str, **opts) -> int:
    from reward_lens import api

    out.event("started", command="forecast")
    if action == "issue":
        record = api.forecast_issue(api.ForecastIssueRequest(claim=opts["claim"]))
    elif action == "resolve":
        record = api.forecast_resolve(
            api.ForecastResolveRequest(forecast_id=_shared.identifier(opts["forecast_id"], field="forecast_id"))
        )
    else:
        record = api.forecast_ledger(api.ForecastLedgerRequest())
    artifacts, project = _shared.artifacts_for_record(record, ".")
    return _shared.deliver_record(out, record, command="forecast", artifacts=artifacts, project=project)


@click.group("forecast", short_help=registry.SUMMARIES["forecast"], epilog=_shared.EPILOG)
def forecast() -> None:
    """Freeze a prediction, resolve it, score the ledger."""


@forecast.command("issue", epilog=_shared.EPILOG)
@click.argument("claim")
@_shared.common
@_shared.verb("forecast")
def issue(**opts) -> int:
    """Freeze a prediction before the run that would settle it."""
    return run(action="issue", **opts)


@forecast.command("resolve", epilog=_shared.EPILOG)
@click.argument("forecast_id")
@_shared.common
@_shared.verb("forecast")
def resolve(**opts) -> int:
    """Settle a frozen prediction against what happened."""
    return run(action="resolve", **opts)


@forecast.command("ledger", epilog=_shared.EPILOG)
@_shared.common
@_shared.verb("forecast")
def ledger(**opts) -> int:
    """Score every prediction this project has frozen."""
    return run(action="ledger", **opts)
