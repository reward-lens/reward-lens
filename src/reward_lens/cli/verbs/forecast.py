"""`reward-lens forecast`: freeze a prediction, resolve it, score the ledger."""

from __future__ import annotations

import json

import click

from .. import output, registry
from . import _shared


def run(*, out, action: str, **opts) -> int:
    from reward_lens import api

    out.event("started", command="forecast")
    if action == "issue":
        record = api.forecast_issue(
            api.ForecastIssueRequest(claim=opts["claim"], record=opts["record"])
        )
    elif action == "resolve":
        record = api.forecast_resolve(
            api.ForecastResolveRequest(
                forecast_id=_shared.identifier(opts["forecast_id"], field="forecast_id"),
                outcome=opts["outcome"],
            )
        )
    else:
        record = api.forecast_ledger(api.ForecastLedgerRequest())
        from reward_lens.product.forecast import _LEDGER_EXTENSION, render_ledger

        if out.format == "text":
            click.echo(render_ledger())
    artifacts, project = _shared.artifacts_for_record(record, ".")
    if action == "ledger" and out.format == "json":
        out.result(json.dumps(record.extensions[_LEDGER_EXTENSION], sort_keys=True))
        document = _shared.envelope_from_record(record, command="forecast", artifacts=artifacts)
        code = output.exit_code_for(document["decision"])
        return 0 if out.exit_zero and code in (1, 2) else code
    return _shared.deliver_record(
        out, record, command="forecast", artifacts=artifacts, project=project
    )


@click.group("forecast", short_help=registry.SUMMARIES["forecast"], epilog=_shared.EPILOG)
def forecast() -> None:
    """Freeze a prediction, resolve it, score the ledger."""


@forecast.command("issue", epilog=_shared.EPILOG)
@click.argument("claim")
@click.option(
    "--record", type=click.Path(path_type=str), help="Checkpoint-bounded JSON issue record."
)
@_shared.common
@_shared.verb("forecast")
def issue(**opts) -> int:
    """Freeze a prediction before the run that would settle it."""
    return run(action="issue", **opts)


@forecast.command("resolve", epilog=_shared.EPILOG)
@click.argument("forecast_id")
@click.option(
    "--outcome",
    required=True,
    help="Frozen forecast outcome: true/false for T1, fraction for T2/T3.",
)
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
