"""`reward-lens trace`: what a training run optimised, and what it selected."""

from __future__ import annotations

from pathlib import Path

import click

from .. import registry
from . import _shared


def run(*, out, **opts) -> int:
    from reward_lens import api

    _shared.budget_guard(
        out,
        seeker=opts.get("seeker", "off"),
        max_budget_usd=opts.get("max_budget_usd"),
        command="trace",
    )
    out.event("started", command="trace")
    run_name = _shared.identifier(opts["run"], field="run")
    request = api.TraceRequest(
        run=run_name,
        reward_functions=tuple(opts["reward_functions"]),
        extra_columns=tuple(opts["extra_columns"]),
        weights=tuple(opts["weights"]),
        group_column=opts["group_column"],
        group_size=opts["group_size"],
    )
    record = api.trace(request)
    # Trace reads an existing run rather than making a reward-lens project record. The assay and
    # report it just produced still live beside that source run, and the envelope names them from
    # exactly that root.
    artifacts, project = _shared.write_trace_artifacts(record, Path(run_name))
    return _shared.deliver_record(
        out, record, command="trace", artifacts=artifacts, project=project
    )


@click.command("trace", short_help=registry.SUMMARIES["trace"], epilog=_shared.EPILOG)
@click.argument("run")
@click.option(
    "--reward-function",
    "reward_functions",
    multiple=True,
    metavar="NAME",
    help="a TRL reward-function column; repeat to declare the complete partition",
)
@click.option(
    "--extra-column",
    "extra_columns",
    multiple=True,
    metavar="NAME",
    help="a TRL log_extra column; repeat to declare the complete partition",
)
@click.option(
    "--weight",
    "weights",
    multiple=True,
    metavar="NAME=VALUE",
    help="a finite weight for every declared reward-function column",
)
@click.option(
    "--group-column",
    type=str,
    default=None,
    metavar="NAME",
    help="the declared log_extra column that names source groups",
)
@click.option(
    "--group-size",
    type=int,
    default=None,
    metavar="K",
    help="the positive source-declared cardinality for each group",
)
@_shared.common
@_shared.measuring
@_shared.verb("trace")
def trace(**opts) -> int:
    """what a training run optimised, and what it selected."""
    return run(**opts)
