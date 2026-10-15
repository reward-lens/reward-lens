"""`reward-lens improve`: start from a goal and run the loop."""

from __future__ import annotations

import json
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
        command="improve",
    )
    out.event("started", command="improve")
    pair = opts.get("mark_pair")
    if (pair is None) != (opts.get("why") is None):
        raise click.UsageError("--mark-pair and --why must be supplied together")
    candidate_path = opts.get("candidate")
    candidate = None
    if candidate_path is not None:
        try:
            candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise click.UsageError(
                f"--candidate must name a readable JSON declaration: {error}"
            ) from error
        if not isinstance(candidate, dict):
            raise click.UsageError("--candidate must contain one JSON object")
    forecast = None
    if opts.get("forecast") is not None:
        try:
            forecast = json.loads(Path(opts["forecast"]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise click.UsageError(
                f"--forecast must name a readable JSON declaration: {error}"
            ) from error
        if not isinstance(forecast, dict):
            raise click.UsageError("--forecast must contain one JSON object")
    request = api.ImproveRequest(
        path=_shared.ids.validate_path_argument(opts["path"], field="path"),
        describe=opts.get("describe"),
        resume=opts.get("resume"),
        mark_pair=tuple(pair) if pair is not None else None,
        why=opts.get("why"),
        candidate_source=opts.get("candidate_source"),
        candidate=candidate,
        name=opts.get("name"),
        forecast=forecast,
        offline=True if opts.get("offline") is None else bool(opts["offline"]),
        seed=api.DEFAULT_SEED if opts.get("seed") is None else int(opts["seed"]),
        sandbox=opts.get("sandbox") or "auto",
        require_tier=opts.get("require_tier"),
        max_budget_usd=opts.get("max_budget_usd"),
        non_interactive=out.non_interactive,
    )
    record = api.improve(request)
    artifacts, project_root = _shared.artifacts_for_record(record, opts["path"])
    return _shared.deliver_record(
        out, record, command="improve", artifacts=artifacts, project=project_root
    )


@click.command("improve", short_help=registry.SUMMARIES["improve"], epilog=_shared.EPILOG)
@click.argument("path", required=False, default=".")
@click.option(
    "--describe",
    type=str,
    default=None,
    help="start a first reward version from described behaviour",
)
@click.option(
    "--candidate-source",
    type=click.Choice(["owner_edit", "agent_draft", "generator_output", "supported_intervention"]),
    default=None,
)
@click.option(
    "--candidate", type=click.Path(path_type=str), default=None, help="candidate payload file"
)
@click.option(
    "--forecast",
    type=click.Path(path_type=str),
    default=None,
    help="closed prospective forecast declaration JSON",
)
@click.option(
    "--mark-pair", nargs=2, type=click.Path(path_type=str), default=None, metavar="LEFT RIGHT"
)
@click.option(
    "--why", type=str, default=None, help="why the two candidate versions are non-equivalent"
)
@_shared.common
@_shared.measuring
@_shared.verb("improve")
def improve(**opts) -> int:
    """start from a goal and run the loop."""
    return run(**opts)
