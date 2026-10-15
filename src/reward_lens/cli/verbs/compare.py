"""`reward-lens compare`: persist and report one or more registered comparison rungs."""

from __future__ import annotations

from pathlib import Path

import click

from .. import registry
from . import _shared


def run(*, out, **opts) -> int:
    from reward_lens import api

    bank_path = opts.get("bank")
    version_a = opts.get("version_a") or opts.get("baseline")
    version_b = opts.get("version_b") or opts.get("candidate")
    if not version_a or not version_b:
        raise click.UsageError(
            "compare needs two evaluator versions as BASELINE CANDIDATE or --version-a/--version-b"
        )
    if bank_path is not None:
        source = Path(bank_path)
    elif opts.get("run_dir") is not None:
        source = Path(opts["run_dir"])
    elif opts.get("short_fork_config") is not None:
        source = Path(opts["short_fork_config"])
    elif opts.get("controlled_update_config") is not None:
        source = Path(opts["controlled_update_config"])
    elif opts.get("transfer_config") is not None:
        source = Path(opts["transfer_config"])
    else:
        source = Path(".")
    request = api.CompareRequest(
        baseline=version_a,
        candidate=version_b,
        bank=bank_path,
        project=opts.get("project"),
        rungs=tuple(opts["rungs"] or (1,)),
        run_dir=opts.get("run_dir"),
        baseline_config=opts.get("baseline_config"),
        candidate_config=opts.get("candidate_config"),
        selection_config=opts.get("selection_config"),
        controlled_update_config=opts.get("controlled_update_config"),
        short_fork_config=opts.get("short_fork_config"),
        transfer_config=opts.get("transfer_config"),
        offline=True if opts.get("offline") is None else bool(opts["offline"]),
        seed=20260911 if opts.get("seed") is None else int(opts["seed"]),
        name=opts.get("name"),
    )
    record = api.compare(request)
    # Projectless Rung 3 writes beside, never inside, its sealed run directory.
    artifact_root = source.parent
    artifacts, project_root = _shared.artifacts_for_record(
        record, opts.get("project"), artifact_root
    )
    return _shared.deliver_record(
        out,
        record,
        command="compare",
        artifacts=artifacts,
        project=project_root,
    )


@click.command("compare", short_help=registry.SUMMARIES["compare"], epilog=_shared.EPILOG)
@click.argument("baseline", required=False)
@click.argument("candidate", required=False)
@click.option("--rung", "rungs", type=click.IntRange(1, 6), multiple=True)
@click.option("--bank", type=click.Path(path_type=Path), default=None, metavar="JSONL")
@click.option("--run-dir", type=click.Path(path_type=Path), default=None, metavar="DIR")
@click.option("--project", type=click.Path(path_type=Path), default=None, metavar="DIR")
@click.option("--version-a", default=None, metavar="VERSION")
@click.option("--version-b", default=None, metavar="VERSION")
@click.option("--baseline-config", type=click.Path(path_type=Path), default=None, metavar="JSON")
@click.option("--candidate-config", type=click.Path(path_type=Path), default=None, metavar="JSON")
@click.option("--selection-config", type=click.Path(path_type=Path), default=None, metavar="JSON")
@click.option(
    "--controlled-update-config", type=click.Path(path_type=Path), default=None, metavar="JSON"
)
@click.option("--short-fork-config", type=click.Path(path_type=Path), default=None, metavar="JSON")
@click.option("--transfer-config", type=click.Path(path_type=Path), default=None, metavar="JSON")
@_shared.common
@_shared.measuring
@_shared.verb("compare")
def compare(**opts) -> int:
    """what changes if the reward changes."""
    return run(**opts)
