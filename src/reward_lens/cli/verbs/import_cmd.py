"""`reward-lens import`: receive, validate, retain, and reopen a record or bundle."""

from __future__ import annotations

from pathlib import Path

import click

from .. import registry
from . import _shared


def run(*, out, path, verify, save_to, **opts) -> int:
    from reward_lens import api

    result = api.import_record(api.ImportRequest(path=Path(path), verify=verify, dest=save_to))
    if result.verification_message:
        out.warn(result.verification_message)
    return _shared.deliver_record(
        out,
        result.record,
        command="import",
        artifacts={"assay": str(result.record_path), "receipt": str(result.receipt_path)},
        project=str(Path(path).parent),
    )


@click.command("import", short_help=registry.SUMMARIES["import"], epilog=_shared.EPILOG)
@click.argument("path")
@click.option(
    "--verify", is_flag=True, help="verify the Export signature before retaining the bytes"
)
@click.option(
    "--save-to",
    type=click.Path(path_type=Path),
    default=None,
    help="directory for source bytes and the ingest receipt",
)
@_shared.common
@_shared.verb("import")
def import_(**opts) -> int:
    """Validate a report and attach it to a project."""
    return run(**opts)
