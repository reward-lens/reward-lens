"""`reward-lens import`: validate a report and attach it to a project."""

from __future__ import annotations

import click

from .. import ids, registry
from . import _shared


def run(*, out, path, **opts) -> int:
    from reward_lens import api

    record = api.open_record(ids.validate_path_argument(path, field="path"))
    return _shared.deliver_record(out, record, command="import")


@click.command("import", short_help=registry.SUMMARIES["import"], epilog=_shared.EPILOG)
@click.argument("path")
@_shared.common
@_shared.verb("import")
def import_(**opts) -> int:
    """Validate a report and attach it to a project."""
    return run(**opts)
