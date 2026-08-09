"""`reward-lens open`: open a report, or the workbench."""

from __future__ import annotations

import click

from .. import ids, registry
from . import _shared


def run(*, out, path, **opts) -> int:
    from reward_lens import api

    record = api.open_record(ids.validate_path_argument(path, field="path"))
    return _shared.deliver_record(out, record, command="open")


@click.command("open", short_help=registry.SUMMARIES["open"], epilog=_shared.EPILOG)
@click.argument("path")
@_shared.common
@_shared.verb("open")
def open_(**opts) -> int:
    """Open a report, or the workbench."""
    return run(**opts)
