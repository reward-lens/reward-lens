"""`reward-lens join`: render D-47's column from one saved Assay."""

from __future__ import annotations

import json
from pathlib import Path

import click

from .. import registry
from . import _shared


def run(*, record: Path, **_: object) -> int:
    from reward_lens.product import join as product_join

    document = json.loads(record.read_text(encoding="utf-8"))
    click.echo(product_join.render_text(product_join.join(document)))
    return 0


@click.command("join", short_help=registry.SUMMARIES["join"], epilog=_shared.EPILOG)
@click.argument("record", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_shared.common
@_shared.verb("join")
def join(**opts: object) -> int:
    """Connect an audit family to its saved trace and compare evidence."""
    return run(**opts)  # type: ignore[arg-type]
