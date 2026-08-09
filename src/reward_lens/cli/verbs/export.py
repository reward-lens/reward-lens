"""`reward-lens export`: write a revision, a bundle, a badge, or a SARIF file.

Two vocabularies wanted the same flag. The shared `--format` names one of the five output formats
of D-19 to D-33 and means the same thing on every verb, so the artefact this command writes is
`--kind`. The commission writes `export --format html` in one place and that still works: a
`--format` value which is not one of the five output formats is read as the kind. A word that is
both, `sarif`, is the output format here as everywhere, and the SARIF artefact is `--kind sarif`.
"""

from __future__ import annotations

import click

from .. import output, registry
from . import _shared

#: The artefact `export` writes.
KINDS = ("html", "sarif", "badge", "revision", "bundle")


def read_kind(opts: dict) -> None:
    """Accept `--format <kind>` for `--kind <kind>`, and only where the two cannot collide.

    An output format stays an output format, so `--format sarif` is the SARIF envelope and not the
    SARIF artefact; a word that is neither is left alone, for `make_output` to refuse as the bad
    format it is.
    """
    asked = (opts.get("format_") or "").strip().lower()
    if not asked or asked in output.FORMATS or asked not in KINDS:
        return
    opts["kind"] = asked
    opts["format_"] = None


def run(*, out, kind, **opts) -> int:
    from reward_lens import api

    request = api.ExportRequest(format=kind)
    result = api.export(request)
    return _shared.deliver_record(out, result, command="export")


@click.command("export", short_help=registry.SUMMARIES["export"], epilog=_shared.EPILOG)
@click.option("--kind", "-t", type=click.Choice(KINDS), default="bundle",
              help="the artefact to write: revision, bundle, badge, sarif or html")
@_shared.common
@_shared.verb("export", normalise=read_kind)
def export(**opts) -> int:
    """Write a revision, a bundle, a badge, or a SARIF file."""
    return run(**opts)
