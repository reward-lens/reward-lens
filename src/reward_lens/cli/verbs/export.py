"""`reward-lens export`: write a revision, a bundle, a badge, or a SARIF file.

Two vocabularies wanted the same flag. The shared `--format` names one of the five output formats
of D-19 to D-33 and means the same thing on every verb, so the artefact this command writes is
`--kind`. The commission writes `export --format html` in one place and that still works: a
`--format` value which is not one of the five output formats is read as the kind. A word that is
both, `sarif`, is the output format here as everywhere, and the SARIF artefact is `--kind sarif`.
"""

from __future__ import annotations

from pathlib import Path

import click

from .. import ids, output, registry
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


def _report_url(artifact, destination: Path) -> str:
    """Return the actual local file a tracker may link after this export has finished."""
    report = destination / "report.html" if artifact.kind.value == "bundle" else destination
    return report.resolve().as_uri()


def _document(*, artifact, destination: Path, report_url: str, tracker=None) -> dict:
    """The public receipt for a completed file export, including its independently checkable ID."""
    artifacts = {
        "export": str(destination),
        "report_url": report_url,
        "kind": artifact.kind.value,
        "tier": artifact.tier.value,
        "manifest_digest": artifact.manifest_digest,
    }
    if tracker is not None:
        artifacts["tracker"] = tracker
    return output.envelope(
        command="export",
        execution={"state": "complete", "offline": True},
        artifacts=artifacts,
    )


def _text(artifact, destination: Path) -> str:
    return (
        f"  Wrote      {artifact.kind.value} export  ·  {destination}\n"
        f"  Manifest   {artifact.manifest_digest}"
    )


def _tracker_receipt(*, target: str, record_path: Path, report_url: str) -> dict:
    """Write to a discovered tracker only after the local export exists, then expose its readback."""
    from reward_lens import api
    from reward_lens.integrations.trackers import RecordSummary, export_to

    reference = export_to(
        target,
        summary=RecordSummary.from_record(api.open_record(record_path)),
        report_url=report_url,
    )
    return {
        "backend": reference.backend,
        "report_url": reference.report_url,
        "path": str(reference.path) if reference.path is not None else None,
        "readback": dict(reference.readback),
    }


def run(
    *,
    out,
    kind,
    record,
    dest,
    tier="A",
    backend="ed25519",
    to=None,
    badge_reach_entry_id=None,
    **opts,
) -> int:
    from reward_lens import api

    if badge_reach_entry_id is not None and kind != "badge":
        raise click.UsageError("--reach-entry is only valid with --kind badge")
    record_path = ids.validate_path_argument(record, field="record")
    destination = ids.validate_path_argument(dest, field="dest")
    request = api.ExportRequest(
        record=record_path,
        dest=destination,
        format=kind,
        tier=tier,
        backend=backend,
        badge_reach_entry_id=badge_reach_entry_id,
        offline=True,
        non_interactive=True,
    )
    artifact = api.export(request)
    report_url = _report_url(artifact, destination)
    tracker = (
        _tracker_receipt(
            target=ids.validate_identifier(to, field="to"),
            record_path=record_path,
            report_url=report_url,
        )
        if to is not None
        else None
    )
    return out.deliver(
        document=_document(
            artifact=artifact,
            destination=destination,
            report_url=report_url,
            tracker=tracker,
        ),
        text=_text(artifact, destination),
        exit_code=0,
    )


@click.command("export", short_help=registry.SUMMARIES["export"], epilog=_shared.EPILOG)
@click.option(
    "--kind",
    "-t",
    type=click.Choice(KINDS),
    default="bundle",
    help="the artefact to write: revision, bundle, badge, sarif or html",
)
@click.option(
    "--record",
    type=click.Path(path_type=Path),
    required=True,
    metavar="ASSAY_JSON",
    help="saved canonical Assay JSON to reopen and export",
)
@click.option(
    "--dest",
    type=click.Path(path_type=Path),
    required=True,
    metavar="PATH",
    help="file destination, or a directory for --kind bundle",
)
@click.option(
    "--tier",
    type=click.Choice(["A", "B", "C"]),
    default="A",
    help="expected sealed embedding tier; the saved record remains authoritative",
)
@click.option(
    "--backend",
    type=click.Choice(["ed25519", "sigstore"]),
    default="ed25519",
    help="bundle attestation backend; sigstore requires its declared extra",
)
@click.option(
    "--reach-entry",
    "badge_reach_entry_id",
    default=None,
    metavar="ENTRY_ID",
    help="the one measured Reach entry to project; valid only for --kind badge",
)
@click.option(
    "--to",
    default=None,
    metavar="TRACKER",
    help="publish the completed export through a configured tracker",
)
@_shared.common
@_shared.verb("export", normalise=read_kind)
def export(**opts) -> int:
    """Write a revision, a bundle, a badge, or a SARIF file."""
    return run(**opts)
