"""The separate maintainer command surface required by D-05 and D-19.

``reward-lens`` is for someone auditing a reward system.  Building project
reference material, the public registry, or an incident namespace is a
different responsibility, so those routes live here and never enter user root
help.  A route may be present before its owning package is integrated; in that
case it refuses by owner and missing capability instead of pretending that a
projection, a proficiency round, or a sweep happened.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import click


def _unavailable(*, route: str, owner: str, capability: str) -> None:
    """Stop at a named unintegrated consumer instead of producing a fake result."""

    raise click.ClickException(f"{route} is unavailable: {owner} is not integrated; {capability}.")


@click.group(
    name="reward-lens-commons",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100},
)
def commons() -> None:
    """Maintainer operations for the reward-lens commons, separate from the user CLI."""


@commons.command("claims")
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--no-unbound", is_flag=True, help="Check tags and evidence references only.")
@click.option("--baseline", type=click.Path(path_type=Path), help="Known unbound-number baseline.")
@click.option("--exclude", multiple=True, metavar="SUBSTRING", help="Skip matching document paths.")
@click.option(
    "--write-baseline", is_flag=True, help="Write the ratchet baseline from current input."
)
@click.option(
    "--store", type=click.Path(path_type=Path), help="Evidence store used to resolve claims."
)
def claims(
    paths: tuple[Path, ...],
    no_unbound: bool,
    baseline: Path | None,
    exclude: tuple[str, ...],
    write_baseline: bool,
    store: Path | None,
) -> None:
    """Run the existing claims ratchet over maintainer-selected documents."""

    from reward_lens.artifacts.claims import baseline_key, check_files, load_baseline
    from reward_lens.core.store import EvidenceStore

    documents: list[Path] = []
    for path in paths:
        documents.extend(sorted(path.rglob("*.md")) if path.is_dir() else [path])
    if exclude:
        documents = [
            path for path in documents if not any(fragment in str(path) for fragment in exclude)
        ]
    evidence_store = EvidenceStore(store) if store is not None else None
    report = check_files(documents, evidence_store)
    if write_baseline:
        if baseline is None:
            raise click.UsageError("--write-baseline needs --baseline")
        keys = sorted(baseline_key(item) for item in report.unbound)
        baseline.write_text(
            "# Numbers in prose with no evidence id, recorded before the check existed.\n"
            "# This list may shrink and must not grow. Bind one and delete its line.\n"
            + "\n".join(keys)
            + "\n",
            encoding="utf-8",
        )
        click.echo(f"wrote {len(keys)} baselined unbound numbers to {baseline}")
        return
    known = load_baseline(baseline) if baseline is not None else set()
    if known:
        before = len(report.unbound)
        report.unbound = [item for item in report.unbound if baseline_key(item) not in known]
        click.echo(f"baseline exempted {before - len(report.unbound)} of {before} unbound numbers.")
    click.echo(report.render())
    if no_unbound:
        passed = all(item.ok for item in report.results) and not report.unresolved_refs
    else:
        passed = report.ok
    raise click.exceptions.Exit(0 if passed else 1)


@commons.group("registry")
def registry() -> None:
    """Build a public registry only from validated public projections."""


@registry.command("build")
@click.option(
    "--out", type=click.Path(path_type=Path), help="Destination for a generated registry."
)
def registry_build(out: Path | None) -> None:
    """Build the registry after P-REGISTRY supplies its validated builder."""
    from reward_lens.commons.registry import build_default_registry, open_registry, write_registry

    destination = out or Path("reward-lens-registry.json")
    try:
        written = write_registry(build_default_registry(), destination)
        reopened = open_registry(written)
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"registry build: entries={len(reopened.entries)} out={written} reopened=yes")


@commons.command("sweep")
@click.option(
    "--battery",
    default="BiasBattery,DistortionV2,RobustnessSNR",
    show_default=True,
    help="Comma-separated observable names to plan.",
)
@click.option("--gpu-budget", type=click.FloatRange(min=0.0), default=0.0, show_default=True)
@click.option(
    "--store", type=click.Path(path_type=Path), help="Evidence store used for cost priors."
)
@click.option("--execute", is_flag=True, help="Request real model scoring after planning.")
def sweep(battery: str, gpu_budget: float, store: Path | None, execute: bool) -> None:
    """Plan and price the existing Atlas sweep; real model scoring remains unavailable here."""

    from reward_lens.artifacts.atlas import Atlas
    from reward_lens.core.provenance import Cost
    from reward_lens.core.store import EvidenceStore

    observables = tuple(part.strip() for part in battery.split(",") if part.strip())
    if not observables:
        raise click.UsageError("--battery needs at least one observable")
    atlas = Atlas.standard()
    evidence_store = EvidenceStore(store) if store is not None else None
    plan = atlas.sweep(
        (entry.fingerprint for entry in atlas.entries),
        observables,
        Cost(gpu_seconds=gpu_budget),
        store=evidence_store,
        execute=False,
    )
    click.echo(plan.summary())
    if execute:
        _unavailable(
            route="sweep --execute",
            owner="P-REGISTRY",
            capability=(
                "the local Atlas can plan from recorded costs but has no integrated real-model "
                "sweep runner"
            ),
        )


@commons.group("reference")
def reference() -> None:
    """Construct and verify locally acquired reference material."""


@reference.command("build")
@click.option("--identifier", default="diagnostic-v3", show_default=True)
def reference_build(identifier: str) -> None:
    """Build the adopted diagnostic-v3 object and verify every local integrity boundary."""

    from reward_lens.reference import load
    from reward_lens.reference.load import load_diagnostic_v3_material, reverify

    material = load(identifier)
    receipt = reverify(material)
    constructed = load_diagnostic_v3_material(material)
    click.echo(
        "reference build: "
        f"{material.identifier} grade={material.grade.value} "
        f"payload_sha256={receipt.verified_digest} pairs={constructed.pair_count} "
        f"dimensions={len(constructed.dimensions)}"
    )


@commons.group("proficiency")
def proficiency() -> None:
    """Run sealed local proficiency rounds with separate public and private artifacts."""


@proficiency.command("run")
@click.option(
    "--charter",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="A sealed charter published before this round.",
)
@click.option(
    "--public-output",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Destination for the coded public projection.",
)
@click.option(
    "--private-output",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Destination for identified reports, written mode 0600.",
)
def proficiency_run(charter: Path, public_output: Path, private_output: Path) -> None:
    """Execute the available local detector methods from an already sealed charter."""

    if public_output.resolve() == private_output.resolve():
        raise click.UsageError("--public-output and --private-output must be distinct")
    from reward_lens.reference.proficiency import (
        ProficiencyRefusal,
        load_sealed_charter,
        persist_round,
        run_local_round,
    )

    try:
        report = run_local_round(load_sealed_charter(charter))
        sealed = persist_round(
            report,
            public_destination=public_output,
            private_destination=private_output,
        )
    except ProficiencyRefusal as exc:
        raise click.ClickException(str(exc)) from exc
    external_bucket = report.external_rows[0].bucket
    click.echo(
        "proficiency run: "
        f"public={sealed.public_path} private={sealed.private_path} "
        f"actual_methods={len(report.actual_results)} external_bucket={external_bucket} "
        "four_way=unavailable (requires four actual detector results)"
    )


@commons.group("rgx")
def rgx() -> None:
    """Allocate local RGX identifiers without mutating a catalogue."""


@rgx.command("alloc")
@click.option("--year", type=click.IntRange(1000, 9999), required=True)
@click.option(
    "--existing",
    multiple=True,
    metavar="RGX-ID",
    help="Additional already-allocated local identifiers; repeated once per identifier.",
)
def rgx_alloc(year: int, existing: tuple[str, ...]) -> None:
    """Allocate the first unused identifier after the verified, integrated seed catalogue."""

    from reward_lens.commons.rgx import allocate_id, seeded_records

    seed_ids = tuple(record.id for record in seeded_records())
    try:
        identifier = allocate_id(year, (*seed_ids, *existing))
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(identifier)


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point, declared only after the dependency writer releases it."""

    try:
        result = commons.main(args=list(argv) if argv is not None else None, standalone_mode=False)
    except click.ClickException as error:
        error.show()
        return error.exit_code
    except click.UsageError as error:
        error.show()
        return error.exit_code
    return result if isinstance(result, int) else 0


if __name__ == "__main__":  # pragma: no cover - console-script behavior
    raise SystemExit(main())


__all__ = ["commons", "main"]
