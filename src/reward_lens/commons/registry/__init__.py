"""The generated public registry, with provenance kept beside every entry.

The registry is deliberately a small document type rather than a second assay
schema. Its entries are projections of records or authenticated historical
sources. A generated document may be copied to the site, but its contents are
always rebuilt from these sources and reopened before a caller receives it.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import rfc8785

from reward_lens.contracts import digest
from reward_lens.render.projections import Projection, View
from reward_lens.render.projections import registry as registry_projection

__all__ = [
    "Assessment",
    "RegistryDocument",
    "RegistryEntry",
    "build_default_registry",
    "entry_from_projection",
    "open_registry",
    "site_projection",
    "write_registry",
]

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_TITLE = re.compile(r"<title>RM Card:\s*([^<]+)</title>", re.IGNORECASE)
_GAPS = re.compile(r'<p class="unvalidated"[^>]*>(.*?)</p>', re.IGNORECASE | re.DOTALL)
_AUTHORITATIVE_X8 = {
    "dossier_sha256": "0dcbcf05d420d2099170ee7338dfa7e2ff305f88d0fe286aec495a9112d99c42",
    "dossier_section": "c.7.3, lines 1047-1072",
}


def _text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _path(value: str) -> str:
    value = _text(value, "path")
    if not value.startswith("/") or "//" in value or "?" in value:
        raise ValueError("path must be an absolute public path without // or ?")
    return value


@dataclass(frozen=True)
class Assessment:
    """One readable retained account of a disputed registry entry."""

    kind: str
    text: str
    source: str

    def __post_init__(self) -> None:
        if self.kind not in {"original", "author_reaudit"}:
            raise ValueError("assessment kind must be original or author_reaudit")
        _text(self.text, "assessment text")
        _text(self.source, "assessment source")


@dataclass(frozen=True)
class RegistryEntry:
    """One generated public record, carrying its origin rather than a new claim."""

    path: str
    title: str
    summary: str
    state: str
    record_type: str
    source_projection_digest: str
    measurement_scope: str
    provenance: str
    measured_here: bool
    primary_link: str | None = None
    migrated: bool = False
    original_scope: str | None = None
    refutation: str | None = None
    assessments: tuple[Assessment, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path(self.path))
        for field in (
            "title",
            "summary",
            "state",
            "record_type",
            "measurement_scope",
            "provenance",
        ):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        if not _SHA256.fullmatch(self.source_projection_digest):
            raise ValueError("source_projection_digest must be a sha256 digest")
        if not isinstance(self.measured_here, bool):
            raise ValueError("measured_here must be a bool")
        if self.primary_link is not None and not self.primary_link.startswith("https://"):
            raise ValueError("primary_link must be an https URL")
        if self.record_type == "public_record":
            if self.measured_here or self.state != "not_measured_here" or self.primary_link is None:
                raise ValueError("public_record must be not_measured_here with a primary_link")
        if self.migrated:
            if self.state != "migrated" or self.measured_here:
                raise ValueError("migrated entry must be state=migrated and not measured here")
            if self.original_scope is None or self.refutation is None:
                raise ValueError("migrated entry requires original scope and refutation text")
            _text(self.original_scope, "original_scope")
            _text(self.refutation, "refutation")
        elif self.original_scope is not None or self.refutation is not None:
            raise ValueError("only a migrated entry may carry original scope or refutation")
        assessments = tuple(self.assessments)
        if any(not isinstance(item, Assessment) for item in assessments):
            raise ValueError("assessments must contain Assessment values")
        if assessments and {item.kind for item in assessments} != {"original", "author_reaudit"}:
            raise ValueError(
                "a disputed entry must retain both original and author_reaudit assessments"
            )
        if len({item.kind for item in assessments}) != len(assessments):
            raise ValueError("a disputed entry may hold one assessment per side")
        object.__setattr__(self, "assessments", assessments)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "title": self.title,
            "summary": self.summary,
            "state": self.state,
            "record_type": self.record_type,
            "source_projection_digest": self.source_projection_digest,
            "measurement_scope": self.measurement_scope,
            "provenance": self.provenance,
            "measured_here": self.measured_here,
            "primary_link": self.primary_link,
            "migrated": self.migrated,
            "original_scope": self.original_scope,
            "refutation": self.refutation,
            "assessments": [asdict(item) for item in self.assessments],
        }


@dataclass(frozen=True)
class RegistryDocument:
    """A checked public registry document, stable enough for static publication."""

    entries: tuple[RegistryEntry, ...]
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError("registry document version must be 1")
        entries = tuple(self.entries)
        if not entries:
            raise ValueError("registry must contain at least one entry")
        if any(not isinstance(entry, RegistryEntry) for entry in entries):
            raise ValueError("registry entries must be RegistryEntry values")
        paths = [entry.path for entry in entries]
        if len(paths) != len(set(paths)):
            raise ValueError("registry paths must be unique")
        object.__setattr__(self, "entries", tuple(sorted(entries, key=lambda entry: entry.path)))

    def as_dict(self) -> dict[str, Any]:
        return {"version": self.version, "entries": [entry.as_dict() for entry in self.entries]}


def _projection_digest(projection: Projection) -> str:
    return digest(
        {
            "view": projection.view.value,
            "fields": dict(projection.fields),
            "source_pointers": list(projection.source_pointers),
        }
    )


def entry_from_projection(
    projection: Projection,
    *,
    path: str,
    title: str,
    summary: str,
    state: str,
    record_type: str,
    measurement_scope: str,
    provenance: str,
    measured_here: bool,
    primary_link: str | None = None,
    assessments: tuple[Assessment, ...] = (),
) -> RegistryEntry:
    """Turn the accepted P-VIEWS registry projection into one validated entry."""
    if projection.view is not View.REGISTRY:
        raise ValueError("registry entries require a View.REGISTRY projection")
    if not {"assay_id", "subject", "decision", "provenance"} <= set(projection.fields):
        raise ValueError("registry projection is missing a required public field")
    return RegistryEntry(
        path=path,
        title=title,
        summary=summary,
        state=state,
        record_type=record_type,
        source_projection_digest=_projection_digest(projection),
        measurement_scope=measurement_scope,
        provenance=provenance,
        measured_here=measured_here,
        primary_link=primary_link,
        assessments=assessments,
    )


_SEED_PREFIX = "reward_lens/commons/registry/seeds"


def _seed_file(*parts: str):
    return resources.files(__package__).joinpath("seeds", *parts)


def _seed_manifest() -> Mapping[str, Any]:
    value = json.loads(_seed_file("manifest.json").read_bytes())
    if not isinstance(value, Mapping) or value.get("version") != 1:
        raise ValueError("registry package seed manifest is invalid")
    return value


def _verified_seed_bytes(*parts: str, expected_sha256: str) -> bytes:
    raw = _seed_file(*parts).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"registry packaged seed {'/'.join(parts)} digest mismatch: {actual}")
    return raw


def _root_seed_bytes(root: Path, *parts: str, expected_sha256: str) -> bytes:
    raw = root.joinpath(*parts).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"registry injected seed {'/'.join(parts)} digest mismatch: {actual}")
    return raw


def _e1_entry(root: Path | None = None) -> RegistryEntry:
    from reward_lens import api

    manifest = _seed_manifest()
    e1 = manifest.get("e1")
    if not isinstance(e1, Mapping) or not isinstance(e1.get("sha256"), str):
        raise ValueError("registry seed manifest has no E1 digest")
    original = e1.get("source")
    if not isinstance(original, str):
        raise ValueError("registry seed manifest has no E1 source")
    raw = (
        _root_seed_bytes(root, *Path(original).parts, expected_sha256=e1["sha256"])
        if root is not None
        else _verified_seed_bytes("e1.assay.json", expected_sha256=e1["sha256"])
    )
    with tempfile.TemporaryDirectory(prefix="reward-lens-registry-e1-") as directory:
        source = Path(directory) / "e1.assay.json"
        source.write_bytes(raw)
        record = api.open_record(source)
    return entry_from_projection(
        registry_projection(record),
        path="/registry/assays/aisi-olmo-e1/",
        title="AISI OLMo reward-hacking E1",
        summary="Source-derived E1 reward-side reconstruction with named absent training-side telemetry.",
        state="measured_here",
        record_type="assay",
        measured_here=True,
        measurement_scope="Pinned rollout parquet and registered E1 analysis; no applied-update, gradient, or effective-sample-size telemetry.",
        provenance=f"{_SEED_PREFIX}/e1.assay.json (privacy-portable derivation of {original}; sha256:{e1['sha256']})",
    )


def _plain_html(fragment: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", fragment).split())


def _migrated_entries(root: Path | None = None) -> tuple[RegistryEntry, ...]:
    """Use P-IMPORT's actual preview without writing beside immutable fixture bytes."""
    from reward_lens.product.ingest import migrate_v3_card

    manifest = _seed_manifest()
    cards = manifest.get("campaign_cards")
    if not isinstance(cards, Mapping) or len(cards) != 10:
        raise ValueError("registry seed manifest must name ten campaign cards")
    entries: list[RegistryEntry] = []
    for name, expected in sorted(cards.items()):
        if not isinstance(name, str) or not isinstance(expected, str):
            raise ValueError("registry seed manifest has an invalid campaign-card digest")
        raw = (
            _root_seed_bytes(
                root,
                "tests",
                "migration",
                "fixtures",
                "campaign-v3-cards",
                name,
                expected_sha256=expected,
            )
            if root is not None
            else _verified_seed_bytes("campaign-v3-cards", name, expected_sha256=expected)
        )
        with tempfile.TemporaryDirectory(prefix="reward-lens-registry-") as directory:
            copied = Path(directory) / name
            copied.write_bytes(raw)
            migration = migrate_v3_card(copied)
        html = migration.preview["rendered_html"]
        title_match = _TITLE.search(html)
        if title_match is None:
            raise ValueError(f"migrated card {name} has no card title")
        gaps_match = _GAPS.search(html)
        explicit_gaps = (
            _plain_html(gaps_match.group(1))
            if gaps_match
            else "No explicit gap paragraph was emitted."
        )
        source_digest = (
            "sha256:"
            + hashlib.sha256(
                rfc8785.dumps(
                    {"preview": migration.preview, "validation": migration.validation_report}
                )
            ).hexdigest()
        )
        identifier = title_match.group(1).strip()
        entries.append(
            RegistryEntry(
                path=f"/registry/migrated/{identifier}/",
                title=identifier,
                summary="Migrated July campaign reward-model card; retained as historical source material.",
                state="migrated",
                record_type="migrated_card",
                source_projection_digest=source_digest,
                measurement_scope="Original campaign card only; intent, level of detail, and composite semantics remain unknown after migration.",
                provenance=(
                    f"{_SEED_PREFIX}/campaign-v3-cards/{name} "
                    f"(byte-identical copy of tests/migration/fixtures/campaign-v3-cards/{name}; "
                    f"sha256:{migration.validation_report['source_sha256']})"
                ),
                measured_here=False,
                migrated=True,
                original_scope="The rendered campaign-v3 card is retained verbatim as the original scope; it is not a contemporary audit record.",
                refutation=f"The source retains its explicit limitation: {explicit_gaps}",
            )
        )
    if len(entries) != 10:
        raise ValueError(
            f"expected ten authenticated migrated campaign cards, found {len(entries)}"
        )
    return tuple(entries)


def _public_record_entries() -> tuple[RegistryEntry, ...]:
    """Primary-linked field records, explicitly never represented as local measurements."""
    manifest = _seed_manifest()
    public = manifest.get("public_records")
    if not isinstance(public, Mapping) or not isinstance(public.get("sha256"), str):
        raise ValueError("registry seed manifest has no public-record digest")
    authority = public.get("authoritative_x8")
    if authority != _AUTHORITATIVE_X8:
        raise ValueError("registry seed manifest has mismatched authoritative X8 identity")
    original = public.get("source")
    if not isinstance(original, str):
        raise ValueError("registry seed manifest has no public-record source")
    rows = json.loads(_verified_seed_bytes("public-records.json", expected_sha256=public["sha256"]))
    if (
        not isinstance(rows, list)
        or len(rows) != 6
        or any(not isinstance(row, Mapping) for row in rows)
    ):
        raise ValueError("registry public-record seed must contain six objects")
    entries: list[RegistryEntry] = []
    for item in rows:
        required = {"key", "arxiv", "authors", "benchmark", "estimand", "sources", "status"}
        if set(item) != required or not all(
            isinstance(item[field], str) and item[field] for field in required - {"sources"}
        ):
            raise ValueError("registry public-record seed has an invalid row")
        sources = item["sources"]
        if (
            not isinstance(sources, list)
            or not sources
            or any(not isinstance(pair, list) or len(pair) != 2 for pair in sources)
        ):
            raise ValueError("registry public-record seed has invalid source rows")
        source = {
            "field_scan": original,
            "key": item["key"],
            "arxiv": item["arxiv"],
            "sources": sources,
            "status": item["status"],
        }
        entries.append(
            RegistryEntry(
                path=f"/registry/public-records/{item['key']}/",
                title=f"{item['authors']}: {item['benchmark']}",
                summary=f"Field record for {item['estimand']}.",
                state="not_measured_here",
                record_type="public_record",
                source_projection_digest="sha256:"
                + hashlib.sha256(rfc8785.dumps(source)).hexdigest(),
                measurement_scope="Published field result as extracted from the checked-in field-scan dossier; reward-lens did not measure this record.",
                provenance=(
                    f"{_SEED_PREFIX}/public-records.json "
                    f"(byte-identical declaration set transcribed from {original}; "
                    f"source lines {', '.join(str(pair[0]) for pair in sources)}; sha256:{public['sha256']}; "
                    f"authoritative X8 dossier sha256:{authority['dossier_sha256']}; "
                    f"section {authority['dossier_section']})"
                ),
                measured_here=False,
                primary_link=f"https://arxiv.org/abs/{item['arxiv']}",
            )
        )
    return tuple(entries)


def build_default_registry(*, root: Path | None = None) -> RegistryDocument:
    """Build the source-backed default registry without fetching or rewriting its inputs."""
    return RegistryDocument(
        entries=(_e1_entry(root), *_migrated_entries(root), *_public_record_entries())
    )


def write_registry(registry: RegistryDocument, destination: Path | str) -> Path:
    """Write JCS bytes and reopen them before returning the public artifact path."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = rfc8785.dumps(registry.as_dict())
    if destination.exists() and destination.read_bytes() != payload:
        raise ValueError("registry destination already contains different generated bytes")
    if not destination.exists():
        destination.write_bytes(payload)
    reopened = open_registry(destination)
    if reopened != registry:
        raise RuntimeError("saved registry did not reopen as the generated registry")
    return destination


def _assessment(value: Mapping[str, Any]) -> Assessment:
    return Assessment(kind=value.get("kind"), text=value.get("text"), source=value.get("source"))


def _entry(value: Mapping[str, Any]) -> RegistryEntry:
    known = {
        "path",
        "title",
        "summary",
        "state",
        "record_type",
        "source_projection_digest",
        "measurement_scope",
        "provenance",
        "measured_here",
        "primary_link",
        "migrated",
        "original_scope",
        "refutation",
        "assessments",
    }
    unknown = set(value) - known
    if unknown:
        raise ValueError("registry entry has unknown fields: " + ", ".join(sorted(unknown)))
    return RegistryEntry(
        path=value.get("path"),
        title=value.get("title"),
        summary=value.get("summary"),
        state=value.get("state"),
        record_type=value.get("record_type"),
        source_projection_digest=value.get("source_projection_digest"),
        measurement_scope=value.get("measurement_scope"),
        provenance=value.get("provenance"),
        measured_here=value.get("measured_here"),
        primary_link=value.get("primary_link"),
        migrated=value.get("migrated", False),
        original_scope=value.get("original_scope"),
        refutation=value.get("refutation"),
        assessments=tuple(_assessment(item) for item in value.get("assessments", ())),
    )


def open_registry(path: Path | str) -> RegistryDocument:
    """Open a generated document and reject malformed, unknown, or incomplete public claims."""
    source = Path(path)
    try:
        value = json.loads(source.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"registry document at {source} is not valid JSON") from error
    if not isinstance(value, Mapping) or set(value) != {"version", "entries"}:
        raise ValueError("registry document must contain exactly version and entries")
    if not isinstance(value["entries"], list):
        raise ValueError("registry document entries must be a list")
    if any(not isinstance(item, Mapping) for item in value["entries"]):
        raise ValueError("registry document entries must be objects")
    return RegistryDocument(
        version=value["version"], entries=tuple(_entry(item) for item in value["entries"])
    )


def site_projection(registry: RegistryDocument) -> dict[str, Any]:
    """The closed static-site projection, retaining the qualifications behind each card.

    ``path`` identifies a registry record.  It is not a promise that a detail
    route exists: the static page renders it as a locator and the readable
    material remains inline.  That makes a generated record useful before a
    publisher elects to add a detail page, without turning absent pages into
    broken links.
    """
    entries = [_site_entry(entry) for entry in registry.entries]
    return {
        "version": 1,
        "supportedFilters": {
            "states": sorted({entry["state"] for entry in entries}),
            "recordTypes": sorted({entry["recordType"] for entry in entries}),
        },
        "entries": entries,
    }


def _checked_in_source(provenance: str) -> str:
    """Extract the checked-in artifact component from this generator's provenance text."""
    artifact = provenance.split(" (", 1)[0].split(", source lines", 1)[0]
    allowed = ("fleet/", "tests/", "field-scan-2026/", f"{_SEED_PREFIX}/")
    if not artifact.startswith(allowed) or ".." in Path(artifact).parts:
        raise ValueError("site projection needs a checked-in public artifact path")
    return artifact


def _site_sources(entry: RegistryEntry) -> list[dict[str, str]]:
    artifact = _checked_in_source(entry.provenance)
    sources = [{"label": "Checked-in source artifact", "url": artifact}]
    if entry.primary_link is not None:
        sources.insert(0, {"label": "Primary public record", "url": entry.primary_link})
    return sources


def _site_assessment(entry: RegistryEntry) -> str:
    if entry.assessments:
        return " ".join(f"{item.kind}: {item.text}" for item in entry.assessments)
    if entry.refutation is not None:
        return entry.refutation
    if entry.record_type == "public_record":
        return entry.summary
    return entry.measurement_scope


def _site_entry(entry: RegistryEntry) -> dict[str, Any]:
    """One exact public row, without internal identifiers or imaginary detail pages."""
    assessment = _site_assessment(entry)
    if not assessment.strip():  # defensive: every entry needs an explicit qualification
        raise ValueError("site projection needs a nonempty assessment")
    return {
        "path": entry.path,
        "title": entry.title,
        "summary": entry.summary,
        "state": entry.state,
        "recordType": entry.record_type,
        "scope": entry.measurement_scope,
        "provenance": entry.provenance,
        "assessment": assessment,
        "sources": _site_sources(entry),
        "primaryLink": entry.primary_link,
        "originalScope": entry.original_scope,
        "refutation": entry.refutation,
        "assessments": [asdict(item) for item in entry.assessments],
    }
