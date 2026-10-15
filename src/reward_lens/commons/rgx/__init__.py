"""Local RGX records for reward-hacking incidents.

RGX is a project record namespace, not a CVE namespace.  The records below
preserve the source catalogue's reported incidents; they make no claim that a
third party has registered, verified, or measured them here.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from enum import StrEnum
from importlib import resources
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import rfc8785

__all__ = [
    "Affected",
    "Assessment",
    "Catalogue",
    "Dispute",
    "EvidenceOrigin",
    "Lifecycle",
    "Range",
    "RGXRecord",
    "allocate_id",
    "seeded_records",
]

_ID = re.compile(r"RGX-[0-9]{4}-[0-9]{4}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_CVE = re.compile(r"CVE-[0-9]{4}-[0-9]{4,}$", re.IGNORECASE)
_CATALOGUE_PATH = "src/reward_lens/commons/rgx/incident_catalog.md"
_CATALOGUE_DIGEST = "sha256:ad5847d1531cf2845a62b3af2d1fc7535357feed42788c8d4baaffc9007d66bf"
_SOURCE_ORIGIN = "product-strategy-notes/02-incident-catalog.md"


class Lifecycle(StrEnum):
    """D-51 lifecycle states.  Fix proposal, implementation, and re-audit differ."""

    DRAFT = "DRAFT"
    PRIVATE_DISCLOSURE = "PRIVATE_DISCLOSURE"
    PUBLISHED = "PUBLISHED"
    DISPUTED = "DISPUTED"
    FIXED = "FIXED"
    RE_AUDITED = "RE_AUDITED"
    WITHDRAWN = "WITHDRAWN"


class EvidenceOrigin(StrEnum):
    """Where the evidence described by this local record came from."""

    PROJECT_CATALOGUE = "project_catalogue"
    EXTERNAL_REPORT = "external_report"
    LOCAL_MEASUREMENT = "local_measurement"


_TRANSITIONS: dict[Lifecycle, frozenset[Lifecycle]] = {
    Lifecycle.DRAFT: frozenset(
        {Lifecycle.PRIVATE_DISCLOSURE, Lifecycle.PUBLISHED, Lifecycle.WITHDRAWN}
    ),
    Lifecycle.PRIVATE_DISCLOSURE: frozenset({Lifecycle.PUBLISHED, Lifecycle.WITHDRAWN}),
    Lifecycle.PUBLISHED: frozenset({Lifecycle.DISPUTED, Lifecycle.FIXED, Lifecycle.WITHDRAWN}),
    Lifecycle.DISPUTED: frozenset({Lifecycle.PUBLISHED, Lifecycle.FIXED, Lifecycle.WITHDRAWN}),
    Lifecycle.FIXED: frozenset({Lifecycle.RE_AUDITED, Lifecycle.WITHDRAWN}),
    Lifecycle.RE_AUDITED: frozenset({Lifecycle.DISPUTED, Lifecycle.FIXED, Lifecycle.WITHDRAWN}),
    Lifecycle.WITHDRAWN: frozenset(),
}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return tuple(sorted(_freeze(item) for item in value))
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    return value


def _collection(value: Any, name: str) -> tuple[Any, ...]:
    """Accept a real collection, never a scalar that happens to be iterable."""
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be a collection")
    try:
        return tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a collection") from exc


def _mapping_collection(value: Any, name: str) -> tuple[Mapping[str, Any], ...]:
    values = _collection(value, name)
    if any(not isinstance(item, Mapping) for item in values):
        raise ValueError(f"{name} must contain mappings")
    return tuple(_freeze(item) for item in values)


@dataclass(frozen=True)
class Range:
    """One OSV affected-range event sequence."""

    events: tuple[Mapping[str, str], ...]

    def __post_init__(self) -> None:
        events = _mapping_collection(self.events, "affected range events")
        if not events or any(
            set(event) - {"introduced", "fixed", "last_affected"} for event in events
        ):
            raise ValueError(
                "an affected range needs only introduced, fixed, or last_affected events"
            )
        if any(len(event) != 1 or not next(iter(event.values())).strip() for event in events):
            raise ValueError("each affected range event needs one nonempty version marker")
        object.__setattr__(self, "events", events)

    def to_osv(self) -> dict[str, Any]:
        return {"type": "ECOSYSTEM", "events": _plain(self.events)}


@dataclass(frozen=True)
class Affected:
    """OSV-shaped affected target. ``ranges=None`` is explicitly unknown."""

    package: str
    ranges: tuple[Range | Mapping[str, Any], ...] | None

    def __post_init__(self) -> None:
        if not self.package.strip():
            raise ValueError("affected package must be nonempty")
        if self.ranges is not None:
            ranges = _collection(self.ranges, "affected ranges")
            if any(not isinstance(item, (Range, Mapping)) for item in ranges):
                raise ValueError("affected ranges must contain ranges or event mappings")
            parsed = tuple(
                item
                if isinstance(item, Range)
                else Range(events=tuple(item["events"]) if "events" in item else (item,))
                for item in ranges
            )
            if not parsed:
                raise ValueError("a known affected range must contain at least one range")
            object.__setattr__(self, "ranges", parsed)

    def to_osv(self) -> dict[str, Any]:
        value: dict[str, Any] = {"package": {"name": self.package}}
        if self.ranges is not None:
            value["ranges"] = [item.to_osv() for item in self.ranges]
        return value


@dataclass(frozen=True)
class Assessment:
    """A cited original report or a separately retained response."""

    text: str
    source: str
    assessed: str

    def __post_init__(self) -> None:
        if not self.text.strip() or not self.source.strip() or not self.assessed.endswith("Z"):
            raise ValueError("assessment needs text, source, and a UTC-Z timestamp")


@dataclass(frozen=True)
class Dispute:
    """Both sides of a dispute, without a replacement operation."""

    original: Assessment
    response: Assessment

    def __post_init__(self) -> None:
        if self.original is self.response or self.original == self.response:
            raise ValueError("original and response assessments must be separate")


@dataclass(frozen=True)
class RGXRecord:
    """Immutable OSV-shaped local exploit record with a citable definition digest."""

    id: str
    definition: Mapping[str, Any]
    published: str
    affected: tuple[Affected, ...]
    evidence_origin: EvidenceOrigin
    measured_here: bool
    evidence_source: str
    evidence_digest: str
    aliases: tuple[str, ...] = ()
    related: tuple[str, ...] = ()
    severity: tuple[Mapping[str, str], ...] = ()
    references: tuple[Mapping[str, str], ...] = ()
    credits: tuple[Mapping[str, str], ...] = ()
    database_specific: Mapping[str, Any] = field(default_factory=dict)
    modified: str | None = None
    disputes: tuple[Dispute, ...] = ()
    _lifecycle: Lifecycle = field(default=Lifecycle.DRAFT, init=False, repr=False)
    withdrawn: str | None = field(default=None, init=False)
    withdrawal: Assessment | None = field(default=None, init=False, repr=False)
    content_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.id):
            raise ValueError("id must match RGX-YYYY-NNNN")
        if not self.definition or not self.published.endswith("Z"):
            raise ValueError("record needs a nonempty definition and UTC-Z published timestamp")
        if not isinstance(self.definition, Mapping):
            raise ValueError("record definition must be a mapping")
        if not isinstance(self.database_specific, Mapping):
            raise ValueError("database_specific must be a mapping")
        affected = _collection(self.affected, "affected targets")
        if not affected or any(not isinstance(item, Affected) for item in affected):
            raise ValueError("record needs at least one affected target")
        packages = [item.package for item in affected]
        if len(packages) != len(set(packages)):
            raise ValueError("affected targets must not repeat a package")
        if not self.evidence_source.strip():
            raise ValueError("evidence source must be nonempty")
        if not _DIGEST.fullmatch(self.evidence_digest):
            raise ValueError("evidence digest must be sha256:<64 lowercase hex>")
        if not isinstance(self.evidence_origin, EvidenceOrigin):
            raise ValueError("evidence origin must be an EvidenceOrigin")
        if self.measured_here != (self.evidence_origin is EvidenceOrigin.LOCAL_MEASUREMENT):
            raise ValueError("measured_here must agree with evidence origin")
        aliases = _collection(self.aliases, "aliases")
        if any(not isinstance(alias, str) or not alias.strip() for alias in aliases):
            raise ValueError("aliases must be nonempty strings")
        if any(_CVE.fullmatch(alias) for alias in aliases):
            raise ValueError("RGX records must not claim CVE aliases")
        related = _collection(self.related, "related identifiers")
        if any(not isinstance(item, str) or not item.strip() for item in related):
            raise ValueError("related identifiers must be nonempty strings")
        disputes = _collection(self.disputes, "disputes")
        if any(not isinstance(item, Dispute) for item in disputes):
            raise ValueError("disputes must contain assessments")
        if disputes:
            raise ValueError("use dispute() to add retained assessments")
        object.__setattr__(self, "definition", _freeze(self.definition))
        object.__setattr__(self, "affected", affected)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "related", related)
        object.__setattr__(self, "database_specific", _freeze(self.database_specific))
        object.__setattr__(self, "severity", _mapping_collection(self.severity, "severity"))
        object.__setattr__(self, "references", _mapping_collection(self.references, "references"))
        object.__setattr__(self, "credits", _mapping_collection(self.credits, "credits"))
        object.__setattr__(self, "disputes", disputes)
        object.__setattr__(self, "modified", self.modified or self.published)
        object.__setattr__(self, "content_id", _definition_digest(self._definition_payload()))

    @property
    def lifecycle(self) -> Lifecycle:
        """Current state; only ``transition()`` and ``dispute()`` can change it."""
        return self._lifecycle

    def _definition_payload(self) -> dict[str, Any]:
        return {
            "definition": _plain(self.definition),
            "affected": [item.to_osv() for item in self.affected],
            "aliases": list(self.aliases),
            "related": list(self.related),
            "severity": _plain(self.severity),
            "references": _plain(self.references),
            "credits": _plain(self.credits),
            "database_specific": _plain(self.database_specific),
        }

    def transition(self, target: Lifecycle, *, modified: str | None = None) -> "RGXRecord":
        if target not in _TRANSITIONS[self.lifecycle]:
            raise ValueError(f"illegal lifecycle transition {self.lifecycle} -> {target}")
        if target is Lifecycle.DISPUTED:
            raise ValueError("use dispute() to retain original and response assessments")
        if target is Lifecycle.WITHDRAWN:
            raise ValueError("use withdraw() to retain a withdrawal assessment")
        return self._copy(modified=modified or self.modified, lifecycle=target)

    def _copy(self, *, lifecycle: Lifecycle | None = None, **changes: Any) -> "RGXRecord":
        clone = replace(self, **changes)
        object.__setattr__(clone, "_lifecycle", self.lifecycle if lifecycle is None else lifecycle)
        return clone

    def with_affected(self, affected: tuple[Affected, ...]) -> "RGXRecord":
        replacement = tuple(affected)
        if any(not isinstance(item, Affected) for item in replacement):
            raise ValueError("affected targets must contain Affected values")
        replacement_by_package = {item.package: item for item in replacement}
        if len(replacement_by_package) != len(replacement):
            raise ValueError("affected targets must not repeat a package")
        for original in self.affected:
            candidate = replacement_by_package.get(original.package)
            if candidate is None:
                raise ValueError("existing affected target cannot be removed")
            if original.ranges is not None and candidate.ranges is None:
                raise ValueError("known affected range cannot be replaced by unknown")
        return self._copy(affected=replacement)

    def dispute(self, dispute: Dispute, *, modified: str | None = None) -> "RGXRecord":
        if Lifecycle.DISPUTED not in _TRANSITIONS[self.lifecycle]:
            raise ValueError(
                f"illegal lifecycle transition {self.lifecycle} -> {Lifecycle.DISPUTED}"
            )
        clone = self._copy(lifecycle=Lifecycle.DISPUTED, modified=modified or self.modified)
        object.__setattr__(clone, "disputes", self.disputes + (dispute,))
        return clone

    def withdraw(self, assessment: Assessment) -> "RGXRecord":
        """Withdraw through an attributable assessment instead of an unrecorded state change."""
        if Lifecycle.WITHDRAWN not in _TRANSITIONS[self.lifecycle]:
            raise ValueError(
                f"illegal lifecycle transition {self.lifecycle} -> {Lifecycle.WITHDRAWN}"
            )
        clone = self._copy(lifecycle=Lifecycle.WITHDRAWN, modified=assessment.assessed)
        object.__setattr__(clone, "withdrawn", assessment.assessed)
        object.__setattr__(clone, "withdrawal", assessment)
        return clone

    def to_osv(self) -> dict[str, Any]:
        """The OSV-shaped projection, with RGX-specific facts namespaced locally."""
        return {
            "id": self.id,
            "aliases": list(self.aliases),
            "related": list(self.related),
            "published": self.published,
            "modified": self.modified,
            **({"withdrawn": self.withdrawn} if self.withdrawn else {}),
            "severity": _plain(self.severity),
            "references": _plain(self.references),
            "credits": _plain(self.credits),
            "affected": [item.to_osv() for item in self.affected],
            "database_specific": {
                **_plain(self.database_specific),
                "rgx": {
                    "content_id": self.content_id,
                    "definition": _plain(self.definition),
                    "lifecycle": self.lifecycle.value,
                    "evidence_origin": self.evidence_origin.value,
                    "measured_here": self.measured_here,
                    "evidence_source": self.evidence_source,
                    "evidence_digest": self.evidence_digest,
                    "disputes": [
                        {"original": item.original.__dict__, "response": item.response.__dict__}
                        for item in self.disputes
                    ],
                    **(
                        {"withdrawal": self.withdrawal.__dict__}
                        if self.withdrawal is not None
                        else {}
                    ),
                },
            },
        }


@dataclass(frozen=True)
class Catalogue:
    """A local collection that refuses allocation collisions before it is published anywhere."""

    records: tuple[RGXRecord, ...]

    def __post_init__(self) -> None:
        ids = [record.id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("RGX identifier collision")


def _definition_digest(definition: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(rfc8785.dumps(definition)).hexdigest()


def allocate_id(year: int, existing: Iterable[str]) -> str:
    """Allocate the first free RGX sequence in a year, refusing malformed or repeated input."""
    if not 1000 <= year <= 9999:
        raise ValueError("year must have four digits")
    seen: set[str] = set()
    for identifier in existing:
        if not _ID.fullmatch(identifier):
            raise ValueError("existing id must match RGX-YYYY-NNNN")
        if identifier in seen:
            raise ValueError("RGX identifier collision")
        seen.add(identifier)
    for sequence in range(1, 10_000):
        candidate = f"RGX-{year:04d}-{sequence:04d}"
        if candidate not in seen:
            return candidate
    raise ValueError("RGX identifier space exhausted for year")


_ROW = re.compile(r"^([A-F][0-9]+)\.\s+(.+)$", re.MULTILINE)
_ALIAS = re.compile(r"\b([A-F][0-9]+)\s*=\s*([A-F][0-9]+)\b")


def _catalogue_bytes() -> bytes:
    """Read and authenticate the checked-in source snapshot before minting seeds."""
    data = resources.files(__package__).joinpath("incident_catalog.md").read_bytes()
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    if digest != _CATALOGUE_DIGEST:
        raise ValueError(f"pinned incident catalogue digest mismatch: {digest}")
    return data


def _seed_rows() -> tuple[tuple[str, str], ...]:
    source = _catalogue_bytes().decode("utf-8")
    rows = tuple(_ROW.findall(source))
    aliases = dict(_ALIAS.findall(source))
    if len(rows) != 40:
        raise ValueError(f"pinned incident catalogue has {len(rows)} rows, expected 40")
    if aliases != {"F1": "A4", "F2": "A10", "E3": "A11", "F3": "A11"}:
        raise ValueError("pinned incident catalogue alias mapping changed")
    distinct = tuple((key, heading) for key, heading in rows if key not in aliases)
    if len(distinct) != 36 or len({key for key, _ in distinct}) != 36:
        raise ValueError("pinned incident catalogue does not resolve to 36 distinct incidents")
    return distinct


def seeded_records() -> tuple[RGXRecord, ...]:
    """Return source-derived, reported-only records from the authenticated snapshot.

    The source's detection-timing section aliases F1=A4, F2=A10, E3=A11, and
    F3=A11.  Those aliases are verified from the source bytes and do not mint
    duplicate local identifiers.
    """
    records = tuple(
        RGXRecord(
            id=f"RGX-2026-{number:04d}",
            definition={"incident_key": key, "source_heading": heading},
            published="2026-09-10T00:00:00Z",
            affected=(Affected(package="external/reported-incident", ranges=None),),
            evidence_origin=EvidenceOrigin.PROJECT_CATALOGUE,
            measured_here=False,
            evidence_source=_CATALOGUE_PATH,
            evidence_digest=_CATALOGUE_DIGEST,
            references=({"type": "REPORT", "url": f"{_CATALOGUE_PATH}#{key}"},),
            database_specific={
                "source_row": key,
                "source_origin": _SOURCE_ORIGIN,
                "reported_only": True,
            },
        ).transition(Lifecycle.PUBLISHED)
        for number, (key, heading) in enumerate(_seed_rows(), start=1)
    )
    Catalogue(records)
    return records
