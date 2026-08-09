"""The runtime validator: the frozen schema, `jsonschema-rs`, and the quantiser rule (D-66, D-10).

Four layers, in this order, each with its own reserved code:

1. the declared schema major (RL0601), which preserves the bytes it was given rather than coercing
   a record this build cannot read;
2. the declared-scale rule (RL0603), which is not a JSON Schema rule and so has to be walked;
3. the schema itself through `jsonschema-rs` (RL0604), zero runtime dependencies and structured
   error paths;
4. a refinement of layer 3: a required-property failure that lands on an entry's evidence is
   reported as RL0602 naming the kind and the field, because "required property 'observed' is
   missing" does not say which of the five kinds was being asked for.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import rfc8785

from .errors import (
    EntryKindEvidenceMissing,
    NumberNotQuantised,
    RecordInvalid,
    SchemaUnknownMajor,
)
from .models import KIND_EVIDENCE, SCHEMA_URL
from .quantise import scale_violations

__all__ = ["SCHEMA_PATH", "SCHEMA_MAJOR", "assay_schema", "validate_record"]

SCHEMA_MAJOR = 1
_SCHEMA_VERSION = re.compile(r"/schema/assay/(\d+)\.(\d+)/assay\.schema\.json$")
_QUOTED = re.compile(r'"([^"]+)" is a required property')

#: The schema ships beside the package in the wheel and beside the repo root in a checkout.
_CANDIDATES = (
    Path(__file__).resolve().parents[1] / "schema" / "assay" / "1.0" / "assay.schema.json",
    Path(__file__).resolve().parents[3] / "schema" / "assay" / "1.0" / "assay.schema.json",
)


def _read_schema() -> dict:
    for candidate in _CANDIDATES:
        if candidate.exists():
            # `json` is imported here and nowhere else in this package, for reading only.
            import json

            return json.loads(candidate.read_text(encoding="utf-8")), candidate  # type: ignore[return-value]
    raise FileNotFoundError(
        "assay.schema.json was not found beside the package or at the repo root; "
        f"looked in {', '.join(str(c) for c in _CANDIDATES)}"
    )


@lru_cache(maxsize=1)
def _loaded() -> tuple[dict, Path]:
    return _read_schema()  # type: ignore[return-value]


def assay_schema() -> dict:
    """The frozen specification, read once."""
    return _loaded()[0]


SCHEMA_PATH = _CANDIDATES[1]


@lru_cache(maxsize=1)
def _validator() -> Any:
    import jsonschema_rs

    return jsonschema_rs.validator_for(assay_schema())


def _declared_major(record: dict) -> int | None:
    declared = record.get("$schema")
    if not isinstance(declared, str):
        return None
    match = _SCHEMA_VERSION.search(declared)
    return int(match.group(1)) if match else None


def _missing_property(error: Any) -> str | None:
    name = getattr(getattr(error, "kind", None), "property", None)
    if isinstance(name, str):
        return name
    found = _QUOTED.search(str(getattr(error, "message", "")))
    return found.group(1) if found else None


def _as_entry_evidence(record: dict, error: Any) -> tuple[str, str] | None:
    """If this error is a missing evidence field on an entry, return (kind, field)."""
    missing = _missing_property(error)
    if missing is None:
        return None
    path = list(getattr(error, "instance_path", []) or [])
    if len(path) < 3 or path[0] != "measurement":
        return None
    section, index = path[1], path[2]
    try:
        entry = record["measurement"][section][index]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(entry, dict):
        return None
    kind = entry.get("kind")
    if kind not in KIND_EVIDENCE:
        return None
    if len(path) == 3 and missing in KIND_EVIDENCE[kind]:
        return kind, missing
    if len(path) == 4 and path[3] == kind:
        return kind, missing
    return None


def validate_record(record: dict, *, raw: bytes | None = None) -> None:
    """Refuse a record that this build cannot read, cannot trust, or cannot reproduce.

    `raw` is the bytes the record arrived as, if the caller still has them. A record from an
    unreadable schema major is returned to the caller with those bytes intact, because the only
    honest thing to do with a record you cannot interpret is to hand it back unaltered (D-64).
    """
    if not isinstance(record, Mapping):
        raise RecordInvalid(
            message=(
                f"the record does not validate against the frozen schema at /: a "
                f"{type(record).__name__} is not a record, which is a JSON object"
            ),
            remediation=(
                "the schema at schema/assay/1.0/assay.schema.json is the specification; the "
                "invalid fixtures beside it name each rule and the layer that catches it"
            ),
            context={"instance_path": "/", "type": type(record).__name__},
        )
    major = _declared_major(record)
    if major is not None and major != SCHEMA_MAJOR:
        preserved = raw if raw is not None else rfc8785.dumps(record)
        raise SchemaUnknownMajor(
            message=(
                f"this record declares assay schema major {major}; this build reads major "
                f"{SCHEMA_MAJOR} ({SCHEMA_URL})"
            ),
            remediation=(
                "read it with a reward-lens whose major matches, or migrate it with the fixtures "
                "beside the schema; the original bytes are on the error, unaltered"
            ),
            context={
                "declared": record.get("$schema"),
                "major": major,
                "expected_major": SCHEMA_MAJOR,
                "original_bytes": preserved,
            },
        )

    violations = scale_violations(record, assay_schema())
    if violations:
        field, _, detail = violations[0].partition(": ")
        scale = int(detail.rsplit(" ", 1)[-1]) if detail else None
        raise NumberNotQuantised(
            message=(
                f"{field} is not quantised to its declared scale: {detail}. Every number in the "
                "record is rounded half-to-even at the record boundary (D-10)"
            ),
            remediation=(
                "quantise the value with reward_lens.contracts.quantise before it enters the "
                "record; a number that needs more digits belongs in a string"
            ),
            context={"field": field, "detail": detail, "scale": scale, "all": violations},
        )

    errors = sorted(
        _validator().iter_errors(record),
        key=lambda e: (len(list(e.instance_path)), [str(p) for p in e.instance_path]),
    )
    if not errors:
        return

    for error in errors:
        evidence = _as_entry_evidence(record, error)
        if evidence is None:
            continue
        kind, missing = evidence
        raise EntryKindEvidenceMissing(
            message=(
                f"a {kind} entry carries {', '.join(KIND_EVIDENCE[kind])}, and the evidence its "
                f"kind calls for; this one is missing {missing}"
            ),
            remediation=(
                f"add {missing} to the {kind} entry, or record it as an absence entry with a "
                "missing access and a remedy (D-11, D-18)"
            ),
            context={
                "kind": kind,
                "missing": missing,
                "instance_path": [p for p in error.instance_path],
                "schema_path": [str(p) for p in error.schema_path],
            },
        )

    first = errors[0]
    instance_path = [p for p in first.instance_path]
    schema_path = [str(p) for p in first.schema_path]
    raise RecordInvalid(
        message=(
            f"the record does not validate against the frozen schema at /"
            f"{'/'.join(str(p) for p in instance_path)}: {first.message}"
        ),
        remediation=(
            "the schema at schema/assay/1.0/assay.schema.json is the specification; the invalid "
            "fixtures beside it name each rule and the layer that catches it"
        ),
        context={
            "instance_path": instance_path,
            "schema_path": schema_path,
            "detail": first.message,
            "errors": len(errors),
        },
    )
