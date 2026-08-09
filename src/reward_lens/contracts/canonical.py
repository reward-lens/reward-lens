"""Canonical bytes and the assay id (D-10).

`canonical_bytes()` is the only producer of hashed bytes in this project. It removes the digest and
signature keys rather than nulling them, then hands the rest to `rfc8785`, which is RFC 8785 JCS
over UTF-8. Python's own JSON serialiser is never used here and never anywhere on a digest path: it is not ES6
number formatting, it writes `1.0` where ES6 writes `1` and `1e-07` where ES6 writes `1e-7`, and it
emits `NaN` unless told not to.

The bytes are a function of the record's values and of nothing else. A model reaches them through
`to_dict()`, which is `model_dump(mode="json", by_alias=True)` with the schema's optional fields
omitted when they hold None, so a record built in code and the same record loaded from disk give
one digest. `exclude_unset` would make the digest depend on the construction path, and P-STORE's
nine dependency digests are taken over exactly these bytes.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping

import rfc8785

__all__ = ["EXCLUDED_FROM_DIGEST", "canonical_bytes", "digest", "strip_for_digest"]

#: Removed before hashing, not nulled: the record's own id, the attestation that is made over the
#: id, the decision signature, and the one block that holds everything machine-specific.
EXCLUDED_FROM_DIGEST: tuple[str, ...] = (
    "assay_id",
    "attestation",
    "decision.signature",
    "environment_excluded_from_digest",
)


def _as_mapping(record: Any) -> dict:
    """The record's values as JSON-ready data. `to_dict()` is the one serialisation (D-10)."""
    to_dict = getattr(record, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(record, Mapping):
        return dict(record)
    from .errors import RecordInvalid

    raise RecordInvalid(
        message=(
            f"the record does not validate against the frozen schema at /: a "
            f"{type(record).__name__} has no JSON form, so it has no canonical bytes"
        ),
        remediation=(
            "canonical_bytes takes an Assay or a mapping of the record's own values: build the "
            "record first, then hash it"
        ),
        context={"instance_path": "/", "type": type(record).__name__},
    )


def strip_for_digest(record: Any) -> dict:
    """The record as it is hashed: a deep copy with `EXCLUDED_FROM_DIGEST` removed."""
    stripped = copy.deepcopy(_as_mapping(record))
    for dotted in EXCLUDED_FROM_DIGEST:
        head, _, tail = dotted.partition(".")
        if not tail:
            stripped.pop(head, None)
            continue
        parent = stripped.get(head)
        if isinstance(parent, dict):
            parent.pop(tail, None)
    return stripped


def canonical_bytes(record: Any) -> bytes:
    """RFC 8785 JCS bytes of the record, with the excluded keys removed. The only hashed bytes."""
    return rfc8785.dumps(strip_for_digest(record))


def digest(record: Any) -> str:
    """`sha256:<64 lowercase hex>` over `canonical_bytes(record)`."""
    return "sha256:" + hashlib.sha256(canonical_bytes(record)).hexdigest()
