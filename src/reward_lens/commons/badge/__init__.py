"""Record-linked, non-verdict Shields payloads for saved canonical Assays."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "BadgeInputError",
    "FORBIDDEN_WORDS",
    "LINKED_RECORD_EXTENSION",
    "badge_from_assay",
    "badge_from_path",
    "read_payload",
    "write_payload",
]

FORBIDDEN_WORDS = frozenset({"passing", "safe", "secure", "certified"})
LINKED_RECORD_EXTENSION = "https://reward-lens.dev/extensions/badge-linked-record"
_WORD = re.compile(r"[a-z]+")
_REQUIRED = frozenset(
    {
        "schemaVersion",
        "label",
        "message",
        "color",
        "labelColor",
        "isError",
        "cacheSeconds",
        "link",
        "record",
        "provenance",
    }
)
_STATUS = {
    "fresh": ("blue", False),
    "stale": ("grey", False),
    "disputed": ("orange", True),
    "withdrawn": ("red", True),
}


class BadgeInputError(ValueError):
    """A canonical record lacks a declared input for the badge projection."""


def _utc(value: str | datetime, *, field: str) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise BadgeInputError(f"{field} must be timezone-aware")
        return value
    if not isinstance(value, str) or not value.endswith("Z"):
        raise BadgeInputError(f"{field} must be a UTC-Z timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise BadgeInputError(f"{field} is not an ISO-8601 timestamp") from error


def _entry_result(entry: Any) -> Mapping[str, Any]:
    result = getattr(entry, "result", None)
    return result if isinstance(result, Mapping) else {}


def _selected_reach(assay: Any, requested: str | None) -> tuple[int, Any, str]:
    candidates: list[tuple[int, Any, str]] = []
    for index, entry in enumerate(assay.measurement.reach):
        result = _entry_result(entry)
        surface = result.get("surface")
        if (
            entry.kind == "estimate"
            and entry.state in {"complete", "partial"}
            and entry.unit == "fraction"
            and type(entry.value) in {int, float}
            and 0 <= entry.value <= 1
            and isinstance(surface, str)
            and surface
        ):
            candidates.append((index, entry, surface))
    if requested is not None:
        matches = [item for item in candidates if item[1].entry_id == requested]
        if len(matches) != 1:
            raise BadgeInputError(
                f"badge reach entry {requested!r} is not one measured fraction; select an existing Reach entry id"
            )
        return matches[0]
    if not candidates:
        raise BadgeInputError(
            "reach is unavailable or not summarised; rerun a measured Reach panel or select a measured Reach entry"
        )
    if len(candidates) != 1:
        raise BadgeInputError(
            "multiple measured Reach entries exist; select one with badge_reach_entry_id rather than averaging them"
        )
    return candidates[0]


def _confirmed_exploit_ids(assay: Any) -> list[str]:
    """Count only reproduced, independently failed Seeker witnesses, never all findings."""
    witnesses: set[str] = set()
    for entry in assay.measurement.exploits:
        result = _entry_result(entry)
        evidence = result.get("outcome_evidence")
        if (
            entry.kind == "witness"
            and entry.state == "complete"
            and entry.method.id == "exploits.seeker"
            and result.get("outcome_passed") is False
            and isinstance(evidence, Mapping)
            and evidence.get("state") == "FAIL"
        ):
            witnesses.add(entry.entry_id)
    return sorted(
        {
            finding.id
            for finding in assay.findings
            if finding.kind == "fail" and witnesses.intersection(finding.entries)
        }
    )


def _linked_context(assay: Any, *, record_url: str) -> tuple[str, str, str | None, bool, bool]:
    extension = (assay.extensions or {}).get(LINKED_RECORD_EXTENSION, {})
    if not isinstance(extension, Mapping):
        raise BadgeInputError("badge linked-record extension must be an object")
    linked = extension.get("record", extension)
    if not isinstance(linked, Mapping):
        raise BadgeInputError("badge linked-record extension record must be an object")
    url = extension.get("url", record_url)
    if not isinstance(url, str) or not url:
        raise BadgeInputError("badge needs a linked record URL")
    identifier = str(extension.get("id") or linked.get("id") or assay.assay_id)
    database = linked.get("database_specific")
    rgx = database.get("rgx") if isinstance(database, Mapping) else {}
    lifecycle = str(linked.get("lifecycle") or (rgx or {}).get("lifecycle") or "").lower()
    withdrawn = bool(linked.get("withdrawn")) or lifecycle == "withdrawn"
    disputes = linked.get("disputes") or (rgx or {}).get("disputes")
    return identifier, url, lifecycle or None, withdrawn, bool(disputes) or lifecycle == "disputed"


def _status(
    *, created: datetime, now: datetime, stale_after: timedelta, withdrawn: bool, disputed: bool
) -> str:
    if withdrawn:
        return "withdrawn"
    if disputed:
        return "disputed"
    return "stale" if now - created >= stale_after else "fresh"


def _reject_verdict_words(payload: Mapping[str, Any]) -> None:
    words = {
        word
        for value in (payload.get("label"), payload.get("message"))
        if isinstance(value, str)
        for word in _WORD.findall(value.lower())
    }
    forbidden = sorted(words & FORBIDDEN_WORDS)
    if forbidden:
        raise BadgeInputError(
            f"badge vocabulary contains forbidden word(s): {', '.join(forbidden)}"
        )


def badge_from_assay(
    assay: Any,
    *,
    record_url: str,
    badge_reach_entry_id: str | None = None,
    now: str | datetime | None = None,
    stale_after: timedelta = timedelta(days=7),
) -> dict[str, Any]:
    """Project one saved canonical Assay to D-52 endpoint JSON without aggregation."""
    from reward_lens.contracts import Assay

    if not isinstance(assay, Assay):
        raise BadgeInputError("badge needs a validated canonical Assay")
    try:
        assay = Assay.model_validate(assay.to_dict())
    except Exception as error:
        raise BadgeInputError("badge needs a validated canonical Assay") from error
    if stale_after <= timedelta(0):
        raise BadgeInputError("badge stale_after must be positive")
    created = _utc(assay.created, field="record created")
    current = _utc(now, field="now") if now is not None else datetime.now(timezone.utc)
    index, reach, surface = _selected_reach(assay, badge_reach_entry_id)
    identifier, url, lifecycle, withdrawn, disputed = _linked_context(assay, record_url=record_url)
    state = _status(
        created=created,
        now=current,
        stale_after=stale_after,
        withdrawn=withdrawn,
        disputed=disputed,
    )
    color, is_error = _STATUS[state]
    exploit_ids = _confirmed_exploit_ids(assay)
    payload: dict[str, Any] = {
        "schemaVersion": 1,
        "label": "reward assay",
        "message": f"reach {float(reach.value):.2f} | exploits {len(exploit_ids)} | {created.date().isoformat()}",
        "color": color,
        "labelColor": "grey",
        "isError": is_error,
        "cacheSeconds": 3600,
        "link": url,
        "record": {
            "id": identifier,
            "url": url,
            "state": state,
            "lifecycle": lifecycle,
            "finding_ids": exploit_ids,
        },
        "provenance": {
            "reach_entry_id": reach.entry_id,
            "reach_surface": surface,
            "reach_pointer": f"/measurement/reach/{index}",
        },
    }
    _reject_verdict_words(payload)
    return payload


def write_payload(payload: Mapping[str, Any], path: str | Path) -> Path:
    if not _REQUIRED <= set(payload):
        raise BadgeInputError("badge payload is missing required endpoint fields")
    _reject_verdict_words(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return destination


def read_payload(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BadgeInputError(f"badge payload cannot be reopened: {error}") from error
    if not isinstance(value, dict) or not _REQUIRED <= set(value):
        raise BadgeInputError("badge payload is not a complete Shields endpoint object")
    _reject_verdict_words(value)
    return value


def badge_from_path(
    path: str | Path,
    *,
    badge_reach_entry_id: str | None = None,
    now: str | datetime | None = None,
    stale_after: timedelta = timedelta(days=7),
) -> dict[str, Any]:
    """Validate and reopen a saved Assay before projecting its endpoint JSON."""
    from reward_lens.api import open_record

    record_path = Path(path).resolve()
    return badge_from_assay(
        open_record(record_path),
        record_url=record_path.as_uri(),
        badge_reach_entry_id=badge_reach_entry_id,
        now=now,
        stale_after=stale_after,
    )
