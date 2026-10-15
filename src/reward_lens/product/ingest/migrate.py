"""Explicit, non-destructive migration for historical v3 cards."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rfc8785

from reward_lens.contracts import RecordInvalid

__all__ = ["Migration", "migrate_v3_card"]


@dataclass(frozen=True)
class Migration:
    preview: dict[str, Any]
    backup_path: Path
    provenance_link: dict[str, str]
    validation_report: dict[str, Any]


def migrate_v3_card(path: Path) -> Migration:
    """Preview one v3 card and retain its source without guessing semantic fields."""
    source = Path(path)
    raw = source.read_bytes()
    old: dict[str, Any]
    source_format: str
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        decoded = None
    if isinstance(decoded, dict) and decoded.get("version") == 3:
        old = decoded
        source_format = "v3-json-card"
    else:
        try:
            html = raw.decode("utf-8")
        except UnicodeDecodeError as failure:
            raise RecordInvalid(
                message=f"historical card at {source} is neither a v3 JSON card nor UTF-8 campaign card HTML",
                remediation="supply an emitted v3 card before requesting a migration preview",
                context={"path": str(source), "original_bytes": raw},
            ) from failure
        title = re.search(r"<title>RM Card:\s*([^<]+)</title>", html, flags=re.IGNORECASE)
        if title is None:
            raise RecordInvalid(
                message=f"historical card at {source} is not an emitted v3 campaign card",
                remediation="supply an emitted v3 card before requesting a migration preview",
                context={"path": str(source), "original_bytes": raw},
            )
        old = {
            "version": 3,
            "id": title.group(1).strip(),
            "record_kind": "campaign_rm_card",
            "rendered_html": html,
        }
        source_format = "campaign-rm-card/html"
    preview = dict(old)
    for field in ("intent", "level_of_detail", "composite"):
        preview.setdefault(field, "unknown")
    backup = source.with_name(source.name + ".v3.backup")
    if backup.exists() and backup.read_bytes() != raw:
        raise RecordInvalid(
            message=f"migration backup {backup} already has different bytes",
            remediation="move the conflicting backup aside, preserving both historical sources",
            context={"path": str(backup)},
        )
    if not backup.exists():
        backup.write_bytes(raw)
    report = {
        "valid": True,
        "source_version": 3,
        "source_format": source_format,
        "source_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "semantic_fields_marked_unknown": [
            field for field in ("intent", "level_of_detail", "composite") if field not in old
        ],
        "preview_digest": "sha256:" + hashlib.sha256(rfc8785.dumps(preview)).hexdigest(),
    }
    report_path = source.with_name(source.name + ".migration.validation.json")
    report_path.write_bytes(rfc8785.dumps(report))
    return Migration(
        preview=preview,
        backup_path=backup,
        provenance_link={
            "source": str(source),
            "backup": str(backup),
            "source_format": source_format,
        },
        validation_report=report,
    )
