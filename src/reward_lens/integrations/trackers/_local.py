"""Standard-library-only persistence helpers for offline tracker fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import unquote, urlparse

from .base import RecordSummary, TrackerRef


def file_report_parent(report_url: str) -> Path:
    parsed = urlparse(report_url)
    if parsed.scheme != "file":
        raise ValueError("offline tracker exports need an absolute file URI for the saved report")
    report = Path(unquote(parsed.path))
    if not report.is_absolute():
        raise ValueError("offline tracker exports need an absolute file URI for the saved report")
    return report.parent


def write_payload(
    path: Path, *, backend: str, summary: RecordSummary, report_url: str
) -> TrackerRef:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"backend": backend, "summary": summary.to_dict(), "report_url": report_url}
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return TrackerRef(backend, report_url, path, json.loads(path.read_text(encoding="utf-8")))
