"""The account-free tracker backend: JSON beside the saved report."""

from __future__ import annotations

from ._local import file_report_parent, write_payload
from .base import Availability, RecordSummary, TrackerRef


class FileTracker:
    name = "file"

    def available(self) -> Availability:
        return Availability.available()

    def write(self, summary: RecordSummary, report_url: str) -> TrackerRef:
        return write_payload(
            file_report_parent(report_url) / "reward-lens-file-tracker.json",
            backend=self.name,
            summary=summary,
            report_url=report_url,
        )


def backend() -> FileTracker:
    return FileTracker()
