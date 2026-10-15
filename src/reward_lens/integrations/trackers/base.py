"""The narrow tracker-export contract shared by every backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from reward_lens import contracts

__all__ = ["Availability", "RecordSummary", "TrackerBackend", "TrackerRef"]


@dataclass(frozen=True)
class Availability:
    is_available: bool
    credential: str | None = None
    detail: str | None = None

    @classmethod
    def available(cls) -> "Availability":
        return cls(True)

    @classmethod
    def unavailable(cls, credential: str, detail: str) -> "Availability":
        return cls(False, credential=credential, detail=detail)


@dataclass(frozen=True)
class RecordSummary:
    """Record-owned values only; derived after P-EXPORT has reopened the saved Assay."""

    assay_id: str
    subject_version: str
    decision: str
    created: str
    producer_version: str

    @classmethod
    def from_record(cls, record: contracts.Assay) -> "RecordSummary":
        return cls(
            record.assay_id,
            record.subject.version.digest,
            record.decision.state,
            record.created,
            record.producer.version,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "assay_id": self.assay_id,
            "subject_version": self.subject_version,
            "decision": self.decision,
            "created": self.created,
            "producer_version": self.producer_version,
        }


@dataclass(frozen=True)
class TrackerRef:
    backend: str
    report_url: str
    path: Path | None
    readback: Mapping[str, Any]


class TrackerBackend(Protocol):
    name: str

    def available(self) -> Availability: ...
    def write(self, summary: RecordSummary, report_url: str) -> TrackerRef: ...
