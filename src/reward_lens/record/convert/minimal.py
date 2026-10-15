"""Read the small declared JSONL trace format used when no framework reader applies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from reward_lens.product.trace.errors import RecordIntegrity
from reward_lens.product.trace.integrity import CheckInput, enforce
from reward_lens.product.trace.record import SERIES_NAMES, Policy, Series, TraceRecord, TraceRow

__all__ = ["READER", "MinimalReader"]


class MinimalReader:
    name = "minimal"

    def can_read(self, path: Any) -> bool:
        candidate = Path(path)
        return (
            candidate.is_file()
            and candidate.suffix == ".jsonl"
            and candidate.name == "records.jsonl"
        )

    def read(self, path: Any, *, policy: Policy = "strict") -> TraceRecord:
        artifact = Path(path)
        root = artifact.parent
        provenance_path = root / "PROVENANCE.json"
        if not provenance_path.is_file():
            raise RecordIntegrity(
                message="PROVENANCE.json is required for minimal JSONL",
                remediation="supply the original JSONL and its provenance record",
                context={
                    "rule": "trace.fixture_provenance",
                    "row": "PROVENANCE.json",
                    "source": str(root),
                },
            )
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        observed = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if provenance.get("artifact", {}).get("sha256") != observed:
            raise RecordIntegrity(
                message="provenance digest differs for records.jsonl",
                remediation="restore the recorded JSONL artifact",
                context={
                    "rule": "trace.fixture_provenance",
                    "row": "records.jsonl",
                    "source": str(root),
                },
            )
        raw_rows = [
            json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines() if line
        ]
        rows = [
            TraceRow(
                index=index,
                step=int(raw["step"]),
                group_id=str(raw["group_id"]) if raw.get("group_id") is not None else None,
                row_id=str(raw["id"]),
                components={name: value for name, value in raw.get("components", {}).items()},
                reward_recorded=raw.get("reward_recorded"),
                reward_applied=raw.get("reward_applied"),
                advantage=raw.get("advantage"),
                selection=raw.get("selection"),
            )
            for index, raw in enumerate(raw_rows)
        ]
        names = sorted({name for raw in raw_rows for name in raw.get("components", {})})
        components = {
            name: Series(name=name, values=tuple(row.components.get(name) for row in rows))
            for name in names
        }
        series = {
            name: Series(name=name, values=tuple(getattr(row, name) for row in rows))
            for name in SERIES_NAMES
        }
        results = enforce(
            CheckInput(
                rows=tuple(rows),
                series=series,
                components=components,
                declared_rows=len(rows),
                source=str(artifact),
            ),
            policy=policy,
        )
        return TraceRecord(
            reader=self.name,
            source=str(artifact),
            run_id=str(provenance.get("run_id", artifact.stem)),
            rows=tuple(rows),
            series=series,
            components=components,
            access_level="RECORD",
            access_statement="RECORD: values declared by the JSONL producer.",
            policy=policy,
            complete=True,
            completeness_statement="the provenance-bound JSONL file was read in full",
            subject_ref=observed,
            depends_on=("digest:minimal_jsonl",),
            framework="minimal-jsonl",
            integrity=results,
        )


READER = MinimalReader()
