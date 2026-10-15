"""Read Inspect logs only through Inspect's documented ``read_eval_log`` API."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from reward_lens.product.trace.errors import RecordIntegrity
from reward_lens.product.trace.integrity import CheckInput, enforce
from reward_lens.product.trace.record import SERIES_NAMES, Policy, Series, TraceRecord, TraceRow

__all__ = ["READER", "InspectEvalReader"]


class InspectEvalReader:
    name = "inspect_eval_wire"

    def can_read(self, path: Any) -> bool:
        candidate = Path(path)
        return candidate.is_file() and candidate.suffix == ".eval"

    def read(self, path: Any, *, policy: Policy = "strict") -> TraceRecord:
        artifact = Path(path)
        provenance_path = artifact.parent / "PROVENANCE.json"
        if not provenance_path.is_file():
            raise RecordIntegrity(
                message="PROVENANCE.json is required for an Inspect eval log",
                remediation="supply the original .eval file and its provenance record",
                context={
                    "rule": "trace.fixture_provenance",
                    "row": "PROVENANCE.json",
                    "source": str(artifact),
                },
            )
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        observed = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if provenance.get("artifact", {}).get("sha256") != observed:
            raise RecordIntegrity(
                message="provenance digest differs for the Inspect .eval artifact",
                remediation="restore the recorded .eval artifact",
                context={
                    "rule": "trace.fixture_provenance",
                    "row": artifact.name,
                    "source": str(artifact),
                },
            )
        from inspect_ai.log import read_eval_log

        header = read_eval_log(artifact, header_only=True)
        log = read_eval_log(artifact)
        samples = list(log.samples or [])
        rows = [
            TraceRow(
                index=index,
                step=int(getattr(sample, "epoch", 0)),
                group_id=None,
                row_id=str(sample.id),
            )
            for index, sample in enumerate(samples)
        ]

        def absent(name: str, reason: str) -> Series:
            return Series(
                name=name, values=tuple(None for _ in rows), present=False, absent_reason=reason
            )

        series = {
            "reward_recorded": absent(
                "reward_recorded",
                "Inspect evaluation logs do not declare a per-rollout reward role",
            ),
            "reward_applied": absent(
                "reward_applied", "Inspect evaluation logs contain no training weight vector"
            ),
            "advantage": absent(
                "advantage", "Inspect evaluation logs contain no trainer advantage"
            ),
            "selection": absent(
                "selection", "Inspect evaluation completion is not update membership"
            ),
        }
        results = enforce(
            CheckInput(
                rows=tuple(rows), series=series, declared_rows=len(rows), source=str(artifact)
            ),
            policy=policy,
        )
        return TraceRecord(
            reader=self.name,
            source=str(artifact),
            run_id=str(header.eval.eval_id),
            rows=tuple(rows),
            series={name: series[name] for name in SERIES_NAMES},
            components={},
            access_level="RECORD",
            access_statement="RECORD: Inspect evaluation outcomes, not a training trace.",
            policy=policy,
            complete=header.status == "success",
            completeness_statement=f"Inspect reported status {header.status!r} before samples were read",
            subject_ref=observed,
            depends_on=("digest:inspect_eval",),
            framework="inspect-ai",
            framework_version=str(provenance.get("writer", {}).get("package_version", "")),
            integrity=results,
        )


READER = InspectEvalReader()
