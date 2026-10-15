"""Read a provenance-bound ``verifiers`` results directory (D-42)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from reward_lens.product.trace.errors import RecordIntegrity
from reward_lens.product.trace.integrity import CheckInput, enforce
from reward_lens.product.trace.record import SERIES_NAMES, Policy, Series, TraceRecord, TraceRow

__all__ = ["READER", "VerifiersResultsReader"]


def _integrity(*, message: str, source: Path, row: str = "PROVENANCE.json") -> RecordIntegrity:
    return RecordIntegrity(
        message=message,
        remediation="supply the original artifact and its matching PROVENANCE.json record",
        context={"rule": "trace.fixture_provenance", "row": row, "source": str(source)},
    )


def _provenance(root: Path, artifact: Path) -> Mapping[str, Any]:
    record = root / "PROVENANCE.json"
    if not record.is_file():
        raise _integrity(message=f"{record.name} is required for {artifact.name}", source=root)
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
        expected = str(data["artifact"]["sha256"])
    except (KeyError, TypeError, ValueError) as error:
        raise _integrity(
            message=f"{record.name} has no usable artifact digest", source=root
        ) from error
    observed = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if expected != observed:
        raise _integrity(
            message=f"provenance digest differs for {artifact.name}", source=root, row=artifact.name
        )
    return data


def _absent(name: str, rows: list[TraceRow], reason: str) -> Series:
    return Series(name=name, values=tuple(None for _ in rows), present=False, absent_reason=reason)


class VerifiersResultsReader:
    """One ``results.jsonl`` plus its writer-produced ``metadata.json`` companion."""

    name = "verifiers_results"

    def can_read(self, path: Any) -> bool:
        root = Path(path)
        return (
            root.is_dir()
            and (root / "results.jsonl").is_file()
            and (root / "metadata.json").is_file()
        )

    def read(self, path: Any, *, policy: Policy = "strict") -> TraceRecord:
        root = Path(path)
        artifact = root / "results.jsonl"
        provenance = _provenance(root, artifact)
        metadata_path = root / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            declared = {str(item["example_id"]): item for item in metadata["examples"]}
        except (KeyError, TypeError, ValueError) as error:
            raise _integrity(
                message="metadata.json must declare examples keyed by example_id",
                source=root,
                row="metadata.json",
            ) from error
        raw_rows = [
            json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines() if line
        ]
        rows: list[TraceRow] = []
        source_rewards: list[float | None] = []
        for index, raw in enumerate(raw_rows):
            example_id = str(raw.get("example_id"))
            if example_id not in declared:
                raise _integrity(
                    message=f"results row {index} has unknown example_id {example_id}",
                    source=root,
                    row=f"results.jsonl:{index + 1}",
                )
            item = declared[example_id]
            reward = raw.get("reward")
            source_rewards.append(float(reward) if isinstance(reward, int | float) else None)
            rows.append(
                TraceRow(
                    index=index,
                    step=int(item.get("step", 0)),
                    group_id=str(item["group_id"]) if item.get("group_id") is not None else None,
                    row_id=f"example:{example_id}:rollout:{index}",
                )
            )
        series = {
            "reward_recorded": _absent(
                "reward_recorded",
                rows,
                "verifiers calls its raw rollout field reward without declaring its role",
            ),
            "reward_applied": _absent(
                "reward_applied", rows, "no applied weight vector is recorded"
            ),
            "advantage": _absent("advantage", rows, "verifiers results do not record advantages"),
            "selection": _absent(
                "selection", rows, "completion state is not evidence of update membership"
            ),
        }
        results = enforce(
            CheckInput(
                rows=tuple(rows),
                series=series,
                declared_rows=len(rows),
                source=str(root),
            ),
            policy=policy,
        )
        return TraceRecord(
            reader=self.name,
            source=str(root),
            run_id=str(metadata.get("run_id", root.name)),
            rows=tuple(rows),
            series={name: series[name] for name in SERIES_NAMES},
            components={},
            access_level="RECORD",
            access_statement="RECORD: saved rollout outcomes; no loss-path or update evidence.",
            policy=policy,
            complete=True,
            completeness_statement="all rows in the provenance-bound results file were read",
            subject_ref=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            depends_on=("digest:verifiers_results",),
            framework="verifiers",
            framework_version=str(provenance.get("writer", {}).get("source_revision", "")),
            integrity=results,
            limitations=("the source reward field is preserved only as a convenience aggregate",),
            convenience={
                "source_reward": Series(
                    name="source_reward",
                    values=tuple(source_rewards),
                    definition="the raw verifiers reward field; its optimisation role is not declared",
                    provenance={"source_column": "reward", "reader": self.name},
                )
            },
        )


READER = VerifiersResultsReader()
