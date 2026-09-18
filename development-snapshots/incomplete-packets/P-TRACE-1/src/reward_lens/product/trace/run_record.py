"""The project's own run record, first in D-42's support order.

Reads the two GRPO fixtures with no `trl` import, no GPU and no network. Everything it needs is
already in the record: the group is declared on the group, the estimator's weight vector is in the
lineage, the abstentions are on the leaves, and the degenerate flag is on the group statistics. So
the four series come out of recorded fields and not out of a reconstruction of them.

Row identity is `step:group:position:trajectory`. The trajectory id alone is a content hash, and a
tiny model writing twelve-token completions produces the same content more than once in a run, so
the trajectory id is not an identity for a row of a trace. Carrying the position alongside it keeps
the identity unique without inventing anything: the group is the record's own, not an adjacency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from reward_lens.core.types import Access, expand_access
from reward_lens.record import scores as score_tree
from reward_lens.record.reader import open_run

from .integrity import CheckInput, enforce
from .record import SERIES_NAMES, Policy, Series, TraceRecord, TraceRow

__all__ = ["READER", "RunRecordReader"]

#: The containment ladder of section 2.3, strongest first.
_LADDER: tuple[tuple[str, Access], ...] = (
    ("BACKWARD", Access.BACKWARD),
    ("FORWARD", Access.FORWARD),
    ("QUERY", Access.QUERY),
    ("RECORD", Access.RECORD),
)

_STATEMENT = {
    "RECORD": (
        "RECORD: the logged values that already exist. That supports evaluator and selection "
        "evidence and a reconstruction of what the optimiser saw; it does not support anything "
        "that needs the grader called again"
    ),
}


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _access_level(matrix: Mapping[Any, Any] | None) -> str:
    if not matrix:
        return "NONE"
    held = []
    for value in matrix.values():
        try:
            held.append(expand_access(Access(int(value))))
        except (TypeError, ValueError):
            return "NONE"
    for name, flag in _LADDER:
        if all(flag in one for one in held):
            return name
    return "NONE"


def _run_ids(root: Path) -> list[str]:
    runs = root / "runs"
    if not runs.is_dir():
        return []
    found = []
    for directory in sorted(runs.iterdir()):
        description = directory / "run.json"
        if description.is_file():
            try:
                found.append(json.loads(description.read_text(encoding="utf-8"))["id"])
            except (ValueError, KeyError, OSError):
                continue
    return found


class RunRecordReader:
    """D-42's first reader: the record this project writes itself."""

    name = "run_record"

    def can_read(self, path: Any) -> bool:
        try:
            return bool(_run_ids(Path(path)))
        except OSError:  # pragma: no cover - an unreadable directory has no opinion
            return False

    def read(
        self, path: Any, *, policy: Policy = "strict", run_id: str | None = None
    ) -> TraceRecord:
        root = Path(path)
        available = _run_ids(root)
        if not available:
            raise FileNotFoundError(f"no run record under {root}")
        chosen = run_id or available[0]
        run = open_run(root, chosen)
        weights = _declared_weights(run)

        rows: list[TraceRow] = []
        component_names: list[str] = []
        per_component: dict[str, list[float | None]] = {}
        index = 0
        for step in run.steps:
            for group in step.groups:
                degenerate = bool(getattr(group.group_stats, "degenerate", False))
                for position, trajectory in enumerate(group.trajectories):
                    values = _component_values(trajectory)
                    for name in values:
                        if name not in per_component:
                            component_names.append(name)
                            per_component[name] = [None] * index
                    for name in component_names:
                        per_component[name].append(values.get(name))
                    recorded = _weighted(values, component_names, None)
                    applied = _weighted(values, component_names, weights)
                    rows.append(
                        TraceRow(
                            index=index,
                            step=int(step.index),
                            group_id=str(group.id),
                            row_id=f"{step.index}:{group.id}:{position}:{trajectory.id}",
                            components=dict(values),
                            reward_recorded=recorded,
                            reward_applied=applied,
                            advantage=_number(trajectory.advantage),
                            selection=0.0 if degenerate else 1.0,
                        )
                    )
                    index += 1

        components = {
            name: Series(
                name=name,
                values=tuple(per_component[name]),
                definition=f"the native score the grader {name!r} returned, as recorded",
            )
            for name in component_names
        }
        vector = tuple(
            float(weights.get(name, 1.0)) if weights else 1.0 for name in component_names
        )
        series = _series(rows, component_names, vector, weights is not None)
        declared = _declared_rows(root, chosen)
        results = enforce(
            CheckInput(
                rows=tuple(rows),
                series=series,
                components=components,
                declared_rows=declared,
                source=str(root),
            ),
            policy=policy,
        )
        level = _access_level(getattr(run, "access", None))
        lineage = getattr(run, "lineage", None)
        return TraceRecord(
            reader=self.name,
            source=str(root),
            run_id=str(chosen),
            rows=tuple(rows),
            series=series,
            components=components,
            access_level=level,
            access_statement=_STATEMENT.get(level, f"{level}: as the record's access matrix states"),
            policy=policy,
            complete=True,
            completeness_statement="the run had finished when it was read; the record is closed",
            subject_ref=_subject_ref(run, chosen),
            depends_on=("digest:training_semantics", "digest:samples", "digest:policy"),
            framework=getattr(lineage, "framework", "") or "",
            framework_version=getattr(lineage, "framework_version", "") or "",
            integrity=results,
        )


def _component_values(trajectory: Any) -> dict[str, float | None]:
    tree = getattr(trajectory, "scores", None)
    if tree is None:
        return {}
    out: dict[str, float | None] = {}
    for leaf in score_tree.leaves(tree):
        out[leaf.name] = None if getattr(leaf, "abstained", False) else _number(leaf.value)
    return out


def _weighted(
    values: Mapping[str, float | None],
    names: list[str],
    weights: Mapping[str, float] | None,
) -> float | None:
    total = 0.0
    seen = False
    for name in names:
        value = values.get(name)
        if value is None:
            continue
        total += (float(weights.get(name, 1.0)) if weights else 1.0) * value
        seen = True
    return total if seen else None


def _declared_weights(run: Any) -> dict[str, float] | None:
    lineage = getattr(run, "lineage", None)
    config = (getattr(lineage, "extra", None) or {}).get("config") or {}
    declared = config.get("reward_weights")
    if not declared:
        return None
    grader = (getattr(run, "components", None) or {})
    names = [str(getattr(ref, "name", key)) for key, ref in grader.items()]
    return {name: float(weight) for name, weight in zip(names, declared)}


def _declared_rows(root: Path, run_id: str) -> int | None:
    manifest = root / "runs" / run_id.replace(":", "_") / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        return int(json.loads(manifest.read_text(encoding="utf-8"))["counts"]["trajectories"])
    except (ValueError, KeyError, OSError, TypeError):
        return None


def _series(
    rows: list[TraceRow],
    names: list[str],
    vector: tuple[float, ...],
    declared: bool,
) -> dict[str, Series]:
    ordered: dict[str, Series] = {}
    ordered["reward_recorded"] = Series(
        name="reward_recorded",
        values=tuple(row.reward_recorded for row in rows),
        weights=tuple(1.0 for _ in names),
        components=tuple(names),
        definition=(
            "the aggregate the trainer displayed: the unit-weighted sum of the native component "
            "scores, with abstentions preserved as missing"
        ),
    )
    ordered["reward_applied"] = Series(
        name="reward_applied",
        values=tuple(row.reward_applied for row in rows),
        weights=vector,
        components=tuple(names),
        definition=(
            "the optimised composite: the same components under the estimator's own weight "
            + ("vector, as the run configuration declares it" if declared else "vector, which this run left unset, so it is the unit vector")
        ),
    )
    ordered["advantage"] = Series(
        name="advantage",
        values=tuple(row.advantage for row in rows),
        definition="the advantage the trainer recorded for this row",
    )
    ordered["selection"] = Series(
        name="selection",
        values=tuple(row.selection for row in rows),
        definition=(
            "whether the row's group could contribute to the update at all, from the recorded "
            "degenerate flag: a group whose scores did not vary carries no gradient"
        ),
    )
    return {name: ordered[name] for name in SERIES_NAMES}


def _subject_ref(run: Any, run_id: str) -> str:
    from reward_lens.contracts import digest

    lineage = getattr(run, "lineage", None)
    components = getattr(run, "components", None) or {}
    return digest(
        {
            "run": str(run_id),
            "kind": str(getattr(run, "kind", "")),
            "framework": getattr(lineage, "framework", "") or "",
            "framework_version": getattr(lineage, "framework_version", "") or "",
            "git_sha": getattr(lineage, "git_sha", "") or "",
            "config_hash": getattr(lineage, "config_hash", None),
            "components": {
                str(key): str(getattr(ref, "name", "")) for key, ref in sorted(
                    components.items(), key=lambda pair: str(pair[0])
                )
            },
        }
    )


READER = RunRecordReader()
