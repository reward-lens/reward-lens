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
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from reward_lens.core.types import Access, expand_access
from reward_lens.record import scores as score_tree
from reward_lens.record.reader import open_run

from .integrity import CheckInput, enforce
from .record import (
    SERIES_NAMES,
    CompositionTelemetry,
    DeclaredReward,
    PlannedContrastDesign,
    Policy,
    PriceSource,
    ScalarLossStep,
    ScalarLossToken,
    Series,
    StepMetric,
    TraceMetadata,
    TraceRecord,
    TraceRow,
    response_text_identity,
)

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
        trace_extra = _trace_extra(run)
        accepted = trace_extra.get("accepted", {})

        rows: list[TraceRow] = []
        component_names: list[str] = []
        per_component: dict[str, list[float | None]] = {}
        declared_group_sizes: dict[str, int] = {}
        group_estimators: dict[str, Mapping[str, object]] = {}
        metrics: list[StepMetric] = []
        applied_weights: dict[str, float] = {}
        index = 0
        for step in run.steps:
            zero_std = _number(
                (getattr(step.optimizer, "extra", None) or {}).get("frac_reward_zero_std")
            )
            if zero_std is not None:
                metrics.append(
                    StepMetric(
                        step=int(step.index),
                        name="frac_reward_zero_std",
                        value=zero_std,
                        scope="optimizer_step",
                        provenance={
                            "reader": self.name,
                            "source_field": "optimizer.extra.frac_reward_zero_std",
                            "source": "recorded optimizer telemetry",
                        },
                    )
                )
            for group in step.groups:
                declared_group_sizes[str(group.id)] = int(group.group_stats.k)
                estimator = getattr(group, "estimator", None)
                canonical = getattr(estimator, "__canonical__", None)
                if callable(canonical):
                    group_estimators[str(group.id)] = dict(canonical())
                for position, trajectory in enumerate(group.trajectories):
                    values = _component_values(trajectory)
                    for name in values:
                        if name not in per_component:
                            component_names.append(name)
                            per_component[name] = [None] * index
                    for name in component_names:
                        per_component[name].append(values.get(name))
                    recorded = _recorded_aggregate(trajectory)
                    applied_weight = (
                        _number(getattr(trajectory, "applied_weight", None))
                        if getattr(trajectory, "applied_weight_source", None) == "recorded"
                        else None
                    )
                    row_id = f"{step.index}:{group.id}:{position}:{trajectory.id}"
                    response_digest, response_codec, response_provenance = _response_identity(
                        trajectory
                    )
                    if applied_weight is not None:
                        applied_weights[row_id] = applied_weight
                    rows.append(
                        TraceRow(
                            index=index,
                            step=int(step.index),
                            group_id=str(group.id),
                            row_id=row_id,
                            components=dict(values),
                            reward_recorded=recorded,
                            reward_applied=None,
                            advantage=_number(trajectory.advantage),
                            selection=None,
                            task_id=str(getattr(trajectory, "task_ref", "")) or None,
                            response_digest=response_digest,
                            response_codec=response_codec,
                            response_provenance=response_provenance,
                            grader_source=_grader_source(run),
                            score_tree=getattr(trajectory, "scores", None),
                            accepted=_recorded_acceptance(accepted, trajectory),
                            applied_weight=applied_weight,
                            applied_weight_source=(
                                "recorded"
                                if getattr(trajectory, "applied_weight_source", None) == "recorded"
                                else None
                            ),
                        )
                    )
                    index += 1

        weights, weight_absence = _declared_weights(run, component_names)
        rows = [
            replace(
                row,
                reward_applied=_weighted(row.components, component_names, weights),
            )
            for row in rows
        ]
        components = {
            name: Series(
                name=name,
                values=tuple(per_component[name]),
                definition=f"the native score the grader {name!r} returned, as recorded",
                provenance={
                    "kind": "raw_native_component_score",
                    "source": "recorded score tree",
                    "reader": self.name,
                    "component": name,
                },
            )
            for name in component_names
        }
        series = _series(rows, component_names, weights, weight_absence)
        declared = _declared_rows(root, chosen)
        results = enforce(
            CheckInput(
                rows=tuple(rows),
                series=series,
                components=components,
                declared_rows=declared,
                declared_group_sizes=declared_group_sizes,
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
            access_statement=_STATEMENT.get(
                level, f"{level}: as the record's access matrix states"
            ),
            policy=policy,
            complete=True,
            completeness_statement="the run had finished when it was read; the record is closed",
            subject_ref=_subject_ref(run, chosen),
            run_identity=_run_identity(chosen),
            depends_on=("digest:training_semantics", "digest:samples", "digest:policy"),
            framework=getattr(lineage, "framework", "") or "",
            framework_version=getattr(lineage, "framework_version", "") or "",
            integrity=results,
            convenience={
                "unit_weighted_component_sum": Series(
                    name="unit_weighted_component_sum",
                    values=tuple(_unit_weighted(row.components, component_names) for row in rows),
                    weights=tuple(1.0 for _ in component_names),
                    components=tuple(component_names),
                    definition=(
                        "a convenience aggregate reconstructed with unit weights; it is not the "
                        "displayed aggregate or the optimised composite"
                    ),
                    provenance={
                        "kind": "derived_convenience_aggregate",
                        "source": "recorded score tree",
                        "reader": self.name,
                    },
                )
            },
            metadata=TraceMetadata(
                declared_reward=DeclaredReward(
                    partition=tuple(component_names),
                    weights=weights,
                    grouping={
                        "kind": "recorded_group",
                        "source": "Group.id and Group.task_ref",
                    },
                    provenance={
                        "reader": self.name,
                        "partition": "recorded score trees",
                        "weights": (
                            "lineage config reward_weights"
                            if weights is not None
                            else weight_absence
                        ),
                    },
                ),
                metrics=tuple(metrics),
                price_source=_price_source(trace_extra),
                planned_contrast=_planned_contrast(trace_extra),
                composition=_composition(trace_extra),
                applied_weights=applied_weights or None,
                group_estimators=group_estimators,
            ),
        )


def _trace_extra(run: Any) -> Mapping[str, Any]:
    """The native run's optional, explicitly named trace metadata envelope.

    ``RunLineage.extra`` is the record schema's preserved extension map.  Only the
    ``trace_metadata`` envelope is interpreted here; missing or malformed optional fields stay
    absent rather than being reconstructed from score, advantage, or optimizer fields.
    """
    lineage = getattr(run, "lineage", None)
    extra = getattr(lineage, "extra", None) or {}
    metadata = extra.get("trace_metadata", {}) if isinstance(extra, Mapping) else {}
    return metadata if isinstance(metadata, Mapping) else {}


def _recorded_acceptance(values: Any, trajectory: Any) -> bool | None:
    if not isinstance(values, Mapping):
        return None
    value = values.get(str(getattr(trajectory, "id", "")))
    return value if isinstance(value, bool) else None


def _price_source(extra: Mapping[str, Any]) -> PriceSource | None:
    value = extra.get("price_source")
    if not isinstance(value, Mapping):
        return None
    name, retrieved_on = value.get("name"), value.get("retrieved_on")
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(retrieved_on, str)
        or not retrieved_on
    ):
        return None
    provenance = value.get("provenance", {})
    return PriceSource(
        name=name,
        retrieved_on=retrieved_on,
        url=value.get("url") if isinstance(value.get("url"), str) else None,
        model=value.get("model") if isinstance(value.get("model"), str) else None,
        provenance={
            "reader": "run_record",
            "source_field": "lineage.extra.trace_metadata.price_source",
            **(
                {str(key): str(item) for key, item in provenance.items()}
                if isinstance(provenance, Mapping)
                else {}
            ),
        },
    )


def _planned_contrast(extra: Mapping[str, Any]) -> PlannedContrastDesign | None:
    value = extra.get("planned_contrast")
    if not isinstance(value, Mapping):
        return None
    try:
        return PlannedContrastDesign(
            standard_error=float(value["standard_error"]),
            alpha=float(value["alpha"]),
            power=float(value["power"]),
            n=int(value["n"]),
            clusters=int(value["clusters"]),
            covariance=value.get("covariance")
            if isinstance(value.get("covariance"), Mapping)
            else None,
            design_effect=None
            if value.get("design_effect") is None
            else float(value["design_effect"]),
            provenance={
                "reader": "run_record",
                "source_field": "lineage.extra.trace_metadata.planned_contrast",
                **(
                    {str(key): str(item) for key, item in value.get("provenance", {}).items()}
                    if isinstance(value.get("provenance"), Mapping)
                    else {}
                ),
            },
        )
    except (KeyError, TypeError, ValueError):
        return None


def _composition(extra: Mapping[str, Any]) -> CompositionTelemetry | None:
    value = extra.get("composition")
    steps = value.get("steps") if isinstance(value, Mapping) else None
    if not isinstance(steps, Mapping):
        return None
    parsed: dict[int, ScalarLossStep] = {}
    try:
        for source_step, source in steps.items():
            if not isinstance(source, Mapping) or not isinstance(source.get("tokens"), list):
                return None
            tokens = tuple(
                ScalarLossToken(
                    row_id=str(token["row_id"]),
                    mask=float(token["mask"]),
                    log_probability=float(token["log_probability"]),
                    old_log_probability=float(token["old_log_probability"]),
                    importance_ratio=float(token["importance_ratio"]),
                    advantage=float(token["advantage"]),
                    kl_term=float(token["kl_term"]),
                    auxiliary_term=float(token["auxiliary_term"]),
                )
                for token in source["tokens"]
                if isinstance(token, Mapping)
            )
            if len(tokens) != len(source["tokens"]):
                return None
            parsed[int(source_step)] = ScalarLossStep(
                trainer_profile=str(source["trainer_profile"]),
                aggregation_denominator=float(source["aggregation_denominator"]),
                minimum_denominator=float(source["minimum_denominator"]),
                clip_low=float(source["clip_low"]),
                clip_high=float(source["clip_high"]),
                kl_coefficient=float(source["kl_coefficient"]),
                observed_loss=None
                if source.get("observed_loss") is None
                else float(source["observed_loss"]),
                tokens=tokens,
            )
    except (KeyError, TypeError, ValueError):
        return None
    return CompositionTelemetry(steps=parsed) if parsed else None


def _component_values(trajectory: Any) -> dict[str, float | None]:
    tree = getattr(trajectory, "scores", None)
    if tree is None:
        return {}
    out: dict[str, float | None] = {}
    for leaf in score_tree.leaves(tree):
        out[leaf.name] = None if getattr(leaf, "abstained", False) else _number(leaf.value)
    return out


def _recorded_aggregate(trajectory: Any) -> float | None:
    """The aggregate the run itself recorded, never a substitute sum made by this reader."""
    features = getattr(trajectory, "features", None) or {}
    return _number(features.get("trl_realised_reward"))


def _weighted(
    values: Mapping[str, float | None],
    names: list[str],
    weights: Mapping[str, float] | None,
) -> float | None:
    """Use only a complete recorded weight map; a missing value never defaults to one."""
    if not names or weights is None or set(weights) != set(names):
        return None
    total = 0.0
    for name in names:
        value = values.get(name)
        if value is None:
            return None
        weight = weights[name]
        if not math.isfinite(weight):
            return None
        total += weight * value
    return total


def _unit_weighted(values: Mapping[str, float | None], names: list[str]) -> float | None:
    """The explicitly labelled convenience sum, separate from an applied composite."""
    if not names:
        return None
    total = 0.0
    for name in names:
        value = values.get(name)
        if value is None:
            return None
        total += value
    return total


def _declared_weights(run: Any, score_components: list[str]) -> tuple[dict[str, float] | None, str]:
    """Align recorded weights to the observed score partition or name the precise absence."""
    lineage = getattr(run, "lineage", None)
    config = (getattr(lineage, "extra", None) or {}).get("config") or {}
    declared = config.get("reward_weights")
    if declared is None:
        return (
            None,
            "the run records no applied reward-weight vector, so an optimised composite cannot be established from this input",
        )
    if not score_components:
        return (
            None,
            "the run records no score-component partition to which reward weights can be aligned",
        )
    if isinstance(declared, Mapping):
        if {str(name) for name in declared} != set(score_components):
            return (
                None,
                "the recorded reward-weight names do not exactly cover the observed score-component partition",
            )
        raw = {str(name): declared[name] for name in declared}
    elif isinstance(declared, (list, tuple)):
        if len(declared) != len(score_components):
            return (
                None,
                "the recorded reward-weight vector does not exactly cover the observed score-component partition",
            )
        raw = dict(zip(score_components, declared))
    else:
        return None, "the recorded reward-weight declaration is not a complete aligned vector"
    try:
        weights = {name: float(raw[name]) for name in score_components}
    except (KeyError, TypeError, ValueError):
        return (
            None,
            "the recorded reward-weight vector is not numeric for every observed score component",
        )
    if not all(math.isfinite(weight) for weight in weights.values()):
        return (
            None,
            "the recorded reward-weight vector is not finite for every observed score component",
        )
    return weights, ""


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
    weights: Mapping[str, float] | None,
    weight_absence: str,
) -> dict[str, Series]:
    declared = weights is not None
    vector = tuple(weights[name] for name in names) if weights is not None else None
    ordered: dict[str, Series] = {}
    ordered["reward_recorded"] = Series(
        name="reward_recorded",
        values=tuple(row.reward_recorded for row in rows),
        weights=None,
        components=tuple(names),
        definition=(
            "the per-row aggregate the run recorded in its realised-reward feature, with missing "
            "values preserved rather than reconstructed"
        ),
        provenance={
            "kind": "recorded_displayed_aggregate",
            "source": "trajectory feature trl_realised_reward",
            "reader": "run_record",
        },
    )
    ordered["reward_applied"] = Series(
        name="reward_applied",
        values=tuple(row.reward_applied for row in rows),
        weights=vector,
        components=tuple(names),
        definition=(
            "the optimised composite: the same components under the estimator's own weight "
            + (
                "vector, as the run configuration declares it"
                if declared
                else "vector is unavailable"
            )
        ),
        present=declared,
        absent_reason="" if declared else weight_absence,
        provenance=(
            {
                "kind": "reconstructed_optimised_composite",
                "source": "lineage config reward_weights",
                "reader": "run_record",
            }
            if declared
            else {}
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
            "whether the row contributed to the update; this record has no observed selection or "
            "loss-mask field"
        ),
        present=False,
        absent_reason=(
            "the run record's group-degenerate statistic is not evidence of selection, a loss "
            "contribution, a gradient, or an update"
        ),
    )
    return {name: ordered[name] for name in SERIES_NAMES}


def _subject_ref(run: Any, _run_id: str) -> str:
    """Digest the audited recorded subject/version, never a run identity or reader location."""
    from reward_lens.contracts import digest

    lineage = getattr(run, "lineage", None)
    components = getattr(run, "components", None) or {}
    return digest(
        {
            "kind": str(getattr(run, "kind", "")),
            "framework": getattr(lineage, "framework", "") or "",
            "framework_version": getattr(lineage, "framework_version", "") or "",
            "git_sha": getattr(lineage, "git_sha", "") or "",
            "config_hash": getattr(lineage, "config_hash", None),
            "components": {
                str(key): str(getattr(ref, "fingerprint", ""))
                for key, ref in sorted(components.items(), key=lambda pair: str(pair[0]))
            },
        }
    )


def _run_identity(run_id: str) -> str:
    from reward_lens.contracts import digest

    return digest({"recorded_run_id": run_id})


def _response_identity(
    trajectory: Any,
) -> tuple[str | None, str | None, Mapping[str, object] | None]:
    """Identify exactly one recorded assistant text; multiple turns stay an explicit absence."""
    turns = [
        turn
        for turn in getattr(trajectory, "turns", ())
        if str(getattr(getattr(turn, "role", None), "value", getattr(turn, "role", ""))).lower()
        == "assistant"
    ]
    if len(turns) != 1 or not isinstance(getattr(turns[0], "text", None), str):
        return None, None, None
    digest, codec = response_text_identity(turns[0].text)
    return digest, codec, {"assistant_turn_index": int(turns[0].index)}


def _grader_source(run: Any) -> Mapping[str, str] | None:
    """Preserve only recorded ComponentRef/lineage source facts, never an inferred digest."""
    components = getattr(run, "components", None) or {}
    grader = next(
        (
            ref
            for key, ref in components.items()
            if str(getattr(key, "name", key)).lower() == "grader"
        ),
        None,
    )
    if grader is None:
        return None
    source = {
        "component_fingerprint": str(getattr(grader, "fingerprint", "")),
        "name": str(getattr(grader, "name", "")),
        "kind": str(getattr(grader, "kind", "")),
    }
    version = getattr(grader, "version", None)
    if version is not None:
        source["version"] = str(version)
    lineage_extra = getattr(getattr(run, "lineage", None), "extra", None) or {}
    for key in ("grader_source_digest", "grader_source", "grader_revision"):
        value = lineage_extra.get(key)
        if isinstance(value, str) and value:
            source[key] = value
    return source if source["component_fingerprint"] else None


READER = RunRecordReader()
