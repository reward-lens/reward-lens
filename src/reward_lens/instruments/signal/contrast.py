"""Recorded signal estimands; unrecorded components remain missing rather than zero."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol


class _SignalRow(Protocol):
    task_id: str
    group_id: str
    reward_applied: float | None
    components: Mapping[str, float | None]


@dataclass(frozen=True)
class Metric:
    value: float
    task_values: tuple[float, ...]
    eligible: int
    missing_components: int = 0
    zero_total_magnitude: int = 0
    task_weights: tuple[float, ...] = ()


def contrast_fraction(rows: Iterable[_SignalRow]) -> Metric:
    by_group: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if row.reward_applied is not None:
            by_group.setdefault((str(row.task_id), str(row.group_id)), []).append(
                float(row.reward_applied)
            )
    by_task: dict[str, list[float]] = {}
    for (task, _), values in by_group.items():
        if len(values) >= 2:
            by_task.setdefault(task, []).append(float(max(values) > min(values)))
    values = tuple(sum(x) / len(x) for _, x in sorted(by_task.items()))
    numerator = sum(sum(x) for x in by_task.values())
    denominator = sum(len(x) for x in by_task.values())
    weights = tuple(float(len(x)) for _, x in sorted(by_task.items()))
    return Metric(
        numerator / denominator if denominator else 0.0, values, len(values), task_weights=weights
    )


def style_share(rows: Iterable[_SignalRow], *, style_component: str = "style") -> Metric:
    rows = tuple(rows)
    names = {name for row in rows for name in row.components}
    by_task: dict[str, list[float]] = {}
    missing = 0
    zero = 0
    for row in rows:
        style = row.components.get(style_component)
        if style is None or not names or any(row.components.get(name) is None for name in names):
            missing += 1
            continue
        observed = [abs(float(row.components[name])) for name in names]
        if sum(observed) == 0:
            zero += 1
            continue
        by_task.setdefault(str(row.task_id), []).append(abs(float(style)) / sum(observed))
    values = tuple(sum(x) / len(x) for _, x in sorted(by_task.items()))
    return Metric(sum(values) / len(values) if values else 0.0, values, len(values), missing, zero)


def component_disagreement(rows: Iterable[_SignalRow]) -> Metric:
    rows = tuple(rows)
    names = {name for row in rows for name in row.components}
    by_task: dict[str, list[float]] = {}
    missing = 0
    for row in rows:
        if len(names) < 2 or any(row.components.get(name) is None for name in names):
            missing += 1
            continue
        values = [float(row.components[name]) for name in names]
        by_task.setdefault(str(row.task_id), []).append(float(min(values) < 0 < max(values)))
    values = tuple(sum(x) / len(x) for _, x in sorted(by_task.items()))
    return Metric(sum(values) / len(values) if values else 0.0, values, len(values), missing)
