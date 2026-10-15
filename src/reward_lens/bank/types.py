"""Immutable response-bank values and their content-derived identifiers."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Mapping, NewType

TaskId = NewType("TaskId", str)


def _freeze(value: Any) -> Any:
    """Recursively detach public bank values from mutable caller-owned containers."""
    if isinstance(value, MappingABC):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class Response:
    """One candidate response, addressed by the task it answers and its content digest."""

    digest: str
    task_id: TaskId
    text: str
    recorded_scores: Mapping[str, Decimal] | None = None

    def __post_init__(self) -> None:
        if self.recorded_scores is not None:
            object.__setattr__(self, "recorded_scores", _freeze(self.recorded_scores))


@dataclass(frozen=True)
class BankProvenance:
    """The source declaration carried by a loaded response-bank fixture."""

    source_path: str
    generating_command: str | None
    revision: str | None
    digest: str


@dataclass(frozen=True)
class ResponseBank:
    """A finite set of candidate responses grouped by the task they answer.

    ``N`` intentionally means the smallest task-level candidate count.  A single selection
    budget applies to every task, so using aggregate rows would permit a budget that some task
    cannot draw without replacement.
    """

    id: str
    digest: str
    provenance: BankProvenance
    tasks: tuple[TaskId, ...]
    responses: Mapping[TaskId, tuple[Response, ...]]
    task_payloads: Mapping[TaskId, Mapping[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tasks = tuple(self.tasks)
        if not tasks:
            raise ValueError("a response bank needs at least one task")
        if len(set(tasks)) != len(tasks):
            raise ValueError("a response bank cannot repeat a task")
        responses = {task: tuple(self.responses[task]) for task in tasks if task in self.responses}
        if tuple(sorted(tasks)) != tuple(sorted(self.responses)):
            raise ValueError("response-bank tasks and response groups must agree")
        if any(not responses[task] for task in tasks):
            raise ValueError("each response-bank task needs at least one response")
        for task in tasks:
            if any(response.task_id != task for response in responses[task]):
                raise ValueError("response task ids must agree with their response-bank group")
        payloads: dict[TaskId, Mapping[str, object]] = {}
        for task in tasks:
            payload = self.task_payloads.get(task, {"id": str(task)})
            if not isinstance(payload, MappingABC):
                raise ValueError("a task payload must be a mapping")
            payloads[task] = _freeze(payload)
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "responses", MappingProxyType(responses))
        object.__setattr__(self, "task_payloads", MappingProxyType(payloads))

    def responses_for(self, task: TaskId) -> tuple[Response, ...]:
        return self.responses[task]

    def task_for(self, task: TaskId) -> Mapping[str, object]:
        return self.task_payloads[task]

    @property
    def n_per_task(self) -> dict[TaskId, int]:
        return {task: len(self.responses[task]) for task in self.tasks}

    @property
    def N(self) -> int:
        """Largest uniform candidate count available to every task in the finite bank."""
        return min(self.n_per_task.values())
