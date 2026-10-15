"""The one projection type shared by all record views."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

from reward_lens.contracts import UsageError
from reward_lens.errors import make

JSONPointer: TypeAlias = str


class View(str, Enum):
    CASE = "case"
    GATE = "gate"
    REPORT = "report"
    INCIDENT = "incident"
    REGISTRY = "registry"


class ProjectionFieldAbsent(UsageError):
    """RL0640: a requested projection value is absent from the saved Assay."""

    exit_status = 4

    def __init__(self, *, view: View, field: str, pointer: JSONPointer) -> None:
        registered = make("RL0640", view=view.value, field=field)
        super().__init__(
            code=registered.code,
            message=registered.message,
            remediation=registered.remediation,
            context={**registered.context, "pointer": pointer},
            exit_code=registered.exit_code,
        )


@dataclass(frozen=True)
class Projection:
    """One selection of canonical Assay values, with no view-specific schema."""

    view: View
    fields: Mapping[str, Any]
    source_pointers: tuple[JSONPointer, ...]
    summary: str

    def __post_init__(self) -> None:
        fields = dict(self.fields)
        if len(fields) != len(self.source_pointers):
            raise ValueError("every projection field needs exactly one source pointer")
        if not self.summary:
            raise ValueError("a projection summary must be derived from its selected record values")
        object.__setattr__(self, "fields", MappingProxyType(fields))
        object.__setattr__(self, "source_pointers", tuple(self.source_pointers))
