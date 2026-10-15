"""Safe, searchable projections of corpus cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .case import Case

__all__ = ["CaseProjection", "project"]


@dataclass(frozen=True)
class CaseProjection:
    """A public teaching projection.  It deliberately has no record source."""

    case_id: str
    fields: Mapping[str, object]


def project(case: Case) -> CaseProjection:
    """Return only the section 7.10 teaching content for search results."""
    return CaseProjection(case_id=case.id, fields=case.to_dict())
