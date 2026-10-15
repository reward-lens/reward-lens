"""Immutable corpus cases and their binding to a stored assay."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from reward_lens.contracts import Assay, digest

from .schema import CaseConclusion, CaseField

__all__ = ["Case", "RecordSource"]

_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _texts(value: object, *, field: str, may_be_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list of non-empty strings")
    result = tuple(_text(item, field=field) for item in value)
    if not result and not may_be_empty:
        raise ValueError(f"{field} must not be empty")
    return result


@dataclass(frozen=True)
class RecordSource:
    """A case's checked binding to an already persisted source record.

    ``path`` is retained so reopening can validate the actual source rather
    than trusting a copied record-shaped object in a corpus JSON file.
    """

    path: Path
    assay_id: str

    @classmethod
    def from_assay(cls, path: Path | str, assay: Assay) -> "RecordSource":
        if not isinstance(assay, Assay):
            raise TypeError("a corpus case source must be a validated Assay")
        location = Path(path).resolve()
        if not location.is_file():
            raise ValueError(f"source assay is not persisted at {location}")
        actual = digest(assay)
        if assay.assay_id != actual:
            raise ValueError("source assay_id does not match its canonical record digest")
        return cls(path=location, assay_id=actual)

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "assay_id": self.assay_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecordSource":
        if set(value) != {"path", "assay_id"}:
            raise ValueError("case source must contain exactly path and assay_id")
        path = value["path"]
        assay_id = value["assay_id"]
        if not isinstance(path, str) or not path:
            raise ValueError("case source path must be a non-empty string")
        if not isinstance(assay_id, str) or not assay_id.startswith("sha256:"):
            raise ValueError("case source assay_id must be a sha256 digest")
        return cls(path=Path(path), assay_id=assay_id)


@dataclass(frozen=True)
class Case:
    """The public, immutable teaching record required by section 7.10."""

    id: str
    failure_or_shortfall: str
    mechanism_evidence: tuple[str, ...]
    candidate_revisions: tuple[str, ...]
    intended_task_effects: tuple[str, ...]
    costs: tuple[str, ...]
    rejected_alternatives: tuple[str, ...]
    stopping_conditions: tuple[str, ...]
    conclusion: CaseConclusion
    source: RecordSource | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _CASE_ID.fullmatch(self.id):
            raise ValueError("case id must be a non-empty portable identifier")
        object.__setattr__(
            self,
            "failure_or_shortfall",
            _text(self.failure_or_shortfall, field=CaseField.FAILURE_OR_SHORTFALL.value),
        )
        for field, empty in (
            (CaseField.MECHANISM_EVIDENCE, False),
            (CaseField.CANDIDATE_REVISIONS, True),
            (CaseField.INTENDED_TASK_EFFECTS, False),
            (CaseField.COSTS, False),
            (CaseField.REJECTED_ALTERNATIVES, False),
            (CaseField.STOPPING_CONDITIONS, False),
        ):
            object.__setattr__(
                self,
                field.value,
                _texts(getattr(self, field.value), field=field.value, may_be_empty=empty),
            )
        if not isinstance(self.conclusion, CaseConclusion):
            raise ValueError("conclusion must be a CaseConclusion")
        if self.conclusion is CaseConclusion.NO_REVISION_WARRANTED and self.candidate_revisions:
            raise ValueError("NO_REVISION_WARRANTED cases cannot carry candidate revisions")
        if self.source is not None and not isinstance(self.source, RecordSource):
            raise TypeError("case source must be a RecordSource")

    def to_dict(self) -> dict[str, object]:
        """The section 7.10 case fields, with no source or acceptance data."""
        return {
            "id": self.id,
            CaseField.FAILURE_OR_SHORTFALL.value: self.failure_or_shortfall,
            CaseField.MECHANISM_EVIDENCE.value: list(self.mechanism_evidence),
            CaseField.CANDIDATE_REVISIONS.value: list(self.candidate_revisions),
            CaseField.INTENDED_TASK_EFFECTS.value: list(self.intended_task_effects),
            CaseField.COSTS.value: list(self.costs),
            CaseField.REJECTED_ALTERNATIVES.value: list(self.rejected_alternatives),
            CaseField.STOPPING_CONDITIONS.value: list(self.stopping_conditions),
            "conclusion": self.conclusion.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Case":
        required = {"id", "conclusion", *(field.value for field in CaseField)}
        missing = sorted(required - set(value))
        unknown = sorted(set(value) - required)
        if missing:
            raise ValueError(f"case is missing required field {missing[0]}")
        if unknown:
            raise ValueError(f"case has unknown field {unknown[0]}")
        try:
            conclusion = CaseConclusion(value["conclusion"])
        except (TypeError, ValueError):
            raise ValueError("conclusion is not a known case conclusion") from None
        return cls(
            id=value["id"],
            failure_or_shortfall=value[CaseField.FAILURE_OR_SHORTFALL.value],
            mechanism_evidence=value[CaseField.MECHANISM_EVIDENCE.value],
            candidate_revisions=value[CaseField.CANDIDATE_REVISIONS.value],
            intended_task_effects=value[CaseField.INTENDED_TASK_EFFECTS.value],
            costs=value[CaseField.COSTS.value],
            rejected_alternatives=value[CaseField.REJECTED_ALTERNATIVES.value],
            stopping_conditions=value[CaseField.STOPPING_CONDITIONS.value],
            conclusion=conclusion,
        )

    def with_source(self, source: RecordSource) -> "Case":
        return replace(self, source=source)
