"""The stable vocabulary for a case corpus entry.

The corpus names teaching evidence.  It is deliberately separate from the
acceptance partition: no acceptance answer belongs to this schema.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["CaseConclusion", "CaseField"]


class CaseField(StrEnum):
    """The seven required per-case fields from section 7.10, in source order."""

    FAILURE_OR_SHORTFALL = "failure_or_shortfall"
    MECHANISM_EVIDENCE = "mechanism_evidence"
    CANDIDATE_REVISIONS = "candidate_revisions"
    INTENDED_TASK_EFFECTS = "intended_task_effects"
    COSTS = "costs"
    REJECTED_ALTERNATIVES = "rejected_alternatives"
    STOPPING_CONDITIONS = "stopping_conditions"


class CaseConclusion(StrEnum):
    """What the evidence licensed at the end of the recorded case."""

    REVISION_WARRANTED = "REVISION_WARRANTED"
    NO_REVISION_WARRANTED = "NO_REVISION_WARRANTED"
