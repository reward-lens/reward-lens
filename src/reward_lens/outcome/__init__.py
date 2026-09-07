"""The independent outcome contract: sealed rounds, the separation proof, panels, qualification.

D-39 and D-40. The check belongs to the owner, the acceptance material belongs to a partition the
candidate side cannot read, and what says the candidate side cannot read it is an attempt that
failed at a named tier. Owned by P-OUTCOME.
"""

from __future__ import annotations

from .errors import AcceptanceRoundExhausted, PanelNotRegistered, usage_refusal
from .panel import Panel, PanelScore, Registration
from .qualify import OutcomeCheck, Qualification, SuiteResult, qualify
from .rounds import Round, RoundLedger, SealedRequest
from .separation import (
    CANARY_NAME,
    CAVEAT,
    NON_CONFINING_TIERS,
    SeparationProof,
    attempt_protected_read,
    certify_separation,
    confines,
    is_inside,
)

__all__ = [
    "CANARY_NAME",
    "CAVEAT",
    "NON_CONFINING_TIERS",
    "AcceptanceRoundExhausted",
    "OutcomeCheck",
    "Panel",
    "PanelNotRegistered",
    "PanelScore",
    "Qualification",
    "Registration",
    "Round",
    "RoundLedger",
    "SealedRequest",
    "SeparationProof",
    "SuiteResult",
    "attempt_protected_read",
    "certify_separation",
    "confines",
    "is_inside",
    "qualify",
    "usage_refusal",
]
