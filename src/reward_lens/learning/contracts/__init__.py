"""GPU-free learning contracts used by the controlled-intervention adapters."""

from __future__ import annotations

from .errors import (
    ContractUndeclaredKey,
    LearningContractError,
    SamplingProvenanceIncomplete,
    TrainerStateIncomplete,
    TrainerUnsupported,
)
from .inputs import RolloutInputs
from .results import ArmResult
from .sampling import DECODING_FIELDS, SamplingProvenance
from .state import EIGHT_OBJECTS, TrainerProfile

__all__ = [
    "ArmResult",
    "ContractUndeclaredKey",
    "DECODING_FIELDS",
    "EIGHT_OBJECTS",
    "LearningContractError",
    "RolloutInputs",
    "SamplingProvenance",
    "SamplingProvenanceIncomplete",
    "TrainerProfile",
    "TrainerStateIncomplete",
    "TrainerUnsupported",
]
