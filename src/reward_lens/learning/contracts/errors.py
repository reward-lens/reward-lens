"""Named learning-contract refusals; numeric catalogue allocation is pending."""

from __future__ import annotations

__all__ = [
    "ContractUndeclaredKey",
    "LearningContractError",
    "SamplingProvenanceIncomplete",
    "TrainerStateIncomplete",
    "TrainerUnsupported",
]


class LearningContractError(ValueError):
    """Base class for a product learning-contract refusal."""


class ContractUndeclaredKey(LearningContractError):
    """CONTRACT_UNDECLARED_KEY: a mapping contains a non-contract field."""

    def __init__(self, *, contract: str, keys: tuple[str, ...]) -> None:
        self.contract = contract
        self.keys = keys
        super().__init__(
            f"CONTRACT_UNDECLARED_KEY: {contract} does not declare "
            f"{', '.join(repr(key) for key in keys)}"
        )


class TrainerUnsupported(LearningContractError):
    """TRAINER_UNSUPPORTED: controlled intervention supports TRL 1.10.0 only."""

    def __init__(self, version: str) -> None:
        self.version = version
        super().__init__(
            f"TRAINER_UNSUPPORTED: TRL version {version!r}; expected '1.10.0'; install trl==1.10.0"
        )


class TrainerStateIncomplete(LearningContractError):
    """TRAINER_STATE_INCOMPLETE: a required mutable state object is absent."""

    def __init__(self, missing: tuple[str, ...]) -> None:
        self.missing = missing
        super().__init__(
            "TRAINER_STATE_INCOMPLETE: missing mutable state object(s) "
            f"{', '.join(repr(name) for name in missing)}"
        )


class SamplingProvenanceIncomplete(LearningContractError):
    """A decoding field or its source row is absent from sampling provenance."""

    def __init__(self, missing: tuple[str, ...], *, location: str) -> None:
        self.missing = missing
        self.location = location
        super().__init__(
            f"SAMPLING_PROVENANCE_INCOMPLETE: {location} lacks "
            f"{', '.join(repr(name) for name in missing)}"
        )
