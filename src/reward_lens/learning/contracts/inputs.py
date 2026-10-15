"""The GPU-free rollout-input contract extracted from the Appendix F snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from .errors import ContractUndeclaredKey

__all__ = ["RolloutInputs"]


@dataclass(frozen=True)
class RolloutInputs:
    """One trainer rollout before loss reduction, with only source input fields."""

    step: int
    group: int
    gen: int
    prompt: str = ""
    completion: str = ""
    token_ids: tuple[int, ...] = ()
    logprobs_sampling: tuple[float, ...] = ()
    logprobs_train: tuple[float, ...] = ()
    logprobs_ref: tuple[float, ...] = ()
    loss_mask: tuple[int, ...] = ()
    component_scores: tuple[float, ...] = ()
    truncated: bool = False
    advantage: float | None = None
    training_passed: bool = False
    oracle_stdout: str = ""
    oracle_expected_n: int = 0
    oracle_completed: bool = True
    oracle_case_passed: tuple[bool, ...] = ()
    solution_code: str = ""
    extra_files: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RolloutInputs":
        """Build a rollout while refusing all undeclared and derived fields."""
        known = {item.name for item in fields(cls)}
        unknown = tuple(sorted(set(mapping) - known))
        if unknown:
            raise ContractUndeclaredKey(contract=cls.__name__, keys=unknown)

        values = dict(mapping)
        for name in (
            "token_ids",
            "logprobs_sampling",
            "logprobs_train",
            "logprobs_ref",
            "loss_mask",
            "component_scores",
            "oracle_case_passed",
        ):
            if name in values and values[name] is not None:
                values[name] = tuple(values[name])
        return cls(**values)
