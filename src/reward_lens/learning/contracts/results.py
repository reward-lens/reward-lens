"""The product arm-result contract, preserved from the Appendix F snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from .errors import ContractUndeclaredKey

__all__ = ["ArmResult"]


@dataclass(frozen=True)
class ArmResult:
    """What one arm over one step range returns, without runtime dependencies."""

    adapter_path: str = ""
    rollouts: tuple[Mapping[str, Any], ...] = ()
    container_ids: tuple[str, ...] = ()
    t_step_s: float | None = None
    t_gen_s: float | None = None
    arm: str = ""
    arm_kind: str = ""
    grader: str = ""
    step_start: int = 0
    step_stop: int = 0
    steps_run: int = 0
    resumed_from: str = ""
    resume_restored: dict[str, Any] | None = None
    group_witness: dict[str, Any] | None = None
    a2_step0: dict[str, Any] | None = None
    step_metrics_path: str = ""
    instance_path: str = ""
    checkpoints: tuple[Mapping[str, Any], ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    instance: Mapping[str, Any] = field(default_factory=dict)
    telemetry: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ArmResult":
        """Build an arm result and refuse every field outside this schema."""
        known = {item.name for item in fields(cls)}
        unknown = tuple(sorted(set(mapping) - known))
        if unknown:
            raise ContractUndeclaredKey(contract=cls.__name__, keys=unknown)

        values = dict(mapping)
        for name in ("rollouts", "container_ids", "checkpoints"):
            if name in values and values[name] is not None:
                values[name] = tuple(values[name])
        return cls(**values)
