"""The minimal state profile required for a controlled trainer intervention."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import TrainerStateIncomplete, TrainerUnsupported

__all__ = ["EIGHT_OBJECTS", "TrainerProfile"]


EIGHT_OBJECTS: tuple[str, ...] = (
    "adapter_weights",
    "adam_moments",
    "scheduler_counter",
    "rng_states",
    "dataloader_position",
    "buffered_inputs",
    "trl_step",
    "state_global_step",
)


@dataclass(frozen=True)
class TrainerProfile:
    """A declared complete mutable-state set for the supported trainer version."""

    version: str
    mutable_objects: tuple[str, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        if self.version != "1.10.0":
            raise TrainerUnsupported(self.version)
        missing = tuple(name for name in EIGHT_OBJECTS if name not in self.mutable_objects)
        if missing:
            raise TrainerStateIncomplete(missing)
