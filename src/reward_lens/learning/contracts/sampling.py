"""A GPU-free representation of registered decoding values and their sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .errors import ContractUndeclaredKey, SamplingProvenanceIncomplete

__all__ = ["DECODING_FIELDS", "SamplingProvenance"]


DECODING_FIELDS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
)


@dataclass(frozen=True)
class SamplingProvenance:
    """The five decoding fields paired one-for-one with their declared sources."""

    values: Mapping[str, float | int | None]
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        expected = set(DECODING_FIELDS)
        for location, mapping in (("values", self.values), ("provenance", self.provenance)):
            unknown = tuple(sorted(set(mapping) - expected))
            if unknown:
                raise ContractUndeclaredKey(contract=f"SamplingProvenance.{location}", keys=unknown)
            missing = tuple(name for name in DECODING_FIELDS if name not in mapping)
            if missing:
                raise SamplingProvenanceIncomplete(missing, location=location)
