"""The result envelope the CLI and the SDK hand back (section 5.7).

Additive only: a reader written against 1.0 keeps working when a field is added. The inner blocks
are typed as open objects here on purpose. Section 5.7 of the commission was not pasted into this
packet's brief, so the field list comes from the wave-1 interface line, and inventing inner shapes
that the section may contradict would be the exact failure the parity gate exists to prevent.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ResultEnvelope", "RESULT_SCHEMA_VERSION"]

RESULT_SCHEMA_VERSION = "assay-result/1.0"


class ResultEnvelope(BaseModel):
    """One command's result, in the one shape every formatter reads."""

    model_config = ConfigDict(strict=True, extra="forbid", populate_by_name=True)

    schema_version: Literal["assay-result/1.0"] = RESULT_SCHEMA_VERSION
    command: str = ""
    execution: dict[str, Any] = Field(default_factory=dict)
    subject: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    findings: list[dict[str, Any]] = Field(default_factory=list)
    holes: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None
    pending: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Every field of 1.0, always, `null` where there is nothing: the additive promise.

        The envelope has no JSON Schema of its own, so its required set is its 1.0 field list,
        which is the same rule the record follows against the frozen schema: a reader written
        against 1.0 reads `result["error"]` and gets `None`, never a `KeyError`. `exclude_unset`
        was the old rule, and under it the same envelope wrote different keys depending on which
        fields the caller happened to pass.
        """
        return self.model_dump(mode="json", by_alias=True)
