"""Typed results that are neither an Assay nor a capability report."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

__all__ = ["ErrorExplanation", "SeekerProposalResult"]


class ErrorExplanation(BaseModel):
    """The catalogue entry shared by SDK, CLI, and agent tools."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    schema_version: str = "error-explanation/1.0"
    command: str = "explain"
    code: str
    title: str
    cause: str
    remedies: list[str]
    exit_code: int
    surface: str
    error: None = None


class SeekerProposalResult(BaseModel):
    """The saved result of one explicitly supplied seeker candidate response.

    `evidence` names only facts already persisted by the Audit engine.  It is
    intentionally a mapping because the sealed actor and supervised outcome
    evidence have their own stable schemas; this surface does not copy or
    reinterpret either one.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    record_path: str
    summary: str
    evidence: dict[str, Any]
