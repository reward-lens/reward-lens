"""Strict, source-bound JSON inputs for executable Compare rungs.

The public dispatcher receives configuration files, never reconstructed training
results.  Each rung's producer validates its own schema and executes its own
factory; this module supplies the shared byte identity and rejects ambiguous
JSON before that producer sees it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from reward_lens import contracts
from reward_lens.contracts import UsageError

__all__ = ["ExecutionConfig", "load_execution_config"]


@dataclass(frozen=True)
class ExecutionConfig:
    """One validated JSON object plus both source and canonical identities."""

    path: Path
    payload: Mapping[str, Any]
    source_digest: str
    canonical_digest: str

    def identity(self) -> dict[str, str]:
        """The persisted binding for the exact bytes and their parsed declaration."""
        return {
            "source_path": str(self.path),
            "source_digest": self.source_digest,
            "canonical_config_digest": self.canonical_digest,
        }


def _no_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_object(raw: bytes, *, path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_no_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise UsageError(
            code="RL0131",
            message=f"compare {field} must be one strict JSON object: {error}",
            remediation="supply a UTF-8 JSON object without duplicate keys or non-finite values",
            context={"field": field, "path": str(path)},
        ) from error
    if not isinstance(value, dict):
        raise UsageError(
            code="RL0131",
            message=f"compare {field} must contain a JSON object",
            remediation="supply the producer's object-shaped executable configuration",
            context={"field": field, "path": str(path)},
        )
    return value


def load_execution_config(path: Path | str | None, *, field: str) -> ExecutionConfig:
    """Read one executable-rung declaration without interpreting its producer fields."""
    if path is None:
        raise UsageError(
            code="RL0131",
            message=f"compare requires --{field.replace('_', '-')} for its executable rung",
            remediation="supply the producer's bounded JSON execution configuration",
            context={"field": field},
        )
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise UsageError(
            code="RL0131",
            message=f"compare could not read {field} from {source}: {error}",
            remediation="supply an existing executable-rung JSON configuration file",
            context={"field": field, "path": str(source)},
        ) from error
    payload = _load_object(raw, path=source, field=field)
    return ExecutionConfig(
        path=source,
        payload=payload,
        source_digest="sha256:" + hashlib.sha256(raw).hexdigest(),
        canonical_digest=contracts.digest(payload),
    )
