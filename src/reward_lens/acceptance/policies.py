"""Immutable policy-file loading and policy digest calculation."""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import rfc8785

from reward_lens.contracts.models import Measurement

from .registry import validate_producer
from .types import Multiplicity, Policy, PolicyCheck, PolicyInvalid

_LOADED: dict[tuple[str, str], str] = {}
_DIGEST = "sha256:"


def _digest(value: dict[str, Any]) -> str:
    return _DIGEST + hashlib.sha256(rfc8785.dumps(value)).hexdigest()


def _require_string(value: Any, name: str, path: Path) -> str:
    if not isinstance(value, str) or not value:
        raise PolicyInvalid(f"{path} has no non-empty {name}")
    return value


def load_policy(path: Path | str) -> Policy:
    """Load one immutable policy file, refusing malformed or in-place revisions."""
    policy_path = Path(path)
    try:
        loaded = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PolicyInvalid(f"{policy_path} does not parse as a policy file: {error}") from error
    if not isinstance(loaded, dict):
        raise PolicyInvalid(f"{policy_path} is not a policy object")

    policy_id = _require_string(loaded.get("id"), "id", policy_path)
    version = _require_string(loaded.get("version"), "version", policy_path)
    raw_checks = loaded.get("checks")
    if not isinstance(raw_checks, list) or not raw_checks:
        raise PolicyInvalid(f"{policy_path} names no required evidence checks")
    checks: list[PolicyCheck] = []
    seen: set[str] = set()
    for raw in raw_checks:
        if not isinstance(raw, dict):
            raise PolicyInvalid(f"{policy_path} has a non-object required evidence check")
        key = _require_string(raw.get("key"), "check key", policy_path)
        if key in seen:
            raise PolicyInvalid(f"{policy_path} names required evidence {key!r} twice")
        seen.add(key)
        source = _require_string(raw.get("source"), "check source", policy_path)
        if source not in {
            "section_measured",
            "all_checks_complete",
            "outcome_qualified",
            "finding_guard",
            "entry_complete",
            "entry_required_check",
            "entry_result_field",
            "entry_any_complete",
            "entry_any_control_field",
            "future_interface",
        }:
            raise PolicyInvalid(f"{policy_path} names evidence key {key!r} with an unknown source")
        section = raw.get("section")
        entry_id = raw.get("entry_id")
        alternate_entry_ids = raw.get("alternate_entry_ids", [])
        result_field = raw.get("result_field")
        if section is not None and not isinstance(section, str):
            raise PolicyInvalid(f"{policy_path} gives {key!r} a non-string section")
        if entry_id is not None and not isinstance(entry_id, str):
            raise PolicyInvalid(f"{policy_path} gives {key!r} a non-string entry id")
        if not isinstance(alternate_entry_ids, list) or not all(
            isinstance(item, str) and item for item in alternate_entry_ids
        ):
            raise PolicyInvalid(f"{policy_path} gives {key!r} invalid alternate entry ids")
        if result_field is not None and not isinstance(result_field, str):
            raise PolicyInvalid(f"{policy_path} gives {key!r} a non-string result field")
        fresh = raw.get("fresh_for_seconds")
        if fresh is not None and (not isinstance(fresh, int) or fresh <= 0):
            raise PolicyInvalid(f"{policy_path} gives {key!r} an invalid freshness interval")
        threshold = raw.get("threshold")
        if threshold is not None and threshold not in {"note", "warning", "error"}:
            raise PolicyInvalid(f"{policy_path} gives {key!r} an invalid severity threshold")
        if source != "outcome_qualified" and section not in Measurement.model_fields:
            raise PolicyInvalid(
                f"{policy_path} names evidence key {key!r} in an unknown record section"
            )
        if source in {
            "entry_complete",
            "entry_required_check",
            "entry_result_field",
            "entry_any_complete",
            "entry_any_control_field",
            "future_interface",
        } and (entry_id is None or section is None or not entry_id.startswith(section + ".")):
            raise PolicyInvalid(
                f"{policy_path} names evidence key {key!r} without a producer entry"
            )
        if any(not item.startswith((section or "") + ".") for item in alternate_entry_ids):
            raise PolicyInvalid(
                f"{policy_path} gives {key!r} an alternate outside its producer section"
            )
        check = PolicyCheck(
            key,
            source,
            section,
            entry_id,
            tuple(alternate_entry_ids),
            result_field,
            None if fresh is None else timedelta(seconds=fresh),
            threshold,
        )
        validate_producer(check)
        checks.append(check)
    try:
        interval = Decimal(str(loaded.get("default_interval")))
    except (InvalidOperation, ValueError) as error:
        raise PolicyInvalid(f"{policy_path} has an invalid default interval") from error
    if not Decimal("0") < interval <= Decimal("1"):
        raise PolicyInvalid(f"{policy_path} has a default interval outside (0, 1]")
    raw_multiplicity = loaded.get("multiplicity")
    if not isinstance(raw_multiplicity, dict):
        raise PolicyInvalid(f"{policy_path} has no multiplicity rule")
    multiplicity = Multiplicity(
        _require_string(raw_multiplicity.get("rule"), "multiplicity.rule", policy_path),
        _require_string(raw_multiplicity.get("family"), "multiplicity.family", policy_path),
    )
    actions = loaded.get("actions")
    if (
        not isinstance(actions, list)
        or not actions
        or not all(isinstance(item, str) and item for item in actions)
    ):
        raise PolicyInvalid(f"{policy_path} has no named actions")
    digest = _digest(loaded)
    identity = (policy_id, version)
    previous = _LOADED.setdefault(identity, digest)
    if previous != digest:
        raise PolicyInvalid(
            f"{policy_path} changed {policy_id!r} version {version!r} in place; make a new version"
        )
    return Policy(policy_id, version, digest, tuple(checks), interval, multiplicity, tuple(actions))
