"""Serialisation helpers for version-local improve evidence.

This module stores only facts acquired by the loop: exact pair bytes, the
planner's pre-execution choice, and pointers to the experiment that ran.  It
does not turn candidate prose or a caller-provided score into evidence.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import state


def attach(path: Path, *, experiment: dict[str, Any]) -> dict[str, Any]:
    payload = state.read(path)
    payload.setdefault("experiments", []).append(experiment)
    payload["state"] = "export"
    state.write(path, payload)
    return payload


def mark_pair(path: Path, *, left: Path | str, right: Path | str, why: str) -> dict[str, Any]:
    """Retain a non-equivalent pair by the exact bytes that make it a pair."""
    left_path, right_path = Path(left), Path(right)
    if not isinstance(why, str) or not why.strip() or left_path == right_path:
        raise ValueError("IMPROVE_RESUME_REQUIRED: pair and why are required")
    try:
        left_bytes, right_bytes = left_path.read_bytes(), right_path.read_bytes()
    except OSError as error:
        raise ValueError(
            "IMPROVE_RESUME_REQUIRED: mark_pair paths must name readable versions"
        ) from error
    payload = state.read(path)
    pair = {
        "left": {
            "path": str(left_path),
            "sha256": "sha256:" + hashlib.sha256(left_bytes).hexdigest(),
        },
        "right": {
            "path": str(right_path),
            "sha256": "sha256:" + hashlib.sha256(right_bytes).hexdigest(),
        },
        "why": why.strip(),
        "regression_case": True,
    }
    if pair["left"]["sha256"] == pair["right"]["sha256"]:
        raise ValueError("IMPROVE_RESUME_REQUIRED: mark_pair versions have identical bytes")
    pairs = list(payload.get("pairs", ()))
    if pair not in pairs:
        pairs.append(pair)
    payload["pairs"] = pairs
    state.write(path, payload)
    return payload


def choice_payload(choice: Any) -> dict[str, Any]:
    """A stable, public subset of a planner choice suitable for the session."""
    cost = getattr(choice, "cost", None)
    return {
        "kind": getattr(choice, "kind", None),
        "rung": getattr(choice, "rung", None),
        "choice_digest": getattr(choice, "choice_digest", None),
        "registration_id": getattr(choice, "registration_id", None),
        "separated_pairs": [list(pair) for pair in getattr(choice, "separated_pairs", ())],
        "unresolved_pairs": [list(pair) for pair in getattr(choice, "unresolved_pairs", ())],
        "expected_outcomes": list(getattr(choice, "expected_outcomes", ())),
        "missing_access": list(getattr(choice, "missing_access", ())),
        "missing_measurement": getattr(choice, "missing_measurement", None),
        "cost": None
        if cost is None
        else {
            "rung": cost.rung,
            "rank": cost.rank,
            "money_usd": str(cost.money_usd),
            "samples": cost.samples,
            "minutes": cost.minutes,
            "access": list(cost.access),
        },
        "alternatives": [
            {
                "rung": item.rung,
                "cost": {
                    "rung": item.cost.rung,
                    "rank": item.cost.rank,
                    "money_usd": str(item.cost.money_usd),
                    "samples": item.cost.samples,
                    "minutes": item.cost.minutes,
                    "access": list(item.cost.access),
                },
                "separates_pairs": [list(pair) for pair in item.separates_pairs],
                "unresolved_pairs": [list(pair) for pair in item.unresolved_pairs],
                "feasible": item.feasible,
                "missing_access": list(item.missing_access),
                "over_budget": item.over_budget,
                "prior_support_case_ids": list(item.prior_support_case_ids),
            }
            for item in getattr(choice, "alternatives", ())
        ],
        "selection_rule": getattr(choice, "selection_rule", None),
        "compared_prior_support": [
            {"rung": rung, "case_ids": list(ids)}
            for rung, ids in getattr(choice, "compared_prior_support", ())
        ],
        "corpus_provenance": getattr(choice, "corpus_provenance", None),
        "cost_table_provenance": getattr(choice, "cost_table_provenance", None),
        "evaluator_contract": (
            None
            if getattr(choice, "evaluator_contract", None) is None
            else choice.evaluator_contract.payload()
        ),
        "prior_case": (
            None
            if getattr(choice, "prior_case", None) is None
            else {
                "id": choice.prior_case.id,
                "explanation_kinds": list(choice.prior_case.explanation_kinds),
                "rung": choice.prior_case.rung,
                "analogy": choice.prior_case.analogy,
                "case_digest": choice.prior_case.case_digest,
                "source_digest": choice.prior_case.source_digest,
                "annotation_digest": choice.prior_case.annotation_digest,
            }
        ),
        "refusal": (
            None
            if getattr(choice, "refusal", None) is None
            else {"code": choice.refusal.code, "detail": choice.refusal.detail}
        ),
    }


def preregister(
    path: Path,
    *,
    choice: Any,
    compare_request: Any,
    candidate: dict[str, Any],
    forecast_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal the expectation and exact execution inputs before Compare starts."""
    from reward_lens import contracts

    choice_data = choice_payload(choice)
    request_data = compare_request.model_dump(mode="json")
    file_inputs = {}
    for name in (
        "bank",
        "run_dir",
        "baseline_config",
        "candidate_config",
        "selection_config",
        "controlled_update_config",
        "short_fork_config",
        "transfer_config",
    ):
        value = getattr(compare_request, name, None)
        if value is not None:
            source = Path(value)
            if source.is_file():
                file_inputs[str(value)] = (
                    "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
                )
            elif source.is_dir():
                digest = hashlib.sha256()
                for item in sorted(item for item in source.rglob("*") if item.is_file()):
                    digest.update(item.relative_to(source).as_posix().encode("utf-8"))
                    digest.update(b"\0")
                    digest.update(item.read_bytes())
                    digest.update(b"\0")
                file_inputs[str(value)] = "sha256:" + digest.hexdigest()
    declared = {
        "planner_choice": choice_data,
        "compare_request": request_data,
        "candidate": candidate,
        "input_digests": file_inputs,
    }
    if forecast_plan is not None:
        declared["forecast_plan"] = forecast_plan
    file = path.parent / (
        "preregister-" + str(choice.registration_id).rsplit(":", 1)[-1][-12:] + ".json"
    )
    if file.exists():
        raw = file.read_bytes()
        payload = json.loads(raw)
        if payload.get("declaration_digest") != contracts.digest(declared) or any(
            payload.get(key) != value for key, value in declared.items()
        ):
            raise ValueError(
                "IMPROVE_RESUME_REQUIRED: preregistered expectation or execution inputs changed"
            )
    else:
        now = datetime.now(timezone.utc)
        payload = {
            "schema": "reward-lens.improve-preregistration/1",
            **declared,
            "declaration_digest": contracts.digest(declared),
            "registered_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "registered_at_precise": now.isoformat(),
        }
        raw = state.write_once(file, payload)
    return {
        "path": str(file),
        "digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "registered_at": payload["registered_at"],
        "registered_at_precise": payload["registered_at_precise"],
        "declaration_digest": payload["declaration_digest"],
    }


def require_registration(registration: dict[str, Any]) -> dict[str, Any]:
    """Reopen the committed expectation; a changed file cannot acquire a new timestamp."""
    raw = Path(registration["path"]).read_bytes()
    if "sha256:" + hashlib.sha256(raw).hexdigest() != registration["digest"]:
        raise ValueError("IMPROVE_RESUME_REQUIRED: immutable preregistration digest changed")
    payload = json.loads(raw)
    if any(
        payload[key] != registration[key]
        for key in ("registered_at", "registered_at_precise", "declaration_digest")
    ):
        raise ValueError("IMPROVE_RESUME_REQUIRED: immutable preregistration receipt changed")
    return payload
