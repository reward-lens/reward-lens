"""Registered T3 outcomes from sealed, actually scored reward groups.

This source measures contrast in the next registered window. It does not infer
optimizer behaviour or turn one proper score into a forecasting skill claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import rfc8785

from reward_lens import contracts
from reward_lens.forecast.evidence import ForecastRefusal
from reward_lens.graders.base import EvidenceEnvelope
from reward_lens.product.trace.follow import follow, seal_chunk


def _digest(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(rfc8785.dumps(dict(value))).hexdigest()


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["digest"] = _digest(payload)
    return payload


def _read(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise ForecastRefusal(
            f"FORECAST_MEASUREMENT_TAMPER {label} is unavailable or malformed"
        ) from error
    if not isinstance(value, dict):
        raise ForecastRefusal(f"FORECAST_MEASUREMENT_TAMPER {label} must be an object")
    digest = value.get("digest")
    payload = dict(value)
    payload.pop("digest", None)
    if digest != _digest(payload):
        raise ForecastRefusal(f"FORECAST_MEASUREMENT_TAMPER {label} digest differs")
    return value


def _issue(project: Any, forecast_id: str) -> Any:
    from reward_lens.product.forecast import _issues, _load, _path

    issued = _issues(_load(_path(project))).get(forecast_id)
    if issued is None or issued.target != "T3":
        raise ForecastRefusal(
            "FORECAST_MEASUREMENT_TAMPER registered forecast is not a sealed T3 issue"
        )
    return issued


def _issue_time(project: Any, forecast_id: str) -> datetime:
    from reward_lens.product.forecast import _load, _path, _recorded_time

    rows = _load(_path(project))
    row = next(
        (
            row
            for row in rows
            if row.get("kind") == "issue" and row.get("sealed_digest") == forecast_id
        ),
        None,
    )
    if row is None:
        raise ForecastRefusal("FORECAST_MEASUREMENT_TAMPER registered issue is missing")
    return _recorded_time(row.get("issued_at"), field="issued_at")


def _registration(project: Any, forecast_id: str, path: Path | str) -> dict[str, Any]:
    _issue(project, forecast_id)
    path = Path(path)
    expected = (
        Path(str(project.root))
        / ".reward-lens"
        / "forecast-measurements"
        / (forecast_id[7:] + ".registration.json")
    )
    if path.resolve() != expected.resolve():
        raise ForecastRefusal("FORECAST_MEASUREMENT_TAMPER registration path differs from forecast")
    value = _read(path, "registration")
    if value.get("forecast_id") != forecast_id:
        raise ForecastRefusal("FORECAST_MEASUREMENT_TAMPER registration forecast id differs")
    from reward_lens.product.forecast import _recorded_time

    if _recorded_time(value.get("registered_at"), field="registered_at") <= _issue_time(
        project, forecast_id
    ):
        raise ForecastRefusal("FORECAST_MEASUREMENT_TAMPER registration did not follow issue")
    return value


def register_measurement(
    project: Any,
    forecast_id: str,
    *,
    context: Mapping[str, Any],
    capture_path: Path,
    expected_groups: Mapping[str, list[str] | tuple[str, ...]] | None = None,
) -> Path:
    """Freeze context and a new capture destination after the T3 issue."""
    issued = _issue(project, forecast_id)
    root = Path(str(project.root)).resolve()
    capture = Path(capture_path).resolve()
    if not capture.is_relative_to(root) or os.path.lexists(capture):
        raise ForecastRefusal(
            "FORECAST_MEASUREMENT_TAMPER capture must be a new project-local path"
        )
    window = context.get("window")
    if (
        not isinstance(window, Mapping)
        or window.get("checkpoint") != issued.checkpoint
        or window.get("start") != issued.checkpoint + 1
        or window.get("end") != window.get("start")
    ):
        raise ForecastRefusal("FORECAST_MEASUREMENT_TAMPER context window differs from issue")
    roster: dict[str, list[str]] | None = None
    if expected_groups is not None:
        if not expected_groups:
            raise ForecastRefusal("FORECAST_MEASUREMENT_TAMPER expected group roster is empty")
        roster = {}
        for task_id, digests in expected_groups.items():
            values = [str(value) for value in digests]
            if not task_id or len(values) < 2 or len(values) != len(set(values)):
                raise ForecastRefusal(
                    "FORECAST_MEASUREMENT_TAMPER expected group roster is invalid"
                )
            roster[str(task_id)] = sorted(values)
    destination = (
        root / ".reward-lens" / "forecast-measurements" / (forecast_id[7:] + ".registration.json")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ForecastRefusal("FORECAST_MEASUREMENT_TAMPER registration already exists")
    registration = _sealed(
        {
            "forecast_id": forecast_id,
            "context": dict(context),
            "capture_path": str(capture),
            "expected_groups": roster,
            "registered_at": datetime.now(UTC).isoformat(),
        }
    )
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(registration, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return destination


def capture_measurement(
    project: Any,
    forecast_id: str,
    *,
    registration: Path,
    collector: Callable[[], list[dict[str, Any]]],
) -> Path:
    """Execute the registered collector once and seal its scored rows."""
    bound = _registration(project, forecast_id, registration)
    capture = Path(bound["capture_path"])
    if not capture.resolve().is_relative_to(Path(str(project.root)).resolve()) or os.path.lexists(
        capture
    ):
        raise ForecastRefusal("FORECAST_MEASUREMENT_TAMPER capture already exists")
    started = datetime.now(UTC).isoformat()
    rows = collector()
    if not isinstance(rows, list) or not rows:
        raise ForecastRefusal("FORECAST_MEASUREMENT_TAMPER collector produced no scored rows")
    seal_chunk(capture, 1, rows)
    finished = datetime.now(UTC).isoformat()
    payload = _sealed(
        {
            "forecast_id": forecast_id,
            "registration_digest": bound["digest"],
            "collection_started_at": started,
            "collection_finished_at": finished,
            "measured_at": datetime.now(UTC).isoformat(),
            "chunk_sha256": "sha256:"
            + hashlib.sha256((capture / "chunk_00001.jsonl").read_bytes()).hexdigest(),
            "row_count": len(rows),
        }
    )
    destination = capture / "FORECAST_CAPTURE.json"
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return destination


def _derive(project: Any, forecast_id: str, registration: Path) -> dict[str, Any]:
    issued = _issue(project, forecast_id)
    bound = _registration(project, forecast_id, registration)
    capture = Path(bound["capture_path"])
    if capture.is_symlink() or not capture.resolve().is_relative_to(
        Path(str(project.root)).resolve()
    ):
        raise ForecastRefusal("FORECAST_MEASUREMENT_TAMPER capture path changed")
    manifest = _read(capture / "FORECAST_CAPTURE.json", "capture")
    if (
        manifest.get("forecast_id") != forecast_id
        or manifest.get("registration_digest") != bound["digest"]
    ):
        raise ForecastRefusal("FORECAST_MEASUREMENT_TAMPER capture differs from registration")
    from reward_lens.product.forecast import _recorded_time

    registered_at = _recorded_time(bound["registered_at"], field="registered_at")
    started_at = _recorded_time(
        manifest.get("collection_started_at"), field="collection_started_at"
    )
    finished_at = _recorded_time(
        manifest.get("collection_finished_at"), field="collection_finished_at"
    )
    measured_at = _recorded_time(manifest.get("measured_at"), field="measured_at")
    if not (registered_at < started_at <= finished_at <= measured_at):
        raise ForecastRefusal("FORECAST_MEASUREMENT_TAMPER collection chronology is invalid")
    chunk_path = capture / "chunk_00001.jsonl"
    try:
        chunk_bytes = chunk_path.read_bytes()
    except OSError as error:
        raise ForecastRefusal("FORECAST_MEASUREMENT_TAMPER recorded chunk is missing") from error
    if "sha256:" + hashlib.sha256(chunk_bytes).hexdigest() != manifest.get("chunk_sha256"):
        raise ForecastRefusal("FORECAST_MEASUREMENT_TAMPER recorded chunk changed")
    try:
        chunks = tuple(follow(capture, policy="strict"))
    except (OSError, ValueError) as error:
        raise ForecastRefusal("FORECAST_MEASUREMENT_TAMPER recorded chunk is invalid") from error
    if len(chunks) != 1 or not chunks[0].sealed or chunks[0].seq != 1:
        raise ForecastRefusal("FORECAST_MEASUREMENT_TAMPER recorded chunk seal is invalid")
    rows = chunks[0].rows
    if len(rows) != manifest.get("row_count"):
        raise ForecastRefusal("FORECAST_MEASUREMENT_TAMPER recorded row count differs")
    context = bound["context"]
    groups: dict[str, list[float]] = defaultdict(list)
    response_roster: dict[str, list[str]] = defaultdict(list)
    seen_pairs: set[tuple[str, str]] = set()
    seen_rows: set[str] = set()
    for row in rows:
        if (
            row.get("step") != context["window"]["start"]
            or row.get("run_id") != issued.source_name
            or row.get("subject_ref") != context["subject_ref"]
            or row.get("task_set") != context["task_set"]
            or row.get("configuration_digest") != context["configuration_digest"]
            or row.get("accepted") is not True
            or row.get("reward_application_profile") != "finite-bank-selector-recorded-groups/1"
        ):
            raise ForecastRefusal("FORECAST_MEASUREMENT_TAMPER row differs from registered context")
        task_id = str(row.get("task_id", ""))
        response_digest = str(row.get("response_digest", ""))
        row_id = str(row.get("row_id", ""))
        if (
            not task_id
            or not response_digest
            or not row_id
            or row.get("group_id") != "recorded-group-" + task_id
            or (task_id, response_digest) in seen_pairs
            or row_id in seen_rows
        ):
            raise ForecastRefusal(
                "FORECAST_MEASUREMENT_TAMPER duplicate or unbound captured response"
            )
        seen_pairs.add((task_id, response_digest))
        seen_rows.add(row_id)
        try:
            envelope = EvidenceEnvelope.from_dict(row["scoring_envelope"])
            value = float(row["reward_applied"])
        except (KeyError, TypeError, ValueError) as error:
            raise ForecastRefusal(
                "FORECAST_MEASUREMENT_TAMPER row has no valid scored reward"
            ) from error
        if (
            envelope.verdict != "scored"
            or envelope.score != value
            or envelope.digest() != row.get("scoring_envelope_digest")
            or not math.isfinite(value)
        ):
            raise ForecastRefusal("FORECAST_MEASUREMENT_TAMPER applied reward differs from scorer")
        groups[task_id].append(value)
        response_roster[task_id].append(response_digest)
    expected_groups = bound.get("expected_groups")
    if expected_groups is not None and (
        set(groups) != set(expected_groups)
        or any(sorted(response_roster[task]) != expected_groups[task] for task in expected_groups)
    ):
        raise ForecastRefusal(
            "FORECAST_MEASUREMENT_TAMPER captured groups differ from registered roster"
        )
    if not groups or any(len(values) < 2 for values in groups.values()):
        raise ForecastRefusal("FORECAST_MEASUREMENT_TAMPER each group needs two scored responses")
    eligible = len(groups)
    contrasted = sum(max(values) > min(values) for values in groups.values())
    outcome = contrasted / eligible
    if eligible < 15:
        uncertainty = contracts.Uncertainty(
            method=contracts.NO_INTERVAL,
            level=0.95,
            reason=f"{eligible} task groups are below the fifteen required for an interval",
            clusters=eligible,
        )
    else:
        z = 1.959963984540054
        denominator = 1 + z * z / eligible
        center = (outcome + z * z / (2 * eligible)) / denominator
        radius = (
            z
            * math.sqrt(outcome * (1 - outcome) / eligible + z * z / (4 * eligible * eligible))
            / denominator
        )
        uncertainty = contracts.Uncertainty(
            method="wilson",
            level=0.95,
            interval=[
                contracts.quantise(max(0.0, center - radius), 6),
                contracts.quantise(min(1.0, center + radius), 6),
            ],
            clusters=eligible,
        )
    entry = contracts.estimate(
        "signal",
        "signal.contrast_fraction",
        "fraction of declared reward groups with any within-group applied-reward contrast",
        outcome,
        "fraction",
        eligible,
        "task",
        uncertainty,
        subject_ref=context["subject_ref"],
        provenance=contracts.EntryProvenance(
            started=datetime.fromisoformat(manifest["collection_started_at"]).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            duration_s=0.0,
            sandbox_tier="T0",
            offline=True,
        ),
        method=contracts.Method(
            id="forecast.recorded_group_contrast",
            version="1.0.0",
            params_digest=contracts.digest(
                {"registration_digest": bound["digest"], "chunk_sha256": manifest["chunk_sha256"]}
            ),
            procedure="count groups with unequal applied rewards among registered scored responses",
        ),
        result={
            "window": context["window"],
            "configuration_digest": context["configuration_digest"],
            "eligible_groups": eligible,
            "contrasted_groups": contrasted,
        },
        scope="evaluator_comparison",
    )
    payload = {
        "profile": "recorded-reward-groups/1",
        "forecast_id": forecast_id,
        "target": "T3",
        "context": context,
        "registration": str(registration),
        "registration_digest": bound["digest"],
        "capture_path": str(capture),
        "capture_digest": manifest["digest"],
        "collection_started_at": manifest["collection_started_at"],
        "collection_finished_at": manifest["collection_finished_at"],
        "measured_at": manifest["measured_at"],
        "sampling_unit": "task",
        "measurand": entry.measurand,
        "native_entry_id": entry.entry_id,
        "outcome_fraction": [contrasted, eligible],
        "outcome": str(outcome),
        "steps": [
            {
                "step": context["window"]["start"],
                "eligible_groups": eligible,
                "contrasted_groups": contrasted,
                "entry": entry.to_dict(),
            }
        ],
    }
    return _sealed(payload)


def derive_outcome(project: Any, forecast_id: str, *, registration: Path) -> dict[str, Any]:
    """Reopen the seal and derive T3 from the recorded applied rewards."""
    return _derive(project, forecast_id, registration)


def verify_outcome(project: Any, forecast_id: str, outcome: Mapping[str, Any]) -> dict[str, Any]:
    """On resume, compare the saved outcome to a fresh derivation from sealed bytes."""
    if not isinstance(outcome, Mapping) or outcome.get("forecast_id") != forecast_id:
        raise ForecastRefusal("FORECAST_MEASUREMENT_TAMPER saved outcome identity differs")
    expected = _derive(project, forecast_id, Path(str(outcome.get("registration"))))
    if dict(outcome) != expected:
        raise ForecastRefusal("FORECAST_MEASUREMENT_TAMPER saved outcome differs from capture")
    return expected


__all__ = ["register_measurement", "capture_measurement", "derive_outcome", "verify_outcome"]
