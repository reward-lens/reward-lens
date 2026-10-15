"""Recorded T3 outcomes refuse replayed responses and resealed bad chronology."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _capture(tmp_path: Path, *, duplicate: bool):
    from reward_lens.graders.base import EvidenceEnvelope
    from reward_lens.product.forecast import issue
    from reward_lens.product.forecast.outcomes import capture_measurement, register_measurement

    project = SimpleNamespace(root=tmp_path)
    names = ("persistence", "climatology", "drift", "covariance")
    quantiles = {"values": [0.1, 0.2, 0.3, 0.4, 0.5]}
    source = tmp_path / "issue.json"
    source.write_text(
        json.dumps(
            {
                "spec": {"horizon": 1},
                "issues": [
                    {
                        "target": "T3",
                        "checkpoint": 4,
                        "source_name": "fixture-run",
                        "source_cutoff": 4,
                        "forecast": quantiles,
                        "baselines": {name: quantiles for name in names},
                        "baseline_provenance": {
                            name: {"cutoff": 4, "fit_digest": "sha256:" + "a" * 64}
                            for name in names
                        },
                        "model_digest": "sha256:" + "b" * 64,
                    }
                ],
            }
        )
    )
    issue(SimpleNamespace(record=source, claim="fixture", offline=True, seed=7), project=project)
    ledger = tmp_path / ".reward-lens" / "forecast-ledger.jsonl"
    forecast_id = next(
        row["sealed_digest"]
        for row in map(json.loads, ledger.read_text().splitlines())
        if row["kind"] == "issue"
    )
    context = {
        "subject_ref": "sha256:" + "c" * 64,
        "task_set": "sha256:" + "d" * 64,
        "configuration_digest": "sha256:" + "e" * 64,
        "window": {"checkpoint": 4, "start": 5, "end": 5},
    }
    registration = register_measurement(
        project,
        forecast_id,
        context=context,
        capture_path=tmp_path / "capture",
        expected_groups={"task-1": ["response-1", "response-2"]},
    )

    def row(index: int, response_digest: str) -> dict[str, object]:
        envelope = EvidenceEnvelope(
            run_id="fixture-run",
            subject_digest=context["subject_ref"],
            manifest_digest="sha256:" + "f" * 64,
            request_digest="sha256:" + "1" * 64,
            start="2026-01-01T00:00:00Z",
            end="2026-01-01T00:00:01Z",
            seed=7,
            verdict="scored",
            score=float(index),
        )
        return {
            "step": 5,
            "row_id": f"row-{index}",
            "response_digest": response_digest,
            "task_id": "task-1",
            "group_id": "recorded-group-task-1",
            "reward_applied": float(index),
            "accepted": True,
            "run_id": "fixture-run",
            "subject_ref": context["subject_ref"],
            "task_set": context["task_set"],
            "configuration_digest": context["configuration_digest"],
            "scoring_envelope": envelope.to_dict(),
            "scoring_envelope_digest": envelope.digest(),
            "reward_application_profile": "finite-bank-selector-recorded-groups/1",
        }

    responses = [row(1, "response-1"), row(2, "response-1" if duplicate else "response-2")]
    capture_measurement(
        project, forecast_id, registration=registration, collector=lambda: responses
    )
    return project, forecast_id, registration


def test_duplicate_captured_response_is_refused_even_with_valid_chunk_seal(tmp_path: Path) -> None:
    from reward_lens.forecast.evidence import ForecastRefusal
    from reward_lens.product.forecast.outcomes import derive_outcome

    project, forecast_id, registration = _capture(tmp_path, duplicate=True)
    with pytest.raises(ForecastRefusal, match="duplicate or unbound captured response"):
        derive_outcome(project, forecast_id, registration=registration)


def test_resealed_capture_cannot_move_collection_before_registration(tmp_path: Path) -> None:
    from reward_lens.forecast.evidence import ForecastRefusal
    from reward_lens.product.forecast.outcomes import _sealed, derive_outcome

    project, forecast_id, registration = _capture(tmp_path, duplicate=False)
    manifest_path = tmp_path / "capture" / "FORECAST_CAPTURE.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("digest")
    manifest["collection_started_at"] = json.loads(registration.read_text())["registered_at"]
    manifest_path.write_text(json.dumps(_sealed(manifest)))
    with pytest.raises(ForecastRefusal, match="chronology"):
        derive_outcome(project, forecast_id, registration=registration)
