"""A refused forecast request must not leave a partial append-only record."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _row(target: str) -> dict[str, object]:
    names = ("persistence", "climatology", "drift", "covariance")
    return {
        "target": target,
        "checkpoint": 4,
        "source_name": "fixture-run",
        "source_cutoff": 4,
        "forecast": 0.8,
        "baselines": {name: 0.5 for name in names},
        "baseline_provenance": {
            name: {"cutoff": 4, "fit_digest": "sha256:" + "a" * 64} for name in names
        },
        "model_digest": "sha256:" + "b" * 64,
    }


def test_invalid_second_issue_does_not_append_header_or_first_issue(tmp_path: Path) -> None:
    from reward_lens.product.forecast import issue

    record = tmp_path / "issues.json"
    invalid = _row("T2")  # Fraction targets require a quantile distribution.
    record.write_text(json.dumps({"issues": [_row("T1"), invalid]}))
    request = SimpleNamespace(record=record, claim="fixture", offline=True, seed=7)
    project = SimpleNamespace(root=tmp_path)
    ledger = tmp_path / ".reward-lens" / "forecast-ledger.jsonl"

    with pytest.raises(ValueError, match="quantile distribution"):
        issue(request, project=project)
    assert not ledger.exists()

    record.write_text(json.dumps({"issues": [_row("T1")]}))
    issue(request, project=project)
    original = ledger.read_bytes()
    record.write_text(json.dumps({"issues": [_row("T2")]}))
    with pytest.raises(ValueError, match="quantile distribution"):
        issue(request, project=project)
    assert ledger.read_bytes() == original


def test_rendered_resolution_time_matches_the_append_only_row(tmp_path: Path) -> None:
    from reward_lens.product.forecast import issue, render_ledger, resolve

    record = tmp_path / "issue.json"
    record.write_text(json.dumps({"issues": [_row("T1")]}))
    project = SimpleNamespace(root=tmp_path)
    issue(SimpleNamespace(record=record, claim="fixture", offline=True, seed=7), project=project)
    ledger = tmp_path / ".reward-lens" / "forecast-ledger.jsonl"
    issued = next(
        row for row in map(json.loads, ledger.read_text().splitlines()) if row["kind"] == "issue"
    )
    resolve(
        SimpleNamespace(forecast_id=issued["sealed_digest"], outcome="true", offline=True, seed=7),
        project=project,
    )
    settled = next(
        row for row in map(json.loads, ledger.read_text().splitlines()) if row["kind"] == "resolve"
    )
    assert settled["resolved_at"] in render_ledger(project)
