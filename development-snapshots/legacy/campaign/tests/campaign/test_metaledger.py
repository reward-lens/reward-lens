"""Tests for the META-LEDGER card: the campaign scoring its own calibration.

The tiny path runs through the real ``run_study`` so the frozen spec adjudicates the exact
registered metric strings; the hand-computed case pins the Brier and coverage arithmetic to
values checkable on paper; the resolver refusal documents that this card is driven by the
reporting layer, never by store reads.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.core.store import EvidenceStore
from reward_lens.studies.runner import run_study

from campaign import metaledger
from campaign.config import task_seed
from campaign.registry import (
    CARDS_BY_ID,
    brier_hypotheses,
    interval_hypotheses,
    required_metrics_for,
)
from campaign.specs import build_spec

CARD = CARDS_BY_ID["META-LEDGER"]


def _run(subjects, tmp_path):
    store = EvidenceStore(tmp_path / "store")
    spec = build_spec(CARD)
    return run_study(spec, subjects=subjects, store=store)


def test_probabilities_payload_matches_the_registry():
    payload = metaledger.probabilities_payload()
    directional = payload["directional"]
    assert len(directional) == len(brier_hypotheses())
    cards = [e["card"] for e in directional]
    assert len(cards) == len(set(cards)), "one headline call per card"
    assert "META-LEDGER" not in cards, "the meta card never scores itself"
    for entry in directional:
        assert 0.0 < entry["prob"] < 1.0
    intervals = payload["intervals"]
    assert len(intervals) == len(interval_hypotheses())
    assert {e["card"] for e in intervals} == {"HUMP", "CAL-TRANSFER"}


def test_tiny_path_emits_required_metrics_and_confirms(tmp_path):
    rng = np.random.default_rng(task_seed("test-metaledger", "tiny"))
    subjects = metaledger.tiny_subjects(rng)
    _, result = _run(subjects, tmp_path)
    for metric in required_metrics_for(CARD):
        assert metric in result.metrics, f"omitted {metric}"
    assert not result.killed, result.killed_by
    # The plant is calibrated by construction: one directional miss, one interval miss.
    assert 0.0 < result.metrics["brier_directional"] < 0.25
    assert result.metrics["interval_coverage_0p8"] == pytest.approx(0.8)
    assert result.metrics["n_directional"] == len(brier_hypotheses())
    assert result.metrics["n_intervals"] == len(interval_hypotheses()) + len(
        metaledger.LADDER_INTERVAL_NAMES)
    assert result.outcomes["H-brier"] == "confirmed"
    assert result.outcomes["H-coverage"] == "confirmed"


def test_brier_and_coverage_arithmetic_by_hand(tmp_path):
    subjects = {
        "outcomes": {
            "CARD-A": {"hypothesis_id": "h-a", "confirmed": True, "prob": None},
            "CARD-B": {"hypothesis_id": "h-b", "confirmed": False, "prob": None},
        },
        "intervals": [{"name": "iv-1", "hit": True}, {"name": "iv-2", "hit": False}],
        "probabilities": {
            "directional": [
                {"card": "CARD-A", "hypothesis": "h-a", "metric": "m", "prob": 0.8},
                {"card": "CARD-B", "hypothesis": "h-b", "metric": "m", "prob": 0.6},
            ],
            "intervals": [],
        },
    }
    _, result = _run(subjects, tmp_path)
    # ((0.8 - 1)^2 + (0.6 - 0)^2) / 2 = (0.04 + 0.36) / 2 = 0.2
    assert result.metrics["brier_directional"] == pytest.approx(0.2)
    assert result.metrics["interval_coverage_0p8"] == pytest.approx(0.5)
    assert result.metrics["n_directional"] == 2.0
    assert result.metrics["n_intervals"] == 2.0
    assert not result.killed


def test_frozen_probabilities_override_outcome_probs(tmp_path):
    # The frozen record wins over whatever probability the outcome entry carries.
    subjects = {
        "outcomes": {"CARD-A": {"hypothesis_id": "h", "confirmed": True, "prob": 0.99}},
        "intervals": [{"name": "iv", "hit": True}],
        "probabilities": {
            "directional": [{"card": "CARD-A", "hypothesis": "h", "metric": "m",
                             "prob": 0.5}],
            "intervals": [],
        },
    }
    _, result = _run(subjects, tmp_path)
    assert result.metrics["brier_directional"] == pytest.approx(0.25)


def test_missing_pre_run_probability_is_an_error(tmp_path):
    subjects = {
        "outcomes": {"UNSCORED": {"hypothesis_id": "h", "confirmed": True}},
        "intervals": [],
        "probabilities": {"directional": [], "intervals": []},
    }
    with pytest.raises(KeyError, match="pre-run probability"):
        _run(subjects, tmp_path)


def test_resolve_subjects_refuses_store_reads(tmp_path):
    store = EvidenceStore(tmp_path / "store")
    with pytest.raises(KeyError, match="reporting layer"):
        metaledger.resolve_subjects(store)
