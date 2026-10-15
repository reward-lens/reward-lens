from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest


def _issued_and_resolved(forecast: float, baselines: dict[str, float]):
    from reward_lens.forecast.evidence import (
        BASELINES,
        BaselineProvenance,
        ForecastSpec,
        issue,
        resolve,
    )

    provenance = {name: BaselineProvenance(3, "sha256:" + ("a" * 64)) for name in BASELINES}
    issued = issue(
        checkpoint=3,
        spec=ForecastSpec.default(),
        target="T1",
        source_name="run",
        source_cutoff=3,
        forecast=forecast,
        baselines=baselines,
        model_digest="sha256:" + ("b" * 64),
        baseline_provenance=provenance,
    )
    return resolve(issued, outcome=True)


def test_t1_eprocesses_are_fixed_per_baseline_and_future_bettor_is_rejected() -> None:
    from reward_lens.forecast.evidence import (
        BASELINES,
        ForecastRefusal,
        future_informed_bet,
        sequential_evidence,
        validate_predictable_bettor,
    )

    row = _issued_and_resolved(
        0.9, {"persistence": 0.8, "climatology": 0.1, "drift": 0.7, "covariance": 0.6}
    )
    evidence = sequential_evidence("T1", (row,))
    assert tuple(evidence.processes) == BASELINES
    assert evidence.processes["climatology"].wealth == Decimal(1)
    assert evidence.processes["persistence"].wealth == Decimal(1)
    assert future_informed_bet((Decimal("0.5"),), Decimal(0)) != future_informed_bet(
        (Decimal("0.5"),), Decimal(1)
    )
    with pytest.raises(ForecastRefusal, match="FORECAST_FUTURE_BET"):
        validate_predictable_bettor(future_informed_bet)


def test_fraction_targets_have_no_sequential_evidence() -> None:
    from reward_lens.forecast.evidence import sequential_evidence

    with pytest.raises(ValueError, match="T2/T3.*sequential"):
        sequential_evidence("T2", ())


def test_registered_1000_stream_null_checks_each_fixed_baseline_and_negative_control() -> None:
    from reward_lens.forecast.evidence import BASELINES, ForecastSpec, null_stream_regression

    results = null_stream_regression(ForecastSpec.default(), length=20)
    assert tuple(results) == BASELINES
    assert all(
        0 <= crossings <= 1000 and p_value >= 0.01 for crossings, p_value in results.values()
    )


def test_headline_keeps_named_crossing_history_after_a_process_falls() -> None:
    from reward_lens.forecast.evidence import BASELINES, sequential_evidence

    wins = tuple(
        SimpleNamespace(
            issued=SimpleNamespace(target="T1"),
            score=Decimal(0),
            baseline_scores={name: Decimal(1) for name in BASELINES},
        )
        for _ in range(15)
    )
    fall = SimpleNamespace(
        issued=SimpleNamespace(target="T1"),
        score=Decimal(1),
        baseline_scores={name: Decimal(0) for name in BASELINES},
    )

    evidence = sequential_evidence("T1", (*wins, fall))

    assert evidence.all_crossed
    assert all(process.wealth < Decimal(20) for process in evidence.processes.values())
    assert evidence.headline_state["qualified"] is True
    assert set(evidence.headline_state["first_crossing_index"]) == set(BASELINES)


def test_alternating_convenient_baselines_cannot_replace_named_processes() -> None:
    from reward_lens.forecast.evidence import BASELINES, T1EProcess, sequential_evidence

    aggregate = T1EProcess()
    rows = []
    for index in range(60):
        convenient = BASELINES[index % len(BASELINES)]
        scores = {name: Decimal(int(name == convenient)) for name in BASELINES}
        rows.append(
            SimpleNamespace(
                issued=SimpleNamespace(target="T1"), score=Decimal(0), baseline_scores=scores
            )
        )
        aggregate.update(forecast_brier=0, baseline_brier=1, resolution_index=index)

    evidence = sequential_evidence("T1", rows)

    assert aggregate.first_crossing_index is not None
    assert evidence.all_crossed is False
    assert evidence.headline_state["crossed_baselines"] == []
    assert all(process.first_crossing_index is None for process in evidence.processes.values())
