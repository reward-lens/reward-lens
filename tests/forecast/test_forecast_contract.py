from __future__ import annotations

from decimal import Decimal

import pytest


def _provenance(cutoff: int) -> dict[str, object]:
    from reward_lens.forecast.evidence import BASELINES, BaselineProvenance

    return {
        name: BaselineProvenance(cutoff=cutoff, fit_digest="sha256:" + ("a" * 64))
        for name in BASELINES
    }


def _baselines() -> dict[str, float]:
    return {"persistence": 0.5, "climatology": 0.4, "drift": 0.5, "covariance": 0.5}


def test_issue_seals_canonical_bytes_and_refuses_future_input() -> None:
    from reward_lens.forecast.evidence import ForecastSpec, issue

    with pytest.raises(ValueError, match=r"FORECAST_FUTURE_INPUT.*checkpoint=10.*source=future"):
        issue(
            checkpoint=10,
            spec=ForecastSpec.default(),
            target="T1",
            source_name="future",
            source_cutoff=11,
            forecast=0.6,
            baselines=_baselines(),
            model_digest="sha256:" + ("b" * 64),
            baseline_provenance=_provenance(10),
        )


def test_resolve_refuses_a_changed_sealed_payload() -> None:
    from reward_lens.forecast.evidence import ForecastSpec, issue, resolve

    issued = issue(
        checkpoint=10,
        spec=ForecastSpec.default(),
        target="T1",
        source_name="history",
        source_cutoff=10,
        forecast=0.6,
        baselines=_baselines(),
        model_digest="sha256:" + ("b" * 64),
        baseline_provenance=_provenance(10),
    )
    with pytest.raises(ValueError, match=r"FORECAST_RESOLUTION_MISMATCH.*sealed_digest="):
        resolve(issued, outcome=True, issued_bytes=issued.issued_bytes + b" ")


def test_baseline_future_window_cannot_change_issued_bytes() -> None:
    from reward_lens.forecast.baselines import (
        d70_climatology,
        d70_covariance,
        d70_drift,
        d70_persistence,
    )
    from reward_lens.forecast.evidence import ForecastSpec, issue

    history = [0.1, 0.2, 0.3, 0.4]
    features = [[0.1] * 6, [0.2] * 6, [0.3] * 6, [0.4] * 6]
    targets = list(history)

    def sealed() -> bytes:
        baselines = {
            "persistence": d70_persistence(history, checkpoint=2),
            "climatology": d70_climatology([0.2, 0.4], reference_cutoff=2),
            "drift": d70_drift(history, checkpoint=2, horizon=2),
            "covariance": d70_covariance(features, targets, checkpoint=2),
        }
        return issue(
            checkpoint=2,
            spec=ForecastSpec.default(),
            target="T1",
            source_name="history",
            source_cutoff=2,
            forecast=0.6,
            baselines=baselines,
            model_digest="sha256:" + ("b" * 64),
            baseline_provenance=_provenance(2),
        ).issued_bytes

    before = sealed()
    history[3] = targets[3] = 0.99
    features[3] = [0.99] * 6
    assert sealed() == before


def test_baseline_fit_provenance_is_mandatory_and_checkpoint_bounded() -> None:
    from reward_lens.forecast.evidence import BASELINES, BaselineProvenance, ForecastSpec, issue

    kwargs = dict(
        checkpoint=10,
        spec=ForecastSpec.default(),
        target="T1",
        source_name="history",
        source_cutoff=10,
        forecast=0.6,
        baselines=_baselines(),
        model_digest="sha256:" + ("b" * 64),
    )
    with pytest.raises(
        ValueError, match="FORECAST_BASELINE_LEAKAGE requires pre-checkpoint provenance"
    ):
        issue(**kwargs)
    provenance = {
        name: BaselineProvenance(cutoff=10, fit_digest="sha256:" + ("a" * 64)) for name in BASELINES
    }
    provenance["drift"] = BaselineProvenance(cutoff=11, fit_digest="sha256:" + ("d" * 64))
    with pytest.raises(ValueError, match=r"FORECAST_BASELINE_LEAKAGE.*checkpoint=10.*source=drift"):
        issue(**kwargs, baseline_provenance=provenance)


def test_fraction_issue_requires_distributions_and_reference_quantiles_are_preissue() -> None:
    from reward_lens.forecast.baselines import (
        d70_covariance_quantiles,
        d70_drift_quantiles,
        d70_persistence_quantiles,
        d70_reference_step_quantiles,
    )
    from reward_lens.forecast.evidence import ForecastSpec, issue

    runs = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]]
    distribution = d70_reference_step_quantiles(runs, checkpoint=1, horizon=1)
    assert distribution["values"] == pytest.approx((0.32, 0.35, 0.4, 0.45, 0.48))
    history = [0.1, 0.2, 0.4, 0.6]
    features = [[value] * 6 for value in history]
    assert len(d70_persistence_quantiles(history, checkpoint=2)["values"]) == 5
    assert len(d70_drift_quantiles(history, checkpoint=2, horizon=1)["values"]) == 5
    assert len(d70_covariance_quantiles(features, history, checkpoint=2)["values"]) == 5
    with pytest.raises(ValueError, match="quantile distribution"):
        issue(
            checkpoint=1,
            spec=ForecastSpec.default(),
            target="T2",
            source_name="history",
            source_cutoff=1,
            forecast=0.4,
            baselines=_baselines(),
            model_digest="sha256:" + ("b" * 64),
            baseline_provenance=_provenance(1),
        )
    issued = issue(
        checkpoint=1,
        spec=ForecastSpec.default(),
        target="T3",
        source_name="history",
        source_cutoff=1,
        forecast=distribution,
        baselines={name: distribution for name in _baselines()},
        model_digest="sha256:" + ("b" * 64),
        baseline_provenance=_provenance(1),
    )
    assert issued.target == "T3"
    assert Decimal(str(issued.forecast["values"][2])) == Decimal("0.4")
    assert set(issued.baseline_provenance) == {"persistence", "climatology", "drift", "covariance"}
