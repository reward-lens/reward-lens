from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _run(name: str, *, onset: int | None, offset: float = 0.0) -> dict[str, object]:
    """Small controlled world. It is conformance data, not a benchmark."""
    observations = []
    for step in range(30):
        hacked = 0.15 + offset + (0.15 if onset is not None and step >= onset else 0.0)
        observations.append(
            {
                "step": step,
                "mean_reward": 0.2 + offset + 0.03 * step,
                "contrast": min(1.0, 0.05 + 0.04 * step + offset),
                "within_group_variance": 0.01 + 0.002 * step,
                "kl": 0.02 + 0.001 * step,
                "length": 5.0 + step,
                "hack_rate": hacked,
            }
        )
    return {"name": name, "observations": observations, "onset_step": onset}


def _references() -> tuple[dict[str, object], ...]:
    return (
        _run("frozen-a", onset=4, offset=0.00),
        _run("frozen-b", onset=5, offset=0.02),
        _run("frozen-c", onset=None, offset=0.04),
        _run("frozen-d", onset=6, offset=0.06),
    )


def test_reference_absence_and_future_feature_are_named_refusals() -> None:
    from reward_lens.forecast.evidence import ForecastSpec
    from reward_lens.forecast.predictors.hazard_v0 import HazardV0, PredictorRefusal

    with pytest.raises(PredictorRefusal, match=r"PREDICTOR_REFERENCE_ABSENT.*reference_runs"):
        HazardV0.fit((), ForecastSpec.default())
    with pytest.raises(PredictorRefusal, match=r"PREDICTOR_FUTURE_FEATURE.*cutoff=2"):
        HazardV0.fit(_references(), ForecastSpec.default()).predict(
            _references()[0], cutoff=2, future=True
        )


def test_frozen_fit_uses_declared_features_loro_inner_selection_and_censoring() -> None:
    from reward_lens.forecast.evidence import ForecastSpec
    from reward_lens.forecast.predictors.hazard_v0 import HazardV0

    fitted = HazardV0.fit(_references(), ForecastSpec.default())
    assert fitted.reference_names == ("frozen-a", "frozen-b", "frozen-c", "frozen-d")
    assert fitted.outer_folds == 4
    assert all(fold["held_out"] in fitted.reference_names for fold in fitted.fold_reports)
    assert all(fold["selection_scope"] == "inner-loro" for fold in fitted.fold_reports)
    assert "hack_rate_level" in fitted.feature_names
    assert fitted.censored_runs == ("frozen-c",)
    assert fitted.digest.startswith("sha256:") and len(fitted.digest) == 71

    prediction = fitted.predict(_references()[0], cutoff=3)
    assert 0.0 <= prediction.onset_probability <= 1.0
    assert len(prediction.hack_rate_quantiles) == 5
    assert len(prediction.contrast_quantiles) == 5
    assert prediction.feature_cutoff == 3
    assert prediction.survival[0] == pytest.approx(1.0)
    assert prediction.survival[-1] <= 1.0


def test_fit_is_frozen_and_digest_binds_coefficients_and_scalers() -> None:
    from reward_lens.forecast.evidence import ForecastSpec
    from reward_lens.forecast.predictors.hazard_v0 import HazardV0

    fitted = HazardV0.fit(_references(), ForecastSpec.default())
    changed = list(_references())
    changed[0] = _run("frozen-a", onset=4, offset=0.90)
    assert fitted.predict(_references()[0], cutoff=3) == fitted.predict(_references()[0], cutoff=3)
    assert fitted.digest != HazardV0.fit(tuple(changed), ForecastSpec.default()).digest
    material = fitted.material()
    assert material["scaler"] and material["hazard_coefficients"]
    assert material["digest"] == fitted.digest


def test_loro_fold_statistics_and_transforms_exclude_its_held_out_run() -> None:
    from reward_lens.forecast.evidence import ForecastSpec
    from reward_lens.forecast.predictors.hazard_v0 import HazardV0

    baseline = HazardV0.fit(_references(), ForecastSpec.default())
    changed = list(_references())
    changed[0] = _run("frozen-a", onset=4, offset=0.70)
    mutated = HazardV0.fit(tuple(changed), ForecastSpec.default())
    original_fold = next(
        report for report in baseline.fold_reports if report["held_out"] == "frozen-a"
    )
    changed_fold = next(
        report for report in mutated.fold_reports if report["held_out"] == "frozen-a"
    )
    assert "frozen-a" not in original_fold["outer_reference_names"]
    assert original_fold["outer_statistics_digest"] == changed_fold["outer_statistics_digest"]
    assert original_fold["outer_transform_digest"] == changed_fold["outer_transform_digest"]
    assert original_fold["selected_penalty"] == changed_fold["selected_penalty"]


def test_digest_binds_frozen_observations_and_reference_statistics() -> None:
    from reward_lens.forecast.evidence import ForecastSpec
    from reward_lens.forecast.predictors.hazard_v0 import HazardV0

    fitted = HazardV0.fit(_references(), ForecastSpec.default())
    material = fitted.material()
    assert material["frozen_reference_material"]
    assert material["reference_statistics"]
    assert material["reference_digest"].startswith("sha256:")
    altered = list(_references())
    altered[1] = _run("frozen-b", onset=5, offset=0.81)
    assert fitted.digest != HazardV0.fit(tuple(altered), ForecastSpec.default()).digest


def test_fitted_predictive_state_is_deeply_immutable_and_alias_safe() -> None:
    from reward_lens.forecast.evidence import ForecastSpec
    from reward_lens.forecast.predictors.hazard_v0 import HazardV0

    references = _references()
    fitted = HazardV0.fit(references, ForecastSpec.default())
    prospective = _run("prospective", onset=5, offset=0.03)
    before = fitted.issue_row(prospective, cutoff=3, target="T2", source_name="controlled")
    old_digest = fitted.digest
    with pytest.raises(TypeError):
        fitted.reference_stats[("hack_rate", 3)] = (99.0, 1.0)  # type: ignore[index]
    with pytest.raises(TypeError):
        fitted.frozen_reference_material[0]["observations"][4]["hack_rate"] = 0.99  # type: ignore[index]
    with pytest.raises(TypeError):
        dict.__setitem__(fitted.frozen_reference_material[0], "name", "forged")
    references[0]["observations"][4]["hack_rate"] = 0.99  # type: ignore[index]
    after = fitted.issue_row(prospective, cutoff=3, target="T2", source_name="controlled")
    assert after == before
    assert fitted.digest == old_digest
    assert HazardV0.fit(references, ForecastSpec.default()).digest != old_digest


def test_baselines_are_independent_frozen_reference_constructions() -> None:
    from reward_lens.forecast.evidence import ForecastSpec
    from reward_lens.forecast.predictors.hazard_v0 import HazardV0

    fitted = HazardV0.fit(_references(), ForecastSpec.default())
    t1 = fitted.issue_row(
        _run("prospective", onset=5, offset=0.03), cutoff=3, target="T1", source_name="controlled"
    )
    t2 = fitted.issue_row(
        _run("prospective", onset=5, offset=0.03), cutoff=3, target="T2", source_name="controlled"
    )
    assert t1["baselines"]["climatology"] != t1["forecast"]
    assert t1["baselines"]["covariance"] != t1["forecast"]
    assert t2["baselines"]["climatology"] != t2["forecast"]
    assert t2["baselines"]["covariance"] != t2["forecast"]
    assert all(
        provenance["fit_digest"] != fitted.digest
        for provenance in t1["baseline_provenance"].values()
    )


def test_fraction_targets_have_their_own_nested_loro_penalty_selection() -> None:
    from reward_lens.forecast.evidence import ForecastSpec
    from reward_lens.forecast.predictors.hazard_v0 import HazardV0

    fitted = HazardV0.fit(_references(), ForecastSpec.default())
    for target in ("T2", "T3"):
        reports = fitted.fraction_fold_reports[target]
        assert len(reports) == len(_references()) * 5
        assert {report["quantile"] for report in reports} == {0.1, 0.25, 0.5, 0.75, 0.9}
        assert all(report["selection_scope"] == "inner-loro" for report in reports)
        assert all("outer_pinball" in report for report in reports)


def test_rowwise_split_is_documented_as_an_inflated_control() -> None:
    from reward_lens.forecast.evidence import ForecastSpec
    from reward_lens.forecast.predictors.hazard_v0 import rowwise_inflation_control

    report = rowwise_inflation_control(_references(), ForecastSpec.default(), seed=19)
    assert report["split"] == "row-wise-negative-control"
    assert report["rowwise_accuracy"] > report["leave_one_run_out_accuracy"]
    assert report["inflation"] > 0.0


def test_fitted_predictor_drives_actual_issue_resolve_and_ledger(tmp_path: Path) -> None:
    from reward_lens.forecast.evidence import ForecastSpec
    from reward_lens.forecast.predictors.hazard_v0 import HazardV0
    from reward_lens.product.forecast import issue, render_ledger, resolve

    spec = ForecastSpec.default()
    fitted = HazardV0.fit(_references(), spec)
    run = _run("prospective-controlled", onset=5, offset=0.03)
    rows = [
        fitted.issue_row(run, cutoff=3, target=target, source_name="controlled-conformance")
        for target in ("T1", "T2", "T3")
    ]
    assert all(row["model_digest"] == fitted.digest for row in rows)
    assert all(row["feature_cutoff"] == 3 for row in rows)
    record = tmp_path / "predictor-record.json"
    record.write_text(json.dumps({"issues": rows}), encoding="utf-8")
    project = SimpleNamespace(root=tmp_path)
    issue(
        SimpleNamespace(record=record, claim="conformance only", offline=True, seed=7),
        project=project,
    )
    path = tmp_path / ".reward-lens" / "forecast-ledger.jsonl"
    issued = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    predictor_rows = [row for row in issued if row["kind"] == "issue"]
    from reward_lens.forecast.evidence import IssuedForecast

    sealed = [
        IssuedForecast.from_issued_bytes(row["issued_bytes"].encode("utf-8"))
        for row in predictor_rows
    ]
    assert {row.model_digest for row in sealed} == {fitted.digest}
    for row, outcome in zip(predictor_rows, ("true", "0.40", "0.28"), strict=True):
        resolve(
            SimpleNamespace(
                forecast_id=row["sealed_digest"], outcome=outcome, offline=True, seed=7
            ),
            project=project,
        )
    rendered = render_ledger(project)
    assert "controlled-conformance" in rendered
    assert "persistence" in rendered and "climatology" in rendered
    assert "campaign | - | - | T1 | - | 0.26" in rendered
