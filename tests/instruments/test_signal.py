from __future__ import annotations

import pytest

from reward_lens import contracts
from reward_lens.instruments.base import RunContext
from reward_lens.instruments.signal import FINDING_CODES, PANEL, SignalInstrument, SignalObservation
from reward_lens.instruments.signal.bootstrap import uncertainty_for_tasks
from reward_lens.instruments.signal.clustering import effective_sample_size
from reward_lens.instruments.signal.contrast import (
    component_disagreement,
    contrast_fraction,
    style_share,
)
from reward_lens.instruments.signal.null_result import null_result
from reward_lens.product.audit.registry import discover
from reward_lens.product.audit.response_scores import ResponseScore
from reward_lens.product.trace import (
    DeclaredReward,
    PlannedContrastDesign,
    StepMetric,
    TraceMetadata,
    TraceRecord,
    TraceRow,
)


def _observations(n_tasks: int = 15) -> tuple[SignalObservation, ...]:
    return tuple(
        SignalObservation(
            task_id=f"task-{task}",
            group_id=f"group-{task}",
            reward_applied=float(position if task % 2 else 1),
            components={"style": 1.0 if task % 2 else -1.0, "task": 1.0},
        )
        for task in range(n_tasks)
        for position in range(2)
    )


def _trace(
    rows: tuple[TraceRow, ...],
    *,
    metrics: tuple[StepMetric, ...] = (),
    planned_contrast: PlannedContrastDesign | None = None,
    partition: tuple[str, ...] | None = ("style", "task"),
) -> TraceRecord:
    return TraceRecord(
        reader="fixture",
        source="fixture://signal",
        run_id="signal-fixture",
        rows=rows,
        series={},
        components={},
        access_level="L0",
        access_statement="fixture",
        policy="strict",
        complete=True,
        completeness_statement="fixture",
        subject_ref=contracts.digest({"fixture": "signal"}),
        depends_on=("digest:training_semantics", "digest:samples"),
        metadata=TraceMetadata(
            declared_reward=None if partition is None else DeclaredReward(partition=partition),
            metrics=metrics,
            planned_contrast=planned_contrast,
        ),
    )


def _trace_rows(*, step: int, contrasting: bool, prefix: str) -> tuple[TraceRow, ...]:
    return tuple(
        TraceRow(
            index=2 * task + position,
            step=step,
            group_id=f"{prefix}-group-{task}",
            row_id=f"{prefix}-{task}-{position}",
            task_id=f"{prefix}-task-{task}",
            reward_applied=float(position) if contrasting else 1.0,
            components={"style": float((-1) ** position), "task": 1.0},
        )
        for task in range(15)
        for position in range(2)
    )


def test_panel_is_public_and_uses_the_frozen_instrument_route() -> None:
    assert PANEL.section == "signal"
    assert len(PANEL.instruments) == 1
    assert isinstance(PANEL.instruments[0], SignalInstrument)
    assert any(panel.section == "signal" for panel in discover())


def test_signal_finding_codes_are_allocated_only_to_the_signal_band() -> None:
    assert set(FINDING_CODES) == {"RL0241", "RL0242", "RL0243", "RL0244", "RL0245"}


def test_unequal_group_anova_icc_uses_m0_and_ma_not_mean_group_size() -> None:
    rows = (
        SignalObservation("a", "g1", 0.0),
        SignalObservation("a", "g1", 1.0),
        SignalObservation("b", "g2", 0.0),
        SignalObservation("b", "g2", 0.0),
        SignalObservation("b", "g2", 1.0),
        SignalObservation("b", "g2", 1.0),
        SignalObservation("b", "g2", 1.0),
    )
    result = effective_sample_size(rows)
    assert result.m_a == pytest.approx(29 / 7)
    assert result.m_0 == pytest.approx(20 / 7)
    assert result.design_effect == pytest.approx(1 + (result.m_a - 1) * result.icc)
    assert result.n_eff == pytest.approx(7 / result.design_effect)
    assert result.mean_group_size != result.m_a


def test_singleton_clusters_do_not_invent_an_icc_or_effective_n() -> None:
    rows = tuple(SignalObservation(f"t{i}", f"g{i}", float(i % 2)) for i in range(16))
    result = effective_sample_size(rows)
    assert result.identifiable is False
    assert result.icc is None and result.n_eff is None
    entries = SignalInstrument().entries_for(rows, RunContext())
    assert next(entry for entry in entries if entry.entry_id == "signal.n_eff").kind == "absence"


def test_icc_carries_a_task_cluster_interval_when_identified() -> None:
    entry = SignalInstrument().contrast_entry(_observations(15), RunContext())
    assert entry.uncertainty.icc_interval is not None
    assert entry.result["icc_interval_method"] == "task_cluster_pairs_bootstrap"


def test_clustered_metrics_use_task_as_the_sampling_unit_and_n_eff_is_not_outcome_n() -> None:
    rows = _observations(40)
    entries = SignalInstrument().entries_for(rows, RunContext())
    assert {entry.entry_id for entry in entries} == {
        "signal.contrast_fraction",
        "signal.n_eff",
        "signal.style_share",
        "signal.component_disagreement",
    }
    for entry in entries:
        assert entry.kind == "estimate"
        assert entry.unit
        assert entry.n is not None
        assert entry.sampling_unit
        assert entry.uncertainty is not None
        assert entry.method.id
    contrast = next(entry for entry in entries if entry.entry_id == "signal.contrast_fraction")
    assert contrast.sampling_unit == "task"
    assert contrast.n == 40
    assert contrast.uncertainty.n_eff is not None
    assert contrast.n != round(contrast.uncertainty.n_eff)


def test_contrast_style_and_disagreement_have_defined_recorded_semantics() -> None:
    rows = (
        SignalObservation("t1", "g1", 0.0, {"style": 2.0, "task": 1.0}),
        SignalObservation("t1", "g1", 1.0, {"style": -1.0, "task": 1.0}),
        SignalObservation("t2", "g2", 1.0, {"style": 1.0, "task": 1.0}),
        SignalObservation("t2", "g2", 1.0, {"style": 1.0}),
    )
    assert contrast_fraction(rows).value == pytest.approx(0.5)
    assert style_share(rows).value == pytest.approx(13 / 24)
    disagreement = component_disagreement(rows)
    assert disagreement.value == pytest.approx(1 / 4)
    assert disagreement.missing_components == 1
    assert disagreement.eligible == 2


def test_contrast_is_a_group_fraction_under_unequal_groups_per_task() -> None:
    rows = [SignalObservation("a", "a", 0.0), SignalObservation("a", "a", 1.0)]
    rows += [SignalObservation("b", f"b{i}", 0.0) for i in range(100) for _ in range(2)]
    assert contrast_fraction(rows).value == pytest.approx(1 / 101)


def test_all_zero_style_vectors_are_counted_as_undefined_not_divided_by_zero() -> None:
    metric = style_share([SignalObservation("t", "g", 0.0, {"style": 0.0, "task": 0.0})])
    assert metric.value == 0.0 and metric.zero_total_magnitude == 1


def test_task_bootstrap_at_40_and_wild_cluster_below_40() -> None:
    forty = uncertainty_for_tasks(list(range(40)), lambda xs: sum(xs) / len(xs), seed=1)
    thirty_nine = uncertainty_for_tasks(list(range(39)), lambda xs: sum(xs) / len(xs), seed=1)
    assert forty.method == "cluster_bootstrap"
    assert forty.resamples >= 999
    assert forty.cluster_unit == "task"
    assert thirty_nine.method == "wild_cluster_bootstrap"
    assert thirty_nine.cluster_unit == "task"
    assert thirty_nine.interval is not None


def test_no_interval_below_15_clusters_has_reason_and_no_endpoints() -> None:
    uncertainty = uncertainty_for_tasks(list(range(14)), lambda xs: sum(xs) / len(xs), seed=1)
    assert uncertainty.method == "no_interval_below_15_clusters"
    assert uncertainty.interval is None
    assert uncertainty.reason
    assert uncertainty.clusters == 14
    assert uncertainty.bounded is False


def test_panel_uses_no_interval_for_every_clustered_estimate_below_15_tasks() -> None:
    entries = SignalInstrument().entries_for(_observations(14), RunContext())
    assert all(entry.uncertainty.method == "no_interval_below_15_clusters" for entry in entries)
    assert all(
        entry.uncertainty.interval is None and entry.uncertainty.bounded is False
        for entry in entries
    )


def test_run_reads_canonical_trace_and_reconciles_one_matching_optimizer_step() -> None:
    class Corpus:
        trace = _trace(
            _trace_rows(step=7, contrasting=True, prefix="s"),
            metrics=(
                StepMetric(
                    step=7,
                    name="frac_reward_zero_std",
                    value=0.0,
                    scope="optimizer_step",
                    provenance={"source": "fixture"},
                ),
            ),
        )

    class Subject:
        source = "grader.py"

    contrast = next(
        entry
        for entry in SignalInstrument().run(Subject(), Corpus(), RunContext())
        if entry.entry_id == "signal.contrast_fraction"
    )
    assert contrast.result["trl_metric_scope"] == "optimizer_step"
    assert contrast.result["trl_metric_step"] == 7
    assert contrast.result["trl_frac_reward_zero_std"] == 0.0
    assert contrast.result["trl_reconciliation_difference"] == 0.0


def test_run_keeps_multiple_optimizer_step_reconciliations_scoped() -> None:
    class Corpus:
        trace = _trace(
            _trace_rows(step=3, contrasting=False, prefix="flat")
            + _trace_rows(step=9, contrasting=True, prefix="contrast"),
            metrics=(
                StepMetric(step=3, name="frac_reward_zero_std", value=1.0, scope="optimizer_step"),
                StepMetric(step=9, name="frac_reward_zero_std", value=0.0, scope="optimizer_step"),
            ),
        )

    contrast = next(
        entry
        for entry in SignalInstrument().run(object(), Corpus(), RunContext())
        if entry.entry_id == "signal.contrast_fraction"
    )
    assert contrast.result["trl_reconciliation"] == "matched_optimizer_steps"
    assert contrast.result["trl_reconciliation_by_step"] == {
        "3": {
            "trl_frac_reward_zero_std": 1.0,
            "computed_frac_reward_zero_std": 1.0,
            "trl_reconciliation_difference": 0.0,
        },
        "9": {
            "trl_frac_reward_zero_std": 0.0,
            "computed_frac_reward_zero_std": 0.0,
            "trl_reconciliation_difference": 0.0,
        },
    }


def test_run_refuses_ordinary_rollouts_without_a_canonical_trace() -> None:
    class Corpus:
        rollouts = _observations(15)

    entry = SignalInstrument().run(object(), Corpus(), RunContext())[0]
    assert entry.kind == "absence"
    assert "no immutable TraceRecord" in entry.absence.missing_access
    assert entry.method == SignalInstrument().method


def test_run_uses_executed_response_bank_scores_only_for_scoped_score_contrast() -> None:
    class Corpus:
        trace = None
        response_scores = tuple(
            ResponseScore(f"{task}-{score}", str(task), float(score), "measured")
            for task in range(15)
            for score in (0, 1)
        )

    entries = SignalInstrument().run(object(), Corpus(), RunContext())
    assert [entry.entry_id for entry in entries] == [
        "signal.response_score_contrast_fraction",
        "signal.response_score_n_eff",
    ]
    contrast = entries[0]
    assert contrast.value == 1.0
    assert contrast.method.id == "signal.response_bank_score"
    assert contrast.result["input_contract"]["grouping"] == "response.task_id, never row adjacency"
    assert "not a recorded applied training reward" in contrast.limitations[0]
    assert all(
        "style" not in entry.entry_id and "component" not in entry.entry_id for entry in entries
    )


def test_response_bank_revalidates_mutated_ess_values_at_declared_record_scales() -> None:
    scores = tuple(
        ResponseScore(
            response_id=f"p{task}-{position}",
            task_id=f"p{task}",
            score=0.0 if task == 0 and position == 1 else float(task % 2 == 0),
            status="measured",
            reason=None,
        )
        for task in range(15)
        for position in range(2 + task % 3)
    )

    contrast, n_eff = SignalInstrument().response_bank_entries(
        scores, RunContext(), subject_ref=contracts.digest({"fixture": "nontrivial-ess"})
    )

    raw_design_effect = contrast.result["design_effect"]
    assert raw_design_effect == pytest.approx(3.084159506499229)
    assert contrast.uncertainty.design_effect == contracts.quantise(raw_design_effect, 4)
    assert contrast.uncertainty.icc == contracts.quantise(contrast.result["icc_raw"], 4)
    assert contrast.uncertainty.n_eff == contracts.quantise(contrast.result["n_eff"], 2)
    assert n_eff.uncertainty.design_effect == contrast.uncertainty.design_effect


def test_run_does_not_invent_null_design_metadata_from_flat_trace_rows() -> None:
    class Corpus:
        trace = _trace(_trace_rows(step=0, contrasting=False, prefix="flat"))

    contrast = next(
        entry
        for entry in SignalInstrument().run(object(), Corpus(), RunContext())
        if entry.entry_id == "signal.contrast_fraction"
    )
    assert contrast.value == 0.0
    assert contrast.power is None
    assert contrast.result["planned_contrast_design"] == {
        "status": "not_declared",
        "reason": "the canonical TraceRecord carries no declared planned contrast design",
    }


def test_run_forwards_full_declared_planned_design_without_reusing_observed_counts() -> None:
    design = PlannedContrastDesign(
        standard_error=0.25,
        alpha=0.1,
        power=0.9,
        n=96,
        clusters=48,
        covariance={"kind": "cluster_robust", "unit": "task"},
        design_effect=1.75,
        provenance={"source": "registered protocol", "id": "signal-v1"},
    )

    class Corpus:
        trace = _trace(
            _trace_rows(step=0, contrasting=False, prefix="flat"), planned_contrast=design
        )

    contrast = next(
        entry
        for entry in SignalInstrument().run(object(), Corpus(), RunContext())
        if entry.entry_id == "signal.contrast_fraction"
    )
    assert contrast.value == 0.0
    assert contrast.n == 15
    assert contrast.result["raw_rollout_n"] == 30
    assert contrast.result["task_clusters"] == 15
    assert contrast.power is not None
    assert (
        contrast.power.alpha,
        contrast.power.power,
        contrast.power.n,
        contrast.power.design_effect,
    ) == (0.1, 0.9, 96, 1.75)
    assert contrast.result["planned_contrast_design"] == {
        "status": "declared",
        "standard_error": 0.25,
        "alpha": 0.1,
        "power": 0.9,
        "n": 96,
        "clusters": 48,
        "covariance": {"kind": "cluster_robust", "unit": "task"},
        "design_effect": 1.75,
        "provenance": {"source": "registered protocol", "id": "signal-v1"},
    }
    assert contrast.power.mde == pytest.approx(0.7316, abs=1e-5)


def test_invalid_canonical_planned_design_is_rejected_before_signal_runs() -> None:
    with pytest.raises(ValueError, match="standard_error"):
        PlannedContrastDesign(standard_error=0.0, alpha=0.05, power=0.8, n=96, clusters=48)


def test_run_preserves_undeclared_optional_design_effect_without_substitution() -> None:
    design = PlannedContrastDesign(standard_error=0.2, alpha=0.05, power=0.8, n=64, clusters=32)

    class Corpus:
        trace = _trace(
            _trace_rows(step=0, contrasting=False, prefix="flat"), planned_contrast=design
        )

    contrast = next(
        entry
        for entry in SignalInstrument().run(object(), Corpus(), RunContext())
        if entry.entry_id == "signal.contrast_fraction"
    )
    assert contrast.power is not None
    assert (
        contrast.power.alpha,
        contrast.power.power,
        contrast.power.n,
        contrast.power.design_effect,
    ) == (0.05, 0.8, 64, None)
    assert contrast.power.mde == pytest.approx(0.560317, abs=1e-6)
    assert contrast.result["planned_contrast_design"]["covariance"] is None
    assert contrast.result["planned_contrast_design"]["design_effect"] is None


def test_run_preserves_missing_declared_task_identity_as_an_absence() -> None:
    class Corpus:
        trace = _trace(
            (TraceRow(index=0, step=0, group_id="g", row_id="r", task_id=None, reward_applied=1.0),)
        )

    entry = SignalInstrument().run(object(), Corpus(), RunContext())[0]
    assert entry.kind == "absence"
    assert "task_id" in entry.absence.missing_access


def test_signal_observation_validation_does_not_misuse_a_finding_code() -> None:
    with pytest.raises(ValueError, match="task_id") as caught:
        SignalObservation.from_value({"group_id": "g"})
    assert "RL0241" not in str(caught.value)


def test_trl_zero_std_is_reconciled_as_a_number() -> None:
    rows = (
        SignalObservation("t1", "g1", 1.0),
        SignalObservation("t1", "g1", 1.0),
        SignalObservation("t2", "g2", 0.0),
        SignalObservation("t2", "g2", 1.0),
    )
    entry = SignalInstrument().contrast_entry(rows, RunContext(), trl_frac_reward_zero_std=0.25)
    assert entry.result["trl_frac_reward_zero_std"] == pytest.approx(0.25)
    assert entry.result["trl_reconciliation_difference"] == pytest.approx(0.25)


def test_null_result_carries_design_inputs_and_mde_without_observed_power() -> None:
    result = null_result(
        clustered_standard_error=0.1, n=50, design_effect=2.0, alpha=0.05, power=0.8
    )
    assert result.alpha == pytest.approx(0.05)
    assert result.power == pytest.approx(0.8)
    assert result.n == 50
    assert result.design_effect == pytest.approx(2.0)
    assert result.mde == pytest.approx(0.280159, abs=1e-6)
    assert not hasattr(result, "observed_power")


def test_no_contrast_public_entry_carries_planned_null_metadata() -> None:
    rows = tuple(SignalObservation(f"t{i}", f"g{i}", 1.0) for i in range(15) for _ in range(2))
    entry = SignalInstrument().contrast_entry(rows, RunContext(), design_standard_error=0.1)
    assert entry.power is not None
    assert (entry.power.alpha, entry.power.power, entry.power.n, entry.power.design_effect) == (
        0.05,
        0.8,
        15,
        entry.result["design_effect"],
    )


def test_flat_null_without_registered_design_variance_carries_no_mde() -> None:
    rows = tuple(SignalObservation(f"t{i}", f"g{i}", 1.0) for i in range(15) for _ in range(2))
    assert SignalInstrument().contrast_entry(rows, RunContext()).power is None


def test_zero_style_denominator_is_an_absence_not_zero_estimate() -> None:
    rows = tuple(
        SignalObservation(f"t{i}", f"g{i}", 0.0, {"style": 0.0, "task": 0.0})
        for i in range(15)
        for _ in range(2)
    )
    style = next(
        e
        for e in SignalInstrument().entries_for(rows, RunContext())
        if e.entry_id == "signal.style_share"
    )
    assert style.kind == "absence"
    assert style.method == SignalInstrument().method


def test_unequal_group_ratio_interval_targets_the_reported_point() -> None:
    rows = [SignalObservation("a", "a", 0.0), SignalObservation("a", "a", 1.0)]
    rows += [SignalObservation(f"b{i}", f"b{i}", 0.0) for i in range(14) for _ in range(2)]
    entry = SignalInstrument().contrast_entry(rows, RunContext())
    assert entry.value == pytest.approx(1 / 15, abs=1e-6)
    assert sum(entry.result["n_eff"] is not None for _ in [0]) == 1


def test_instrument_preserves_missing_trl_metadata_as_unreconciled() -> None:
    entry = SignalInstrument().contrast_entry(_observations(15), RunContext())
    assert entry.result["trl_reconciliation"] == "not_available_in_recorded_input"
    assert "trl_frac_reward_zero_std" not in entry.result
