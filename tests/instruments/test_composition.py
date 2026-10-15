"""Composition keeps observed loss evidence separate from gradients and updates."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from reward_lens.instruments import composition
from reward_lens.instruments.base import Instrument, Panel, RunContext
from reward_lens.product.audit import registry
from reward_lens.product.trace import open_trace
from reward_lens.product.trace.record import Series, TraceMetadata, TraceRecord, TraceRow
from reward_lens.record.schema import EstimatorSpec


@dataclass
class Subject:
    source: str = "def score(): return 1"
    grader_path: Path = Path("grader.py")
    entrypoint: str = "score"
    project_dir: Path | None = None
    tasks_path: Path | None = None
    responses_path: Path | None = None
    trainer: str | None = "trl"
    name: str = "composition-fixture"


@dataclass
class Corpus:
    composition_record: Any | None = None
    trace: TraceRecord | None = None
    rollouts: tuple[Any, ...] = ()
    tasks: tuple[dict[str, Any], ...] = ()
    origin: str = "composition fixture"
    n_tasks: int = 0

    def as_list(self) -> list[Any]:
        return list(self.rollouts)

    def __len__(self) -> int:
        return len(self.rollouts)


def context() -> RunContext:
    return RunContext(offline=True)


def trace(
    *, recorded=(1.0, -1.0), applied=(2.0, 0.0), metadata: TraceMetadata | None = None
) -> TraceRecord:
    rows = tuple(
        TraceRow(
            index=index,
            step=0,
            group_id="g-1",
            row_id=f"row-{index}",
            components={"correct": value},
            reward_applied=value,
            advantage=recorded[index],
        )
        for index, value in enumerate(applied)
    )
    return TraceRecord(
        reader="fixture",
        source="fixture",
        run_id="fixture-run",
        rows=rows,
        series={
            "reward_recorded": Series(name="reward_recorded", values=applied),
            "reward_applied": Series(
                name="reward_applied", values=applied, weights=(1.0,), components=("correct",)
            ),
            "advantage": Series(name="advantage", values=recorded),
            "selection": Series(name="selection", present=False),
        },
        components={"correct": Series(name="correct", values=applied)},
        access_level="trainer_log",
        access_statement="fixture trainer log",
        policy="strict",
        complete=True,
        completeness_statement="fixture complete",
        subject_ref="sha256:" + "1" * 64,
        depends_on=("digest:training_semantics",),
        metadata=metadata or TraceMetadata(),
    )


def record(*, telemetry: composition.CompositionTelemetry | None = None, **overrides: Any) -> Any:
    tokens = (
        composition.ScalarLossToken("row-0", 1.0, 0.0, 0.0, 1.0, 1.0, 0.2, 0.1),
        composition.ScalarLossToken("row-1", 1.0, 0.0, 0.0, 1.0, -1.0, 0.2, 0.0),
    )
    telemetry = telemetry or composition.CompositionTelemetry(
        {
            0: composition.ScalarLossStep(
                trainer_profile="grpo_ppo_token_mean_kl_v1",
                aggregation_denominator=2.0,
                minimum_denominator=1e-8,
                clip_low=0.8,
                clip_high=1.2,
                kl_coefficient=0.5,
                observed_loss=0.15,
                tokens=tokens,
            )
        }
    )
    fields: dict[str, Any] = {
        "trace": trace(
            metadata=TraceMetadata(
                composition=telemetry,
                applied_weights={"row-0": -0.075, "row-1": -0.075},
            )
        ),
        "estimators": {
            "g-1": EstimatorSpec(
                family="grpo",
                group_centred=True,
                std_normalised=False,
                degenerate_policy="zero",
            )
        },
    }
    fields.update(overrides)
    return composition.CompositionRecord(**fields)


def test_panel_is_discoverable_and_has_a_real_instrument() -> None:
    assert isinstance(composition.PANEL, Panel)
    assert composition.PANEL.section == "trace"
    assert len(composition.PANEL.instruments) == 1
    assert isinstance(composition.PANEL.instruments[0], Instrument)
    assert "trace" in [panel.section for panel in registry.discover()]


def test_advantages_replay_declared_grouping_and_normalisation_as_numeric_agreement() -> None:
    agreement = composition.recompute_advantages(record())
    assert agreement.observed is True
    assert agreement.agreement == 1.0
    assert agreement.reconstructed == (1.0, -1.0)
    assert agreement.recorded == (1.0, -1.0)


def test_advantage_disagreement_keeps_both_numbers_and_is_not_adjusted() -> None:
    checked = record(trace=trace(recorded=(0.0, 0.0)))
    agreement = composition.recompute_advantages(checked)
    assert agreement.agreement == 0.0
    assert agreement.reconstructed == (1.0, -1.0)
    assert agreement.recorded == (0.0, 0.0)


def test_price_arithmetic_delegates_to_the_existing_ledger_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def supplied(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return "price-result"

    monkeypatch.setattr(composition.advantages, "selection_differential", supplied)
    assert composition.price_differential(record()) == "price-result"
    assert calls


def test_applied_weight_identity_passes_and_failure_carries_both_disagreeing_numbers() -> None:
    passed = composition.check_applied_weight_identity(record())[0]
    assert passed.available is True
    assert passed.passed is True
    failed = composition.check_applied_weight_identity(record(losses={0: 0.9}))[0]
    assert failed.available is True
    assert failed.passed is False
    assert failed.sum_applied_weight == -0.15
    assert failed.negative_loss == -0.9


def test_identity_failure_emits_rl0261_from_an_instrument_entry() -> None:
    entries = composition.CompositionInstrument().run(
        Subject(), Corpus(record(losses={0: 0.9})), context()
    )
    identity = next(
        entry for entry in entries if entry.entry_id == "trace.composition.applied_weight_identity"
    )
    assert identity.kind == "check"
    assert identity.check.passed is False
    findings = composition.findings_for(identity)
    assert len(findings) == 1
    assert findings[0].code == "RL0261"
    assert identity.result["sum_applied_weight"] == -0.15
    assert identity.result["negative_loss"] == -0.9


def test_missing_numerical_loss_data_only_refuses_scalar_level_and_zero_denominator_is_never_reported() -> (
    None
):
    incomplete = record(telemetry=composition.CompositionTelemetry({}))
    entries = composition.CompositionInstrument().run(Subject(), Corpus(incomplete), context())
    scalar = next(entry for entry in entries if entry.entry_id == "trace.composition.scalar_loss")
    assert scalar.kind == "absence"
    assert scalar.absence.missing_access == "no numerical scalar-loss telemetry was recorded"

    zero = record(
        telemetry=composition.CompositionTelemetry(
            {0: replace(record().trace.metadata.composition.steps[0], aggregation_denominator=0.0)}
        )
    )
    zero_entries = composition.CompositionInstrument().run(Subject(), Corpus(zero), context())
    zero_scalar = next(
        entry for entry in zero_entries if entry.entry_id == "trace.composition.scalar_loss"
    )
    assert "denominator is non-positive or unstable" in zero_scalar.absence.missing_access


def test_controls_keep_loss_gradient_and_optimizer_transform_distinct() -> None:
    loss_control = composition.equal_loss_different_gradient()
    optimizer_control = composition.equal_gradient_different_optimizer()
    assert loss_control.same_loss is True
    assert loss_control.left_loss == loss_control.right_loss == 1.0
    assert loss_control.left_gradient == 2.0
    assert loss_control.right_gradient == -2.0
    assert loss_control.measured_training_gradient is False
    assert loss_control.update_equivalence_claimed is False
    assert optimizer_control.same_gradient is True
    assert optimizer_control.transforms_differ is True
    assert optimizer_control.left_control_delta != optimizer_control.right_control_delta
    assert optimizer_control.measured_training_update is False
    assert optimizer_control.update_equivalence_claimed is False


def test_scalar_loss_reconstruction_records_numeric_terms_and_signed_shares() -> None:
    canonical_trace = record().trace
    entries = composition.CompositionInstrument().run(
        Subject(), Corpus(trace=canonical_trace), context()
    )
    scalar = next(entry for entry in entries if entry.entry_id == "trace.composition.scalar_loss")
    assert scalar.kind == "check"
    assert scalar.check.passed is True
    step = scalar.result["steps"][0]
    assert step["reconstructed_loss"] == pytest.approx(0.15)
    assert step["observed_loss"] == pytest.approx(0.15)
    assert step["aggregation_denominator"] == 2.0
    assert step["row_signed_negative_loss_contributions"] == {
        "row-0": pytest.approx(0.4),
        "row-1": pytest.approx(-0.55),
    }
    assert sum(step["row_shares"].values()) == pytest.approx(1.0)


def test_scalar_loss_mismatch_retains_both_numbers_and_missing_observation_is_scoped_absence() -> (
    None
):
    step = record().trace.metadata.composition.steps[0]
    mismatched = record(
        telemetry=composition.CompositionTelemetry({0: replace(step, observed_loss=0.2)})
    )
    entries = composition.CompositionInstrument().run(Subject(), Corpus(mismatched), context())
    scalar = next(entry for entry in entries if entry.entry_id == "trace.composition.scalar_loss")
    assert scalar.kind == "check"
    assert scalar.check.passed is False
    result = scalar.result["steps"][0]
    assert result["reconstructed_loss"] == pytest.approx(0.15)
    assert result["observed_loss"] == pytest.approx(0.2)

    missing = record(
        telemetry=composition.CompositionTelemetry({0: replace(step, observed_loss=None)})
    )
    absent_entries = composition.CompositionInstrument().run(Subject(), Corpus(missing), context())
    absent = next(
        entry for entry in absent_entries if entry.entry_id == "trace.composition.scalar_loss"
    )
    assert absent.kind == "absence"
    assert "did not record the scalar loss" in absent.absence.missing_access


def test_each_material_loss_input_changes_or_refuses_the_scalar_reconstruction() -> None:
    baseline = composition.reconstruct_scalar_loss(record()).steps[0].reconstructed_loss
    step = record().trace.metadata.composition.steps[0]
    first, second = step.tokens
    variants = (
        replace(step, tokens=(replace(first, mask=0.5), second)),
        replace(
            step,
            tokens=(replace(first, log_probability=math.log(1.5), importance_ratio=1.5), second),
        ),
        replace(
            step,
            clip_high=1.4,
            tokens=(replace(first, log_probability=math.log(1.5), importance_ratio=1.5), second),
        ),
        replace(step, aggregation_denominator=4.0),
        replace(step, kl_coefficient=0.75),
        replace(step, tokens=(replace(first, auxiliary_term=0.3), second)),
    )
    for variant in variants:
        rebuilt = composition.reconstruct_scalar_loss(
            record(telemetry=composition.CompositionTelemetry({0: variant}))
        ).steps[0]
        assert rebuilt.available is True
        assert rebuilt.reconstructed_loss != pytest.approx(baseline)

    inconsistent_ratio = replace(step, tokens=(replace(first, importance_ratio=1.5), second))
    refused = composition.reconstruct_scalar_loss(
        record(telemetry=composition.CompositionTelemetry({0: inconsistent_ratio}))
    ).steps[0]
    assert refused.available is False
    assert "log probabilities" in refused.reason

    inconsistent_advantage = replace(step, tokens=(replace(first, advantage=0.5), second))
    refused_advantage = composition.reconstruct_scalar_loss(
        record(telemetry=composition.CompositionTelemetry({0: inconsistent_advantage}))
    ).steps[0]
    assert refused_advantage.available is False
    assert "recorded trace advantage" in refused_advantage.reason


def test_instrument_without_trace_join_is_an_honest_absence() -> None:
    entries = composition.CompositionInstrument().run(Subject(), Corpus(), context())
    assert len(entries) == 1
    assert entries[0].kind == "absence"
    assert entries[0].entry_id == "trace.composition.trace_join"


def test_discovered_public_panel_refuses_an_actual_trace_without_the_missing_loss_join() -> None:
    panel = next(panel for panel in registry.discover() if panel is composition.PANEL)
    actual_trace = open_trace(Path("tests/fixtures/grpo_run/short"))
    entries = panel.instruments[0].run(Subject(), Corpus(trace=actual_trace), context())
    advantages = next(
        entry for entry in entries if entry.entry_id == "trace.composition.advantages"
    )
    scalar = next(entry for entry in entries if entry.entry_id == "trace.composition.scalar_loss")
    assert advantages.kind == "absence"
    assert "recorded estimator" in advantages.absence.missing_access
    assert scalar.absence.missing_access == "no numerical scalar-loss telemetry was recorded"
