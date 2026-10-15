"""Controls for the record-backed reward-statistics panel."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from statistics import stdev
from typing import Any

import pytest

from reward_lens import contracts
from reward_lens.instruments.base import Panel, RunContext
from reward_lens.instruments.reward_statistics import (
    PANEL,
    LatencyPercentiles,
    RewardPrice,
    RewardSemantics,
    findings_for,
)
from reward_lens.instruments.reward_statistics._recorded import (
    PriceSource,
    RecordedCompletion,
    TraceReview,
    calls,
)
from reward_lens.product.audit import registry
from reward_lens.product.trace import (
    DeclaredReward,
    TraceMetadata,
    TraceRecord,
    TraceRow,
)
from reward_lens.product.trace import (
    PriceSource as TracePriceSource,
)
from reward_lens.record import GraderCallRef, Leaf, Piecewise, WeightedSum

run_module = import_module("reward_lens.product.audit.run")
_UNSET = object()


@dataclass
class Subject:
    grader_path: Path
    source: str = "def score(): return 1.0\n"
    entrypoint: str = "score"
    project_dir: Path | None = None
    tasks_path: Path | None = None
    responses_path: Path | None = None
    trainer: str | None = None
    name: str = "recorded-reward"


@dataclass
class OrdinaryCorpus:
    rollouts: tuple[Any, ...] = ()
    origin: str = "ordinary response bank"
    tasks: tuple[dict[str, Any], ...] = ()
    n_tasks: int = 0

    def as_list(self) -> list[Any]:
        return list(self.rollouts)

    def __len__(self) -> int:
        return len(self.rollouts)


@dataclass
class TraceCorpus:
    """The public consumer fixture: Audit supplies only its canonical trace field."""

    trace: TraceRecord | None


def ctx() -> RunContext:
    return RunContext(offline=True)


def subject(tmp_path: Path) -> Subject:
    return Subject(grader_path=tmp_path / "grader.py")


def tree(index: int) -> Piecewise:
    """A skewed, unequal-call fixture: odd completions have a third recorded call."""
    leaves = [
        Leaf(
            "quality",
            0.9,
            GraderCallRef("quality", latency_s=0.05 + index * 0.01, cost_usd=0.01 + index * 0.001),
        ),
        Leaf(
            "safety",
            None,
            GraderCallRef(
                "safety",
                outcome="raised",
                latency_s=0.2 + index * 0.02,
                cost_usd=0.02 + index * 0.001,
            ),
            abstained=True,
        ),
    ]
    weights: list[float] = [0.7, 0.3]
    if index % 2:
        leaves.append(
            Leaf(
                "style",
                0.4,
                GraderCallRef("style", latency_s=3.0 + index, cost_usd=0.03 + index * 0.001),
            )
        )
        weights.append(0.1)
    return Piecewise(
        "clipped_total",
        WeightedSum("total", tuple(leaves), weights=tuple(weights)),
        knots=((-1.0, -1.0), (1.0, 1.0)),
        extrapolate="clamp",
    )


def review(
    n: int = 20, *, declared: dict[str, Any] | None = None, price_source: PriceSource | None = None
) -> TraceReview:
    source = price_source or PriceSource(
        "fixture hosted scorer price card",
        "2026-09-26",
        "https://example.invalid/pricing",
        "fixture-v1",
    )
    return TraceReview(
        tuple(
            RecordedCompletion(f"c-{index}", tree(index), accepted=index % 2 == 0)
            for index in range(n)
        ),
        declared_reward=declared,
        price_source=source,
    )


def trace_record(
    n: int = 20,
    *,
    price_source: Any = _UNSET,
    acceptance: Any = _UNSET,
) -> TraceRecord:
    source = (
        TracePriceSource(
            "fixture hosted scorer price card",
            "2026-09-26",
            "https://example.invalid/pricing",
            "fixture-v1",
        )
        if price_source is _UNSET
        else price_source
    )
    rows = tuple(
        TraceRow(
            index=index,
            step=index,
            group_id=f"g-{index // 2}",
            row_id=f"trace-{index}",
            score_tree=tree(index),
            accepted=(index % 2 == 0 if acceptance is _UNSET else acceptance),
        )
        for index in range(n)
    )
    return TraceRecord(
        reader="P-STATS fixture",
        source="fixture://canonical-trace",
        run_id="fixture-trace-run",
        rows=rows,
        series={},
        components={},
        access_level="fixture",
        access_statement="offline canonical trace fixture",
        policy="strict",
        complete=True,
        completeness_statement="fixture complete",
        subject_ref="sha256:fixture",
        depends_on=("digest:fixture",),
        metadata=TraceMetadata(
            declared_reward=DeclaredReward(
                partition=("quality", "safety", "style"), weights={"quality": 0.7, "safety": 0.3}
            ),
            price_source=source,
        ),
    )


def by_id(entries: list[contracts.Entry]) -> dict[str, contracts.Entry]:
    return {entry.entry_id: entry for entry in entries}


def test_panel_is_public_and_discovered() -> None:
    assert isinstance(PANEL, Panel)
    assert PANEL.section == "reward_statistics"
    assert [item.id for item in PANEL.instruments] == [
        "reward_statistics.semantics",
        "reward_statistics.price",
        "reward_statistics.latency",
    ]
    assert "reward_statistics" in [panel.section for panel in registry.discover()]


def test_ordinary_public_corpus_names_the_absent_canonical_trace(tmp_path: Path) -> None:
    ordinary = OrdinaryCorpus()
    for instrument in PANEL.instruments:
        entry = instrument.run(subject(tmp_path), ordinary, ctx())[0]
        assert entry.kind == "absence"
        assert entry.absence.missing_access == "corpus.trace is absent"


def test_public_run_converts_the_canonical_trace_record_into_panel_inputs(tmp_path: Path) -> None:
    corpus = TraceCorpus(trace_record())
    semantics = RewardSemantics().run(subject(tmp_path), corpus, ctx())[0]
    prices = by_id(RewardPrice().run(subject(tmp_path), corpus, ctx()))
    latencies = by_id(LatencyPercentiles().run(subject(tmp_path), corpus, ctx()))
    assert semantics.kind == "estimate"
    assert semantics.result["declared"]["partition"] == ["quality", "safety", "style"]
    assert semantics.n == 20
    assert prices["reward_statistics.price.per_call"].kind == "estimate"
    assert prices["reward_statistics.price.per_accepted_completion"].n == 10
    assert (
        prices["reward_statistics.price.per_call"].result["price_source"]["retrieved"]
        == "2026-09-26"
    )
    assert latencies["reward_statistics.latency.p95"].kind == "estimate"


def test_public_run_names_missing_canonical_price_and_acceptance_fields(tmp_path: Path) -> None:
    no_price = RewardPrice().run(
        subject(tmp_path), TraceCorpus(trace_record(price_source=None)), ctx()
    )
    assert len(no_price) == 1
    assert no_price[0].kind == "absence"
    assert no_price[0].absence.missing_access == "corpus.trace.metadata.price_source is absent"

    unknown_acceptance = by_id(
        RewardPrice().run(subject(tmp_path), TraceCorpus(trace_record(acceptance=None)), ctx())
    )
    assert unknown_acceptance["reward_statistics.price.per_call"].kind == "estimate"
    accepted = unknown_acceptance["reward_statistics.price.per_accepted_completion"]
    assert accepted.kind == "absence"
    assert "trace-0" in accepted.absence.missing_access
    assert "no recorded acceptance" in accepted.absence.missing_access


def test_semantics_read_every_field_from_typed_trace_review_and_keep_weight_disagreement(
    tmp_path: Path,
) -> None:
    entry = RewardSemantics().analyze(subject(tmp_path), review(), ctx())
    result = entry.result
    assert result["components"] == ["quality", "safety", "style"]
    assert len(result["weight_configurations"]) == 2
    assert {row["weights"]["quality"] for row in result["weight_configurations"]} == {0.7}
    assert {len(row["completion_ids"]) for row in result["weight_configurations"]} == {10}
    assert all(
        row["clipping"][0]["extrapolate"] == "clamp" for row in result["weight_configurations"]
    )
    assert (
        result["missing_value_semantics"] == "abstained scores are missing measurements, not zero"
    )
    assert entry.n == 20
    assert entry.sampling_unit == "recorded completion"
    assert entry.uncertainty.method == "cluster_bootstrap"


def test_current_audit_harvest_collects_rl0251_from_the_package(tmp_path: Path) -> None:
    entry = RewardSemantics().analyze(
        subject(tmp_path), review(declared={"weights": {"quality": 1.0}}), ctx()
    )
    direct = findings_for(entry)
    harvested = run_module._panel_findings(RewardSemantics(), [entry])
    assert len(direct) == len(harvested) == 1
    finding = harvested[0]
    assert finding.code == "RL0251"
    assert finding.entries == [entry.entry_id]
    assert "declared" in finding.message and "observed" in finding.message


def test_price_units_denominators_n_and_sampling_units_match_what_is_displayed(
    tmp_path: Path,
) -> None:
    entries = by_id(RewardPrice().analyze(subject(tmp_path), review(), ctx()))
    assert set(entries) == {
        "reward_statistics.price.per_call",
        "reward_statistics.price.per_accepted_completion",
        "reward_statistics.price.per_thousand_completions",
    }
    call, accepted, thousand = (
        entries["reward_statistics.price.per_call"],
        entries["reward_statistics.price.per_accepted_completion"],
        entries["reward_statistics.price.per_thousand_completions"],
    )
    assert (call.n, call.sampling_unit, call.result["denominator"]) == (
        50,
        "recorded grader call",
        "50 recorded grader calls",
    )
    assert (accepted.n, accepted.sampling_unit, accepted.result["denominator"]) == (
        10,
        "accepted recorded completion",
        "10 accepted completion(s); rejected completion costs remain in the numerator",
    )
    assert (thousand.n, thousand.sampling_unit, thousand.result["denominator"]) == (
        20,
        "recorded completion",
        "20 recorded completions",
    )
    assert accepted.uncertainty.cluster_unit == "recorded completion, stratified by accepted status"
    assert accepted.uncertainty.interval[0] > 0
    assert (
        accepted.result["interval_sampling_design"]
        == "completion-cluster bootstrap stratified by observed accepted status; accepted and rejected cluster counts are fixed, so the interval conditions on the observed acceptance allocation"
    )
    for entry in entries.values():
        assert entry.result["price_source"]["name"] == "fixture hosted scorer price card"
        assert entry.result["price_source"]["retrieved"] == "2026-09-26"
        assert entry.uncertainty.method == "cluster_bootstrap"
        assert entry.uncertainty.resamples == 2000
        assert entry.uncertainty.interval[0] <= entry.value <= entry.uncertainty.interval[1]
    assert (
        call.uncertainty.cluster_unit == thousand.uncertainty.cluster_unit == "recorded completion"
    )


def test_rare_acceptance_uses_a_disclosed_conditional_ratio_bootstrap(tmp_path: Path) -> None:
    base = review(15)
    rare = TraceReview(
        tuple(
            RecordedCompletion(completion.id, completion.scores, accepted=index == 0)
            for index, completion in enumerate(base.completions)
        ),
        price_source=base.price_source,
    )
    entry = by_id(RewardPrice().analyze(subject(tmp_path), rare, ctx()))[
        "reward_statistics.price.per_accepted_completion"
    ]
    total_cost = sum(
        call.cost_usd
        for completion in rare.completions
        for call in calls(completion)
        if call.cost_usd is not None
    )
    assert entry.value == pytest.approx(total_cost)
    assert (entry.n, entry.sampling_unit) == (1, "accepted recorded completion")
    assert entry.uncertainty.method == "cluster_bootstrap"
    assert entry.uncertainty.cluster_unit == "recorded completion, stratified by accepted status"
    assert entry.uncertainty.interval[0] > 0
    assert (
        "conditions on the observed acceptance allocation"
        in entry.result["interval_sampling_design"]
    )


def test_undated_price_source_is_refused_and_missing_actual_cost_is_an_absence(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="retrieved"):
        PriceSource("undated", "")
    without_source = TraceReview(review(1).completions)
    source_absence = RewardPrice().analyze(subject(tmp_path), without_source, ctx())[0]
    assert source_absence.kind == "absence"
    assert "price_source" in source_absence.absence.missing_access
    missing = TraceReview(
        (
            RecordedCompletion(
                "no-cost", Leaf("quality", 1.0, GraderCallRef("quality", cost_usd=None)), True
            ),
        ),
        price_source=PriceSource("dated", "2026-09-26"),
    )
    entry = RewardPrice().analyze(subject(tmp_path), missing, ctx())[0]
    assert entry.kind == "absence"
    assert "no recorded cost_usd" in entry.absence.missing_access


def test_latency_percentile_uses_a_quantile_bootstrap_not_a_mean_standard_error(
    tmp_path: Path,
) -> None:
    entries = by_id(LatencyPercentiles().analyze(subject(tmp_path), review(), ctx()))
    p95 = entries["reward_statistics.latency.p95"]
    assert p95.n == 50
    assert p95.sampling_unit == "recorded grader call"
    assert p95.uncertainty.method == "cluster_bootstrap"
    assert p95.uncertainty.cluster_unit == "recorded completion"
    assert p95.uncertainty.interval[0] <= p95.value <= p95.uncertainty.interval[1]
    latencies = (
        [0.05 + i * 0.01 for i in range(20)]
        + [0.2 + i * 0.02 for i in range(20)]
        + [3.0 + i for i in range(20) if i % 2]
    )
    mean_se = stdev(latencies) / len(latencies) ** 0.5
    mean_interval = [round(p95.value - 1.96 * mean_se, 6), round(p95.value + 1.96 * mean_se, 6)]
    assert p95.uncertainty.interval != mean_interval


def test_small_completion_count_names_the_no_interval_rule(tmp_path: Path) -> None:
    priced = RewardPrice().analyze(subject(tmp_path), review(2), ctx())
    latencies = LatencyPercentiles().analyze(subject(tmp_path), review(2), ctx())
    assert all(entry.uncertainty.method == "no_interval_below_15_clusters" for entry in priced)
    assert all(entry.uncertainty.method == "no_interval_below_15_clusters" for entry in latencies)


def test_an_instrument_exception_is_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failing = RewardSemantics()
    monkeypatch.setattr(
        failing, "analyze", lambda *_: (_ for _ in ()).throw(RuntimeError("broken trace"))
    )
    with pytest.raises(RuntimeError, match="broken trace"):
        failing.analyze(subject(tmp_path), review(), ctx())
