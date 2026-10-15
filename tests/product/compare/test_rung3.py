"""Rung 3 replays the recorded score-to-advantage transform."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from reward_lens.contracts import UsageError
from reward_lens.product.compare.case_evidence import CASE_EXTENSION, build_case_evidence
from reward_lens.product.compare.registry import Rung, get
from reward_lens.product.compare.rung3 import (
    R3_ENTRY_ID,
    ReplayVersion,
    _replay_group,
    advantage_replay,
)
from reward_lens.product.trace import reader_for
from reward_lens.record.schema import EstimatorSpec

REPO = Path(__file__).resolve().parents[3]
REAL_RUN = REPO / "tests" / "fixtures" / "grpo_run" / "short"


@pytest.fixture(scope="module")
def run_record():
    """The checked-in GRPO record, read by the product's actual run-record reader."""
    return reader_for(REAL_RUN).read(REAL_RUN, policy="strict")


def _version(name: str, *, weight: str = "1", offset: str = "0") -> ReplayVersion:
    return ReplayVersion(name, {"length_reward": Decimal(weight)}, Decimal(offset))


def test_replays_both_versions_from_the_real_run_record(run_record) -> None:
    result = advantage_replay(run_record, _version("base"), _version("offset", offset="-3"))

    assert result.scope == "reconstructed_training_pressure"
    assert result.reading.verdict == "unchanged pressure"
    assert result.reading.displayed_score_delta == Decimal("-3")
    assert result.reading.advantage_delta == Decimal("0")
    assert result.reading.advantages_a
    assert result.reading.advantages_b
    assert result.reading.advantages_a.keys() == result.reading.advantages_b.keys()
    assert all(
        abs(result.reading.advantages_a[row_id] - result.reading.advantages_b[row_id])
        <= Decimal("0.000000000001")
        for row_id in result.reading.advantages_a
    )
    assert "reconstruction" in result.establishes
    assert "gradient" in result.does_not_establish
    assert "rank" in result.does_not_establish


def test_candidate_replay_witnesses_bind_measured_local_costs_to_the_decisive_pair(
    run_record,
) -> None:
    result = advantage_replay(run_record, _version("base"), _version("offset", offset="-3"))

    decisive = next(entry for entry in result.entries if entry.entry_id == R3_ENTRY_ID)
    candidates = tuple(entry for entry in result.entries if entry.entry_id != R3_ENTRY_ID)
    assert [entry.result["candidate_id"] for entry in candidates] == ["base", "offset"]
    assert all(entry.kind == "witness" for entry in candidates)
    assert all(entry.provenance.counters.calls == 1 for entry in candidates)
    assert all(entry.provenance.counters.wall_s >= 0 for entry in candidates)
    assert all(entry.provenance.counters.cpu_s >= 0 for entry in candidates)
    assert all(entry.provenance.counters.api_calls == 0 for entry in candidates)
    assert all(entry.provenance.counters.usd == "0.00" for entry in candidates)
    assert all(entry.provenance.budget_usd == "0.00" for entry in candidates)
    assert all(
        any("local replay execution" in limitation for limitation in entry.limitations)
        for entry in candidates
    )

    case = build_case_evidence(decisive, candidates)
    assert case["scope"] == "local_replay_execution"
    assert case["decision"]["entry_id"] == R3_ENTRY_ID
    assert case["decision"]["verdict"] == "unchanged pressure"
    assert [row["candidate_id"] for row in case["decisive_pair"]] == ["base", "offset"]
    assert [row["candidate_id"] for row in case["candidate_costs"]] == ["base", "offset"]
    assert [row["entry_id"] for row in case["decisive_pair"]] == [
        row["entry_id"] for row in case["candidate_costs"]
    ]
    assert [row["digest"] for row in case["decisive_pair"]] == [
        row["digest"] for row in case["candidate_costs"]
    ]
    assert all(row["usd"] == "0.00" and row["api_calls"] == 0 for row in case["candidate_costs"])
    assert [
        {field: row[field] for field in ("wall_s", "cpu_s", "usd", "api_calls")}
        for row in case["candidate_costs"]
    ] == [
        {
            field: candidate.to_dict()["provenance"]["counters"][field]
            for field in ("wall_s", "cpu_s", "usd", "api_calls")
        }
        for candidate in candidates
    ]

    missing_counter = candidates[0].model_copy(
        update={"provenance": candidates[0].provenance.model_copy(update={"counters": None})}
    )
    with pytest.raises(ValueError, match="counters"):
        build_case_evidence(decisive, (missing_counter, candidates[1]))

    foreign_subject = candidates[1].model_copy(update={"subject_ref": "sha256:" + "0" * 64})
    with pytest.raises(ValueError, match="subject_ref"):
        build_case_evidence(decisive, (candidates[0], foreign_subject))

    assert CASE_EXTENSION == "https://reward-lens.github.io/case/v1"


def test_group_constant_score_offset_changes_display_not_centered_unclipped_pressure(
    run_record,
) -> None:
    result = advantage_replay(run_record, _version("before"), _version("after", offset="2.5"))

    assert result.reading.displayed_score_delta == Decimal("2.5")
    assert result.reading.advantage_delta == Decimal("0")
    assert result.reading.verdict == "unchanged pressure"


def test_positive_rescaling_scales_centered_only_advantages_separately() -> None:
    spec = EstimatorSpec(family="grpo", group_centred=True, degenerate_policy="zero")
    before = _replay_group([Decimal("1"), Decimal("3")], spec, group_id="control")
    after = _replay_group([Decimal("2"), Decimal("6")], spec, group_id="control")

    assert after == pytest.approx([2 * value for value in before])


def test_standardization_is_not_declared_invariant_when_its_recorded_epsilon_changes_the_denominator() -> (
    None
):
    spec = EstimatorSpec(
        family="grpo",
        group_centred=True,
        std_normalised=True,
        std_epsilon=0.1,
        std_ddof=0,
        degenerate_policy="zero",
        clip_low=0.2,
        clip_high=0.2,
    )
    before = _replay_group([Decimal("1"), Decimal("3")], spec, group_id="control")
    after = _replay_group([Decimal("2"), Decimal("6")], spec, group_id="control")

    assert after != pytest.approx(before)


def test_refuses_missing_run_record_with_the_reserved_input_contract() -> None:
    with pytest.raises(UsageError) as caught:
        advantage_replay(None, _version("a"), _version("b"))

    assert caught.value.code == "RL0131"
    assert caught.value.exit_code == 4
    assert "run record" in caught.value.message
    assert "group" in caught.value.message


def test_refuses_undeclared_group_instead_of_reconstructing_one_from_adjacency(run_record) -> None:
    broken = replace(
        run_record, rows=(replace(run_record.rows[0], group_id=None), *run_record.rows[1:])
    )

    with pytest.raises(UsageError, match="group_id"):
        advantage_replay(broken, _version("a"), _version("b"))


def test_refuses_missing_component_weight_instead_of_silently_defaulting_to_one(run_record) -> None:
    missing_weight = ReplayVersion("missing", {}, Decimal("0"))

    with pytest.raises(UsageError, match="weights"):
        advantage_replay(run_record, missing_weight, _version("b"))


def test_refuses_missing_recorded_component_data_instead_of_imputing_it(run_record) -> None:
    broken = replace(
        run_record, rows=(replace(run_record.rows[0], components={}), *run_record.rows[1:])
    )

    with pytest.raises(UsageError, match="missing recorded component"):
        advantage_replay(broken, _version("a"), _version("b"))


def test_module_registers_the_real_rung_three_engine() -> None:
    assert get(Rung.ADVANTAGE_REPLAY).__module__ == "reward_lens.product.compare.rung3"
