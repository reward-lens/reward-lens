"""Trace's caller-declared TRL partition metadata is a typed API contract."""

from __future__ import annotations

import pytest

from reward_lens import api, contracts


def test_trace_request_carries_only_explicit_trl_declarations() -> None:
    request = api.TraceRequest(
        run="trl-run",
        reward_functions=("length_reward", "format_reward"),
        extra_columns=("group",),
        weights=("length_reward=0.5", "format_reward=2"),
        group_column="group",
        group_size=4,
    )
    assert request.reward_functions == ("length_reward", "format_reward")
    assert request.extra_columns == ("group",)
    assert request.weights == ("length_reward=0.5", "format_reward=2")
    assert request.group_column == "group"
    assert request.group_size == 4


def test_compare_request_carries_an_optional_typed_response_bank(tmp_path) -> None:
    bank = tmp_path / "fixed-bank.jsonl"
    request = api.CompareRequest(baseline="v1", candidate="v2", project=tmp_path, bank=bank)
    assert request.project == tmp_path
    assert request.bank == bank


@pytest.mark.parametrize(
    "weights",
    [
        ("length_reward=nan",),
        ("length_reward=inf",),
        ("length_reward=1", "length_reward=2"),
        ("length_reward",),
        ("=1",),
    ],
)
def test_trace_request_refuses_ambiguous_weight_declarations(weights: tuple[str, ...]) -> None:
    with pytest.raises(contracts.UsageError) as caught:
        api.TraceRequest(run="trl-run", weights=weights)
    assert caught.value.code == "RL0003"
    assert "weights" in caught.value.message


@pytest.mark.parametrize("size", [0, -1])
def test_trace_request_refuses_nonpositive_declared_group_size(size: int) -> None:
    with pytest.raises(contracts.UsageError) as caught:
        api.TraceRequest(run="trl-run", group_size=size)
    assert caught.value.code == "RL0003"
    assert "group_size" in caught.value.message
