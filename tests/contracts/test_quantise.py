"""The declared-scale rule of D-10, at the model layer and at the validator."""

from __future__ import annotations

import copy
import json

from decimal import Decimal

import pytest
import rfc8785
from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import load, FIXTURES

from reward_lens.contracts import Assay, Cost, validate_record
from reward_lens.contracts.errors import NumberNotQuantised
from reward_lens.contracts.quantise import MAX_SIGNIFICANT_DIGITS, quantise


@given(
    x=st.floats(
        allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6, width=64
    ),
    scale=st.integers(min_value=0, max_value=6),
)
@settings(max_examples=400, deadline=None)
def test_jcs_round_trips_every_quantised_value(x, scale):
    q = quantise(x, scale)
    assert json.loads(rfc8785.dumps(q)) == q


@given(
    x=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    scale=st.integers(min_value=0, max_value=6),
)
@settings(max_examples=400, deadline=None)
def test_quantising_twice_changes_nothing(x, scale):
    q = quantise(x, scale)
    assert quantise(q, scale) == q


@given(
    x=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    scale=st.integers(min_value=0, max_value=6),
)
@settings(max_examples=200, deadline=None)
def test_a_quantised_value_carries_at_most_its_scale(x, scale):
    """Counted on the JCS rendering, which is what the bytes carry: Python's repr writes 0.0
    where ES6 writes 0, so repr is not the thing to count."""
    rendered = rfc8785.dumps(quantise(x, scale)).decode("utf-8")
    fraction = rendered.split("e")[0].partition(".")[2]
    assert len(fraction) <= scale


@pytest.mark.parametrize(
    ("value", "scale", "expected"),
    [
        (0.5, 0, 0.0),
        (1.5, 0, 2.0),
        (2.5, 0, 2.0),
        (0.12345, 4, 0.1234),
        (0.12355, 4, 0.1236),
        (1.0, 3, 1.0),
        (-0.0005, 3, -0.0),
    ],
)
def test_round_half_even(value, scale, expected):
    assert quantise(value, scale) == expected


def test_quantise_refuses_nan_and_infinity():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            quantise(bad, 3)


def test_quantise_refuses_more_than_fifteen_significant_digits():
    assert MAX_SIGNIFICANT_DIGITS == 15
    with pytest.raises(ValueError):
        quantise(1234567890.1234567, 7)


def test_nan_and_infinity_are_refused_at_the_model_layer():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(Exception):
            Cost(wall_s=bad, cpu_s=0.0, usd="0.00", api_calls=0)


def test_a_model_quantises_on_construction():
    cost = Cost(wall_s=1.23456789, cpu_s=0.0, usd="0.00", api_calls=0)
    assert cost.wall_s == 1.235  # x-scale 3 on cost.wall_s


def test_money_is_a_two_place_string():
    cost = Cost(wall_s=1.0, cpu_s=0.0, usd="0.00", api_calls=0)
    assert cost.usd == "0.00"
    with pytest.raises(Exception):
        Cost(wall_s=1.0, cpu_s=0.0, usd="0.0", api_calls=0)
    with pytest.raises(Exception):
        Cost(wall_s=1.0, cpu_s=0.0, usd=0.0, api_calls=0)


def test_the_validator_rejects_an_over_precise_number(reference):
    record = copy.deepcopy(reference)
    record["cost"]["wall_s"] = 31.234567
    with pytest.raises(NumberNotQuantised) as excinfo:
        validate_record(record)
    assert "cost.wall_s" in str(excinfo.value)


def test_the_validator_accepts_the_quantised_form(reference):
    record = copy.deepcopy(reference)
    record["cost"]["wall_s"] = 31.235
    validate_record(record)


def test_an_assay_round_trips_through_canonical_json(reference):
    assay = Assay.model_validate(reference)
    again = Assay.model_validate(json.loads(rfc8785.dumps(assay.to_dict())))
    assert again.to_dict() == assay.to_dict()


def test_the_quantiser_fixture_is_refused_by_field_name():
    """`measured_disagreement` is declared x-scale 4; the fixture carries seven places."""
    inst = load(FIXTURES / "invalid" / "number_not_quantised.json")
    assert inst["intent"]["outcome_check"]["measured_disagreement"] == 0.1234567
    with pytest.raises(NumberNotQuantised) as excinfo:
        validate_record(inst)
    assert "intent.outcome_check.measured_disagreement" in str(excinfo.value)
    assert excinfo.value.code == "RL0603"
