"""D-75: an uncertainty block may state that no interval is available, and must say why.

Section 7.11 rule 9 stops reporting an interval below about fifteen clusters. The frozen schema
required an `interval` on every uncertainty while admitting the method that says there is none, so
the only way to write the record rule 9 demands was to invent endpoints. Both refusals are checked
here at both layers: the schema refuses an interval under that method and a missing interval under
any other, and the models refuse the same two. The measured value on the estimate is untouched
either way, and `bounded` is the surface the evaluator policy (D-72, P-POLICY) reads; no rule about
what an unbounded estimate does to a decision lives in this package.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from conftest import INDEX, invalid_instance, valid_instance

from reward_lens.contracts import NO_INTERVAL, Assay, RewardLensError, Uncertainty, validate_record

REASON = "9 task clusters, below the 15 at which rule 9 stops reporting an interval"
BOUNDED = {"method": "wilson", "level": 0.95, "interval": [0.7354, 1.0]}
UNBOUNDED = {
    "method": NO_INTERVAL,
    "level": 0.95,
    "clusters": 9,
    "cluster_unit": "task",
    "reason": REASON,
}

MANUFACTURED = "uncertainty_interval_under_stated_absence"
MISSING = "uncertainty_missing_interval_under_wilson"


def the_estimate(record: dict) -> dict:
    for entries in record["measurement"].values():
        for entry in entries:
            if entry["kind"] == "estimate":
                return entry
    raise AssertionError("the fixture carries no estimate entry")


# --- the two shapes, at the model layer ---------------------------------------------------------


def test_a_named_interval_is_bounded():
    assert Uncertainty.model_validate(BOUNDED).bounded is True


def test_the_stated_absence_of_an_interval_is_not_bounded():
    block = Uncertainty.model_validate(UNBOUNDED)
    assert block.bounded is False
    assert block.interval is None
    assert block.reason == REASON
    assert block.clusters == 9


def test_the_models_refuse_manufactured_endpoints():
    with pytest.raises(ValueError, match="invented"):
        Uncertainty.model_validate(dict(UNBOUNDED, interval=[0.0, 1.0]))


def test_the_models_refuse_an_unbounded_block_that_does_not_say_why():
    with pytest.raises(ValueError, match="says why"):
        Uncertainty.model_validate({"method": NO_INTERVAL, "level": 0.95, "clusters": 9})


def test_the_models_refuse_an_unbounded_block_without_its_cluster_count():
    with pytest.raises(ValueError, match="cluster count"):
        Uncertainty.model_validate({"method": NO_INTERVAL, "level": 0.95, "reason": REASON})


def test_the_models_refuse_a_missing_interval_under_a_method_that_produces_one():
    with pytest.raises(ValueError, match="carries the interval"):
        Uncertainty.model_validate({"method": "wilson", "level": 0.95})


# --- the two shapes, whole-record, at both layers -----------------------------------------------


def test_the_unbounded_record_validates_under_the_schema(rs_validator):
    assert list(rs_validator.iter_errors(valid_instance("unbounded_estimate"))) == []


def test_the_unbounded_record_validates_under_the_models_and_round_trips():
    record = valid_instance("unbounded_estimate")
    assay = Assay.model_validate(record)
    assert assay.to_dict() == record


def test_the_unbounded_record_passes_the_runtime_validator():
    validate_record(valid_instance("unbounded_estimate"))


def test_the_unbounded_records_estimate_keeps_its_measured_value():
    """D-75 removes the interval, not the measurement: value, unit, n and method all stay."""
    assay = Assay.model_validate(valid_instance("unbounded_estimate"))
    for entries in assay.measurement.model_dump().values():
        for row in entries:
            if row["kind"] == "estimate":
                assert row["value"] is not None
                assert row["uncertainty"]["method"] == NO_INTERVAL
                assert row["uncertainty"].get("interval") is None


@pytest.mark.parametrize("name", [MANUFACTURED, MISSING])
def test_the_schema_refuses_both_shapes(rs_validator, name):
    """At the uncertainty block of the entry, by the conditional D-75 added: the `then/not` branch
    for the manufactured interval, the `else/required` branch for the missing one. A refusal
    somewhere else in a 426-line fixture would not be this rule."""
    row = INDEX[name]
    landed = {
        (tuple(str(p) for p in e.instance_path), tuple(str(p) for p in e.schema_path))
        for e in rs_validator.iter_errors(invalid_instance(name))
    }
    want = (tuple(row["instance_path"]), tuple(row["schema_path"]))
    assert want in landed, f"{name}: refused, but not at {want}; got {sorted(landed)}"
    assert want[0][-1] == "uncertainty"
    assert want[1][:2] == ("$defs", "uncertainty")


@pytest.mark.parametrize("name", [MANUFACTURED, MISSING])
def test_the_models_refuse_both_shapes(name):
    """The model layer lands on the same block, which is what makes the two layers one rule."""
    with pytest.raises(RewardLensError) as excinfo:
        Assay.model_validate(invalid_instance(name))
    cause = excinfo.value.__cause__
    assert isinstance(cause, ValidationError)
    assert excinfo.value.code == "RL0604"
    locs = {tuple(str(p) for p in error["loc"]) for error in cause.errors()}
    want = tuple(INDEX[name]["model_loc"])
    assert want in locs, f"{name}: refused, but not at {want}; got {sorted(locs)}"
    assert want[-1] == "uncertainty"


@pytest.mark.parametrize("name", [MANUFACTURED, MISSING])
def test_validate_record_refuses_both_shapes_with_the_instance_path(name):
    """RL0604 and no new code: D-75 is a rule of the record, not a new failure mode."""
    from reward_lens.contracts import RewardLensError

    with pytest.raises(RewardLensError) as excinfo:
        validate_record(invalid_instance(name))
    error = excinfo.value
    assert error.code == "RL0604", error.code
    assert error.exit_code == 4
    path = error.context["instance_path"]
    assert path[0] == "measurement"
    assert path[-1] == "uncertainty", path
    assert "/measurement/" in error.message


def test_the_unbounded_estimate_is_the_only_difference_from_the_reference(reference):
    """The valid fixture is the reference record with one uncertainty block replaced, so a
    difference anywhere else would mean the fixture is testing something it does not name."""
    record = valid_instance("unbounded_estimate")
    reference_estimate = the_estimate(reference)
    fixture_estimate = the_estimate(record)
    assert fixture_estimate["entry_id"] == reference_estimate["entry_id"]
    assert fixture_estimate["value"] == reference_estimate["value"]
    assert not Uncertainty.model_validate(fixture_estimate["uncertainty"]).bounded
    assert Uncertainty.model_validate(reference_estimate["uncertainty"]).bounded
    differences = [k for k in set(record) | set(reference) if record.get(k) != reference.get(k)]
    assert sorted(differences) == ["decision", "measurement"], differences
