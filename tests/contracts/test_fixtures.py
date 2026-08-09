"""The fixture corpus, under both layers.

The reference instance validates under `jsonschema-rs` and under the pydantic models; every
invalid instance fails on the rule its INDEX entry names, at the layer that entry names, and at
the place that entry names. A fixture refused for the wrong reason is a test of nothing, so each
row of `fixtures/invalid/INDEX.json` carries where the refusal lands as well as what it is for:

    rule            what the instance breaks, in words
    caught_by       "schema" or "quantiser"
    refusal         the RewardLensError subclass `validate_record()` raises
    code            its RL code
    instance_path   where in the instance the refusal lands (schema layer)
    schema_path     which keyword of the frozen schema refuses it (schema layer)
    model_loc       the `loc` of the pydantic error (schema layer)
    field           the over-precise field (quantiser layer)

The paths are the contract, not a transcript of one library's phrasing: no assertion here reads a
message string.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from conftest import (
    INDEX,
    QUANTISER_LAYER,
    SCHEMA_LAYER,
    VALID_NAMES,
    invalid_instance,
    valid_instance,
)

from conftest import INDEXED_WITHOUT_FIXTURE, INVALID_NAMES, ORPHAN_FIXTURES

from reward_lens.contracts import Assay, RecordInvalid, RewardLensError, validate_record
from reward_lens.contracts.quantise import scale_violations


def named(name: str, key: str):
    """One recorded expectation, or a failure that says the INDEX row is incomplete."""
    row = INDEX[name]
    assert key in row, f"{name}: INDEX names no {key} for `{row['rule']}`"
    return row[key]


def test_reference_validates_under_jsonschema_rs(rs_validator, reference):
    assert list(rs_validator.iter_errors(reference)) == []


def test_reference_has_no_scale_violation(hand_schema, reference):
    assert scale_violations(reference, hand_schema) == []


def test_reference_validates_under_the_models(reference):
    assay = Assay.model_validate(reference)
    assert assay.to_dict() == reference


def test_reference_passes_the_runtime_validator(reference):
    validate_record(reference)


@pytest.mark.parametrize("name", SCHEMA_LAYER)
def test_schema_layer_instance_rejected_by_jsonschema_rs_at_its_named_rule(rs_validator, name):
    """The schema refuses it, and one of the errors is the rule the INDEX row names: the keyword
    of the frozen schema that fires, at the place in the instance where it fires."""
    landed = {
        ([str(p) for p in e.instance_path].__str__(), [str(p) for p in e.schema_path].__str__())
        for e in rs_validator.iter_errors(invalid_instance(name))
    }
    assert landed, f"{name} was accepted by the schema, but INDEX says {INDEX[name]['rule']}"
    want = (str(named(name, "instance_path")), str(named(name, "schema_path")))
    assert want in landed, f"{name}: refused, but not at {want[0]} by {want[1]}; got {sorted(landed)}"


@pytest.mark.parametrize("name", SCHEMA_LAYER)
def test_schema_layer_instance_rejected_by_the_models_at_its_named_loc(name):
    """The models refuse it at the same place. `loc` is pydantic's own path into the instance,
    so this is the model layer's answer to the question the schema path answers above.

    `Assay.model_validate` answers in RL0604, the code the record layer reserves for a record
    that does not validate, and keeps pydantic's own error as the cause: every location is still
    there, under `__cause__`."""
    with pytest.raises(RecordInvalid) as excinfo:
        Assay.model_validate(invalid_instance(name))
    cause = excinfo.value.__cause__
    assert isinstance(cause, ValidationError)
    assert excinfo.value.code == "RL0604"
    locs = {tuple(str(p) for p in error["loc"]) for error in cause.errors()}
    want = tuple(named(name, "model_loc"))
    assert want in locs, f"{name}: refused, but not at {want}; got {sorted(locs)}"


@pytest.mark.parametrize("name", QUANTISER_LAYER)
def test_quantiser_layer_instance_passes_the_schema(rs_validator, hand_schema, name):
    """The quantiser rule is not a schema rule: the schema accepts these, the walker does not."""
    inst = invalid_instance(name)
    assert list(rs_validator.iter_errors(inst)) == []
    violations = scale_violations(inst, hand_schema)
    assert violations, f"{name}: no scale violation found"
    assert violations[0].startswith(named(name, "field") + ":"), violations


@pytest.mark.parametrize("name", QUANTISER_LAYER)
def test_quantiser_layer_instance_is_quantised_by_the_models(name):
    """D-10 quantises at the record boundary, so the model accepts and rounds; the validator
    is what refuses the over-precise number, below."""
    inst = invalid_instance(name)
    assay = Assay.model_validate(inst)
    assert assay.to_dict() != inst


@pytest.mark.parametrize("name", QUANTISER_LAYER)
def test_quantiser_layer_instance_is_refused_by_the_field_it_names(name):
    with pytest.raises(RewardLensError) as excinfo:
        validate_record(invalid_instance(name))
    error = excinfo.value
    assert type(error).__name__ == named(name, "refusal"), error
    assert error.code == named(name, "code")
    assert error.context["field"] == named(name, "field")


@pytest.mark.parametrize("name", sorted(INDEX))
def test_every_invalid_instance_is_refused_by_validate_record(name):
    with pytest.raises(RewardLensError) as excinfo:
        validate_record(invalid_instance(name))
    error = excinfo.value
    assert type(error).__name__ == named(name, "refusal"), error
    assert error.code == named(name, "code")
    assert error.code.startswith("RL06")


@pytest.mark.parametrize("name", SCHEMA_LAYER)
def test_schema_layer_refusal_names_the_path_the_index_records(name):
    """What the caller is told. `validate_record()` reports one refusal out of the several a
    broken record can produce, and which one it reports is part of the contract."""
    with pytest.raises(RewardLensError) as excinfo:
        validate_record(invalid_instance(name))
    context = excinfo.value.context
    assert [str(p) for p in context["instance_path"]] == named(name, "instance_path")
    assert [str(p) for p in context["schema_path"]] == named(name, "schema_path")


@pytest.mark.parametrize("name", VALID_NAMES)
def test_valid_corpus_instance_validates_under_both_layers(rs_validator, hand_schema, name):
    """`fixtures/valid/` holds the records the specification admits beside the reference, and the
    freeze checker reads the same directory."""
    instance = valid_instance(name)
    assert list(rs_validator.iter_errors(instance)) == []
    assert scale_violations(instance, hand_schema) == []
    assay = Assay.model_validate(instance)
    assert assay.to_dict() == instance
    validate_record(instance)


def test_the_invalid_corpus_and_its_index_are_the_same_set():
    """An orphan fixture cannot go untested, and an INDEX row cannot name a file that is gone.

    The parametrisation reads the directory, so a fixture written without its row lands here
    rather than sitting on disk demonstrating nothing. The other direction is the same failure
    seen from the INDEX: a row whose fixture was renamed or deleted names a rule the corpus no
    longer shows.
    """
    assert ORPHAN_FIXTURES == [], f"fixtures on disk with no INDEX row: {ORPHAN_FIXTURES}"
    assert INDEXED_WITHOUT_FIXTURE == [], (
        f"INDEX rows with no fixture on disk: {INDEXED_WITHOUT_FIXTURE}"
    )
    assert INVALID_NAMES, "the invalid corpus is empty"
