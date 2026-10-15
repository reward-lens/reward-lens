"""Content and executable checks for the published record standard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
STANDARD = ROOT / "docs" / "standard"


def _published() -> str:
    return "\n".join(
        (STANDARD / name).read_text(encoding="utf-8")
        for name in ("index.md", "assay.md", "run-record.md", "validation.md")
    )


def test_standard_is_normative_and_names_the_authoritative_schemas() -> None:
    text = _published()
    assert "schema/assay/1.0/assay.schema.json" in text
    assert "RECORD_SCHEMA_VERSION" in text
    assert "MUST" in text
    assert "additive" in text
    assert "major" in text


def test_standard_defines_recorded_applied_and_component_fields() -> None:
    text = _published()
    for field in ("reward_recorded", "reward_applied", "advantage", "selection", "components"):
        assert field in text
    assert "MUST NOT name a standard field `reward`" in text
    assert "recorded output" in text
    assert "optimizer-applied contribution" in text


def test_standard_has_two_real_validation_routes_and_no_dummy_optional_reader() -> None:
    text = _published()
    assert "python -m reward_lens.examples.e1" in text
    assert "validate_record" in text
    assert "tests/fixtures/readers/verifiers/" in text
    assert "verifiers_results" in text
    assert "PROVENANCE.json" in text
    assert "verl" in text and "unsupported" in text
    assert "prime-rl" in text and "unsupported" in text


def test_standard_names_required_refusals() -> None:
    text = _published()
    for refusal in (
        "unknown field",
        "incompatible major",
        "invalid instance",
        "missing version",
        "unverifiable example",
    ):
        assert refusal in text


def test_generated_e1_assay_validates_against_the_actual_schema() -> None:
    from reward_lens.contracts import validate_record
    from reward_lens.examples.e1 import build

    validate_record(build().contract_assay().to_dict())


def test_verifiers_instance_is_opened_through_the_actual_reader() -> None:
    from reward_lens.product.trace import open_trace
    from reward_lens.tap.adapters.verifiers import score_tree

    fixture = ROOT / "tests" / "fixtures" / "readers" / "verifiers"
    record = open_trace(fixture, reader="verifiers_results")
    assert record.reader == "verifiers_results"
    assert tuple(record.series) == (
        "reward_recorded",
        "reward_applied",
        "advantage",
        "selection",
    )
    assert record.components == {}

    source_row = json.loads((fixture / "results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    component = score_tree(source_row["metrics"], weights={"correct": 1.0})
    assert component.name == "correct"
    assert component.value == source_row["metrics"]["correct"]


def test_actual_validator_refuses_unknown_field_incompatible_major_and_invalid_instance() -> None:
    from reward_lens.contracts import RecordInvalid, SchemaUnknownMajor, validate_record

    fixture_root = ROOT / "schema" / "assay" / "1.0" / "fixtures"
    reference = json.loads((fixture_root / "reference.json").read_text(encoding="utf-8"))

    unknown = dict(reference, unrecognised_standard_field=True)
    with pytest.raises(RecordInvalid):
        validate_record(unknown)

    unreadable = dict(reference, **{"$schema": reference["$schema"].replace("/1.0/", "/2.0/")})
    raw = json.dumps(unreadable, sort_keys=True).encode("utf-8")
    with pytest.raises(SchemaUnknownMajor) as incompatible:
        validate_record(unreadable, raw=raw)
    assert incompatible.value.context["original_bytes"] == raw

    invalid = json.loads(
        (fixture_root / "invalid" / "missing_schema_url.json").read_text(encoding="utf-8")
    )
    with pytest.raises(RecordInvalid):
        validate_record(invalid)
