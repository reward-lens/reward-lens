"""The D-66 parity gate: the frozen schema and the pydantic models give the same verdict.

Three assertions, following note 25 (d): differential verdicts over the fixture corpus, generated
instances in both directions, and a keyword-level structural comparison that ignores titles,
descriptions and order. `x-` keywords and `format` are stripped before generation, because
`hypothesis-jsonschema` reads neither.

Four edits are made to the schema the generator draws from, and to nothing else: the judge is
always the specification's own schema. `bound_quantised_numbers`, `strip_annotations`,
`pin_patterns` and `relax_conditionals` are each defended where they are defined in
`structure.py`. Together they took the nine per-`$def` generation tests from over 190 seconds for
one of them to 1.6 seconds for all nine (A-003). They were not enough for the two whole-record
tests, which live in `test_parity_slow.py` behind `REWARD_LENS_SLOW=1` with their measurement.
"""

from __future__ import annotations

import re

import jsonschema_rs
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis_jsonschema import from_schema

from conftest import INDEX, VALID_NAMES, invalid_instance, valid_instance
from structure import (
    PATTERN_SAMPLES,
    bound_quantised_numbers,
    diff,
    keymap,
    patterns_in,
    pin_patterns,
    relax_conditionals,
    strip_annotations,
)

from reward_lens.contracts import NO_INTERVAL, Assay, model_for_def
from reward_lens.contracts.quantise import scale_violations

CORPUS = ["reference"] + ["valid/" + n for n in VALID_NAMES] + sorted(INDEX)

GENERATION = settings(
    max_examples=10,
    deadline=None,
    database=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
        HealthCheck.large_base_example,
    ],
)


def model_accepts(instance) -> bool:
    try:
        Assay.model_validate(instance)
    except Exception:
        return False
    return True


@pytest.mark.parametrize("name", CORPUS)
def test_identical_verdicts_over_the_fixture_corpus(rs_validator, reference, name):
    if name == "reference":
        instance = reference
    elif name.startswith("valid/"):
        instance = valid_instance(name.split("/", 1)[1])
    else:
        instance = invalid_instance(name)
    assert rs_validator.is_valid(instance) == model_accepts(instance), (
        f"{name}: schema and models disagree"
    )


@pytest.fixture(scope="session")
def judgement_schema(hand_schema):
    """The specification's own sub-schemas, used as the judge in `parity_over`. Only the
    annotations a JSON Schema validator does not read are dropped; every rule stays."""
    return strip_annotations(hand_schema)


@pytest.fixture(scope="session")
def generation_schema(hand_schema):
    """The hand schema as the generator sees it: quantised numbers bounded, annotations dropped,
    patterns pinned, conditionals relaxed. Each edit is defended where it is defined."""
    return relax_conditionals(
        pin_patterns(strip_annotations(bound_quantised_numbers(hand_schema)))
    )


@pytest.fixture(scope="session")
def model_schema():
    """The model's own schema, unpinned: assertion 3 compares this one, patterns included."""
    return strip_annotations(Assay.model_json_schema(mode="validation"))


def test_the_quantiser_layer_is_the_one_divergence(judgement_schema):
    """The models are the schema plus the quantiser layer, which is what `check_fixtures.py`
    already reports as two layers. This is the divergence, stated rather than generated around:
    at scale 3 the schema accepts 1e12 and the models refuse it, because writing it at that
    scale needs sixteen significant digits and a double round-trips fifteen (D-10, RL0603)."""
    probe = {"cpu_s": 1e12}
    judge = jsonschema_rs.validator_for(sub_schema(judgement_schema, "counters"))
    assert judge.is_valid(probe)
    with pytest.raises(Exception):
        model_for_def("counters").model_validate(probe)


def test_the_generator_sees_no_pattern(hand_schema):
    """The pinning table and the schema must agree, or generation quietly stops being cheap (a
    pattern the table misses) or the table quietly stops being checked (a sample that has gone
    illegal, or an entry for a pattern the schema dropped)."""
    stripped = strip_annotations(hand_schema)
    assert patterns_in(stripped) == set(PATTERN_SAMPLES), "PATTERN_SAMPLES and the schema differ"
    assert not patterns_in(pin_patterns(stripped))
    for pattern, samples in PATTERN_SAMPLES.items():
        assert samples, pattern
        for sample in samples:
            assert re.fullmatch(pattern, sample), (pattern, sample)


def sub_schema(schema: dict, ref: str) -> dict:
    """One `$def`, carrying the whole `$defs` block so its own `$ref`s still resolve."""
    return {**schema["$defs"][ref], "$defs": schema["$defs"]}


def parity_over(judge: dict, gen: dict, model, scaled: dict) -> None:
    """Generate from one `$def` and assert the schema and the model agree on every instance.

    `judge` is the sub-schema as the specification writes it and decides the verdict; `gen` is
    the same sub-schema after the generator's edits and only decides what gets drawn. They are
    two schemas because `relax_conditionals` must not reach the judge: a validator that has lost
    the record's `if`/`then` would accept what the models correctly reject, and the gate would
    report that disagreement as a defect in the models.

    The validator and the model are resolved once: rebuilding a `jsonschema_rs` validator inside
    the property function compiles the whole `$defs` block on every example.
    """
    sub_validator = jsonschema_rs.validator_for(judge)
    wrapper = {
        "type": "object",
        "properties": {"probe": gen},
        "required": ["probe"],
        "$defs": gen["$defs"],
    }

    @GENERATION
    @given(from_schema(wrapper))
    def check(instance):
        probe = instance["probe"]
        try:
            built = model.model_validate(probe)
            accepted = True
        except Exception:
            built = None
            accepted = False
        assert sub_validator.is_valid(probe) == accepted, probe
        if built is not None:
            out = built.to_dict()
            # Serialisation is a function of the values and not of how the object was built, so
            # a second trip through the model changes nothing.
            assert model.model_validate(out).to_dict() == out, out
            # And the specification's own sub-schema still admits what the serialiser wrote: a
            # required nullable property is written as `null`, never dropped.
            assert sub_validator.is_valid(out), out
            # And what the model reads back out is what went in, for every generated instance the
            # record layer admits. The exception is the quantiser, the one stated divergence: a
            # generated number carrying more decimals than its `x-scale` is rounded on the way in,
            # and such an instance is one `validate_record()` refuses (RL0603).
            if not scale_violations(probe, scaled):
                assert out == probe, probe

    check()


@pytest.mark.parametrize(
    "ref",
    ["entry", "finding", "uncertainty", "power", "method", "entry_provenance", "counters"],
)
def test_generated_from_each_def_gets_the_same_verdict(
    hand_schema, judgement_schema, generation_schema, ref
):
    """Sub-schema generation reaches the entry kinds and the metrology blocks, which the
    whole-record strategy rarely populates."""
    parity_over(
        sub_schema(judgement_schema, ref),
        sub_schema(generation_schema, ref),
        model_for_def(ref),
        sub_schema(hand_schema, ref),
    )


@pytest.mark.parametrize("method", ["wilson", NO_INTERVAL])
def test_generated_uncertainties_cover_both_shapes(
    hand_schema, judgement_schema, generation_schema, method
):
    """D-75 put two shapes in one `$def`: an interval with its method, and the stated absence of
    one with its reason. A strategy free to pick the method reaches the rarer branch only by luck,
    so each branch is generated on its own and checked against the same conditional. Only the
    generator is narrowed; the judge keeps the whole `$def`, conditional included."""
    gen = sub_schema(generation_schema, "uncertainty")
    narrowed = {**gen, "properties": {**gen["properties"], "method": {"const": method}}}
    parity_over(
        sub_schema(judgement_schema, "uncertainty"),
        narrowed,
        model_for_def("uncertainty"),
        sub_schema(hand_schema, "uncertainty"),
    )


def test_keyword_level_structural_parity(hand_schema, model_schema):
    problems = diff(keymap(strip_annotations(hand_schema)), keymap(model_schema))
    assert not problems, "\n".join(problems[:40])


def test_the_models_forbid_unknown_fields(reference):
    """Without extra=forbid pydantic emits no additionalProperties:false and the gate lies."""
    import copy

    junk = copy.deepcopy(reference)
    junk["surprise"] = 1
    assert not model_accepts(junk)


def test_the_model_schema_declares_the_scales(hand_schema):
    """`x-scale` is part of the specification, so the model's schema carries it too."""
    emitted = Assay.model_json_schema(mode="validation")
    hand = keymap(hand_schema)
    model = keymap(emitted)
    scaled = {p: kw["x-scale"] for p, kw in hand.items() if "x-scale" in kw}
    assert scaled
    for pointer, scale in scaled.items():
        assert model.get(pointer, {}).get("x-scale") == scale, pointer
