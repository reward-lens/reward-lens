"""BLK-048: the recording contract's schema, writers and refusals.

The run's field list lives in the corpus and the rehearsal that drives it lives
in `experiments/chain_c1/recorder/`. What is tested here is the library's half:
the machinery that makes a two-out-of-three field unrepresentable, the three
naming rules, the refusal, and the tap signature the emitter is written against.
"""

from __future__ import annotations

import pytest

from reward_lens.record.contract import (
    ContractField,
    ContractRecord,
    ContractRefusal,
    ContractSchema,
    ContractWriter,
    SchemaIncomplete,
    type_ok,
)
from reward_lens.tap.adapters import trl_signature


def f(name="oracle_label", type="enum", grain="rollout", **kw):
    kw.setdefault("writer", "the oracle harness")
    kw.setdefault("refusal", "Without it the label is the reward.")
    return ContractField(name=name, type=type, grain=grain, **kw)


# ----------------------------------------------------- three edits, or none


def test_a_schema_entry_with_no_writer_is_a_promise():
    with pytest.raises(SchemaIncomplete, match="promise"):
        ContractField(
            name="oracle_label", type="enum", grain="rollout", writer="", refusal="something"
        )


def test_a_writer_with_no_refusal_is_a_silent_null():
    with pytest.raises(SchemaIncomplete, match="silently stays null"):
        ContractField(
            name="oracle_label", type="enum", grain="rollout", writer="something", refusal=""
        )


def test_an_optional_field_may_have_neither():
    ContractField(name="spare", type="float", grain="run", required=False)


def test_a_field_named_reward_cannot_be_declared():
    with pytest.raises(SchemaIncomplete, match="no column anywhere"):
        f(name="reward", type="float")


def test_a_grain_outside_the_seven_is_refused():
    with pytest.raises(SchemaIncomplete, match="grain"):
        f(grain="epoch")


def test_a_type_that_is_not_a_type_is_refused():
    with pytest.raises(SchemaIncomplete, match="not a real type"):
        f(type="whatever")


def test_a_field_named_twice_is_two_fields():
    with pytest.raises(SchemaIncomplete, match="named twice"):
        ContractSchema((f(), f()))


# --------------------------------------------------------------- the refusal


def _schema():
    return ContractSchema(
        (
            f(name="oracle_label", type="enum", grain="rollout"),
            f(name="oracle_fraction", type="float", grain="rollout"),
            f(name="loss_type", type="str", grain="run"),
        )
    )


def _full():
    r = ContractRecord(schema=_schema())
    for k in ("0:0:0", "0:0:1"):
        r.put("rollout", k, "oracle_label", "EXPLOIT")
        r.put("rollout", k, "oracle_fraction", 0.25)
    r.put("run", "run", "loss_type", "grpo")
    return r


EXPECT = {"rollout": ["0:0:0", "0:0:1"], "run": ["run"]}


def test_a_complete_record_verifies():
    _full().verify(expect=EXPECT)


def test_a_missing_field_is_refused_and_named():
    r = _full()
    del r.values["rollout"]["0:0:1"]["oracle_fraction"]
    with pytest.raises(ContractRefusal, match="`oracle_fraction`"):
        r.verify(expect=EXPECT)


def test_the_refusal_carries_the_writer_that_did_not_run():
    r = _full()
    del r.values["rollout"]["0:0:1"]["oracle_label"]
    with pytest.raises(ContractRefusal, match="the oracle harness"):
        r.verify(expect=EXPECT)


def test_a_field_present_on_one_rollout_and_not_another_is_refused():
    """The case a row count passes: the writer stopped after the first group."""
    r = _full()
    del r.values["rollout"]["0:0:1"]["oracle_label"]
    assert len(r.values["rollout"]) == 2  # the count is still right
    with pytest.raises(ContractRefusal):
        r.verify(expect=EXPECT)


def test_a_present_but_wrong_typed_field_is_refused():
    r = _full()
    r.values["rollout"]["0:0:0"]["oracle_fraction"] = "0.25"
    with pytest.raises(ContractRefusal, match="declares float"):
        r.verify(expect=EXPECT)


def test_a_field_written_at_the_wrong_grain_is_refused():
    r = _full()
    with pytest.raises(ContractRefusal, match="rollout-grain field"):
        r.put("run", "run", "oracle_label", "EXPLOIT")


def test_a_field_not_in_the_contract_is_refused():
    r = _full()
    with pytest.raises(ContractRefusal, match="not in the recording contract"):
        r.put("rollout", "0:0:0", "invented_field", 1.0)


def test_a_column_named_reward_is_refused_at_write_time():
    r = _full()
    with pytest.raises(ContractRefusal, match="no column anywhere"):
        r.put("rollout", "0:0:0", "reward", 1.0)


# --------------------------------------------------- the three naming rules


def _series_schema():
    return ContractSchema(
        (
            f(name="reward_components", type="array<float>", grain="rollout"),
            f(name="reward_component_index", type="array<int>", grain="run"),
            f(name="reward_component_max", type="array<float>", grain="run"),
        )
    )


def test_a_component_series_keyed_on_a_duplicate_index_is_refused():
    r = ContractRecord(schema=_series_schema())
    with pytest.raises(ContractRefusal, match="9.000000000"):
        r.set_component_series("0:0:0", [1.0, 2.0], [0, 0], [1.0, 1.0])


def test_a_component_series_with_no_attainable_maximum_is_refused():
    r = ContractRecord(schema=_series_schema())
    with pytest.raises(ContractRefusal, match="attainable maximum"):
        r.set_component_series("0:0:0", [1.0], [0], [None])


def test_a_component_series_shorter_than_its_index_is_refused():
    r = ContractRecord(schema=_series_schema())
    with pytest.raises(ContractRefusal, match="keyed on something else"):
        r.set_component_series("0:0:0", [1.0], [0, 1], [1.0, 1.0])


def test_a_valid_component_series_lands_at_three_grains():
    r = ContractRecord(schema=_series_schema())
    r.set_component_series("0:0:0", [0.25, 1.0], [0, 1], [1.0, 1.0])
    assert r.get("rollout", "0:0:0", "reward_components") == [0.25, 1.0]
    assert r.get("run", "run", "reward_component_index") == [0, 1]
    assert r.get("run", "run", "reward_component_max") == [1.0, 1.0]


# ------------------------------------------------------------------- types


@pytest.mark.parametrize(
    "spec,value,ok",
    [
        ("float", 1.0, True),
        ("float", True, False),
        ("float", "1.0", False),
        ("int", 1, True),
        ("int", True, False),
        ("int", 1.0, False),
        ("bool", True, True),
        ("bool", 1, False),
        ("str", "x", True),
        ("str", 1, False),
        ("object", {"a": 1}, True),
        ("object", [1], False),
        ("array<float>", [1.0, 2.0], True),
        ("array<float>", 1.0, False),
        ("array<float>", "ab", False),
        ("array<int>", [1, 2], True),
        ("array<int>", [1.5], False),
        ("array<str>", ["a"], True),
        ("array<str>", [1], False),
        ("enum<A<B<C>", "B", True),
        ("enum<A<B<C>", "", False),
    ],
)
def test_type_ok(spec, value, ok):
    assert type_ok(spec, value) is ok


# ------------------------------------------------------------ the signature


def test_the_documented_tap_signature_matches_the_live_tap():
    """A documented signature that drifts is a wrong answer with a citation."""
    trl_signature.verify()


def test_the_signature_check_can_fail():
    class Drifted:
        pass

    with pytest.raises(trl_signature.SignatureDrift):
        trl_signature.verify(Drifted)


def test_the_declared_signature_names_the_seams_an_emitter_overrides():
    text = trl_signature.declared()
    for name in ("wrap", "attach", "finish", "_features_for", "_turns_for"):
        assert name in text


# --------------------------------------------------------------- the writer


def test_the_contract_is_written_beside_the_record(tmp_path):
    r = _full()
    d = ContractWriter(root=tmp_path, run_id="chain-c1").write(r)
    assert (d / "rollout.json").exists()
    assert (d / "run.json").exists()
    assert (d / "schema.json").exists()
    assert d.parent.parent.name == "runs"
