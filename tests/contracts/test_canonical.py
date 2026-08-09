"""`canonical_bytes()` is the only producer of hashed bytes (D-10)."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from typing import get_args, get_origin

import pytest
import rfc8785
from pydantic import BaseModel

from conftest import CONTRACTS_SRC, ROOT, VECTOR, VECTOR_DIGEST, load

from reward_lens.contracts import Assay, absence, canonical_bytes, digest
from reward_lens.contracts.canonical import EXCLUDED_FROM_DIGEST, strip_for_digest
from reward_lens.contracts.models import (
    MODEL_FOR_DEF,
    Cost,
    Decision,
    Embedding,
    EntryProvenance,
    Intent,
    Measurement,
    Producer,
    Provenance,
    Subject,
    omitted_when_none,
)
from reward_lens.contracts.envelope import ResultEnvelope
from reward_lens.contracts.models import ContextRef, Digests, VersionRef
from reward_lens.contracts.validate import assay_schema

DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def test_canonical_bytes_is_rfc8785_over_the_stripped_record(reference):
    assert canonical_bytes(reference) == rfc8785.dumps(strip_for_digest(reference))


def test_canonical_bytes_is_utf8_bytes(reference):
    out = canonical_bytes(reference)
    assert isinstance(out, bytes)
    out.decode("utf-8")


def test_the_excluded_keys_are_removed_not_nulled(reference):
    stripped = strip_for_digest(reference)
    assert "assay_id" not in stripped
    assert "attestation" not in stripped
    assert "environment_excluded_from_digest" not in stripped
    assert "signature" not in stripped["decision"]
    assert set(EXCLUDED_FROM_DIGEST) == {
        "assay_id",
        "attestation",
        "decision.signature",
        "environment_excluded_from_digest",
    }


def test_changing_an_excluded_key_does_not_change_the_digest(reference):
    other = copy.deepcopy(reference)
    other["assay_id"] = "sha256:" + "f" * 64
    other["environment_excluded_from_digest"] = {"host": "somewhere-else"}
    other["attestation"] = {
        "statement_digest": "sha256:" + "a" * 64,
        "backend": "sigstore",
    }
    assert digest(other) == digest(reference)


def test_changing_a_measured_value_does_change_the_digest(reference):
    other = copy.deepcopy(reference)
    other["cost"]["wall_s"] = reference["cost"]["wall_s"] + 1
    assert digest(other) != digest(reference)


def test_digest_shape(reference):
    assert DIGEST_RE.match(digest(reference))


def test_an_assay_and_its_dict_hash_the_same(reference):
    assert digest(Assay.model_validate(reference)) == digest(reference)


def test_integral_floats_render_the_es6_way():
    """`json.dumps` writes 1.0 and 1e-07; ES6 and therefore JCS write 1 and 1e-7 (note 25 (a))."""
    assert rfc8785.dumps({"a": 1.0}) == b'{"a":1}'
    assert rfc8785.dumps({"a": 1e-7}) == b'{"a":1e-7}'


def test_no_json_dumps_anywhere_in_the_contracts_package():
    hits = subprocess.run(
        ["/usr/bin/grep", "-rn", "--include=*.py", "json.dumps", str(CONTRACTS_SRC)],
        capture_output=True,
        text=True,
    )
    assert hits.returncode == 1, f"json.dumps found on a contracts path:\n{hits.stdout}"


def test_vector_digest_matches_the_checked_in_value():
    assert VECTOR.exists() and VECTOR_DIGEST.exists()
    expected = VECTOR_DIGEST.read_text(encoding="utf-8").strip()
    assert DIGEST_RE.match(expected)
    assert digest(load(VECTOR)) == expected


def test_the_vector_is_itself_a_valid_record():
    from reward_lens.contracts import validate_record

    validate_record(load(VECTOR))


@pytest.mark.parametrize("seed", ["0", "1", "524287"])
def test_digest_is_stable_across_pythonhashseed_in_a_fresh_interpreter(seed):
    env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(ROOT / "src"))
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys;from reward_lens.contracts import digest;"
            "print(digest(json.load(open(sys.argv[1]))))",
            str(VECTOR),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == VECTOR_DIGEST.read_text(encoding="utf-8").strip()


def test_the_digest_is_sha256_of_the_canonical_bytes(reference):
    expected = hashlib.sha256(canonical_bytes(reference)).hexdigest()
    assert digest(reference) == f"sha256:{expected}"


def a_record_built_in_code(reference: dict) -> Assay:
    """The record an SDK caller builds: constructors and `absence()`, nothing off disk.

    The blocks that carry no entries (producer, subject, intent, cost, embedding, provenance)
    are the reference instance's, because what this test is about is the measurement and the
    serialisation, not the scaffolding around them.
    """
    entry = absence(
        "signal",
        "signal.gradient_probe",
        "the update response of the scorer under a one-step fork",
        "no training loop was reachable in this environment",
        "re-run inside the trainer with the tap attached",
        ("signal.update_response",),
        subject_ref=reference["subject"]["version"]["digest"],
        provenance=EntryProvenance(
            started="2026-09-13T10:15:00Z",
            duration_s=0.412,
            sandbox_tier="L2",
            offline=True,
        ),
    )
    sections = {name: [] for name in Measurement.model_fields}
    sections["signal"] = [entry]
    built = Assay(
        schema_url=reference["$schema"],
        assay_id=reference["assay_id"],
        created=reference["created"],
        producer=Producer.model_validate(reference["producer"]),
        subject=Subject.model_validate(reference["subject"]),
        intent=Intent.model_validate(reference["intent"]),
        measurement=Measurement.model_validate(sections),
        holes=[],
        findings=[],
        decision=Decision.model_validate(reference["decision"]),
        cost=Cost.model_validate(reference["cost"]),
        embedding=Embedding.model_validate(reference["embedding"]),
        provenance=Provenance.model_validate(reference["provenance"]),
        environment_excluded_from_digest=dict(reference["environment_excluded_from_digest"]),
    )
    built.holes_from_entries()
    return built


def test_the_digest_of_a_built_record_equals_the_digest_of_its_reload(reference):
    """Canonical bytes are a function of the record's values, never of how it was built.

    `to_dict()` used to be `model_dump(exclude_unset=True)`, which reads the object's history:
    a record built through the constructors and the same record loaded from disk could hash
    differently, and P-STORE's dependency digests are taken over exactly these bytes. Three
    assertions pin the property, and they are the round trip through disk. The history itself is
    tested one function down, by construction: two constructor paths that set different fields,
    rather than a record whose `__pydantic_fields_set__` was reached into and rewritten.
    """
    built = a_record_built_in_code(reference)
    on_disk = rfc8785.dumps(built.to_dict())
    reloaded = Assay.model_validate(json.loads(on_disk.decode("utf-8")))

    assert reloaded.to_dict() == built.to_dict()
    assert canonical_bytes(reloaded) == canonical_bytes(built)
    assert digest(reloaded) == digest(built)
    assert rfc8785.dumps(reloaded.to_dict()) == on_disk


def test_the_built_record_round_trips_through_model_validate(reference):
    """`to_dict(Assay.model_validate(d)) == d` for a record the constructors produced, which is
    the same identity the fixture corpus asserts, taken from the other end."""
    record = a_record_built_in_code(reference).to_dict()
    assert Assay.model_validate(record).to_dict() == record


@pytest.mark.parametrize("ref", sorted(MODEL_FOR_DEF))
def test_the_serialiser_omits_exactly_what_the_schema_leaves_optional_in_each_def(
    hand_schema, ref
):
    """The rule is the frozen schema's `required` list, read back the other way.

    The serialiser decides what to omit from the model's own field defaults, so this test is what
    stops the two specifications drifting: for every `$def` the parity gate generates from, the
    set of keys the serialiser may omit is exactly the set of properties the schema does not
    require. A field that gains or loses a default without the schema moving fails here.
    """
    node = hand_schema["$defs"][ref]
    model = MODEL_FOR_DEF[ref]
    names = {field.alias or name for name, field in model.model_fields.items()}
    optional = set(node["properties"]) - set(node.get("required", []))
    assert optional <= names, sorted(optional - names)
    assert omitted_when_none(model) & names == optional


def test_the_record_omits_exactly_what_the_schema_leaves_optional(hand_schema):
    """The same rule at the top level, where `$schema` is the one aliased field."""
    names = {field.alias or name for name, field in Assay.model_fields.items()}
    optional = set(hand_schema["properties"]) - set(hand_schema["required"])
    assert optional <= names, sorted(optional - names)
    assert omitted_when_none(Assay) & names == optional


def test_a_required_nullable_field_survives_the_round_trip_as_null(reference):
    """Rule 1 of the serialisation: a field the schema requires is written even when it is null.
    The reference record carries fourteen of them, and dropping any one would leave a record the
    frozen schema refuses."""
    written = Assay.model_validate(reference).to_dict()
    nulls = 0
    for pointer, value in walk_leaves(reference):
        if value is None:
            nulls += 1
            assert at(written, pointer) is None, pointer
    assert nulls == 14, nulls


def walk_leaves(node, pointer=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk_leaves(value, pointer + (key,))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from walk_leaves(value, pointer + (index,))
    else:
        yield pointer, node


def at(node, pointer):
    for step in pointer:
        node = node[step]
    return node


# --- path independence, by construction ---------------------------------------------------------
#
# The record is built twice, both times through the constructors and never through
# `model_validate`, and the two builds state different fields. `_build` walks the fixture and the
# model graph together: a nested object becomes its model's constructor call, a list of objects
# becomes a list of them. With `state_defaults` it also passes the model's own default for a key
# the fixture omits, wherever the model admits that default; without it, a value that already
# equals its field default is left unstated. Both builds must give the fixture's dict, its
# canonical bytes and its digest.


def _model_in(annotation):
    stack = [annotation]
    while stack:
        item = stack.pop()
        if isinstance(item, type) and issubclass(item, BaseModel):
            return item
        stack.extend(get_args(item))
    return None


def _shape(annotation):
    """("model" | "list" | "dict", the model inside) or ("plain", None)."""
    stack = [annotation]
    while stack:
        item = stack.pop()
        if isinstance(item, type) and issubclass(item, BaseModel):
            return ("model", item)
        origin = get_origin(item)
        if origin is list and _model_in(item) is not None:
            return ("list", _model_in(item))
        if origin is dict and _model_in(item) is not None:
            return ("dict", _model_in(item))
        stack.extend(get_args(item))
    return ("plain", None)


def _admits_none(annotation):
    stack = [annotation]
    while stack:
        item = stack.pop()
        if item is type(None):
            return True
        stack.extend(get_args(item))
    return False


def refuses_its_own_default(model):
    """Optional fields declared non-nullable with a `None` default: strict mode refuses that None.

    `value: float = None` is the house idiom for "optional, and absent when unset". Under
    `strict=True` the default cannot be passed back in, so there is no construction path that
    states these fields the way a caller states the others. They are named here rather than
    quietly skipped.
    """
    out = []
    for name, field in model.model_fields.items():
        if field.is_required():
            continue
        default = field.get_default(call_default_factory=True)
        if default is None and not _admits_none(field.annotation):
            out.append(name)
    return out


def _build(model, data, *, state_defaults):
    kwargs = {}
    refused = set(refuses_its_own_default(model))
    for name, field in model.model_fields.items():
        alias = field.alias or name
        if alias in data:
            value = data[alias]
        elif name in data:
            value = data[name]
        else:
            if state_defaults and not field.is_required() and name not in refused:
                kwargs[name] = field.get_default(call_default_factory=True)
            continue
        if (
            not state_defaults
            and not field.is_required()
            and value == field.get_default(call_default_factory=True)
        ):
            continue
        kind, inner = _shape(field.annotation)
        if kind == "model" and isinstance(value, dict):
            value = _build(inner, value, state_defaults=state_defaults)
        elif kind == "list" and isinstance(value, list):
            value = [
                _build(inner, item, state_defaults=state_defaults) if isinstance(item, dict) else item
                for item in value
            ]
        elif kind == "dict" and isinstance(value, dict):
            value = {
                key: (_build(inner, item, state_defaults=state_defaults) if isinstance(item, dict) else item)
                for key, item in value.items()
            }
        kwargs[name] = value
    return model(**kwargs)


def _fields_set_by_path(obj, path="", out=None):
    out = {} if out is None else out
    if isinstance(obj, BaseModel):
        out[path] = set(obj.model_fields_set)
        for name in type(obj).model_fields:
            _fields_set_by_path(getattr(obj, name), f"{path}.{name}", out)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _fields_set_by_path(item, f"{path}[{i}]", out)
    elif isinstance(obj, dict):
        for key, item in obj.items():
            _fields_set_by_path(item, f"{path}.{key}", out)
    return out


def test_two_construction_paths_give_the_fixture_bytes_and_digest(reference, monkeypatch):
    """Canonical bytes are a function of the record's values, shown by building it two ways.

    This replaces a test that reached in and rewrote `__pydantic_fields_set__`, which proved a
    property of a record no caller can produce. Here both records are built the way an SDK caller
    builds one, through the constructors: `model_validate` is made to raise for the duration, so
    neither path can borrow the loaded record's field set. They differ in which optional fields
    they state, and under the old `exclude_unset` rule that difference would reach the bytes.
    """

    def refuse(*args, **kwargs):
        raise AssertionError("this path is meant to use the constructors, not model_validate")

    monkeypatch.setattr(BaseModel, "model_validate", classmethod(refuse))
    stated = _build(Assay, reference, state_defaults=True)
    unset = _build(Assay, reference, state_defaults=False)
    monkeypatch.undo()

    assert stated.to_dict() == reference
    assert unset.to_dict() == reference
    assert canonical_bytes(stated) == canonical_bytes(reference) == canonical_bytes(unset)
    assert digest(stated) == digest(reference) == digest(unset)

    left, right = _fields_set_by_path(stated), _fields_set_by_path(unset)
    differ = [path for path in left if left[path] != right.get(path)]
    assert differ, "the two paths set the same fields everywhere: the test would prove nothing"


# --- the one rule, in the two modules that used to read the object's history --------------------

NESTED_NODES = {
    "context": (("subject", "context"), ContextRef),
    "decision": (("decision",), Decision),
    "digests": (("subject", "digests"), Digests),
    "version": (("subject", "version"), VersionRef),
}


def _schema_node(schema, path):
    node = schema
    for step in path:
        node = node["properties"][step]
    return node


@pytest.mark.parametrize("label", sorted(NESTED_NODES))
def test_the_serialiser_follows_a_nested_required_list_changed_in_memory(label):
    """The omission set is the schema's, read at call time, not the model's own defaults.

    Moving a property out of a nested `required` list in memory has to move it into the set the
    serialiser may omit. If the rule were read off the model's defaults, or cached at import, the
    two specifications could drift without any test noticing.
    """
    path, model = NESTED_NODES[label]
    node = _schema_node(assay_schema(), path)
    kept = list(node["required"])
    by_alias = {(f.alias or n): n for n, f in model.model_fields.items()}
    nullable = [
        name
        for name in kept
        if name in by_alias and _admits_none(model.model_fields[by_alias[name]].annotation)
    ]
    assert nullable, f"{label}: no required nullable property to move"
    moved = nullable[0]
    assert moved not in omitted_when_none(model)
    try:
        node["required"] = [name for name in kept if name != moved]
        assert moved in omitted_when_none(model), f"{label}: the serialiser kept the old required list"
    finally:
        node["required"] = kept
    assert moved not in omitted_when_none(model)


def test_the_result_envelope_writes_every_field_of_1_0_even_when_it_is_null():
    """`exclude_unset` made the envelope's keys depend on which fields the caller passed.

    The envelope has no schema of its own, so its required set is its 1.0 field list, and the
    additive promise is that a reader written against 1.0 reads `result["error"]` and gets None
    rather than a KeyError.
    """
    out = ResultEnvelope().to_dict()
    assert set(out) == set(ResultEnvelope.model_fields)
    assert out["error"] is None
    assert out["pending"] is None
    assert out["subject"] is None
    assert ResultEnvelope(command="audit").to_dict().keys() == out.keys()
