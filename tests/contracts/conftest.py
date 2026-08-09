"""Shared paths and loaders for the P-CONTRACT suite.

Everything here is read-only: the frozen schema, the reference instance, the invalid corpus and its
INDEX. The suite never writes into `schema/assay/1.0/` except the vector files it owns.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "schema" / "assay" / "1.0"
SCHEMA_PATH = SCHEMA_DIR / "assay.schema.json"
FIXTURES = SCHEMA_DIR / "fixtures"
INVALID = FIXTURES / "invalid"
VALID = FIXTURES / "valid"
VECTOR = FIXTURES / "vector.json"
VECTOR_DIGEST = FIXTURES / "vector.sha256"
PROJECT_SCHEMA = ROOT / "schema" / "project" / "1.0" / "rewardlens.schema.json"
CONTRACTS_SRC = ROOT / "src" / "reward_lens" / "contracts"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def invalid_index() -> dict:
    return load(INVALID / "INDEX.json")


INDEX = invalid_index()
#: The corpus is parametrised from the files on disk, never from the INDEX's own keys. A fixture
#: added without an INDEX row would otherwise be a file every test walks past: the row is what
#: names the rule and the layer, so a fixture without one is untested, and an INDEX row without a
#: fixture is a rule nothing demonstrates. `test_the_invalid_corpus_and_its_index_are_the_same_set`
#: is where either shows up by name.
INVALID_FILES = sorted(q.stem for q in INVALID.glob("*.json") if q.stem != "INDEX")
INVALID_NAMES = INVALID_FILES
ORPHAN_FIXTURES = [name for name in INVALID_FILES if name not in INDEX]
INDEXED_WITHOUT_FIXTURE = sorted(name for name in INDEX if name not in set(INVALID_FILES))
#: Every record under `fixtures/valid/` is one the specification admits, beside the reference.
VALID_NAMES = sorted(q.stem for q in VALID.glob("*.json")) if VALID.is_dir() else []
SCHEMA_LAYER = [n for n in INVALID_NAMES if INDEX.get(n, {}).get("caught_by") == "schema"]
QUANTISER_LAYER = [n for n in INVALID_NAMES if INDEX.get(n, {}).get("caught_by") == "quantiser"]


@pytest.fixture(scope="session")
def hand_schema() -> dict:
    return load(SCHEMA_PATH)


@pytest.fixture(scope="session")
def reference() -> dict:
    return load(FIXTURES / "reference.json")


@pytest.fixture(scope="session")
def rs_validator(hand_schema):
    import jsonschema_rs

    return jsonschema_rs.validator_for(hand_schema)


def invalid_instance(name: str) -> dict:
    return load(INVALID / f"{name}.json")


def valid_instance(name: str) -> dict:
    return load(VALID / (name + ".json"))
