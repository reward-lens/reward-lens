"""The two whole-record halves of the D-66 parity gate, which do not fit the module's budget.

A-003 step 3. Generation from the *whole* assay record costs what generation from its parts does
not. Measured on this worktree with `.venv/bin/python` (3.11.13), one example of the whole-record
strategy cost 36.7s and ten cost more than 190s. Three edits to the generation schema fixed the
per-`$def` tests, which went from over 190s for one of the nine to 1.6s for all nine: patterns
pinned to legal values, `if`/`then`/`else`/`not` dropped (`hypothesis-jsonschema` cannot invert
them, so it filters, and with ten conditionals in the record almost every draw is thrown away),
and `x-scale` numbers bounded to the range where the schema and the models agree. The same three
edits apply here and the whole-record strategy is still over 150s for ten examples: what is left
is the size of the record itself, an object of objects of arrays with twenty `oneOf` branches in
it, and no edit short of weakening the gate makes it cheap.

So these two run on request rather than on every invocation, with nothing marked `xfail` and
nothing deleted:

    REWARD_LENS_SLOW=1 PYTHONPATH=src .venv/bin/python -m pytest -q tests/contracts/test_parity_slow.py

The timing of that run is in `fleet/HANDOFFS/P-CONTRACT.md` under `## Attempt 2b`. The rest of the
gate, including both generative halves at the level of every `$def` and both shapes of D-75's
uncertainty, runs in `test_parity.py` on every invocation.
"""

from __future__ import annotations

import os

import pytest

if os.environ.get("REWARD_LENS_SLOW") != "1":
    pytest.skip(
        "whole-record generation; set REWARD_LENS_SLOW=1", allow_module_level=True
    )

from hypothesis import given  # noqa: E402
from hypothesis_jsonschema import from_schema  # noqa: E402

from structure import (  # noqa: E402
    bound_quantised_numbers,
    pin_patterns,
    relax_conditionals,
    strip_annotations,
)
from test_parity import GENERATION, model_accepts  # noqa: E402

from reward_lens.contracts import Assay  # noqa: E402


def for_generation(raw: dict) -> dict:
    """The four edits `test_parity.py` makes, each defended where it is defined."""
    return relax_conditionals(pin_patterns(strip_annotations(bound_quantised_numbers(raw))))


@pytest.fixture(scope="session")
def hand_generation_schema(hand_schema):
    return for_generation(hand_schema)


@pytest.fixture(scope="session")
def model_generation_schema():
    return for_generation(Assay.model_json_schema(mode="validation"))


def test_generated_from_the_hand_schema_gets_the_same_verdict(
    rs_validator, hand_generation_schema
):
    @GENERATION
    @given(from_schema(hand_generation_schema))
    def check(instance):
        assert rs_validator.is_valid(instance) == model_accepts(instance)

    check()


def test_generated_from_the_model_schema_gets_the_same_verdict(
    rs_validator, model_generation_schema
):
    @GENERATION
    @given(from_schema(model_generation_schema))
    def check(instance):
        assert rs_validator.is_valid(instance) == model_accepts(instance)

    check()
