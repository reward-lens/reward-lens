"""`make` builds the contracts' error type, with the right exit code and the golden shape."""

from __future__ import annotations

import json
import re

import pytest

from reward_lens.contracts import (
    BudgetExceeded,
    CapabilityUnavailable,
    InternalFailure,
    PendingDecision,
    RewardLensError,
    UsageError,
)
from reward_lens.errors import CATALOGUE, explain, make, render_two_line

# This public literal protects the rendered error contract without a private transcript.
TYPE_FOR_EXIT = {
    3: PendingDecision,
    4: UsageError,
    5: CapabilityUnavailable,
    6: BudgetExceeded,
    7: InternalFailure,
}


def test_error_render_has_stable_three_line_shape() -> None:
    assert render_two_line(make("RL0341", path="./tests")) == (
        "error: the outcome check at ./tests collected 0 tests, so it cannot qualify a correctness claim\n"
        "help:  check the path, or run: reward-lens doctor --outcome ./tests\n"
        "       reward-lens explain RL0341"
    )

def test_make_builds_the_type_that_matches_the_exit_code() -> None:
    for code, spec in CATALOGUE.items():
        error = make(code)
        assert isinstance(error, RewardLensError), code
        assert error.code == code
        assert error.exit_code == spec.exit_code, code
        assert type(error) is TYPE_FOR_EXIT.get(spec.exit_code, RewardLensError), code


def test_make_carries_the_first_remedy_as_the_remediation() -> None:
    for code, spec in CATALOGUE.items():
        error = make(code)
        assert error.remediation, code
        assert error.message, code


def test_make_interpolates_the_context_it_is_given() -> None:
    error = make("RL0341", path="./suite")
    assert "./suite" in error.message
    assert "./suite" in error.remediation
    assert error.context["path"] == "./suite"


def test_a_missing_context_key_renders_as_a_placeholder_and_never_raises() -> None:
    error = make("RL0341")
    assert "<path>" in error.message


def test_the_json_twin_carries_exactly_three_fields() -> None:
    payload = json.loads(make("RL0341", path="./tests").to_json())
    assert set(payload) == {"code", "message", "remediation"}
    assert payload["code"] == "RL0341"


def test_a_pending_decision_keeps_its_extra_fields() -> None:
    error = make("RL0801", decision="which policy to apply")
    assert isinstance(error, PendingDecision)
    assert error.exit_code == 3


@pytest.mark.parametrize("bad", ["RL9999", "", "rl0341", "RL341", None, 341])
def test_make_refuses_a_code_it_does_not_hold(bad: object) -> None:
    with pytest.raises(RewardLensError) as caught:
        make(bad)  # type: ignore[arg-type]
    assert caught.value.code == "RL0001"
    assert caught.value.exit_code == 4


def test_every_rendered_error_is_three_lines() -> None:
    for code in CATALOGUE:
        lines = render_two_line(make(code)).splitlines()
        assert len(lines) == 3, code
        assert lines[0].startswith("error: ")
        assert lines[1].startswith("help:  ")
        assert lines[2] == f"       reward-lens explain {code}"


def test_rl0703_names_the_verb_it_refuses_and_explains_it_with_no_slot_left() -> None:
    # The refusal A-026 puts in front of the parser for a verb whose engine is not in the build.
    # The caller passes the verb and the program name; the long form is read by someone who has
    # no invocation in hand, so it names no verb and leaves nothing to decode.
    error = make("RL0703", verb="trace", command="reward-lens")
    assert error.exit_code == 5
    assert "trace" in error.message
    lines = render_two_line(error).splitlines()
    assert len(lines) == 3
    assert (
        lines[0] == "error: `trace` is not in this build, so this invocation cannot be carried out"
    )
    assert lines[1].startswith("help:  ")
    assert lines[2] == "       reward-lens explain RL0703"

    long_form = explain("RL0703")
    assert "trace" not in long_form
    assert not re.search(r"\{[A-Za-z_][A-Za-z0-9_]*\}", long_form)
    assert not re.search(r"<[A-Za-z_][A-Za-z0-9_]*>", long_form)
