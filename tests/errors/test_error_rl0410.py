"""RL0410 has two surfaces, and which one applies turns on whose limit was breached.

A breached execution limit is only reward-lens's failure when the limit was reward-lens's own. A
graded process that runs out of time or memory has told us something about that process, which is
a verdict the record carries, not a defect in the tool that measured it. Exiting 7 on it would
report our own failure every time a graded process was slow.
"""

from __future__ import annotations

import typing

from reward_lens.contracts.models import Verdict
from reward_lens.errors import CATALOGUE, explain, make
from reward_lens.errors.catalogue import surface_for

SPEC = CATALOGUE["RL0410"]


def test_an_internal_limit_is_an_error_and_exits_seven() -> None:
    assert surface_for("RL0410", internal=True) == ("error", 7)
    error = make("RL0410", dimension="wall clock")
    assert error.code == "RL0410"
    assert error.exit_code == 7
    assert "wall clock" in error.message
    assert "reward-lens reached its own" in error.message


def test_a_limit_the_graded_process_hit_is_a_verdict_and_has_no_exit() -> None:
    surface, exit_code = surface_for("RL0410", internal=False)
    assert surface == "verdict"
    assert exit_code is None, "a verdict on the graded process is not a status the tool exits with"
    assert SPEC.verdicts == ("timeout", "resource_exhausted")
    assert set(SPEC.verdicts) <= set(typing.get_args(Verdict)), (
        "the verdicts have to be the record's own vocabulary, not a second one invented here"
    )


def test_a_code_with_one_surface_is_an_error_whoever_breached_it() -> None:
    for code, spec in CATALOGUE.items():
        if spec.verdicts:
            continue
        assert surface_for(code, internal=False) == ("error", spec.exit_code), code
        assert surface_for(code, internal=True) == ("error", spec.exit_code), code


def test_rl0410_is_the_only_code_with_a_second_surface() -> None:
    assert [code for code, spec in CATALOGUE.items() if spec.verdicts] == ["RL0410"]


def test_explain_says_which_surface_is_which() -> None:
    text = explain("RL0410")
    assert SPEC.verdict_rule in text
    assert "timeout" in text and "resource_exhausted" in text
    assert "reward-lens exits 7" in text


def test_the_cause_is_the_internal_limit_and_not_the_graded_process() -> None:
    # The cause is the text of the error surface, and the error surface is the internal one. The
    # attempt-1 entry read "the graded process hit the ... limit" while carrying exit 7, which
    # blamed reward-lens for the graded code's behaviour.
    assert "reward-lens reached its own" in SPEC.cause
    assert "the graded process hit" not in SPEC.cause
