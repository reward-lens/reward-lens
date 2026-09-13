"""BLK-027: the tap leaves the per-token fields its own schema declares empty.

``record/turns.py`` declares four per-token fields on ``Turn`` and this adapter set none of
them: ``_turns_for`` returned exactly ``Turn(index=0, role="user", text=prompt)`` and
``Turn(index=1, role="assistant", text=completion)``. ``worlds/RECORDING.md`` §3.4 lists the
same per-token fields as contract rows, so the released artifact was short of its own stated
contract.

**One of the four is filled here and three are not**, and the three are named with their
obstacle rather than left silent. See ``_turns_for``'s docstring. This file asserts both halves:
that ``token_ids`` now arrives and round-trips, and that the other three are ``None`` rather than
an empty tuple, because an abstention and a measurement of zero are different claims.

The fixture is BLK-010's, reused deliberately: index 1 is one character and 211 tokens, so any
implementation reaching for ``len(text)`` fails every assertion.
"""

from __future__ import annotations

from test_blk010_completion_tokens import (  # noqa: E402
    COMPLETION_IDS,
    COMPLETION_TEXTS,
    GENEROUS,
    drive,
    rollouts,
)

from reward_lens.tap.adapters.trl import TRLTap


def _turns(run):
    return [t.turns for t in rollouts(run)]


def test_the_assistant_turn_carries_the_token_ids_it_was_given():
    """At the pinned SHA this fails: `token_ids` is None on every turn."""
    run = drive(TRLTap(run_id="blk027", budget=GENEROUS))
    turns = _turns(run)
    assert len(turns) == len(COMPLETION_IDS)
    for got, expected in zip(turns, COMPLETION_IDS):
        assistant = got[1]
        assert assistant.role == "assistant"
        assert assistant.token_ids is not None, (
            "Turn.token_ids is declared in record/turns.py and the tap left it None, "
            "which is BLK-027"
        )
        assert tuple(assistant.token_ids) == expected


def test_the_ids_are_the_count_and_not_the_character_length():
    run = drive(TRLTap(run_id="blk027b", budget=GENEROUS))
    got = [len(t[1].token_ids) for t in _turns(run)]
    assert got == [3, 211, 2, 5]
    assert got != [len(t) for t in COMPLETION_TEXTS], "a character proxy would give 40, 1, 3, 7"


def test_n_tokens_now_reports_the_real_count():
    """`Turn.n_tokens` was 0 on every record because token_ids was None."""
    run = drive(TRLTap(run_id="blk027c", budget=GENEROUS))
    assert [t[1].n_tokens for t in _turns(run)] == [3, 211, 2, 5]


def test_the_three_unreachable_fields_abstain_rather_than_report_zero():
    """An abstention and a measurement of zero are different claims.

    `loss_mask` and `logprobs_sampling` need the point before
    `shuffle_sequence_dict` (BLK-009's override); `logprobs_train` needs the
    `_compute_loss` override (BLK-004, P2-LOSS). None is reachable from a reward
    function, so all three stay None and none becomes an empty tuple.
    """
    run = drive(TRLTap(run_id="blk027d", budget=GENEROUS))
    for turn_pair in _turns(run):
        a = turn_pair[1]
        for field in ("loss_mask", "logprobs_sampling", "logprobs_train"):
            assert getattr(a, field) is None, (
                f"{field} is not reachable from a reward function and must abstain, "
                f"not report an empty sequence"
            )


def test_the_prompt_turn_abstains_because_prompt_ids_are_not_passed():
    run = drive(TRLTap(run_id="blk027e", budget=GENEROUS))
    for turn_pair in _turns(run):
        assert turn_pair[0].role == "user"
        assert turn_pair[0].token_ids is None


def test_the_switch_restores_the_old_behaviour_exactly():
    run = drive(TRLTap(run_id="blk027f", budget=GENEROUS, record_token_ids=False))
    for turn_pair in _turns(run):
        assert turn_pair[1].token_ids is None


def test_a_length_mismatch_makes_the_step_abstain_rather_than_mismatch_rows():
    """The same refusal `_completion_lengths_for` already makes."""
    run = drive(TRLTap(run_id="blk027g", budget=GENEROUS), completion_ids=COMPLETION_IDS[:2])
    for turn_pair in _turns(run):
        assert turn_pair[1].token_ids is None, (
            "a shorter completion_ids means a different batch, and putting ids on the "
            "rows it can match would attach them to the wrong rollout"
        )
