"""A panel registers its membership and its estimand before it scores anything (D-40)."""

from __future__ import annotations

import pytest

from reward_lens.contracts.errors import UsageError
from reward_lens.outcome import Panel, PanelNotRegistered

MEMBERS = ("rater-a", "rater-b", "rater-c")
ESTIMAND = "the share of responses a majority of the panel calls correct"


def test_scoring_before_registering_refuses(panel_id: str = "panel-1") -> None:
    panel = Panel(panel_id)
    with pytest.raises(PanelNotRegistered) as caught:
        panel.score(item_id="item-1", member="rater-a", label=1.0)
    assert caught.value.code == "RL0001"
    assert caught.value.exit_code == 4
    assert "estimand" in caught.value.message
    assert caught.value.context["panel_id"] == "panel-1"
    assert panel.scores() == ()


def test_the_refusal_is_a_usage_error_the_cli_can_exit_on() -> None:
    assert issubclass(PanelNotRegistered, UsageError)


def test_registering_then_scoring_works() -> None:
    panel = Panel("panel-1")
    registration = panel.register(MEMBERS, ESTIMAND)
    assert registration.membership == MEMBERS
    assert registration.estimand == ESTIMAND
    assert registration.digest.startswith("sha256:")
    score = panel.score(item_id="item-1", member="rater-a", label=1.0)
    assert score.registration_digest == registration.digest
    assert panel.scores() == (score,)


def test_the_registration_digest_covers_the_membership_and_the_estimand() -> None:
    one = Panel("panel-1").register(MEMBERS, ESTIMAND)
    same = Panel("panel-1").register(MEMBERS, ESTIMAND)
    different_members = Panel("panel-1").register(MEMBERS[:2], ESTIMAND)
    different_estimand = Panel("panel-1").register(MEMBERS, "the mean label")
    assert one.digest == same.digest
    assert one.digest != different_members.digest
    assert one.digest != different_estimand.digest


def test_a_member_outside_the_registered_membership_refuses() -> None:
    panel = Panel("panel-1")
    panel.register(MEMBERS, ESTIMAND)
    with pytest.raises(UsageError) as caught:
        panel.score(item_id="item-1", member="rater-z", label=1.0)
    assert "rater-z" in caught.value.message


def test_re_registering_after_scoring_refuses_rather_than_backfilling() -> None:
    panel = Panel("panel-1")
    first = panel.register(MEMBERS, ESTIMAND)
    panel.score(item_id="item-1", member="rater-a", label=1.0)
    with pytest.raises(UsageError) as caught:
        panel.register(MEMBERS, "the mean label")
    assert "already scored" in caught.value.message
    assert panel.registration is not None
    assert panel.registration.digest == first.digest


def test_re_registering_before_any_score_is_allowed() -> None:
    panel = Panel("panel-1")
    panel.register(MEMBERS, ESTIMAND)
    second = panel.register(MEMBERS, "the mean label")
    assert panel.registration is not None
    assert panel.registration.digest == second.digest


def test_the_registration_is_timestamped_before_the_first_score() -> None:
    panel = Panel("panel-1")
    registration = panel.register(MEMBERS, ESTIMAND)
    score = panel.score(item_id="item-1", member="rater-a", label=0.5)
    assert registration.registered_at <= score.scored_at
