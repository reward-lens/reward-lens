"""Structural controls for the live Seeker actor base class."""

from __future__ import annotations

import pytest

from reward_lens.instruments.seeker import (
    AgentActor,
    ApiActor,
    BaseActor,
    Budget,
    LocalActor,
    Pricing,
    RecordedTransport,
    ScoreResult,
    ScriptedActor,
    SeekerState,
)
from reward_lens.instruments.seeker.actors.agent import propose_tool


class FixedScorer:
    threshold = 0.9

    def score(self, task: dict, response: str) -> ScoreResult:
        return ScoreResult(reward=0.0, outcome_passed=True)


class FixedEndpoint:
    def complete(self, payload: dict) -> dict:
        return completion("local response")


def completion(text: str) -> dict:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def state() -> SeekerState:
    return SeekerState(task={"id": "task", "prompt": "Say hello"}, intent="hello")


def agent_caller(session: str):
    def call(payload: dict) -> dict:
        return propose_tool(
            {
                "session": session,
                "task_id": payload["task_id"],
                "response": "agent response",
            }
        )

    return call


def test_base_actor_is_not_instantiable() -> None:
    with pytest.raises(TypeError, match="abstract"):
        BaseActor(scorer=FixedScorer())


def test_subclass_without_propose_is_not_instantiable() -> None:
    class IncompleteActor(BaseActor):
        pass

    with pytest.raises(TypeError, match="abstract"):
        IncompleteActor(scorer=FixedScorer())


@pytest.mark.parametrize(
    "actor",
    [
        lambda: ScriptedActor(["scripted response"], scorer=FixedScorer()),
        lambda: ApiActor(
            RecordedTransport([completion("api response")]),
            scorer=FixedScorer(),
            budget=Budget("1.00"),
            pricing=Pricing("0.001", "0.001"),
        ),
        lambda: LocalActor(FixedEndpoint(), scorer=FixedScorer()),
        lambda: AgentActor(scorer=FixedScorer(), caller=agent_caller),
    ],
    ids=["scripted", "api", "local", "agent"],
)
def test_shipped_actors_remain_concrete_and_propose(actor) -> None:
    proposal = actor().propose(state())
    assert proposal.response.endswith("response")
