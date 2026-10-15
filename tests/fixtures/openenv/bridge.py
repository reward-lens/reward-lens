"""In-process OpenEnv 0.6.0 fixture: it never opens a socket or starts a provider."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server import Action, Observation, State


class Submit(Action):
    prompt: str
    response: str


class KnownObservation(Observation):
    value: str


class FixtureState(State):
    pass


class LocalKnownScoreClient(EnvClient[Submit, KnownObservation, FixtureState]):
    def __init__(
        self, *, base_url: str, connect_timeout_s: float, message_timeout_s: float
    ) -> None:
        super().__init__(
            base_url=base_url,
            connect_timeout_s=connect_timeout_s,
            message_timeout_s=message_timeout_s,
        )
        self.connected = 0
        self.closed = 0
        self.reset_calls: list[dict[str, Any]] = []
        self.step_calls: list[Submit] = []
        self.reward: float | None = 0.75
        self.done = True

    async def _connect_async(self):
        self.connected += 1
        return self

    async def _close_async(self) -> None:
        self.closed += 1

    async def _reset_async(self, **kwargs: Any) -> StepResult[KnownObservation]:
        self.reset_calls.append(dict(kwargs))
        return StepResult(KnownObservation(value="reset"), reward=None, done=False)

    async def _step_async(self, action: Submit, **kwargs: Any) -> StepResult[KnownObservation]:
        assert isinstance(action, Submit)
        assert not kwargs
        self.step_calls.append(action)
        return StepResult(KnownObservation(value="scored"), reward=self.reward, done=self.done)

    def _step_payload(self, action: Submit) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[KnownObservation]:
        raise AssertionError("fixture overrides _step_async")

    def _parse_state(self, payload: dict[str, Any]) -> FixtureState:
        return FixtureState()


LAST_CLIENT: LocalKnownScoreClient | None = None


def make_client(
    *, base_url: str, connect_timeout_s: float, message_timeout_s: float
) -> LocalKnownScoreClient:
    global LAST_CLIENT
    LAST_CLIENT = LocalKnownScoreClient(
        base_url=base_url, connect_timeout_s=connect_timeout_s, message_timeout_s=message_timeout_s
    )
    return LAST_CLIENT


def reset_from_task(task: dict) -> dict[str, object]:
    return {"seed": task["seed"], "episode_id": task["id"]}


def action_from_task_response(task: dict, response: str | dict) -> Submit:
    return Submit(prompt=task["prompt"], response=str(response))
