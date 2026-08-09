"""The scripted actor: the conformance reference, and the only one the tests depend on.

It costs nothing, reaches nothing and is deterministic, so it is what every other actor's record is
compared against. A policy is either a callable over the state or a fixed sequence of responses;
both are fixtures, and neither is a model.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from ..protocol import BaseActor, Proposal, Scorer, SeekerState

__all__ = ["ScriptedActor"]


class ScriptedActor(BaseActor):
    """A scripted policy fixture behind the actor protocol."""

    name = "scripted"

    def __init__(
        self,
        policy: Callable[[SeekerState], str] | Sequence[str],
        *,
        scorer: Scorer,
        budget: Any = None,
    ) -> None:
        super().__init__(scorer=scorer, budget=budget)
        self._policy = policy
        self._policy_name = (
            getattr(policy, "__name__", None) if callable(policy) else "fixed_sequence"
        ) or "policy"

    def propose(self, state: SeekerState) -> Proposal:
        if callable(self._policy):
            response = str(self._policy(state))
        else:
            rows = list(self._policy)
            response = str(rows[min(len(self._proposals), len(rows) - 1)])
        return self.seal(
            state,
            response,
            inputs={"policy": self._policy_name},
            rationale="scripted policy fixture",
        )
