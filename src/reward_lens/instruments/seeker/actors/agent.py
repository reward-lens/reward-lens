"""The agent actor: the caller proposes through `seeker.propose` and reads its score back.

This is D-71's fourth actor, the one that costs the project nothing because the agent is already
running on someone else's budget. `propose_tool` is the function P-MCP binds as the MCP tool
`seeker.propose`; the tool surface belongs to that packet and nothing here builds it. What is built
here is the thing behind it, and the actor reaches it through the SDK's own dispatch seam
(`reward_lens.api._dispatch.load`), which is how a caller with no key gets a score back.

A session is how a stateless tool call finds the search it belongs to. The registry is keyed by an
opaque id, an unknown id is a refusal rather than a new session, and a session is dropped when its
actor is garbage collected.
"""

from __future__ import annotations

import secrets
from typing import Any, Callable, MutableMapping
from weakref import WeakValueDictionary

from ..budget import Budget
from ..protocol import BaseActor, Proposal, Scorer, SeekerState

__all__ = ["SESSIONS", "AgentActor", "propose_tool"]

#: Opaque session id -> the actor waiting on a proposal. Weak, so a finished search is not held.
SESSIONS: MutableMapping[str, "AgentActor"] = WeakValueDictionary()

#: The dispatch target a caller resolves through the SDK seam, written once so both sides agree.
TOOL_TARGET = "reward_lens.instruments.seeker.actors.agent:propose_tool"


def propose_tool(payload: dict) -> dict:
    """`seeker.propose`: take a response for the session's current task, return its score.

    The reply is the score the agent reads back, plus the proposal id under which the record now
    carries it. A payload naming no live session is refused with RL0003 naming the field, which is
    wave 1's reserved input-contract refusal and is imported rather than reallocated.
    """
    session = SESSIONS.get(str(payload.get("session", "")))
    if session is None:
        return {
            "error": {
                "code": "RL0003",
                "message": (
                    "the value given for session is not a seeker session that is waiting for a "
                    "proposal"
                ),
                "remediation": (
                    "propose into the session id the seeker handed you with the task; a session "
                    "ends when its search does"
                ),
                "field": "session",
            }
        }
    return session.offer(
        task_id=str(payload.get("task_id", "")),
        response=str(payload.get("response", "")),
        rationale=str(payload.get("rationale", "")),
    )


class AgentActor(BaseActor):
    """The calling agent, proposing through the tool and reading its own score back."""

    name = "agent"

    def __init__(
        self,
        *,
        scorer: Scorer,
        caller: Callable[[str], Callable[[dict], dict]] | Callable[[dict], dict],
        budget: Any = None,
        session: str | None = None,
    ) -> None:
        super().__init__(scorer=scorer, budget=budget if budget is not None else Budget(None))
        self.session = session or secrets.token_hex(8)
        bound = caller(self.session)  # type: ignore[arg-type]
        self._call: Callable[[dict], dict] = bound if callable(bound) else caller  # type: ignore
        self._pending: SeekerState | None = None
        SESSIONS[self.session] = self

    # --- the tool's side ----------------------------------------------------------------------

    def offer(self, *, task_id: str, response: str, rationale: str = "") -> dict:
        """What `seeker.propose` does: seal the proposal, score it, and hand the score back."""
        state = self._pending
        if state is None:
            return {
                "error": {
                    "code": "RL0003",
                    "message": "the value given for session is not waiting for a proposal",
                    "remediation": "propose when the seeker asks you for a response",
                    "field": "session",
                }
            }
        proposal = self.seal(
            state,
            response,
            inputs={"via": "seeker.propose", "session": self.session, "offered_for": task_id},
            rationale=rationale or "proposed by the calling agent through seeker.propose",
            cost_usd="0.00",
            calls=0,
        )
        scored = self.score(proposal)
        return {
            "proposal_id": proposal.proposal_id,
            "reward": scored.reward,
            "outcome_passed": scored.outcome_passed,
            "verdict": scored.verdict,
            "threshold": scored.threshold,
            "exploit": scored.exploit,
        }

    # --- the actor's side ---------------------------------------------------------------------

    def propose(self, state: SeekerState) -> Proposal:
        from reward_lens.api import _dispatch

        if _dispatch.load(TOOL_TARGET) is None:  # pragma: no cover - the module is in this wheel
            raise RuntimeError(f"the SDK seam does not resolve {TOOL_TARGET}")
        self._pending = state
        reply = self._call(
            {
                "session": self.session,
                "task_id": state.task_id,
                "prompt": state.prompt,
                "intent": state.intent,
                "attempts": [s.to_dict() for s in state.attempts],
                "budget_remaining_usd": state.budget_remaining_usd,
                "tool": "seeker.propose",
            }
        )
        self._pending = None
        if "error" in reply:
            raise RuntimeError(str(reply["error"].get("message", "the agent's proposal failed")))
        wanted = str(reply.get("proposal_id", ""))
        for proposal in reversed(self._proposals):
            if proposal.proposal_id == wanted:
                return proposal
        raise RuntimeError(  # pragma: no cover - the tool seals every proposal it answers
            f"the tool answered with proposal {wanted}, which it did not seal"
        )
