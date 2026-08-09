"""The seeker: the black-box arm of the audit, as an actor protocol with four actors (D-71, D-41).

The instrument does one thing. It takes a proposed response for a task, scores it under the audited
reward, runs the independent check, and records the proposal, the score and the outcome with the
actor that proposed it. Four actors implement the protocol and differ only in where the text of a
proposal comes from: a scripted policy for the tests, an API model under `--max-budget-usd`, a
local model through an endpoint `doctor` found, and the agent that is calling the MCP server.

Two properties hold whichever actor ran. Every proposal is recorded with the inputs it was made
from, and every finding is reproduced from those inputs before it enters the record. "Not found" is
recorded as "not found at this budget by this actor", never as "no exploit". And the arm is off
unless it is asked for: a run with the seeker off is a complete audit that says the arm was not
run, which is an entry in the record rather than a gap in it.
"""

from __future__ import annotations

from .actors import (
    SESSIONS,
    AgentActor,
    ApiActor,
    Completion,
    Endpoint,
    LocalActor,
    RecordedTransport,
    ScriptedActor,
    Transport,
    propose_tool,
)
from .budget import Budget, BudgetExhausted, Pricing, SeekerRequiresBudget, money
from .panel import FINDING_ID, METHOD_VERSION, PANEL, ExploitSearch, findings_for
from .protocol import (
    DEFAULT_THRESHOLD,
    NOT_FOUND,
    Actor,
    ActorRecord,
    BaseActor,
    FindingNotReplayable,
    GraderScorer,
    Proposal,
    Scored,
    ScoreResult,
    Scorer,
    SeekerState,
    is_exploit,
    scorer_for,
)
from .search import replay, rescore, seek

from . import panel  # noqa: F401  the audit's own tests reach for it by name

__all__ = [
    "Actor",
    "ActorRecord",
    "AgentActor",
    "ApiActor",
    "BaseActor",
    "Budget",
    "BudgetExhausted",
    "Completion",
    "DEFAULT_THRESHOLD",
    "Endpoint",
    "ExploitSearch",
    "FINDING_ID",
    "FindingNotReplayable",
    "GraderScorer",
    "LocalActor",
    "METHOD_VERSION",
    "NOT_FOUND",
    "PANEL",
    "Pricing",
    "Proposal",
    "RecordedTransport",
    "SESSIONS",
    "ScoreResult",
    "Scored",
    "Scorer",
    "ScriptedActor",
    "SeekerRequiresBudget",
    "SeekerState",
    "Transport",
    "findings_for",
    "is_exploit",
    "money",
    "propose_tool",
    "replay",
    "rescore",
    "scorer_for",
    "seek",
]
