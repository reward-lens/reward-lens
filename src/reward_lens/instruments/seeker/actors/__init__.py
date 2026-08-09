"""The four actors of D-71, each behind the same three methods."""

from __future__ import annotations

from .agent import SESSIONS, AgentActor, propose_tool
from .api import ApiActor, Completion, RecordedTransport, Transport
from .local import Endpoint, LocalActor
from .scripted import ScriptedActor

__all__ = [
    "AgentActor",
    "ApiActor",
    "Completion",
    "Endpoint",
    "LocalActor",
    "RecordedTransport",
    "SESSIONS",
    "ScriptedActor",
    "Transport",
    "propose_tool",
]
