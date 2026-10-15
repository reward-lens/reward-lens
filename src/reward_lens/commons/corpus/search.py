"""Context-safe corpus search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from .projection import CaseProjection, project
from .store import CorpusStore

__all__ = ["ActorContext", "SearchResult", "SearchStatus", "search"]


class ActorContext(StrEnum):
    """The explicit public contexts that may use corpus search."""

    CANDIDATE = "candidate"
    REVIEWER = "reviewer"


class SearchStatus(StrEnum):
    MATCHES = "MATCHES"
    NO_MATCHES = "NO_MATCHES"
    PROTECTED_ACCEPTANCE_MAPPING_BLOCKED = "PROTECTED_ACCEPTANCE_MAPPING_BLOCKED"


@dataclass(frozen=True)
class SearchResult:
    status: SearchStatus
    matches: tuple[CaseProjection, ...]
    reason: str | None = None
    provenance: Mapping[str, object] | None = None


def _asks_for_protected_acceptance_mapping(query: str) -> bool:
    words = set(query.lower().replace("-", " ").replace("_", " ").split())
    return "acceptance" in words and "mapping" in words


def search(
    query: str, *, actor_context: ActorContext, corpus: CorpusStore | None = None
) -> SearchResult:
    """Search public projections, refusing an acceptance-mapping request before inspection."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("search query must be a non-empty string")
    if not isinstance(actor_context, ActorContext):
        raise ValueError("actor_context must be an explicit ActorContext")
    if corpus is None or not isinstance(corpus, CorpusStore):
        raise TypeError("search requires a CorpusStore")
    if _asks_for_protected_acceptance_mapping(query):
        return SearchResult(
            status=SearchStatus.PROTECTED_ACCEPTANCE_MAPPING_BLOCKED,
            matches=(),
            reason=SearchStatus.PROTECTED_ACCEPTANCE_MAPPING_BLOCKED.value,
        )
    terms = tuple(part for part in query.lower().split() if part)
    matches: list[CaseProjection] = []
    for case in corpus.all():
        candidate = project(case)
        haystack = json.dumps(candidate.fields, sort_keys=True).lower()
        if any(term in haystack for term in terms):
            matches.append(candidate)
    if not matches:
        return SearchResult(status=SearchStatus.NO_MATCHES, matches=())
    return SearchResult(status=SearchStatus.MATCHES, matches=tuple(matches))
