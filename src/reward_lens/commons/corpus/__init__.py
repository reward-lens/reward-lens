"""Public case-corpus API.  Planner protocol and seed remain separately owned."""

from .case import Case, RecordSource
from .schema import CaseConclusion, CaseField
from .search import ActorContext, SearchResult, SearchStatus, search
from .store import CorpusStore

__all__ = [
    "ActorContext",
    "Case",
    "CaseConclusion",
    "CaseField",
    "CorpusStore",
    "RecordSource",
    "SearchResult",
    "SearchStatus",
    "search",
]


class Corpus:
    """Public case lookup over the accepted seed or a source-bound local store.

    Planning meaning for saved cases is explicit through PlanningAnnotation.
    Lookup never uploads records or exposes their private source paths.
    """

    def __init__(self, store=None, *, planning_annotations=None, seed_path=None):
        from pathlib import Path

        if store is not None and not isinstance(store, CorpusStore):
            raise TypeError("Corpus requires a CorpusStore or the packaged seed")
        if store is not None and seed_path is not None:
            raise ValueError("Corpus accepts either a store or a seed path")
        if store is None and planning_annotations is not None:
            raise ValueError("planning annotations require a CorpusStore")
        self.store = store
        self.seed_path = None if seed_path is None else Path(seed_path)
        self.planning_annotations = (
            None if planning_annotations is None else tuple(planning_annotations)
        )

    def _dataset(self):
        """Reopen source identities, retaining only digests in public provenance."""
        import hashlib
        from dataclasses import replace
        from pathlib import Path

        from reward_lens.contracts import digest

        from .projection import CaseProjection
        from .protocol import load_planner_cases

        if self.store is None:
            source = (
                self.seed_path
                or Path(__file__).with_name("seed") / "planner" / "planner_cases.json"
            )
            # Hash the exact bytes that the protocol parses, and detect an intervening change.
            before = source.read_bytes()
            cases = load_planner_cases(self.seed_path)
            if source.read_bytes() != before:
                raise ValueError("corpus seed changed during lookup")
            source_digest = "sha256:" + hashlib.sha256(before).hexdigest()
            projections = tuple(
                CaseProjection(
                    case.id,
                    {
                        "explanation_kinds": list(case.explanation_kinds),
                        "rung": case.rung,
                        "analogy": case.analogy,
                    },
                )
                for case in cases
            )
            cases = tuple(
                replace(case, case_digest=digest(projection.fields), source_digest=source_digest)
                for case, projection in zip(cases, projections, strict=True)
            )
            provenance = {
                "kind": "accepted_seed" if self.seed_path is None else "registered_seed",
                "source_digest": source_digest,
                "dataset_digest": digest(
                    {
                        "cases": [
                            {
                                "id": case.id,
                                "case_digest": case.case_digest,
                                "source_digest": case.source_digest,
                            }
                            for case in cases
                        ]
                    }
                ),
            }
        else:
            from .projection import project

            saved = self.store.all()
            projections = tuple(project(case) for case in saved)
            cases = (
                ()
                if self.planning_annotations is None
                else load_planner_cases(corpus=self.store, annotations=self.planning_annotations)
            )
            provenance = {
                "kind": "registered_store",
                "dataset_digest": digest(
                    {
                        "cases": [
                            {
                                "id": case.id,
                                "case_digest": digest(case.to_dict()),
                                "source_digest": case.source.assay_id,
                            }
                            for case in saved
                        ],
                        "planning_annotations": []
                        if self.planning_annotations is None
                        else [annotation.payload() for annotation in self.planning_annotations],
                    }
                ),
            }
        return projections, cases, provenance

    def cases_like(self, query: str, *, actor_context: ActorContext) -> SearchResult:
        """Find public lexical matches; return a named nonresult before protected reads."""
        import json

        from .search import _asks_for_protected_acceptance_mapping

        if not isinstance(query, str) or not query.strip():
            raise ValueError("cases_like query must be a non-empty string")
        if not isinstance(actor_context, ActorContext):
            raise ValueError("actor_context must be an explicit ActorContext")
        if _asks_for_protected_acceptance_mapping(query):
            return SearchResult(
                SearchStatus.PROTECTED_ACCEPTANCE_MAPPING_BLOCKED,
                (),
                SearchStatus.PROTECTED_ACCEPTANCE_MAPPING_BLOCKED.value,
            )
        projections, _cases, provenance = self._dataset()
        if self.store is not None:
            # Use the existing Wave 6 safe public projection/search route.
            result = search(query, actor_context=actor_context, corpus=self.store)
            # A source may not change between sealed provenance and the public search.
            if self._dataset()[2] != provenance:
                raise ValueError("corpus dataset changed during lookup")
            return SearchResult(result.status, result.matches, result.reason, provenance)
        terms = tuple(query.lower().split())
        matches = tuple(
            item
            for item in projections
            if any(term in json.dumps(item.fields, sort_keys=True).lower() for term in terms)
        )
        return SearchResult(
            SearchStatus.MATCHES if matches else SearchStatus.NO_MATCHES,
            matches,
            provenance=provenance,
        )


__all__.append("Corpus")
