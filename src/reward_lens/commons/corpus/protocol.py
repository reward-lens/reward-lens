"""Narrow, immutable prior-case protocol used by the improve planner.

This is deliberately smaller than the Wave 6 corpus.  It gives the planner a
nameable case, the explanations it actually resembles, and the analogy that
may later survive or fail.  It never supplies a likelihood or a planning score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from reward_lens.contracts import digest

from .store import CorpusStore

__all__ = ["PlannerCase", "PlanningAnnotation", "load_planner_cases", "consult_corpus"]


@dataclass(frozen=True)
class PlannerCase:
    """A prior experiment that can be cited, checked, and later contradicted."""

    id: str
    explanation_kinds: tuple[str, ...]
    rung: int
    analogy: str
    case_digest: str | None = None
    source_digest: str | None = None
    annotation_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise TypeError("planner case id must be a non-empty string")
        if len(self.explanation_kinds) < 2 or any(
            not isinstance(kind, str) or not kind for kind in self.explanation_kinds
        ):
            raise ValueError("planner case needs at least two non-empty explanation kinds")
        if len(set(self.explanation_kinds)) != len(self.explanation_kinds):
            raise ValueError("planner case explanation kinds must be unique")
        if type(self.rung) is not int or self.rung not in range(1, 7):
            raise ValueError("planner case rung must be an integer from 1 through 6")
        if not isinstance(self.analogy, str) or not self.analogy:
            raise TypeError("planner case analogy must be a non-empty string")
        for field, value in (
            ("case_digest", self.case_digest),
            ("source_digest", self.source_digest),
            ("annotation_digest", self.annotation_digest),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71
            ):
                raise ValueError(f"planner case {field} must be a sha256 digest when present")


@dataclass(frozen=True)
class PlanningAnnotation:
    """Caller-declared planner meaning for one saved Case, with source identity pinned.

    The corpus schema intentionally has no explanation-kind or rung fields.  This
    annotation supplies those research choices explicitly and can only cite exact
    mechanism-evidence and cost entries that the reopened immutable case holds.
    """

    case_id: str
    case_digest: str
    source_assay_id: str
    explanation_kinds: tuple[str, ...]
    rung: int
    mechanism_evidence: tuple[str, ...]
    costs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise TypeError("planning annotation case id must be non-empty")
        for field, value in (
            ("case digest", self.case_digest),
            ("source assay id", self.source_assay_id),
        ):
            if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
                raise ValueError(f"planning annotation {field} must be a sha256 digest")
        if len(self.explanation_kinds) < 2 or len(set(self.explanation_kinds)) != len(
            self.explanation_kinds
        ):
            raise ValueError("planning annotation needs two or more distinct explanation kinds")
        if any(not isinstance(item, str) or not item for item in self.explanation_kinds):
            raise ValueError("planning annotation explanation kinds must be non-empty strings")
        if type(self.rung) is not int or self.rung not in range(1, 7):
            raise ValueError("planning annotation rung must be an integer from 1 through 6")
        for field, values in (
            ("mechanism evidence", self.mechanism_evidence),
            ("costs", self.costs),
        ):
            if not values or any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"planning annotation {field} must be non-empty strings")

    def payload(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "case_digest": self.case_digest,
            "source_assay_id": self.source_assay_id,
            "explanation_kinds": list(self.explanation_kinds),
            "rung": self.rung,
            "mechanism_evidence": list(self.mechanism_evidence),
            "costs": list(self.costs),
        }

    @property
    def annotation_digest(self) -> str:
        return digest(self.payload())


def load_planner_cases(
    path: Path | str | None = None,
    *,
    corpus: CorpusStore | None = None,
    annotations: Iterable[PlanningAnnotation] | None = None,
) -> tuple[PlannerCase, ...]:
    """Load legacy packaged seed or explicitly annotated persisted corpus cases.

    The corpus route never uses ``search`` and therefore cannot read a protected
    acceptance mapping.  It reopens each source-bound Case through CorpusStore.
    """

    if corpus is not None:
        if path is not None:
            raise ValueError("planner corpus loading accepts either a seed path or a CorpusStore")
        if annotations is None:
            raise ValueError("planner CorpusStore loading requires explicit planning annotations")
        declared = tuple(annotations)
        if not declared:
            raise ValueError(
                "planner CorpusStore loading requires at least one planning annotation"
            )
        if any(not isinstance(item, PlanningAnnotation) for item in declared):
            raise TypeError("planner annotations must be PlanningAnnotation values")
        if len({item.case_id for item in declared}) != len(declared):
            raise ValueError("planner annotations may name each saved case only once")
        result: list[PlannerCase] = []
        for annotation in declared:
            saved = corpus.reopen(annotation.case_id)
            if saved.source is None or saved.source.assay_id != annotation.source_assay_id:
                raise ValueError(
                    f"planning annotation source digest does not match saved case {annotation.case_id!r}"
                )
            if digest(saved.to_dict()) != annotation.case_digest:
                raise ValueError(
                    f"planning annotation case digest does not match saved case {annotation.case_id!r}"
                )
            if not set(annotation.mechanism_evidence).issubset(saved.mechanism_evidence):
                raise ValueError(
                    f"planning annotation mechanism evidence is not present in saved case {annotation.case_id!r}"
                )
            if not set(annotation.costs).issubset(saved.costs):
                raise ValueError(
                    f"planning annotation costs are not present in saved case {annotation.case_id!r}"
                )
            analogy = (
                f"saved case {saved.id}: declared mechanism evidence "
                f"{' | '.join(annotation.mechanism_evidence)}; declared costs "
                f"{' | '.join(annotation.costs)}"
            )
            result.append(
                PlannerCase(
                    id=saved.id,
                    explanation_kinds=annotation.explanation_kinds,
                    rung=annotation.rung,
                    analogy=analogy,
                    case_digest=annotation.case_digest,
                    source_digest=saved.source.assay_id,
                    annotation_digest=annotation.annotation_digest,
                )
            )
        return tuple(result)
    if annotations is not None:
        raise ValueError("planning annotations require a CorpusStore")

    source = (
        Path(path)
        if path is not None
        else Path(__file__).with_name("seed") / "planner" / "planner_cases.json"
    )
    try:
        loaded = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"planner prior-case seed is missing: {source}") from None
    if not isinstance(loaded, list):
        raise ValueError("planner prior-case seed must be a JSON list")
    cases = tuple(
        PlannerCase(
            id=row["id"],
            explanation_kinds=tuple(row["explanation_kinds"]),
            rung=row["rung"],
            analogy=row["analogy"],
        )
        for row in loaded
    )
    if not cases:
        raise ValueError("planner prior-case seed must not be empty")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("planner prior-case seed contains duplicate ids")
    return cases


def consult_corpus(corpus, query: str) -> tuple[tuple[PlannerCase, ...], dict[str, object]]:
    """Consult the public facade and pin precisely the dataset that matched."""
    from . import ActorContext, Corpus, SearchStatus

    if not isinstance(corpus, Corpus):
        raise TypeError("planner facade consultation requires a Corpus")
    result = corpus.cases_like(query, actor_context=ActorContext.CANDIDATE)
    if result.status is SearchStatus.PROTECTED_ACCEPTANCE_MAPPING_BLOCKED:
        raise ValueError("planner corpus lookup refused a protected acceptance mapping")
    if corpus.store is not None and corpus.planning_annotations is None:
        raise ValueError("planner Corpus store loading requires explicit planning annotations")
    _projections, cases, provenance = corpus._dataset()
    if result.provenance != provenance:
        raise ValueError("corpus dataset changed between lookup and planning")
    matched = {item.case_id for item in result.matches}
    return tuple(case for case in cases if case.id in matched), {
        **provenance,
        "query": query,
        "status": result.status.value,
        "matched_case_ids": sorted(matched),
    }
