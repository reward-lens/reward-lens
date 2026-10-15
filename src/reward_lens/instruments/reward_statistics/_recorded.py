"""Typed canonical-trace facts consumed by the reward-statistics calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from reward_lens.record import GraderCallRef, ScoreTree
from reward_lens.record.scores import leaves


@dataclass(frozen=True)
class PriceSource:
    """A dated source identifying a price card, never a fabricated current price."""

    name: str
    retrieved: str
    url: str | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("price_source.name is required")
        if not self.retrieved:
            raise ValueError("price_source.retrieved is required")

    def as_dict(self) -> dict[str, str]:
        out = {"name": self.name, "retrieved": self.retrieved}
        if self.url is not None:
            out["url"] = self.url
        if self.model is not None:
            out["model"] = self.model
        return out


@dataclass(frozen=True)
class RecordedCompletion:
    """One completion and the score tree trace recorded for it."""

    id: str
    scores: ScoreTree
    accepted: bool | None


@dataclass(frozen=True)
class TraceReview:
    """The facts P-STATS reads from one canonical ``TraceRecord``."""

    completions: tuple[RecordedCompletion, ...]
    declared_reward: Mapping[str, Any] | None = None
    price_source: PriceSource | None = None
    price_source_missing: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "completions", tuple(self.completions))
        ids = [completion.id for completion in self.completions]
        if len(set(ids)) != len(ids):
            raise ValueError("TraceReview completion ids must be unique")


class TraceReviewUnavailable(ValueError):
    """The canonical trace lacks a field needed by every P-STATS reader."""

    def __init__(self, missing: str, remedy: str) -> None:
        super().__init__(missing)
        self.missing = missing
        self.remedy = remedy


def review_from_corpus(corpus: Any) -> TraceReview:
    """Convert only ``corpus.trace`` into the panel's explicit calculation input.

    Audit owns construction of the one canonical ``TraceRecord``.  This consumer never opens a
    directory or treats another corpus field as trace evidence.
    """
    from reward_lens.product.trace import TraceRecord

    trace = getattr(corpus, "trace", None)
    if trace is None:
        raise TraceReviewUnavailable(
            "corpus.trace is absent",
            "supply a canonical TraceRecord through the audit trace bridge",
        )
    if not isinstance(trace, TraceRecord):
        raise TraceReviewUnavailable(
            "corpus.trace is not a canonical TraceRecord",
            "supply the immutable TraceRecord produced by the audit trace bridge",
        )
    completions: list[RecordedCompletion] = []
    for row in trace.rows:
        if not row.row_id:
            raise TraceReviewUnavailable(
                "corpus.trace contains a row without a stable row_id",
                "record a stable row identifier in the canonical trace before auditing statistics",
            )
        if row.score_tree is None:
            raise TraceReviewUnavailable(
                f"corpus.trace row {row.row_id!r} has no recorded score_tree",
                "record score trees for every completion whose reward statistics are audited",
            )
        if row.accepted is not None and not isinstance(row.accepted, bool):
            raise TraceReviewUnavailable(
                f"corpus.trace row {row.row_id!r} has a non-boolean accepted field",
                "record accepted as true, false, or unavailable in the canonical trace",
            )
        completions.append(RecordedCompletion(row.row_id, row.score_tree, row.accepted))
    declared = _declared_reward(trace.metadata.declared_reward)
    price_source, price_source_missing = _price_source(trace.metadata.price_source)
    try:
        return TraceReview(
            tuple(completions),
            declared_reward=declared,
            price_source=price_source,
            price_source_missing=price_source_missing,
        )
    except ValueError as error:
        raise TraceReviewUnavailable(
            str(error),
            "record unique stable row identifiers in the canonical trace",
        ) from error


def _declared_reward(declared: Any) -> Mapping[str, Any] | None:
    if declared is None:
        return None
    result: dict[str, Any] = {"partition": list(declared.partition)}
    for source, target in (
        ("weights", "weights"),
        ("grouping", "grouping"),
        ("estimator", "estimator"),
        ("normalisation", "normalisation"),
        ("clipping", "clipping"),
        ("missing_value_policy", "missing_value_policy"),
        ("provenance", "provenance"),
    ):
        value = getattr(declared, source)
        if value is not None:
            result[target] = dict(value)
    return result


def _price_source(source: Any) -> tuple[PriceSource | None, str | None]:
    if source is None:
        return None, "corpus.trace.metadata.price_source is absent"
    if not source.name:
        return None, "corpus.trace.metadata.price_source.name is absent"
    if not source.retrieved_on:
        return None, "corpus.trace.metadata.price_source.retrieved_on is absent"
    return PriceSource(source.name, source.retrieved_on, source.url, source.model), None


def calls(completion: RecordedCompletion) -> tuple[GraderCallRef, ...]:
    """Every recorded grader call belonging to one completion."""
    return tuple(
        leaf.grader_call for leaf in leaves(completion.scores) if leaf.grader_call is not None
    )


def subject_ref(subject: Any) -> str:
    """A stable subject reference when the audit has not supplied one separately."""
    from reward_lens import contracts

    return contracts.digest({"source": getattr(subject, "source", "") or ""})
