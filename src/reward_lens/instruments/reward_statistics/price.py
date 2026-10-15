"""Observed grader cost, with displayed quantities bootstrapped at completion level."""

from __future__ import annotations

from typing import Any, Sequence

from reward_lens import contracts
from reward_lens.instruments.base import RunContext

from ._recorded import (
    RecordedCompletion,
    TraceReview,
    TraceReviewUnavailable,
    calls,
    review_from_corpus,
    subject_ref,
)
from ._statistics import completion_bootstrap, stratified_completion_bootstrap

VERSION = "1.2.0"
__all__ = ["PriceSourceError", "RewardPrice"]


class PriceSourceError(ValueError):
    """Recorded costs without a named, dated source row are refused before gate 17."""


def _priced(review: TraceReview) -> tuple[tuple[RecordedCompletion, tuple[float, ...]], ...]:
    out: list[tuple[RecordedCompletion, tuple[float, ...]]] = []
    for completion in review.completions:
        actual = calls(completion)
        if not actual:
            raise ValueError(f"completion {completion.id} contains no recorded grader call")
        if any(call.cost_usd is None for call in actual):
            raise ValueError(
                f"completion {completion.id} contains a grader call with no recorded cost_usd"
            )
        out.append((completion, tuple(float(call.cost_usd) for call in actual)))
    return tuple(out)


def _costs(sample: Sequence[object]) -> list[float]:
    return [cost for _, costs in sample for cost in costs]  # type: ignore[misc]


def _per_call(sample: Sequence[object]) -> float:
    costs = _costs(sample)
    return sum(costs) / len(costs)


def _per_completion(sample: Sequence[object]) -> float:
    return sum(_costs(sample)) / len(sample)


def _per_accepted(sample: Sequence[object]) -> float:
    accepted = sum(bool(completion.accepted) for completion, _ in sample)  # type: ignore[misc]
    if not accepted:
        raise ValueError(
            "USD per accepted completion is undefined for a sample with no accepted completion"
        )
    return sum(_costs(sample)) / accepted


def _accepted_completion_bootstrap(priced: Sequence[object]) -> contracts.Uncertainty:
    return stratified_completion_bootstrap(
        priced,
        _per_accepted,
        lambda item: bool(item[0].accepted),  # type: ignore[index]
        stratum_name="accepted status",
    )


class RewardPrice:
    """Report observed USD per call, accepted completion and thousand completions."""

    id = "reward_statistics.price"
    section: contracts.Section = "reward_statistics"
    method = contracts.Method(
        id=id,
        version=VERSION,
        params_digest=contracts.digest(
            {
                "input": "TraceReview",
                "resampling": "completion cluster bootstrap; accepted-completion ratio conditional on observed accepted/rejected allocation",
            }
        ),
        procedure="aggregate actual GraderCallRef.cost_usd values by completion; bootstrap per-call and per-thousand statistics over whole completions, and bootstrap accepted-completion cost within observed acceptance strata so every ratio replicate has an accepted denominator",
    )

    def run(self, subject: Any, corpus: Any, ctx: RunContext) -> list[contracts.Entry]:
        try:
            review = review_from_corpus(corpus)
        except TraceReviewUnavailable as error:
            return [
                ctx.absence(
                    self.section,
                    self.id,
                    "price per call, accepted completion and thousand completions",
                    error.missing,
                    error.remedy,
                    ("no claim about grader price from an ordinary response-bank audit",),
                    subject_ref=subject_ref(subject),
                    depends_on=("digest:samples",),
                )
            ]
        return self.analyze(subject, review, ctx)

    def analyze(self, subject: Any, review: TraceReview, ctx: RunContext) -> list[contracts.Entry]:
        reference = subject_ref(subject)
        if not review.completions:
            return [
                ctx.absence(
                    self.section,
                    self.id,
                    "price per call, accepted completion and thousand completions",
                    "the trace review contains no recorded completions",
                    "import a trace with recorded grader calls and costs",
                    ("no claim about the price of this grader",),
                    subject_ref=reference,
                    depends_on=("digest:samples",),
                )
            ]
        try:
            priced = _priced(review)
        except ValueError as error:
            return [
                ctx.absence(
                    self.section,
                    self.id,
                    "price per call, accepted completion and thousand completions",
                    str(error),
                    "record cost_usd for every grader call in every completion used for this price estimate",
                    ("no claim about the price of this grader",),
                    subject_ref=reference,
                    depends_on=("digest:samples",),
                )
            ]
        if review.price_source is None:
            return [
                ctx.absence(
                    self.section,
                    self.id,
                    "price per call, accepted completion and thousand completions",
                    review.price_source_missing
                    or "the trace review has no named, dated price_source",
                    "record a named price source and retrieval date in corpus.trace.metadata.price_source",
                    ("no claim about grader price from an undated or absent source",),
                    subject_ref=reference,
                    depends_on=("digest:samples",),
                )
            ]
        source = review.price_source.as_dict()
        costs, completion_count = _costs(priced), len(priced)
        accepted_unknown = [
            completion.id for completion, _ in priced if completion.accepted is None
        ]
        accepted_count = sum(completion.accepted is True for completion, _ in priced)
        entries = [
            self._entry(
                "per_call",
                "USD per recorded grader call",
                _per_call(priced),
                len(costs),
                "recorded grader call",
                f"{len(costs)} recorded grader calls",
                completion_bootstrap(priced, _per_call),
                source,
                reference,
                ctx,
            ),
            self._entry(
                "per_thousand_completions",
                "USD per thousand recorded completions",
                _per_completion(priced) * 1000.0,
                completion_count,
                "recorded completion",
                f"{completion_count} recorded completions",
                completion_bootstrap(priced, lambda sample: _per_completion(sample) * 1000.0),
                source,
                reference,
                ctx,
            ),
        ]
        if accepted_unknown:
            entries.append(
                ctx.absence(
                    self.section,
                    f"{self.id}.per_accepted_completion",
                    "price per accepted completion",
                    f"corpus.trace rows have no recorded acceptance: {', '.join(accepted_unknown)}",
                    "record accepted as true or false for every completion before estimating accepted-completion cost",
                    ("no claim about price per accepted completion when acceptance is unknown",),
                    subject_ref=reference,
                    depends_on=("digest:samples",),
                )
            )
        elif accepted_count:
            entries.append(
                self._entry(
                    "per_accepted_completion",
                    "USD per accepted completion",
                    _per_accepted(priced),
                    accepted_count,
                    "accepted recorded completion",
                    f"{accepted_count} accepted completion(s); rejected completion costs remain in the numerator",
                    _accepted_completion_bootstrap(priced),
                    source,
                    reference,
                    ctx,
                    interval_sampling_design="completion-cluster bootstrap stratified by observed accepted status; accepted and rejected cluster counts are fixed, so the interval conditions on the observed acceptance allocation",
                )
            )
        else:
            entries.append(
                ctx.absence(
                    self.section,
                    f"{self.id}.per_accepted_completion",
                    "price per accepted completion",
                    "corpus.trace contains no recorded accepted completion",
                    "record at least one accepted completion before estimating price per accepted completion",
                    (
                        "no claim about price per accepted completion without an accepted denominator",
                    ),
                    subject_ref=reference,
                    depends_on=("digest:samples",),
                )
            )
        return entries

    def _entry(
        self,
        suffix: str,
        unit: str,
        value: float,
        n: int,
        sampling_unit: str,
        denominator: str,
        uncertainty: contracts.Uncertainty,
        source: dict[str, str],
        reference: str,
        ctx: RunContext,
        *,
        interval_sampling_design: str | None = None,
    ) -> contracts.Entry:
        result: dict[str, Any] = {"price_source": source, "denominator": denominator}
        limitations = ["price includes only completions whose every grader call recorded cost_usd"]
        if interval_sampling_design is not None:
            result["interval_sampling_design"] = interval_sampling_design
            limitations.append(
                "the accepted-completion interval conditions on the observed accepted/rejected completion allocation and does not quantify acceptance-rate uncertainty"
            )
        return contracts.estimate(
            self.section,
            f"{self.id}.{suffix}",
            unit,
            value,
            unit,
            n,
            sampling_unit,
            uncertainty,
            subject_ref=reference,
            provenance=ctx.provenance(),
            method=self.method,
            result=result,
            depends_on=("digest:samples",),
            limitations=tuple(limitations),
        )
