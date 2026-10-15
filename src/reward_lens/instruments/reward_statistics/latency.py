"""Sample-quantile latency estimates with completion-cluster bootstrap intervals."""

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
from ._statistics import completion_bootstrap, quantile

VERSION = "1.1.0"
__all__ = ["LatencyPercentiles"]


def _latencies(review: TraceReview) -> tuple[tuple[RecordedCompletion, tuple[float, ...]], ...]:
    groups: list[tuple[RecordedCompletion, tuple[float, ...]]] = []
    for completion in review.completions:
        values = tuple(
            float(call.latency_s) for call in calls(completion) if call.latency_s is not None
        )
        if values:
            groups.append((completion, values))
    return tuple(groups)


def _percentile(sample: Sequence[object], percentile: int) -> float:
    values = [latency for _, latencies in sample for latency in latencies]  # type: ignore[misc]
    return quantile(values, percentile / 100.0)


class LatencyPercentiles:
    """Report p50, p95 and p99 of actual per-call latency, with quantile intervals."""

    id = "reward_statistics.latency"
    section: contracts.Section = "reward_statistics"
    method = contracts.Method(
        id=id,
        version=VERSION,
        params_digest=contracts.digest(
            {
                "input": "TraceReview",
                "percentiles": [50, 95, 99],
                "resampling": "completion cluster bootstrap",
            }
        ),
        procedure="calculate linearly interpolated sample quantiles from recorded latency_s values and bootstrap each quantile by resampling whole completions",
    )

    def run(self, subject: Any, corpus: Any, ctx: RunContext) -> list[contracts.Entry]:
        try:
            review = review_from_corpus(corpus)
        except TraceReviewUnavailable as error:
            return [
                ctx.absence(
                    self.section,
                    self.id,
                    "latency percentiles for grader calls",
                    error.missing,
                    error.remedy,
                    ("no claim about grader latency from an ordinary response-bank audit",),
                    subject_ref=subject_ref(subject),
                    depends_on=("digest:samples",),
                )
            ]
        return self.analyze(subject, review, ctx)

    def analyze(self, subject: Any, review: TraceReview, ctx: RunContext) -> list[contracts.Entry]:
        grouped, reference = _latencies(review), subject_ref(subject)
        if not grouped:
            return [
                ctx.absence(
                    self.section,
                    self.id,
                    "latency percentiles for grader calls",
                    "the trace review contains no recorded GraderCallRef.latency_s values",
                    "record per-call latency in trace before running this panel",
                    ("no claim about grader latency percentiles",),
                    subject_ref=reference,
                    depends_on=("digest:samples",),
                )
            ]
        n_calls = sum(len(values) for _, values in grouped)
        entries: list[contracts.Entry] = []
        for percentile in (50, 95, 99):
            value = _percentile(grouped, percentile)
            entries.append(
                contracts.estimate(
                    self.section,
                    f"{self.id}.p{percentile}",
                    f"p{percentile} latency of recorded grader calls",
                    value,
                    "seconds per recorded grader call",
                    n_calls,
                    "recorded grader call",
                    completion_bootstrap(
                        grouped, lambda sample, q=percentile: _percentile(sample, q)
                    ),
                    subject_ref=reference,
                    provenance=ctx.provenance(),
                    method=self.method,
                    result={
                        "percentile": percentile,
                        "percentile_method": "linear interpolation on sorted recorded call latencies",
                        "completion_clusters": len(grouped),
                    },
                    depends_on=("digest:samples",),
                    limitations=(
                        "the percentile uses only calls that recorded latency_s; its interval resamples complete latency-bearing completions",
                    ),
                )
            )
        return entries
