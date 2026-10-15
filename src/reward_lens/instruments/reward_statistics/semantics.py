"""Reward semantics read from an explicit trace review, never a declaration alone."""

from __future__ import annotations

from typing import Any

from reward_lens import contracts
from reward_lens.instruments.base import RunContext
from reward_lens.record import Piecewise, WeightedSum
from reward_lens.record.scores import leaves, walk

from ._recorded import TraceReview, TraceReviewUnavailable, review_from_corpus, subject_ref
from ._statistics import completion_bootstrap

VERSION = "1.1.0"
FINDING_CODE = "RL0251"
FINDING_ID = "RGX-local-0251"

__all__ = ["FINDING_CODE", "FINDING_ID", "RewardSemantics", "findings_for"]


def _component_count(sample: list[object]) -> float:
    return float(len({leaf.name for completion in sample for leaf in leaves(completion.scores)}))  # type: ignore[misc]


def _observed(review: TraceReview) -> dict[str, Any]:
    """Every semantic configuration seen in the recorded completion population."""
    configurations: dict[str, dict[str, Any]] = {}
    component_names: list[str] = []
    abstained = total_leaves = 0
    type_names = {
        "WeightedSum": "weighted_sum",
        "Product": "product",
        "Override": "override",
        "Lexicographic": "lexicographic",
        "Piecewise": "piecewise",
        "GroupGate": "group_gate",
        "Mask": "mask",
    }
    for completion in review.completions:
        weights: dict[str, float] = {}
        aggregation: list[str] = []
        clipping: list[dict[str, Any]] = []
        for node in walk(completion.scores):
            label = type_names.get(type(node).__name__)
            if label is not None and label not in aggregation:
                aggregation.append(label)
            if isinstance(node, WeightedSum):
                for child, weight in zip(node.children, node.weights):
                    weights[child.name] = float(weight)
            if isinstance(node, Piecewise) and node.extrapolate == "clamp":
                clipping.append(
                    {
                        "node": node.name,
                        "extrapolate": node.extrapolate,
                        "knots": [[float(x), float(y)] for x, y in node.knots],
                    }
                )
        for leaf in leaves(completion.scores):
            if leaf.name not in component_names:
                component_names.append(leaf.name)
            total_leaves += 1
            abstained += int(leaf.abstained)
        key = str({"weights": weights, "aggregation": aggregation, "clipping": clipping})
        current = configurations.setdefault(
            key,
            {
                "weights": weights,
                "aggregation": aggregation,
                "clipping": clipping,
                "completion_ids": [],
            },
        )
        current["completion_ids"].append(completion.id)
    return {
        "components": component_names,
        "weight_configurations": list(configurations.values()),
        "missing_value_semantics": "abstained scores are missing measurements, not zero",
        "abstained_leaf_calls": abstained,
        "recorded_leaf_calls": total_leaves,
    }


class RewardSemantics:
    """Components, weights, aggregation, clipping and missing values as trace recorded them."""

    id = "reward_statistics.semantics"
    section: contracts.Section = "reward_statistics"
    method = contracts.Method(
        id=id,
        version=VERSION,
        params_digest=contracts.digest({"input": "TraceReview", "declared": "comparison only"}),
        procedure="read each recorded score tree in an explicit TraceReview; retain a declaration only to compare it with the observed trace",
    )

    def run(self, subject: Any, corpus: Any, ctx: RunContext) -> list[contracts.Entry]:
        try:
            review = review_from_corpus(corpus)
        except TraceReviewUnavailable as error:
            return [
                ctx.absence(
                    self.section,
                    self.id,
                    "recorded reward components, weights, aggregation, clipping and missing-value semantics",
                    error.missing,
                    error.remedy,
                    ("no claim about reward composition from an ordinary response-bank audit",),
                    subject_ref=subject_ref(subject),
                    depends_on=("digest:samples",),
                )
            ]
        return [self.analyze(subject, review, ctx)]

    def analyze(self, subject: Any, review: TraceReview, ctx: RunContext) -> contracts.Entry:
        reference = subject_ref(subject)
        if not review.completions:
            return ctx.absence(
                self.section,
                self.id,
                "recorded reward components, weights, aggregation, clipping and missing-value semantics",
                "the trace review contains no recorded completions",
                "import a trace with per-completion score trees",
                ("no claim about reward composition or missing-value semantics",),
                subject_ref=reference,
                depends_on=("digest:samples",),
            )
        observed = _observed(review)
        result: dict[str, Any] = dict(observed)
        if review.declared_reward is not None:
            result["declared"] = dict(review.declared_reward)
            result["observed"] = {
                key: observed[key]
                for key in ("components", "weight_configurations", "missing_value_semantics")
            }
            result["declared_matches_observed"] = result["declared"] == result["observed"]
        return contracts.estimate(
            self.section,
            self.id,
            "the reward composition carried by the recorded trace",
            float(len(observed["components"])),
            "recorded reward components",
            len(review.completions),
            "recorded completion",
            completion_bootstrap(review.completions, _component_count),
            subject_ref=reference,
            provenance=ctx.provenance(),
            method=self.method,
            result=result,
            depends_on=("digest:samples",),
            limitations=(
                "a trace without a score tree remains absent and is not reconstructed from a declaration",
            ),
        )


def findings_for(entry: contracts.Entry) -> tuple[contracts.Finding, ...]:
    """Emit RL0251 when a declaration differs from the observed trace record."""
    result = dict(entry.result or {})
    if "declared" not in result or result.get("declared_matches_observed") is not False:
        return ()
    declared, observed = result["declared"], result["observed"]
    return (
        contracts.Finding(
            id=FINDING_ID,
            rule=entry.entry_id,
            level="warning",
            kind="fail",
            severity_rationale="the score trees are the observed trace; the declaration remains beside them so the disagreement can be inspected",
            scope="evaluator_comparison",
            entries=[entry.entry_id],
            partial_fingerprint=contracts.digest(
                {
                    "code": FINDING_CODE,
                    "entry": entry.entry_id,
                    "declared": declared,
                    "observed": observed,
                }
            ).removeprefix("sha256:")[:16],
            message=f"declared reward semantics differ from observed trace: declared={declared!r}; observed={observed!r}",
            code=FINDING_CODE,
            arm="static",
            witness_path=f"/measurement/reward_statistics/{entry.entry_id}/result",
        ),
    )
