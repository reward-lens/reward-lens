"""The shapes a reader hands back (section 7.4, wave-2 interfaces section 5).

Four series, named separately and always all four present, because the first rule of the run-record
standard is that the native component scores, the displayed aggregate, the optimised composite and
any convenience aggregate are distinct named things. A series that could not be built is still
named; it carries `present=False` and the reason. Naming an absence is the point: E1 was paid for
by a single column whose name said "the total" and whose values were a batch mean.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping

import rfc8785

if TYPE_CHECKING:  # pragma: no cover - typing only
    from reward_lens.contracts import Entry
    from reward_lens.record.scores import ScoreTree

    from .integrity import RuleResult

__all__ = [
    "Policy",
    "PlannedContrastDesign",
    "CompositionTelemetry",
    "DeclaredReward",
    "PriceSource",
    "ScalarLossStep",
    "ScalarLossToken",
    "RESPONSE_TEXT_CODEC",
    "response_text_identity",
    "SERIES_NAMES",
    "Series",
    "StepMetric",
    "TraceMetadata",
    "TornTail",
    "TraceRecord",
    "TraceRow",
]

#: Declared by the caller and recorded in every record a follow produces (section 7.4).
Policy = Literal["strict", "permissive"]

#: A response join is defined on exactly one recorded assistant text, encoded as a JSON string.
#: Turn ordering is distinct row provenance and is deliberately excluded from this content identity.
RESPONSE_TEXT_CODEC = "rfc8785-response-text/1"


def response_text_identity(text: str) -> tuple[str, str]:
    """Return the versioned identity of one exact recorded assistant response string.

    This is intentionally narrower than :func:`reward_lens.contracts.digest`: response identities
    are RFC 8785 canonical bytes for a JSON string itself, rather than a surrounding record map.
    A caller must decide whether it has exactly one assistant response before calling this function.
    """
    if not isinstance(text, str):
        raise TypeError("response text identity requires one exact str")
    return "sha256:" + hashlib.sha256(rfc8785.dumps(text)).hexdigest(), RESPONSE_TEXT_CODEC


def _freeze(value: Any) -> Any:
    """Make trace facts safe to hand to multiple panels without a mutable side channel."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _codec_frozen(value: Any) -> Any:
    """Deep-freeze codec-safe payloads; ValueCodec explicitly accepts Mapping at serialization."""
    return _freeze(value)


def _score_tree_snapshot(tree: "ScoreTree | None") -> "ScoreTree | None":
    """Detach every tree node and deeply freeze the mutable grader-call payload beneath it.

    This belongs at the Trace producer boundary.  It deliberately does not modify the shared
    score-tree classes, whose mutable `GraderCallRef` remains source-record compatibility.  The
    snapshot keeps every public node class and therefore counterfactual and codec behaviour.
    """
    if tree is None:
        return None
    from reward_lens.record.scores import (
        GraderCallRef,
        GroupGate,
        Leaf,
        Lexicographic,
        Mask,
        Override,
        Piecewise,
        Product,
        WeightedSum,
    )

    if isinstance(tree, Leaf):
        call = tree.grader_call
        if call is not None:
            call = GraderCallRef(
                grader=call.grader,
                outcome=call.outcome,
                facets={},
                latency_s=call.latency_s,
                cost_usd=call.cost_usd,
                raw=None,
                raw_is_repr=call.raw_is_repr,
                error_type=call.error_type,
                error_message=call.error_message,
                seq=call.seq,
                step=call.step,
            )
            object.__setattr__(call, "facets", _codec_frozen(tree.grader_call.facets))
            object.__setattr__(call, "raw", _codec_frozen(tree.grader_call.raw))
        return Leaf(tree.name, tree.value, call, tree.abstained)
    if isinstance(tree, WeightedSum):
        return WeightedSum(
            tree.name,
            tuple(_score_tree_snapshot(child) for child in tree.children),
            tree.weights,
            tree.bias,
        )
    if isinstance(tree, Product):
        return Product(tree.name, tuple(_score_tree_snapshot(child) for child in tree.children))
    if isinstance(tree, Override):
        return Override(
            tree.name, tree.condition, tree.constant, _score_tree_snapshot(tree.otherwise)
        )
    if isinstance(tree, Lexicographic):
        return Lexicographic(
            tree.name,
            _score_tree_snapshot(tree.primary),
            _score_tree_snapshot(tree.secondary),
            tree.threshold,
            tree.cap,
            tree.include_primary,
        )
    if isinstance(tree, Piecewise):
        return Piecewise(tree.name, _score_tree_snapshot(tree.child), tree.knots, tree.extrapolate)
    if isinstance(tree, GroupGate):
        return GroupGate(
            tree.name,
            _score_tree_snapshot(tree.child),
            tree.accuracy_above,
            tree.entropy_below,
            tree.mode,
            tree.entropy_feature,
        )
    if isinstance(tree, Mask):
        return Mask(tree.name, _score_tree_snapshot(tree.child), tree.when)
    raise TypeError(f"unsupported ScoreTree node {type(tree).__name__}")


@dataclass(frozen=True)
class DeclaredReward:
    """The source or caller declaration of the reward, without a defaulted component."""

    partition: tuple[str, ...]
    weights: Mapping[str, float] | None = None
    grouping: Mapping[str, object] | None = None
    estimator: Mapping[str, object] | None = None
    normalisation: Mapping[str, object] | None = None
    clipping: Mapping[str, object] | None = None
    missing_value_policy: Mapping[str, object] | None = None
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "partition", tuple(self.partition))
        for name in (
            "weights",
            "grouping",
            "estimator",
            "normalisation",
            "clipping",
            "missing_value_policy",
            "provenance",
        ):
            value = getattr(self, name)
            object.__setattr__(self, name, None if value is None else _freeze(value))


@dataclass(frozen=True)
class PriceSource:
    """A dated source for a recorded price, never a guessed model tariff."""

    name: str
    retrieved_on: str
    url: str | None = None
    model: str | None = None
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", _freeze(self.provenance))


@dataclass(frozen=True)
class PlannedContrastDesign:
    """Declared planning inputs for a contrast, never a reconstruction from observed outcomes."""

    standard_error: float
    alpha: float
    power: float
    n: int
    clusters: int
    covariance: Mapping[str, object] | None = None
    design_effect: float | None = None
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("standard_error", "alpha", "power"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"planned contrast {name} must be finite")
            object.__setattr__(self, name, value)
        if self.standard_error <= 0:
            raise ValueError("planned contrast standard_error must be positive")
        if not 0 < self.alpha < 1 or not 0 < self.power < 1:
            raise ValueError(
                "planned contrast alpha and power must be strictly between zero and one"
            )
        if self.n <= 0 or self.clusters <= 0:
            raise ValueError("planned contrast n and clusters must be positive")
        if self.design_effect is not None:
            value = float(self.design_effect)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("planned contrast design_effect must be positive and finite")
            object.__setattr__(self, "design_effect", value)
        object.__setattr__(
            self, "covariance", None if self.covariance is None else _freeze(self.covariance)
        )
        object.__setattr__(self, "provenance", _freeze(self.provenance))


@dataclass(frozen=True)
class StepMetric:
    """One source metric at one training step, with the scope that produced it."""

    step: int
    name: str
    value: float | None
    scope: str
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", _freeze(self.provenance))


@dataclass(frozen=True)
class ScalarLossToken:
    """One recorded token contribution to the supported scalar-loss profile."""

    row_id: str
    mask: float
    log_probability: float
    old_log_probability: float
    importance_ratio: float
    advantage: float
    kl_term: float
    auxiliary_term: float


@dataclass(frozen=True)
class ScalarLossStep:
    """The token-level observations required to check one scalar loss, if logged."""

    trainer_profile: str
    aggregation_denominator: float
    minimum_denominator: float
    clip_low: float
    clip_high: float
    kl_coefficient: float
    observed_loss: float | None
    tokens: tuple[ScalarLossToken, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokens", tuple(self.tokens))


@dataclass(frozen=True)
class CompositionTelemetry:
    """Recorded scalar-loss evidence.  Missing telemetry stays ``None`` on metadata."""

    steps: Mapping[int, ScalarLossStep]

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", MappingProxyType(dict(self.steps)))


@dataclass(frozen=True)
class TraceMetadata:
    """Record-wide declarations and telemetry shared by all Audit panels."""

    declared_reward: DeclaredReward | None = None
    price_source: PriceSource | None = None
    planned_contrast: PlannedContrastDesign | None = None
    metrics: tuple[StepMetric, ...] = ()
    composition: CompositionTelemetry | None = None
    #: Only source-recorded signed row weights.  A missing mapping is not an advantage fallback.
    applied_weights: Mapping[str, float] | None = None
    group_estimators: Mapping[str, Mapping[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(
            self,
            "applied_weights",
            None if self.applied_weights is None else _freeze(self.applied_weights),
        )
        object.__setattr__(self, "group_estimators", _freeze(self.group_estimators))


#: The four, in the order every record names them.
SERIES_NAMES: tuple[str, ...] = (
    "reward_recorded",
    "reward_applied",
    "advantage",
    "selection",
)


@dataclass(frozen=True)
class Series:
    """One named series, with the weight vector beside it when it is a composite.

    `values` is one entry per row of the record, in row order, and a `None` is the trainer's own
    missing value preserved rather than normalised to zero.
    """

    name: str
    values: tuple[float | None, ...] = ()
    weights: tuple[float, ...] | None = None
    components: tuple[str, ...] = ()
    definition: str = ""
    present: bool = True
    absent_reason: str = ""
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))
        object.__setattr__(self, "weights", None if self.weights is None else tuple(self.weights))
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "provenance", _freeze(self.provenance))


@dataclass(frozen=True)
class TraceRow:
    """One completion as the optimiser saw it.

    `group_id` is the group the source declared. It is `None` when the source declared none, and
    it is never filled in from what the neighbouring rows look like (D-42).
    """

    index: int
    step: int
    group_id: str | None
    row_id: str
    components: Mapping[str, float | None] = field(default_factory=dict)
    reward_recorded: float | None = None
    reward_applied: float | None = None
    advantage: float | None = None
    selection: float | None = None
    task_id: str | None = None
    response_digest: str | None = None
    response_codec: str | None = None
    response_provenance: Mapping[str, object] | None = None
    grader_source: Mapping[str, str] | None = None
    score_tree: "ScoreTree | None" = None
    accepted: bool | None = None
    applied_weight: float | None = None
    applied_weight_source: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", _freeze(self.components))
        object.__setattr__(
            self,
            "response_provenance",
            None if self.response_provenance is None else _freeze(self.response_provenance),
        )
        object.__setattr__(
            self,
            "grader_source",
            None if self.grader_source is None else _freeze(self.grader_source),
        )
        object.__setattr__(self, "score_tree", _score_tree_snapshot(self.score_tree))


@dataclass(frozen=True)
class TornTail:
    """A partial write at the end of a chunk, kept visible rather than silently dropped."""

    path: str
    seq: int
    bytes_dropped: int
    text: str
    allowed_by: str


@dataclass(frozen=True)
class TraceRecord:
    """What one reader made of one artifact."""

    reader: str
    source: str
    run_id: str
    rows: tuple[TraceRow, ...]
    series: Mapping[str, Series]
    components: Mapping[str, Series]
    access_level: str
    access_statement: str
    policy: Policy
    complete: bool
    completeness_statement: str
    subject_ref: str
    depends_on: tuple[str, ...]
    run_identity: str | None = None
    framework: str = ""
    framework_version: str = ""
    integrity: tuple["RuleResult", ...] = ()
    torn_tail: TornTail | None = None
    limitations: tuple[str, ...] = ()
    convenience: Mapping[str, Series] = field(default_factory=dict)
    metadata: TraceMetadata = field(default_factory=TraceMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "series", _freeze(self.series))
        object.__setattr__(self, "components", _freeze(self.components))
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        object.__setattr__(self, "integrity", tuple(self.integrity))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        object.__setattr__(self, "convenience", _freeze(self.convenience))

    def entries(self) -> tuple["Entry", ...]:
        """The record-level half of the join: one entry per job, each naming its subject."""
        from .entries import trace_entries

        return trace_entries(self)

    def require_measurements(self, *names: str) -> None:
        """Require named measurements without discarding lower measurements this record supports."""
        from .integrity import require_series

        require_series(self.series, source=self.source, names=names)
