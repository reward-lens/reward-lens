"""F1, the Price ledger: what this optimiser step selected for, and what moved anyway.

**What this is, and what it is not, because a reviewer will ask and the distinction is real.**
Steven A. Frank (arXiv 2507.18549, 2025; journal version PMC12651060) partitions optimiser updates
with the Price equation, unifying SGD, Newton, Adam, Bayesian updating, Langevin dynamics and
natural selection under `Δθ = Mf + b + ξ`. That work derives **update rules in parameter space**.
This module is a **measurement instrument on behavioural traits in a live run**: it reads a training
record, computes the selection differential on named features of the policy's own rollouts, and
reports what the step's selection pressure does not explain. Same equation, different use. Frank's
partition never touches the selection-differential-versus-gradient distinction, Lande, the breeder's
equation, correlated response, RL or language models; this one never derives an update rule. Both
statements were checked against the paper rather than assumed, and neither subsumes the other.

**The identity.** Let `f(y)` be any measurable feature of a rollout and `z(θ) = E_{y~π_θ}[f(y)]`.
Under a Fisher-preconditioned step the update is exactly an exponential tilt, `π_new ∝ π_old·exp(ηA)`,
and to first order in the step size

    Δz_i = η · Cov_group(A, f_i) + ρ_i

Both sides are separately measurable inside a live run and neither needs a model. The left side
needs `f` measured at step `t` and at step `t+1`. The right side needs the advantages and the
features at step `t` and costs nothing, because the numbers are already in the record. **`ρ` is what
moved for reasons this step's selection pressure does not explain**, and naming it rather than
discarding it is the whole point: its candidate sources are off-policy staleness, the KL pull toward
the reference, entropy bonuses, optimiser momentum, gradient interference from other prompts in the
same batch, and the curvature term dropped in the `O(η²)`. Each is separately testable and none of
them is visible if the residual is thrown away.

**Four things this cannot do, stated here rather than on a caveats page.**

This is a first-order expansion about the current step and it is meaningless if the step is large,
which the `LINEAR_RESPONSE` envelope condition checks and F2's `Λ` measures. It assumes the group
has spread, so it means nothing on all-fail groups, which `GROUP_NONDEGENERATE` checks. It assumes
one generating policy per trajectory, which `NEAR_POLICY` checks. And `Δz` is an expectation over the
policy **and over whatever prompts the batch drew**: when step `t` and step `t+1` sample disjoint
prompt sets, the measured `Δz` carries prompt-resampling noise that no amount of selection pressure
was ever going to explain, which inflates `ρ` and deflates `Λ`. That last one is not in the
specification's envelope and it is measured here anyway, as `task_overlap` on every reading, with a
paired estimator over shared tasks available when the overlap is not zero.

**On the covariance operator.** `Cov_group` is the **within-group** covariance: both `A` and `f` are
centred inside their own prompt group before the product is taken, and the pooled denominator is
`n - G`. SPEC-ERRATA E17 settles that this is the operator §3.1.1 and §3.1.2 mean, and it is not the
pooled covariance: between the two sits prompt-to-prompt heterogeneity, which is a property of the
task distribution rather than of the policy. Every reading names the operator it used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import (
    BiasStatement,
    CostModel,
    EstimatorEntry,
    register_estimator,
)
from reward_lens.core.reading import Reading, Refusal, RefusalReason, refuse_incomplete
from reward_lens.core.types import (
    Access,
    AccessMatrix,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.base import BaseObservable, Context, PreflightResult, run
from reward_lens.measure.ledger.features import TrajectoryFeaturiser, matrix_of
from reward_lens.measure.rate.regime import MEASURED_BY
from reward_lens.record.schema import Run, SelectionQuantity, SelectionWeights, Step

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence

#: Half-open ``[lo, hi)`` over step indices, matching `StepStream.slice` and `measure.rate.regime`.
Window = tuple[int, int]


def whole_run(run_: Run) -> Window:
    """The window covering every recorded step, resolved from the record's own index list.

    Every window in this module is resolved through here rather than defaulted to a sentinel pair,
    and that is not style. `RecordReader.partitions_for` builds `range(first, last + 1, chunk)` over
    the requested bounds before intersecting with the partitions that exist, so a call with wide
    sentinel bounds enumerates a range of order `2**63` and never returns. Asking the record what it
    holds costs one attribute read and cannot hang.
    """
    indices = run_.steps.indices
    return (min(indices), max(indices) + 1) if indices else (0, 0)


#: The ledger reads a record and a featuriser and nothing else. §5's access line for F1 and F2 is
#: "RECORD + a featuriser", and the featuriser is an argument rather than an access level because it
#: is code the caller supplies rather than a thing they must be granted.
LEDGER_ACCESS: AccessMatrix = {Component.RECORD: Access.RECORD}

#: F1's envelope, verbatim from the catalogue's `envelope_requires`. `LINEAR_RESPONSE` is measured
#: by F2's own `selection.explained_fraction`, which is why the two instruments in this package are
#: not independent: Λ is the certificate that licenses reading a first-order ledger row at all, and
#: an F1 reading taken without it is refused rather than run with the check skipped.
LEDGER_ENVELOPE = EnvelopeSpec(
    requires=frozenset(
        {
            RegimeCondition.LINEAR_RESPONSE,
            RegimeCondition.GROUP_NONDEGENERATE,
            RegimeCondition.NEAR_POLICY,
        }
    ),
    measured_by={
        c: MEASURED_BY[c]
        for c in (
            RegimeCondition.LINEAR_RESPONSE,
            RegimeCondition.GROUP_NONDEGENERATE,
            RegimeCondition.NEAR_POLICY,
        )
    },
    on_violation="refuse",
)


# ---------------------------------------------------------------------------
# The sample: one step, reduced to what the ledger reads
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepSample:
    """One optimiser step's rollouts, reduced to the four arrays the ledger needs.

    Deliberately not a `Step`. The ledger's arithmetic is the same whether the rows came from a
    `Run` or from a labelled rollout table somebody else published, and putting the record type in
    the middle of it would make the second case a rewrite rather than an adapter. `steps_from_run`
    builds these from a record; `measure.ledger.labelled` builds them from a table.

    ``advantages`` is NaN where the rollout received none. That is TRL's own convention for a masked
    row and it is the one the record layer keeps: an abstained grader produces no advantage, and a
    zero there is a real number that flows into a covariance.

    ``advantage_source`` is ``"recorded"`` when the trainer's own advantages were read off the
    record and ``"reconstructed"`` when they were recomputed from group rewards. The two are not
    interchangeable and a reading that does not say which it used cannot be checked.
    """

    index: int
    names: tuple[str, ...]
    features: np.ndarray
    advantages: np.ndarray
    group_ids: np.ndarray
    task_ids: tuple[str, ...]
    advantage_source: str = "recorded"
    #: `a-tilde` per rollout, when the trainer recorded one. `None` means it did not, and the
    #: ledger reads the advantage instead **under its own label**, rather than passing one quantity
    #: off as the other (BLK-015). NaN inside it is the same masked-row convention `advantages` uses.
    applied_weights: SelectionWeights | None = None
    n_dropped: int = 0
    detail: str = ""

    def __post_init__(self) -> None:
        n, k = self.features.shape
        if k != len(self.names):
            raise ValueError(
                f"step {self.index}: features has {k} columns and {len(self.names)} names. A "
                f"column without a name cannot be reported and a name without a column is a "
                f"feature nobody measured."
            )
        for label, arr in (
            ("advantages", self.advantages),
            ("group_ids", self.group_ids),
            ("task_ids", np.asarray(self.task_ids, dtype=object)),
        ):
            if arr.shape[0] != n:
                raise ValueError(
                    f"step {self.index}: {label} has {arr.shape[0]} rows against {n} feature rows. "
                    f"A misaligned column pairs one rollout's advantage with another's features."
                )

    @property
    def n(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_scored(self) -> int:
        """Rollouts carrying a finite advantage, which is what the covariance is taken over."""
        return int(np.count_nonzero(np.isfinite(self.advantages)))

    @property
    def tasks(self) -> frozenset[str]:
        return frozenset(self.task_ids)

    def weights(self) -> SelectionWeights:
        """The scalar to take the differential against, carrying what it is.

        The applied weight when the record has one, the advantage otherwise, and **labelled either
        way**. A caller who needs `S_applied` reads `Differential.as_applied`, which refuses the
        second case rather than returning the first case's number under the second case's name.
        """
        if self.applied_weights is not None:
            return self.applied_weights
        return SelectionWeights(self.advantages, SelectionQuantity.ADVANTAGE, self.advantage_source)


def advantages_from_rewards(
    rewards: Sequence[float] | np.ndarray,
    group_ids: Sequence[int] | np.ndarray,
    *,
    std_normalised: bool,
    std_epsilon: float = 0.0,
    std_ddof: int | None = None,
) -> np.ndarray:
    """GRPO's group-relative advantage, recomputed from rewards. NaN where the reward is NaN.

    ``A_i = (r_i - mean_g r) / (std_g r + eps)`` over the rollout's own prompt group, or the
    numerator alone under Dr.GRPO's un-normalised variant. This exists for two reasons. A published
    rollout table carries rewards and labels and no advantages, so reconstructing them is the only
    way the ledger runs on somebody else's corpus. And the generated invariance test needs a path
    that recomputes `A` from scores, because an advantage read off a record does not move when the
    reward is rescaled and a test that transforms the reward would then assert nothing.

    **``std_normalised`` has no default and ``std_ddof`` is required whenever it is true.** Both
    used to be optional and both used to disagree with the rest of the library: this function
    defaulted to standardising while `EstimatorSpec` defaults to not, and it divided by numpy's
    population standard deviation while the library's own verified framework table records that
    every framework in scope applies Bessel's correction. The two conventions differ by
    `sqrt(K/(K-1))`, which is 3.3 per cent at a group of sixteen and 15.5 per cent at a group of
    four, so the disagreement was worth about three per cent on any fitted step size that ran
    through here. A caller that has not established the convention is asked to say so rather than
    inheriting one.

    The `eps` in the denominator is why the z-scored advantage is only *approximately* invariant
    under `reward.affine`: the numerator scales by `a` and the denominator goes to `a·std + eps`
    rather than `a·(std + eps)`. At the ``std_epsilon = 1e-4`` these records carry, the violation is
    around `1e-4` in relative terms rather than the `1e-7` SPEC-ERRATA E13 measured at `1e-8`, so
    the generated test's tolerance has to be set from the record's own epsilon and not from a
    constant. Under the centred convention the epsilon never enters, which is why it defaults to
    zero here rather than to the value a standardising trainer would use: an epsilon carried into
    an arithmetic that does not use it is a number in a provenance string that describes nothing.
    """
    if std_normalised and std_ddof is None:
        raise ValueError(
            "advantages_from_rewards(std_normalised=True) needs std_ddof. The population and "
            "sample standard deviations differ by sqrt(K/(K-1)), which is 3.3 per cent at K=16, "
            "and which one a trainer used is a property of that trainer rather than a convention "
            "this function may pick. Every framework in scope uses 1."
        )
    if not std_normalised and std_epsilon:
        raise ValueError(
            f"advantages_from_rewards(std_normalised=False) was given std_epsilon={std_epsilon!r}. "
            f"The centred convention has no divisor, so an epsilon here would be recorded in the "
            f"provenance of an arithmetic that never used it."
        )
    ddof = int(std_ddof or 0)
    r = np.asarray(rewards, dtype=np.float64).ravel()
    g = np.asarray(group_ids).ravel()
    out = np.full(r.shape, np.nan, dtype=np.float64)
    for label in np.unique(g):
        mask = g == label
        present = mask & np.isfinite(r)
        if not np.any(present):
            continue
        values = r[present]
        centred = values - values.mean()
        if std_normalised:
            spread = float(values.std(ddof=ddof)) if values.size > ddof else 0.0
            out[present] = centred / (spread + std_epsilon)
        else:
            out[present] = centred
    return out


def steps_from_run(
    run_: Run,
    featuriser: TrajectoryFeaturiser,
    *,
    window: Window | None = None,
) -> list[StepSample]:
    """Every step in the window, featurised, in step order.

    Group labels are assigned per step from the record's own `Group` boundaries, so the within-group
    centring matches the partition the trainer actually normalised over. Trajectories the featuriser
    declines are dropped and counted; trajectories with no advantage keep their row and carry NaN,
    because they were still sampled from the policy and still belong in `z`.
    """
    lo, hi = window if window is not None else whole_run(run_)
    out: list[StepSample] = []
    for step in sorted(run_.steps.slice(lo, hi), key=lambda s: s.index):
        out.append(_sample_of(step, featuriser))
    return out


def _sample_of(step: Step, featuriser: TrajectoryFeaturiser) -> StepSample:
    rows: list[np.ndarray] = []
    advantages: list[float] = []
    applied: list[float] = []
    applied_sources: set[str] = set()
    groups: list[int] = []
    tasks: list[str] = []
    dropped = 0
    for ordinal, group in enumerate(step.groups):
        matrix, kept = matrix_of(group.trajectories, featuriser)
        dropped += len(group.trajectories) - len(kept)
        for row, index in zip(matrix, kept):
            trajectory = group.trajectories[index]
            rows.append(row)
            advantage = trajectory.advantage
            advantages.append(np.nan if advantage is None else float(advantage))
            weight = trajectory.applied_weight
            applied.append(np.nan if weight is None else float(weight))
            if weight is not None:
                applied_sources.add(trajectory.applied_weight_source or "unknown")
            groups.append(ordinal)
            tasks.append(str(group.task_ref))
    features = (
        np.asarray(rows, dtype=np.float64)
        if rows
        else np.zeros((0, len(featuriser.names)), dtype=np.float64)
    )
    weights = None
    if applied_sources:
        # One source per step or none. Two different provenances mixed into one column is a column
        # nobody can label, so it is refused rather than reported as whichever sorted first.
        if len(applied_sources) > 1:
            raise ValueError(
                f"step {step.index}: applied weights arrived from {sorted(applied_sources)}. One "
                f"step's weights have one provenance or the column cannot be labelled"
            )
        weights = SelectionWeights(
            np.asarray(applied, dtype=np.float64),
            SelectionQuantity.APPLIED_WEIGHT,
            applied_sources.pop(),
        )
    return StepSample(
        index=step.index,
        names=tuple(featuriser.names),
        features=features,
        advantages=np.asarray(advantages, dtype=np.float64),
        group_ids=np.asarray(groups, dtype=np.int64),
        task_ids=tuple(tasks),
        advantage_source="recorded",
        applied_weights=weights,
        n_dropped=dropped,
        detail=(
            f"{len(rows)} rollouts over {len(step.groups)} groups"
            + (f"; {dropped} produced no features and were dropped" if dropped else "")
        ),
    )


# ---------------------------------------------------------------------------
# The two halves of the identity
# ---------------------------------------------------------------------------


class UnlabelledSelectionWeights(ValueError):
    """A differential was asked for as `S_applied` and was not computed on the applied weight.

    Raised rather than returned, and raised on `UNLABELLED` as well as on the wrong quantity: an
    array that arrived with no declaration is not evidence that it was the advantage, it is evidence
    that nobody said (BLK-015).
    """

    def __init__(self, quantity: SelectionQuantity, source: str) -> None:
        self.quantity = quantity
        self.source = source
        super().__init__(
            f"this differential was computed on {quantity.value!r} from {source!r}, and "
            f"`S_applied` is defined on the per-rollout applied weight. Pass a "
            f"`SelectionWeights(values, SelectionQuantity.APPLIED_WEIGHT)` to "
            f"`selection_differential`, or read this through `as_dict` under its own name. The "
            f"arithmetic is identical for every quantity, which is why the label is the only thing "
            f"that can tell them apart afterwards"
        )


@dataclass(frozen=True)
class Differential:
    """`Cov_group(A, f)` per feature, with the group-level spread that decides whether to believe it.

    ``by_group`` holds each group's own unbiased covariance contribution, which is the natural
    clustering unit: two rollouts of one prompt are not two independent observations of the policy's
    selection pressure, and a standard error that treats them as such is too narrow by roughly the
    square root of the group size.
    """

    names: tuple[str, ...]
    value: np.ndarray
    standard_error: np.ndarray
    n_scored: int
    n_groups: int
    n_degenerate: int
    operator: str = "within_group"
    #: Which per-rollout scalar this differential was taken against. `UNLABELLED` when the caller
    #: passed a bare array, which is the state every in-repository caller was in before BLK-015.
    quantity: SelectionQuantity = SelectionQuantity.UNLABELLED
    #: Where those numbers came from: recorded, reconstructed, or unknown. Not the same question as
    #: `quantity`, and the ledger used to answer only this one.
    weight_source: str = "unknown"

    def as_dict(self) -> dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, self.value)}

    def as_applied(self) -> dict[str, float]:
        """The same numbers, readable **only** if they are `S_applied`.

        `C4` is a claim about the scalar the loss applied. `Cov_group(A, f)` and
        `Cov_group(a-tilde, f)` are the same arithmetic on different inputs and they differ by
        whatever the aggregation did, so a released artifact that does not say which one produced
        the headline ratio cannot be audited at all. This raises rather than returning, because the
        alternative is a number that reads correct and is a different quantity.
        """
        if self.quantity is not SelectionQuantity.APPLIED_WEIGHT:
            raise UnlabelledSelectionWeights(self.quantity, self.weight_source)
        return self.as_dict()


def selection_differential(
    features: np.ndarray,
    advantages: SelectionWeights | np.ndarray,
    group_ids: np.ndarray,
    names: Sequence[str],
) -> Differential:
    """The within-group covariance of the advantage with each feature, pooled over groups.

    `S_i = Σ_g Σ_j (A_gj - Ā_g)(f_gji - f̄_gi) / (n - G)`, which is the pooled unbiased estimator of
    the within-group covariance: each group's own unbiased covariance weighted by its `k_g - 1`.
    Groups of one contribute nothing and are excluded from both sums, because a single rollout has
    no within-group spread to covary with.

    Both variables are re-centred inside the group even though GRPO has already centred the
    advantage. On a clean group that is arithmetically a no-op; on a group where the grader abstained
    on one rollout it is not, because the surviving advantages no longer sum to zero and the
    uncentred product would pick up the group's mean advantage times the feature mean.

    The standard error is clustered at the group level: the per-group contributions are treated as
    the independent observations and `se = sd(c_g)/sqrt(G)`. It is reported per feature and it is the
    number that decides whether a residual is a finding or a sampling fluctuation.
    """
    if isinstance(advantages, SelectionWeights):
        quantity, weight_source = advantages.quantity, advantages.source
        advantages = advantages.values
    else:
        # A bare array carries no declaration, and the arithmetic below cannot tell `A-hat` from
        # `a-tilde`. Recorded as UNLABELLED rather than assumed to be the advantage: assuming is the
        # substitution BLK-015 exists to stop, and it is silent by construction.
        quantity, weight_source = SelectionQuantity.UNLABELLED, "unknown"
        advantages = np.asarray(advantages, dtype=np.float64)
    if advantages.shape[0] != features.shape[0]:
        raise ValueError(
            f"{advantages.shape[0]} weights against {features.shape[0]} feature rows. A misaligned "
            f"column pairs one rollout's weight with another's features"
        )
    finite = np.isfinite(advantages) & np.all(np.isfinite(features), axis=1)
    f = features[finite]
    a = advantages[finite]
    g = group_ids[finite]
    k = features.shape[1]
    labels = np.unique(g) if g.size else np.asarray([], dtype=np.int64)

    total = np.zeros(k, dtype=np.float64)
    per_group: list[np.ndarray] = []
    weights: list[float] = []
    n_used = 0
    n_degenerate = 0
    for label in labels:
        mask = g == label
        size = int(np.count_nonzero(mask))
        if size < 2:
            n_degenerate += 1
            continue
        fa = f[mask] - f[mask].mean(axis=0)
        aa = a[mask] - a[mask].mean()
        cross = fa.T @ aa
        total += cross
        per_group.append(cross / (size - 1))
        weights.append(float(size - 1))
        n_used += size
    n_groups = len(per_group)
    if n_groups == 0:
        nan = np.full(k, np.nan, dtype=np.float64)
        return Differential(
            names=tuple(names),
            value=nan,
            standard_error=nan,
            n_scored=0,
            n_groups=0,
            n_degenerate=n_degenerate,
            quantity=quantity,
            weight_source=weight_source,
        )
    denominator = float(n_used - n_groups)
    value = total / denominator if denominator > 0 else np.full(k, np.nan)
    stacked = np.vstack(per_group)
    if n_groups > 1:
        se = stacked.std(axis=0, ddof=1) / np.sqrt(n_groups)
    else:
        se = np.full(k, np.nan, dtype=np.float64)
    return Differential(
        names=tuple(names),
        value=value,
        standard_error=se,
        n_scored=n_used,
        n_groups=n_groups,
        n_degenerate=n_degenerate,
        quantity=quantity,
        weight_source=weight_source,
    )


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LedgerRow:
    """One feature, one step: what moved, what was selected for, and what is left over."""

    feature: str
    delta_z: float
    covariance: float
    eta: float
    selection: float
    residual: float
    se_delta_z: float
    se_covariance: float
    z_before: float
    z_after: float

    @property
    def selection_share(self) -> float:
        """`η·Cov / Δz`, or NaN when nothing moved. 1.0 is a fully explained step."""
        return float(self.selection / self.delta_z) if self.delta_z else float("nan")

    @property
    def residual_dominates(self) -> bool:
        """Whether the unexplained half is the larger one. F1's kill condition, per feature."""
        return abs(self.residual) > abs(self.selection)

    def render(self) -> str:
        return (
            f"{self.feature:<20} Δz {self.delta_z:+.5g}  = selection {self.selection:+.5g} "
            f"+ residual {self.residual:+.5g}   (Cov {self.covariance:+.5g}, "
            f"se(Δz) {self.se_delta_z:.3g})"
        )


@dataclass(frozen=True)
class StepLedger:
    """The ledger for one step pair, per feature, with what decides whether to read it.

    ``task_overlap`` is the Jaccard overlap of the two steps' prompt sets. At zero, `Δz` is a
    difference between two different task samples as well as between two policies, and the prompt
    half of that difference is noise the selection term was never going to explain. It is reported
    on every ledger because a run that resamples prompts every step will produce a low `Λ` for a
    reason that has nothing to do with the step being large.
    """

    step: int
    next_step: int
    rows: tuple[LedgerRow, ...]
    eta: float
    eta_source: str
    task_overlap: float
    n_before: int
    n_after: int
    n_scored: int
    n_groups: int
    basis: str = "all_rollouts"
    operator: str = "within_group"
    advantage_source: str = "recorded"
    #: Which per-rollout scalar the covariance on these rows was actually taken against, carried
    #: through from the `Differential` (BLK-015). Not derivable from `advantage_source`: that field
    #: says where the *advantages* came from and is `"recorded"` on a step whose covariance was
    #: taken against `a-tilde` instead, which is the substitution this row exists to make visible.
    quantity: SelectionQuantity = SelectionQuantity.UNLABELLED
    #: The provenance of those numbers: recorded, reconstructed, or unknown. Equal to
    #: `advantage_source` only in the fallback case where no applied weight was recorded.
    weight_source: str = "unknown"
    notes: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(r.feature for r in self.rows)

    def row(self, feature: str) -> LedgerRow:
        for r in self.rows:
            if r.feature == feature:
                return r
        raise KeyError(f"no ledger row for {feature!r}; this ledger holds {list(self.names)}")

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """``(Δz, Cov)`` in feature order, which is what the Λ fit consumes."""
        return (
            np.asarray([r.delta_z for r in self.rows], dtype=np.float64),
            np.asarray([r.covariance for r in self.rows], dtype=np.float64),
        )

    def render(self) -> str:
        head = (
            f"ledger {self.step} -> {self.next_step}  η = {self.eta:.4g} ({self.eta_source}), "
            f"{self.n_scored} scored rollouts in {self.n_groups} groups, "
            f"task overlap {self.task_overlap:.2f}"
        )
        return "\n".join([head, *("    " + r.render() for r in self.rows)])


def _feature_stats(
    sample: StepSample, tasks: frozenset[str] | None
) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean, variance and count of each feature over a step, optionally restricted to some tasks."""
    if tasks is None:
        rows = sample.features
    else:
        keep = np.asarray([t in tasks for t in sample.task_ids], dtype=bool)
        rows = sample.features[keep]
    if rows.shape[0] == 0:
        k = sample.features.shape[1]
        return np.full(k, np.nan), np.full(k, np.nan), 0
    ddof = 1 if rows.shape[0] > 1 else 0
    return rows.mean(axis=0), rows.var(axis=0, ddof=ddof), int(rows.shape[0])


def ledger_between(
    before: StepSample,
    after: StepSample,
    *,
    eta: float,
    eta_source: str = "supplied",
    basis: str = "all_rollouts",
) -> StepLedger:
    """The Price ledger for one step pair.

    ``basis="all_rollouts"`` takes `z` over every rollout of each step. ``basis="shared_tasks"``
    restricts both `z` estimates to the prompts the two steps have in common, which removes the
    prompt-composition half of `Δz` at the cost of every rollout on a prompt that appeared once. Use
    the second when the overlap is high enough to leave a sample; it is refused rather than silently
    emptied when the overlap is zero.
    """
    if before.names != after.names:
        raise ValueError(
            f"the two steps carry different feature bases: {list(before.names)} against "
            f"{list(after.names)}. A `Δz` between two bases is a difference between two different "
            f"quantities."
        )
    shared = before.tasks & after.tasks
    union = before.tasks | after.tasks
    overlap = float(len(shared) / len(union)) if union else 0.0
    if basis == "shared_tasks":
        if not shared:
            raise ValueError(
                f"steps {before.index} and {after.index} share no prompt, so a shared-task Δz has "
                f"no sample. Use basis='all_rollouts' and read the task_overlap of 0.0 as the "
                f"caveat it is."
            )
        restrict: frozenset[str] | None = shared
    else:
        restrict = None

    mean_before, var_before, n_before = _feature_stats(before, restrict)
    mean_after, var_after, n_after = _feature_stats(after, restrict)
    # `before.weights()` rather than `before.advantages`: the applied weight when the record has
    # one, the advantage otherwise, labelled either way, so the ledger row says which it read
    # (BLK-015).
    differential = selection_differential(
        before.features, before.weights(), before.group_ids, before.names
    )
    delta = mean_after - mean_before
    se = np.sqrt(
        np.where(n_before > 0, var_before / max(n_before, 1), np.nan)
        + np.where(n_after > 0, var_after / max(n_after, 1), np.nan)
    )
    rows = tuple(
        LedgerRow(
            feature=name,
            delta_z=float(delta[i]),
            covariance=float(differential.value[i]),
            eta=float(eta),
            selection=float(eta * differential.value[i]),
            residual=float(delta[i] - eta * differential.value[i]),
            se_delta_z=float(se[i]),
            se_covariance=float(differential.standard_error[i]),
            z_before=float(mean_before[i]),
            z_after=float(mean_after[i]),
        )
        for i, name in enumerate(before.names)
    )
    notes: list[str] = []
    if overlap == 0.0:
        notes.append(
            "the two steps share no prompt, so Δz is a difference between two task samples as well "
            "as between two policies. The prompt half of it is noise the selection term cannot "
            "explain and it inflates every residual here."
        )
    if differential.n_degenerate:
        notes.append(
            f"{differential.n_degenerate} group(s) held fewer than two scored rollouts and "
            f"contributed nothing to the covariance."
        )
    return StepLedger(
        step=before.index,
        next_step=after.index,
        rows=rows,
        eta=float(eta),
        eta_source=eta_source,
        task_overlap=overlap,
        n_before=n_before,
        n_after=n_after,
        n_scored=differential.n_scored,
        n_groups=differential.n_groups,
        basis=basis,
        operator=differential.operator,
        advantage_source=before.advantage_source,
        # Straight off the `Differential`, not re-derived here. The estimator is the only place
        # that knows which of the two scalars `before.weights()` handed it (BLK-015).
        quantity=differential.quantity,
        weight_source=differential.weight_source,
        notes=tuple(notes),
    )


def ledger_series(
    samples: Sequence[StepSample],
    *,
    eta: float | str = "supplied",
    eta_by_step: Mapping[int, float] | None = None,
    basis: str = "all_rollouts",
) -> list[StepLedger]:
    """One ledger per consecutive pair. The last step has no successor and therefore no row.

    ``eta_by_step`` supplies a per-step step size, which is what a run with a schedule needs; a
    scalar ``eta`` applies one value to every pair. A step whose size is not available is skipped
    and named by the caller rather than defaulted, because a default step size is a number nobody
    measured entering the one term the ledger exists to separate.
    """
    out: list[StepLedger] = []
    for before, after in zip(samples, samples[1:]):
        if eta_by_step is not None:
            value = eta_by_step.get(before.index)
            if value is None:
                continue
            source = "schedule"
        else:
            value = float(eta)  # type: ignore[arg-type]
            source = "supplied"
        out.append(ledger_between(before, after, eta=value, eta_source=source, basis=basis))
    return out


def learning_rates(run_: Run, window: Window | None = None) -> dict[int, float]:
    """`Step.schedule["learning_rate"]` per step, for the steps that recorded one.

    This is the optimiser's Euclidean step size and the identity is derived for a
    Fisher-preconditioned one, so substituting it makes the *split* between the selection term and
    the residual only as good as that substitution. The ratio between the two is what F2's `η_eff`
    measures, and a reading whose `eta_source` is ``"schedule"`` should be read next to it.
    """
    lo, hi = window if window is not None else whole_run(run_)
    out: dict[int, float] = {}
    for step in run_.steps.slice(lo, hi):
        value = step.schedule.get("learning_rate")
        if value is not None:
            out[int(step.index)] = float(value)
    return out


# ---------------------------------------------------------------------------
# The instruments
# ---------------------------------------------------------------------------


#: What to do about each envelope condition the ledger declares, as an instruction.
#:
#: `BaseObservable.preflight` writes one remedy for every violated condition, "restrict the window
#: to a span where the condition holds", and for two of these three that is not advice a reader can
#: act on: no window makes a large step small, and no window makes an all-fail group informative.
#: A remedy is a user interface, so the instruments here replace it with the sentence that fits the
#: condition that actually failed.
ENVELOPE_REMEDIES: Mapping[RegimeCondition, str] = {
    RegimeCondition.LINEAR_RESPONSE: (
        "Lambda is below the threshold, so the first-order term is not carrying this run and a "
        "ledger row would attribute movement to a term that does not explain it. Read "
        "`SelectionResidual` and F3's cost book, which do not assume the expansion holds; or run "
        "the same window at a smaller learning rate, which is the only intervention that makes a "
        "first-order expansion valid. If Lambda came back unknown rather than low, compute it with "
        "`SelectionExplainedFraction` and pass it as `RegimeInputs.explained_fraction`."
    ),
    RegimeCondition.GROUP_NONDEGENERATE: (
        "Too many groups have no reward spread, so their advantages are zero or noise and the "
        "covariance is taken over rollouts that taught the optimiser nothing. Restrict the window "
        "to steps whose groups score above the floor, or raise the task difficulty spread so the "
        "groups separate. Instrument E4 measures what a z-scoring estimator does to an all-fail "
        "group and is the reading to take first."
    ),
    RegimeCondition.NEAR_POLICY: (
        "Rollouts in this window are stale or were generated by more than one policy version, so "
        "`Delta z` is a difference between two mixtures rather than between two policies. Restrict "
        "the window to steps whose trajectories are single-policy, or supply an importance "
        "correction and read the ledger at the corrected weights."
    ),
}


def _remedy_for(refusal: Refusal, envelope: EnvelopeSpec, regime: Any) -> Refusal:
    """Replace the generic envelope remedy with the one for the condition that failed."""
    if refusal.reason is not RefusalReason.ENVELOPE_VIOLATED:
        return refusal
    violated = [v.condition for v in envelope.violations(regime)]
    lines = [ENVELOPE_REMEDIES[c] for c in violated if c in ENVELOPE_REMEDIES]
    if not lines:
        return refusal
    return Refusal(
        instrument=refusal.instrument,
        reason=refusal.reason,
        detail=refusal.detail,
        remedy=" ".join(lines),
        partial=refusal.partial,
        provenance=refusal.provenance,
        statistics=refusal.statistics,
    )


class _LedgerInstrument(BaseObservable):
    """Shared plumbing for F1's two quantities: one computation, two registered readings.

    `selection.term` and `selection.residual` are the two halves of one subtraction and are computed
    together, but §4.2 gives an instrument one quantity so that two rungs of one ladder can be
    compared. Splitting them into two classes over one computation is what `measure.composition`
    does for the abstention pair and it is the pattern followed here.
    """

    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to: str | None = "Price equation (section 3.1.1)"
    deviations: tuple[str, ...] = (
        "the identity is derived for a Fisher-preconditioned (natural-gradient) step and every "
        "trainer takes a Euclidean one. When `eta_source` is 'schedule' the logged learning rate "
        "stands in for the natural-gradient step size, and the ratio between the two is exactly "
        "what F2's `eta_eff` measures. Read the two together or read `eta_eff` instead.",
        "`z` is estimated from the step's own rollouts, so `Delta z` carries the sampling noise of "
        "two finite batches and, when the batches drew different prompts, the difference between "
        "two task samples. `task_overlap` and `se_delta_z` are on every reading for that reason.",
    )

    requires: AccessMatrix = LEDGER_ACCESS
    substrates = frozenset(Substrate)
    #: Not PRE_RUN, which has no record, and not DEPLOYED, where only the artifact survives.
    phases = frozenset({Phase.IN_RUN, Phase.POST_RUN})
    envelope = LEDGER_ENVELOPE
    #: The advantage is z-scored inside its group, so an affine rescaling of the reward leaves it
    #: unchanged up to the estimator's epsilon, and `Delta z` never touches the reward at all. So
    #: the whole ledger is invariant rather than covariant under `reward.affine`, which is the
    #: substantive difference from the shipped `chi`: `Cov(f, r)` scales by `a` and `Cov(f, A)` does
    #: not. SPEC-ERRATA E13 records the epsilon caveat that makes the invariance approximate.
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    baselines = ("baseline.random_feature", "baseline.permuted_advantage")
    rung = 0

    def __init__(
        self,
        run_: Run,
        featuriser: TrajectoryFeaturiser,
        *,
        window: Window | None = None,
        eta: float | str = "schedule",
        basis: str = "all_rollouts",
    ) -> None:
        self.run = run_
        self.featuriser = featuriser
        self.window = window
        self.eta = eta
        self.basis = basis
        self._computed: list[StepLedger] | None = None

    # -- the two methods of section 4.2, plus a better remedy ---------------

    def preflight(self, ctx: Context) -> PreflightResult:
        """The base preflight, with the envelope remedy specialised to the condition that failed."""
        pre = super().preflight(ctx)
        if pre.ok or pre.refusal is None or self.envelope is None:
            return pre
        from dataclasses import replace as _replace

        return _replace(pre, refusal=_remedy_for(pre.refusal, self.envelope, ctx.regime_reading))

    # -- the computation ----------------------------------------------------

    def compute(self) -> list[StepLedger] | Refusal:
        """Every ledger row in the window, or the refusal that says why there are none."""
        indices = sorted(self.run.steps.indices)
        lo, hi = self.window if self.window is not None else whole_run(self.run)
        inside = [i for i in indices if lo <= i < hi]
        if not inside:
            have = f"steps {min(indices)} to {max(indices)}" if indices else "no steps at all"
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.VOID,
                detail=(
                    f"the window [{lo}, {hi}) contains no recorded steps of run {self.run.id}, "
                    f"which holds {have}."
                ),
                remedy=(
                    "Ask for a window inside the recorded range. If the record genuinely has no "
                    "steps, the run is void rather than negative, and the reason to record is that "
                    "nothing was written rather than that nothing happened."
                ),
                statistics={"window": [lo, hi], "recorded": len(indices)},
            )
        if len(inside) < 2:
            return refuse_incomplete(
                self.name,
                field="a successor step inside the window",
                subject=f"window [{lo}, {hi}) of run {self.run.id}, which holds step {inside[0]}",
                remedy=(
                    f"Widen the window to include step {inside[0] + 1}: the ledger differences a "
                    f"feature mean between consecutive steps, so a one-step window has nothing to "
                    f"difference against. If step {inside[0]} is the last step of the run then it "
                    f"has no ledger row by construction and the series ends at "
                    f"{inside[0] - 1} -> {inside[0]}."
                ),
                window=[lo, hi],
                steps_in_window=len(inside),
            )

        samples = steps_from_run(self.run, self.featuriser, window=(lo, hi))
        empty = [s.index for s in samples if s.n == 0]
        if len(empty) > len(samples) - 2:
            return refuse_incomplete(
                self.name,
                field="rollouts the featuriser could read",
                subject=(
                    f"run {self.run.id} over [{lo}, {hi}): "
                    f"{len(empty)} of {len(samples)} steps produced no feature rows"
                ),
                remedy=(
                    "Supply a featuriser that reads what this record carries. `SurfaceFeatures` "
                    "needs assistant turn text; `RecordedFeatures` needs the converter to have "
                    "written `Trajectory.features`. Check one trajectory by hand before running "
                    "the series."
                ),
                n_empty=len(empty),
                n_steps=len(samples),
            )

        if isinstance(self.eta, str) and self.eta == "schedule":
            rates = learning_rates(self.run, (lo, hi))
            if not rates:
                return refuse_incomplete(
                    self.name,
                    field="schedule['learning_rate']",
                    subject=f"every step of run {self.run.id} in [{lo}, {hi})",
                    remedy=(
                        "Pass `eta=` explicitly, or read F2's `eta_eff`, which fits the step size "
                        "from the movement itself and needs no schedule. The raw covariance and "
                        "`Delta z` are on every ledger row regardless, so the pair is readable "
                        "without a step size; only the split into selection and residual needs one."
                    ),
                )
            ledgers = ledger_series(samples, eta_by_step=rates, basis=self.basis)
        else:
            ledgers = ledger_series(samples, eta=float(self.eta), basis=self.basis)  # type: ignore[arg-type]
        if not ledgers:
            return refuse_incomplete(
                self.name,
                field="a step pair with a step size",
                subject=f"run {self.run.id} over [{lo}, {hi})",
                remedy=(
                    "No consecutive pair in this window has both steps recorded and a learning "
                    "rate on the earlier one. Pass `eta=` explicitly to use one step size for the "
                    "whole window."
                ),
            )
        return ledgers

    # -- the two methods of section 4.2 ------------------------------------

    def estimate(self, ctx: Context) -> Reading:
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        out = self.compute()
        if isinstance(out, Refusal):
            return out
        self._computed = out
        try:
            return run(self, ctx)  # type: ignore[arg-type,no-any-return]
        finally:
            self._computed = None

    def measure(self, ctx: Context) -> "Evidence":
        out = self._computed if self._computed is not None else self.compute()
        if isinstance(out, Refusal):
            raise ValueError(
                f"{self.name}.measure was called on a window that declines to produce Evidence: "
                f"{out.reason.name}. Call `estimate`, which returns the refusal as a value with "
                f"its remedy."
            )
        return ctx.emit(self.payload(out))

    def payload(self, ledgers: Sequence[StepLedger]) -> dict[str, Any]:  # pragma: no cover - base
        raise NotImplementedError


def _agreed_quantity(ledgers: Sequence[StepLedger]) -> SelectionQuantity:
    """The window's quantity when every step agrees, `UNLABELLED` when they do not.

    `_sample_of` already refuses two provenances inside one step. Across the steps of one window
    nothing does, and a window where some steps recorded `a-tilde` and others did not would emit
    the first step's label over a mixed column: the same silent substitution one level up. So a
    disagreement is reported as unlabelled, which is what `as_applied` refuses on.
    """
    seen = {led.quantity for led in ledgers}
    return seen.pop() if len(seen) == 1 else SelectionQuantity.UNLABELLED


def _agreed_weight_source(ledgers: Sequence[StepLedger]) -> str:
    """The window's weight source when every step agrees, `"unknown"` when they do not."""
    seen = {led.weight_source for led in ledgers}
    return seen.pop() if len(seen) == 1 else "unknown"


def _common_payload(ledgers: Sequence[StepLedger]) -> dict[str, Any]:
    """The fields every ledger reading emits, including what the covariance was taken against.

    `quantity` and `weight_source` are here because this dict *is* the released artifact: a reader
    auditing `C4` has the numbers and, before BLK-015, no way to ask whether they were computed on
    the advantage or on the weight the loss applied. `advantage_source` does not answer that. It
    answers where the advantages came from and reads `"recorded"` either way, so the two fields are
    two questions rather than one field under two names.
    """
    return {
        "n_pairs": len(ledgers),
        "steps": [[led.step, led.next_step] for led in ledgers],
        "features": list(ledgers[0].names) if ledgers else [],
        "operator": ledgers[0].operator if ledgers else "within_group",
        "advantage_source": ledgers[0].advantage_source if ledgers else "unknown",
        "quantity": _agreed_quantity(ledgers).value,
        "weight_source": _agreed_weight_source(ledgers),
        "eta": [led.eta for led in ledgers],
        "eta_source": ledgers[0].eta_source if ledgers else "unknown",
        "basis": ledgers[0].basis if ledgers else "all_rollouts",
        "task_overlap": [led.task_overlap for led in ledgers],
        "n_scored": [led.n_scored for led in ledgers],
        "n_groups": [led.n_groups for led in ledgers],
        "notes": sorted({n for led in ledgers for n in led.notes}),
    }


class SelectionTerm(_LedgerInstrument):
    """F1's first half: `η · Cov_group(A, f_i)`, the movement this step's selection pressure buys.

    Reports the raw covariance alongside the term, because the covariance is what the record
    determines and the term is the covariance times a step size whose provenance is a separate
    question. A reading whose `eta_source` is ``"schedule"`` is reporting the optimiser's Euclidean
    learning rate in a place the derivation wants a natural-gradient step, and F2's `eta_eff` is the
    measured alternative.

    The kill condition in the catalogue is about the other half: if `ρ` is the same size as this
    term for every feature, the first-order picture explains nothing. `SelectionResidual` reports
    that comparison directly and `Λ` reports it over a window.
    """

    name = "SelectionTerm"
    quantity = "selection.term"

    def payload(self, ledgers: Sequence[StepLedger]) -> dict[str, Any]:
        return {
            **_common_payload(ledgers),
            "selection": [[r.selection for r in led.rows] for led in ledgers],
            "covariance": [[r.covariance for r in led.rows] for led in ledgers],
            "se_covariance": [[r.se_covariance for r in led.rows] for led in ledgers],
            "delta_z": [[r.delta_z for r in led.rows] for led in ledgers],
        }


class SelectionResidual(_LedgerInstrument):
    """F1's second half: `ρ_i = Δz_i − η·Cov_group(A, f_i)`, what moved for some other reason.

    This is the interesting half and it is the one a decomposition normally discards. Its candidate
    sources are separately testable: off-policy staleness, the KL pull toward the reference, entropy
    bonuses, optimiser momentum, gradient interference between prompts in the same batch, and the
    `O(η²)` curvature term. F4 budgets them; this instrument names the total.

    ``se_delta_z`` travels with every row because the comparison that matters is between `ρ` and the
    sampling noise in `Δz`, not between `ρ` and zero. A residual smaller than the standard error of
    the movement it is a residual of is not evidence of an unmodelled term.
    """

    name = "SelectionResidual"
    quantity = "selection.residual"

    def payload(self, ledgers: Sequence[StepLedger]) -> dict[str, Any]:
        return {
            **_common_payload(ledgers),
            "residual": [[r.residual for r in led.rows] for led in ledgers],
            "delta_z": [[r.delta_z for r in led.rows] for led in ledgers],
            "selection": [[r.selection for r in led.rows] for led in ledgers],
            "se_delta_z": [[r.se_delta_z for r in led.rows] for led in ledgers],
            "residual_dominates": [
                [bool(r.residual_dominates) for r in led.rows] for led in ledgers
            ],
        }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_LEDGER_BIAS = BiasStatement(
    direction="unknown",
    why=(
        "the covariance is unbiased for the within-group population covariance under the pooled "
        "n-G denominator, and `Delta z` is unbiased for the change in the feature mean. The split "
        "between them is not: it inherits whatever error is in the step size, and a step size read "
        "off the schedule is a Euclidean one standing in for a natural-gradient one. The direction "
        "of that error is a property of the run's curvature and is not knowable from the record."
    ),
)

_LEDGER_COST = CostModel(
    note="one pass over the window's rollouts plus the featuriser; no grader calls, no GPU"
)


def _register() -> None:
    """One rung each for the two halves. The catalogue gives F1 exactly one and this is it."""
    for quantity, impl in (
        ("selection.term", "selection.term.record_pair"),
        ("selection.residual", "selection.residual.record_pair"),
    ):
        register_estimator(
            EstimatorEntry(
                quantity=quantity,
                impl=impl,
                requires=LEDGER_ACCESS,
                envelope=LEDGER_ENVELOPE,
                rung=0,
                bias=_LEDGER_BIAS,
                cost=_LEDGER_COST,
                phases=frozenset({Phase.IN_RUN, Phase.POST_RUN}),
                run=None,
            )
        )


_register()


# ===========================================================================
# BLK-014: the contrast between two labelled populations
# ===========================================================================
#
# SCOPE NOTE for the integrator. Everything below is additive and nothing above it was touched.
# `P2-LOSS` also writes this file. This packet's whole edit to `price.py` is this one block plus
# five names appended to `__all__`.
#
# `selection_differential` above returns `Cov_group(A, f)`. For a 0/1 label that is the
# fixed-effects contrast multiplied by the label's mean within-group variance and by `n / (n - G)`,
# and it carries no per-population mean, no per-population count and no interval. So Part 6.1's
# `Delta_gap` was reachable only by dividing one number by another somewhere downstream, which the
# gap ledger's own rule says does not close a gap.
#
# **That factor was written here as `G / (n - G)` and it is `n / (n - G)`.** At this design's
# geometry the two are 1.2 and 0.2, so anybody backing a contrast out of a recorded covariance with
# the shipped comment in hand was out by `n / G`, the group size, which is 6. The derivation:
# `sum_j (A_gj - Abar_g)(d_gj - dbar_g)` is `w_g * Delta_g` with `w_g = n_high * n_low / k_g`, so
# summing over groups gives `beta * sum_g w_g`, and `sum_g w_g = sum_g k_g * Var_g(d) = n * Vbar`.
# The estimator's own denominator is `n - G`. Recomputed against the shipped estimator at
# `chain/repair/proofs/BLK-014-a/residual_factor.py`, which reproduces `n / (n - G)` to nine figures
# and puts `G / (n - G)` out by exactly 6.000000000.
#
# Part 6.1's estimand, verbatim:
#
#     Delta_gap = E[A^vuln - A^corr | EXPLOIT] - E[A^vuln - A^corr | LEGITIMATE]
#     estimated as a regression of `A^vuln - A^corr` on the label with prompt-group fixed effects,
#     clustered at the group level.
#
# and Part 6.3 makes it three-way: "the exploit-versus-failure contrast reported alongside the
# exploit-versus-legitimate one, so a reader can see which of the two any claim leans on".
#
# **Nothing about the interval is chosen here.** BLK-065 records that eleven registered rows resolve
# on a bootstrap and no document gives a replicate count, an interval type or a resampling level,
# and that the resampling level is contradictory across Parts 5.1, 6.1 and 8.4. So `cluster_ids`,
# `n_resamples` and `ci_level` are required arguments, and `cluster_ids` is separate from
# `group_ids` precisely because the design does not agree with itself about what a cluster is.


class CrossedClusterDesign(ValueError):
    """A prompt group appears in more than one cluster, so the groups are not nested in them.

    `labelled_contrast` aggregates **group-level** scores into clusters, both for the analytic
    standard error and for the resampling unit the bootstrap draws. That aggregation assumes each
    contributing group sits in exactly one cluster. When it does not, there is no such thing as
    "the group's cluster", and the only reason a number came back is that a dict comprehension over
    pairs kept whichever cluster happened to come last (BLK-014-a).

    A `ValueError` subclass, so callers that already catch the structural refusals this module
    raises keep working, and so a test can assert this refusal and no other.
    """

    def __init__(self, group: Any, clusters: Sequence[Any], n_crossed: int, n_groups: int) -> None:
        self.group = group
        self.clusters = tuple(clusters)
        self.n_crossed = int(n_crossed)
        self.n_groups = int(n_groups)
        super().__init__(
            f"prompt group {group!r} appears in {len(self.clusters)} clusters "
            f"({', '.join(repr(c) for c in self.clusters)}), and {n_crossed} of the {n_groups} "
            f"groups contributing to this contrast are crossed with the clustering the same way. "
            f"The interval this function computes aggregates one score per group into clusters, so "
            f"a crossed design is not the design it estimates. Either cluster at a unit the groups "
            f"nest inside (pass `group_ids` as `cluster_ids` to cluster at the group level), or use "
            f"an estimator for crossed random effects. Collapsing the crossing silently is what "
            f"this refusal exists to stop: on a partially crossed sample it returned "
            f"[0.1009, 1.3379] where a bootstrap on the same input gave [0.6707, 0.7681]"
        )


def _cluster_of_group(groups: np.ndarray, clusters: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """The cluster each contributing group belongs to, refusing when a group belongs to several.

    Only the groups in ``kept`` are checked, because they are the only ones whose scores enter the
    aggregation. A degenerate group that straddles clusters contributes no score and so cannot move
    the interval, and refusing on it would make the refusal depend on which contrast was requested
    rather than on whether the answer is computable.
    """
    seen: dict[Any, list[Any]] = {}
    for g, c in zip(groups.tolist(), clusters.tolist()):
        bucket = seen.setdefault(g, [])
        if c not in bucket:
            bucket.append(c)
    contributing = kept.tolist()
    crossed = [g for g in contributing if len(seen.get(g, ())) > 1]
    if crossed:
        raise CrossedClusterDesign(crossed[0], seen[crossed[0]], len(crossed), len(contributing))
    return np.asarray([seen[g][0] for g in contributing])


@dataclass(frozen=True)
class PopulationMean:
    """One label's mean, its clustered standard error, and the counts behind both."""

    label: str
    mean: float
    standard_error: float
    n: int
    n_groups: int


@dataclass(frozen=True)
class Contrast:
    """The within-group difference between two labelled populations, with its interval.

    ``value`` is the prompt-group fixed-effects contrast: the weighted mean of each group's own
    difference, weighted by `n_high * n_low / (n_high + n_low)`, which is the exact coefficient a
    two-level factor regression with group fixed effects returns. ``pooled_value`` is the naive
    difference of the two population means, reported beside it because the gap between them is how
    much the group composition was doing, and a corrected number with no uncorrected one beside it
    is not checkable.

    ``n_groups_informative`` counts the groups carrying both labels, which are the only ones
    contributing; ``n_groups_degenerate`` counts the rest. Part 6.1's estimand is explicitly over
    "prompt groups carrying variation in the exploit label", so a contrast estimated on eight of
    forty groups is a different measurement from one estimated on all forty and the two are
    indistinguishable without the counts.

    ``ci_method`` is ``"clustered-normal"`` when ``n_resamples`` is zero and ``"cluster-percentile"``
    otherwise. Neither is registered; see BLK-065.
    """

    high: str
    low: str
    value: float
    pooled_value: float
    standard_error: float
    ci_low: float
    ci_high: float
    ci_level: float
    ci_method: str
    n_resamples: int
    n_groups_informative: int
    n_groups_degenerate: int
    cluster_level_n: int


@dataclass(frozen=True)
class LabelledContrast:
    """Every population's mean and every requested contrast, from one call."""

    populations: tuple[PopulationMean, ...]
    contrasts: tuple[Contrast, ...]
    n_scored: int
    n_dropped: int
    n_groups: int
    detail: str

    def contrast(self, high: str, low: str) -> Contrast:
        for c in self.contrasts:
            if c.high == high and c.low == low:
                return c
        raise KeyError(f"no contrast between {high!r} and {low!r} was requested")


def _group_difference(
    values: np.ndarray,
    is_high: np.ndarray,
    is_low: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per group: its own high-minus-low difference, its regression weight, and its id.

    The weight is `n_high * n_low / (n_high + n_low)`, which is `sum_j (d_gj - dbar_g)^2` for a 0/1
    indicator inside the group. Weighting each group's difference by it and dividing by the total is
    algebraically the fixed-effects coefficient, so this is the registered estimator computed
    directly rather than a second one that happens to agree.
    """
    labels = np.unique(groups)
    diffs, weights, kept = [], [], []
    for label in labels:
        in_group = groups == label
        high = in_group & is_high
        low = in_group & is_low
        n_high = int(np.count_nonzero(high))
        n_low = int(np.count_nonzero(low))
        if n_high == 0 or n_low == 0:
            continue
        diffs.append(float(values[high].mean() - values[low].mean()))
        weights.append(n_high * n_low / (n_high + n_low))
        kept.append(label)
    return (
        np.asarray(diffs, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        np.asarray(kept),
    )


def labelled_contrast(
    values: Any,
    labels: Any,
    group_ids: Any,
    *,
    cluster_ids: Any,
    contrasts: Sequence[tuple[str, str]],
    n_resamples: int,
    ci_level: float,
    seed: int = 0,
) -> LabelledContrast:
    """Part 6.1's `Delta_gap` and Part 6.3's second contrast, from one call.

    ``values`` is the per-rollout quantity being decomposed, ``labels`` the behavioural label on
    each rollout (`EXPLOIT`, `LEGITIMATE`, `FAIL`), ``group_ids`` the prompt group, and
    ``cluster_ids`` the unit the interval resamples.

    ``group_ids`` and ``cluster_ids`` are two arguments and not one. The fixed effects are on the
    prompt group because that is what Part 6.1 registers, and the resampling unit is open: BLK-065
    records Part 6.1 saying "the clone-adjusted prompt group", Part 5.1 saying "the run" and Part
    8.4 saying "The run", and this packet is not the place that is settled. Pass the same array
    twice to cluster at the group level.

    ``n_resamples = 0`` gives the clustered-normal interval; anything positive gives a cluster
    percentile bootstrap through `stats.ess.cluster_bootstrap`, which is this package's cluster
    resampling and is reused rather than rewritten. Neither the count, the type nor the level has a
    registered value, so all three arrive from the caller.

    Raises rather than returning a pooled difference when no group carries both labels of a
    requested contrast: the within-group estimand is not defined there, and answering a different
    question under the same name is how a row gets resolved by an estimator nobody chose.

    Raises `CrossedClusterDesign` when a contributing prompt group appears in more than one cluster.
    The interval aggregates one score per group into clusters, so the groups have to nest inside
    them; a crossed design is a different estimator, not a wider interval (BLK-014-a).
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    lab = np.asarray(labels, dtype=object).ravel()
    grp = np.asarray(group_ids).ravel()
    clu = np.asarray(cluster_ids).ravel()
    if not (v.size == lab.size == grp.size == clu.size):
        raise ValueError(
            f"values, labels, group_ids and cluster_ids must be the same length; got "
            f"{v.size}, {lab.size}, {grp.size}, {clu.size}"
        )
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must lie strictly inside (0, 1); got {ci_level}")
    if n_resamples < 0:
        raise ValueError(f"n_resamples must not be negative; got {n_resamples}")

    finite = np.isfinite(v)
    n_dropped = int(v.size - np.count_nonzero(finite))
    v, lab, grp, clu = v[finite], lab[finite], grp[finite], clu[finite]
    if v.size == 0:
        raise ValueError("every value is non-finite; there is nothing to contrast")

    present = set(lab.tolist())
    for high, low in contrasts:
        for name in (high, low):
            if name not in present:
                raise ValueError(
                    f"the contrast ({high!r}, {low!r}) names {name!r}, which appears on no rollout. "
                    f"Present labels: {sorted(map(str, present))}"
                )

    populations = []
    for name in sorted(present, key=str):
        mask = lab == name
        rows = v[mask]
        clusters = np.unique(clu[mask])
        per_cluster = np.asarray([rows[clu[mask] == c].mean() for c in clusters])
        se = (
            float(per_cluster.std(ddof=1) / np.sqrt(per_cluster.size))
            if per_cluster.size > 1
            else float("nan")
        )
        populations.append(
            PopulationMean(
                label=str(name),
                mean=float(rows.mean()),
                standard_error=se,
                n=int(rows.size),
                n_groups=int(np.unique(grp[mask]).size),
            )
        )
    by_label = {p.label: p for p in populations}

    out: list[Contrast] = []
    for high, low in contrasts:
        is_high, is_low = lab == high, lab == low
        diffs, weights, kept = _group_difference(v, is_high, is_low, grp)
        n_informative = int(kept.size)
        if n_informative == 0:
            raise ValueError(
                f"no group carries both {high!r} and {low!r}, so the within-group contrast Part 6.1 "
                f"registers is not estimable on this sample. A pooled difference is a different "
                f"estimand and is not returned under this name"
            )
        n_degenerate = int(np.unique(grp[is_high | is_low]).size - n_informative)
        point = float(np.sum(weights * diffs) / np.sum(weights))

        group_cluster = _cluster_of_group(grp, clu, kept)
        cluster_labels = np.unique(group_cluster)
        n_clusters = int(cluster_labels.size)
        denominator = float(np.sum(weights))
        scores = np.asarray(
            [
                float(
                    np.sum(weights[group_cluster == c] * diffs[group_cluster == c])
                    - point * np.sum(weights[group_cluster == c])
                )
                for c in cluster_labels
            ]
        )
        if n_clusters > 1:
            se = float(np.sqrt(n_clusters / (n_clusters - 1) * np.sum(scores**2)) / denominator)
        else:
            se = float("nan")

        if n_resamples == 0:
            from scipy.stats import t as student_t

            # Student's t on `C - 1` degrees of freedom, not a normal. A cluster-robust standard
            # error is an average of `C` squared scores, so at small `C` it is downward-biased and a
            # normal critical value gives an interval that is too narrow. Measured on this
            # estimator at the geometry BLK-065 is about: at 60 clusters the normal interval covers
            # at 0.955 against a nominal 0.95, and at 5 clusters it covers at 0.897. Five is the
            # number of runs this design has, so that is not a corner case here, it is the design.
            crit = float(student_t.ppf(0.5 + ci_level / 2, df=max(n_clusters - 1, 1)))
            lo, hi = point - crit * se, point + crit * se
            method = "clustered-t"
        else:
            from reward_lens.stats.ess import cluster_bootstrap

            def _statistic(index: np.ndarray, _d=diffs, _w=weights) -> float:
                rows = index.astype(np.int64)
                total = float(np.sum(_w[rows]))
                return float(np.sum(_w[rows] * _d[rows]) / total) if total > 0 else float("nan")

            booted = cluster_bootstrap(
                np.arange(diffs.size, dtype=np.float64),
                list(group_cluster),
                statistic=_statistic,
                n_resamples=int(n_resamples),
                ci=ci_level,
                seed=seed,
            )
            lo, hi = float(booted.ci_low), float(booted.ci_high)
            method = "cluster-percentile"

        out.append(
            Contrast(
                high=str(high),
                low=str(low),
                value=point,
                pooled_value=float(by_label[str(high)].mean - by_label[str(low)].mean),
                standard_error=se,
                ci_low=lo,
                ci_high=hi,
                ci_level=float(ci_level),
                ci_method=method,
                n_resamples=int(n_resamples),
                n_groups_informative=n_informative,
                n_groups_degenerate=n_degenerate,
                cluster_level_n=n_clusters,
            )
        )

    return LabelledContrast(
        populations=tuple(populations),
        contrasts=tuple(out),
        n_scored=int(v.size),
        n_dropped=n_dropped,
        n_groups=int(np.unique(grp).size),
        detail=(
            f"{v.size} rollouts over {np.unique(grp).size} prompt groups, "
            f"{len(populations)} populations, {len(out)} contrasts"
            + (f"; {n_dropped} non-finite values dropped" if n_dropped else "")
        ),
    )


__all__ = [
    "LEDGER_ACCESS",
    "LEDGER_ENVELOPE",
    "Contrast",
    "CrossedClusterDesign",
    "UnlabelledSelectionWeights",
    "Differential",
    "LabelledContrast",
    "LedgerRow",
    "PopulationMean",
    "SelectionResidual",
    "SelectionTerm",
    "StepLedger",
    "StepSample",
    "Window",
    "advantages_from_rewards",
    "labelled_contrast",
    "ledger_between",
    "ledger_series",
    "learning_rates",
    "selection_differential",
    "steps_from_run",
    "whole_run",
]
