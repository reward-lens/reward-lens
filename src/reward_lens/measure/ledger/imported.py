"""What has to be established about a training run somebody else published, before it is read.

A record this library wrote carries its own estimator specification, its own group boundaries and
its own step index, so nothing downstream has to guess. An imported run carries none of that. It
carries columns, and the four things that decide what those columns mean are properties of the
trainer rather than of the table: **what the reward was made of, what the optimiser's group was,
which advantage convention was applied, and which axis the step index is on.**

Every one of those four was assumed rather than established in this library's first adapter, and
every one of the four was wrong on the first real artifact it was pointed at. That is not a story
about one publisher. It is what happens when a keyword argument with a plausible default stands in
for a determination, because a default is an answer nobody had to write down and nobody can audit.

So the four are types here, and they are required rather than defaulted on the imported path. The
cost is that reading somebody else's table now takes four sentences of setup. The benefit is that
each of those sentences is a claim with a source attached, and a claim with a source can be wrong
in a way a caller can see.

**The refusal that matters most is the first one.** A reward component published as a binarisation
of itself is a missing component, not a coarse one, because the difference between a score and its
own indicator is reward the reconstruction cannot see. `ComposedReward.operational` returns
`RECORD_INCOMPLETE` in that case and names the column and the resolution the remedy needs. What it
does not do is return the best available composite under the name of the reward, which is what a
caller wants and is exactly the substitution that produces a confident wrong number.
`ComposedReward.proxy` returns that composite under a name that says what it is.

**None of this is specific to one publisher.** Any rollout table plus any Hugging Face
`trainer_state.json` fits these types, and the four determinations are the four a reader of any
published run has to make.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from reward_lens.core.reading import Refusal, RefusalReason, refuse_incomplete, refuse_undefined

# ---------------------------------------------------------------------------
# 1. Step axes, and the alignment between them
# ---------------------------------------------------------------------------


class StepAxis(enum.Enum):
    """Which clock a series is indexed by.

    Four of these appear in a single published run and they do not agree. The rollout file index
    counts evaluation artifacts in whatever order they were written; the trainer log step counts
    optimiser steps as the trainer numbered them; the optimiser step is what the update was applied
    at; the checkpoint step is what a directory name says. Joining two series is only meaningful
    once somebody has said which axis each is on and what the offset between them is.
    """

    #: The chronological index of the eval file a batch was written to. Inferred from file order.
    ROLLOUT_FILE_INDEX = "rollout_file_index"
    #: The `step` field of a `trainer_state.json` entry. Recorded by the trainer.
    TRAINER_LOG_STEP = "trainer_log_step"
    #: The optimiser update ordinal. Equal to the log step on every trainer in scope, and named
    #: separately because that equality is a fact about those trainers rather than a definition.
    OPTIMISER_STEP = "optimiser_step"
    #: The integer in a `checkpoint-N` directory name.
    CHECKPOINT_STEP = "checkpoint_step"


class AlignmentStrength(enum.Enum):
    """How well an offset between two axes is established, worst to best.

    The ordering is the point. A lead time computed across an `ASSUMED` alignment is a lead time
    denominated in somebody's guess about file ordering, and the whole reason this enum exists is
    that such a guess is invisible once the number is written down.
    """

    #: Nothing but the order the files came in. This is a guess and it is labelled as one.
    ASSUMED = "assumed"
    #: A residual is smallest at this offset. A fit, so it degrades when the predictor is poor.
    FITTED = "fitted"
    #: An exact arithmetic identity holds at this offset and at no other. Does not degrade when a
    #: reconstruction is incomplete, which is what makes it worth separating from a fit.
    IDENTITY = "identity"
    #: The artifact states the offset. Nothing to establish.
    RECORDED = "recorded"


@dataclass(frozen=True)
class AxisAlignment:
    """`target = source + offset`, with the evidence for the offset and how strong it is.

    Carried as an object rather than passed as an integer because an integer in a function call is
    a number somebody typed and an object is a determination somebody made. The difference shows up
    the first time a reader asks where the `+1` came from.
    """

    source: StepAxis
    target: StepAxis
    offset: int
    strength: AlignmentStrength
    evidence: str
    statistics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source is self.target and self.offset != 0:
            raise ValueError(
                f"an alignment from {self.source.value!r} to itself has offset {self.offset}. An "
                f"axis is not displaced from itself; one of the two axes is misnamed."
            )
        if not self.evidence.strip():
            raise ValueError(
                f"the {self.source.value} -> {self.target.value} alignment carries no evidence. An "
                f"offset with no stated reason is the assumption this type exists to make visible."
            )

    @property
    def is_assumed(self) -> bool:
        return self.strength is AlignmentStrength.ASSUMED

    def to(self, index: Sequence[float] | np.ndarray) -> np.ndarray:
        """Source-axis values as target-axis values."""
        return np.asarray(index, dtype=np.float64).ravel() + float(self.offset)

    def inverse(self) -> "AxisAlignment":
        return AxisAlignment(
            source=self.target,
            target=self.source,
            offset=-self.offset,
            strength=self.strength,
            evidence=self.evidence,
            statistics=dict(self.statistics),
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "target": self.target.value,
            "offset": self.offset,
            "strength": self.strength.value,
            "evidence": self.evidence,
            "statistics": dict(self.statistics),
        }


def identity_alignment(axis: StepAxis) -> AxisAlignment:
    """The zero-offset alignment of an axis with itself, for callers joining like with like."""
    return AxisAlignment(
        source=axis,
        target=axis,
        offset=0,
        strength=AlignmentStrength.RECORDED,
        evidence="the two series are on the same axis, so no displacement applies",
    )


@dataclass(frozen=True)
class JoinedSeries:
    """Two series paired step by step, with what the pairing dropped and what it was built on."""

    source_values: np.ndarray
    target_values: np.ndarray
    target_steps: np.ndarray
    alignment: AxisAlignment
    n_source_unmatched: int
    n_target_unmatched: int

    @property
    def n(self) -> int:
        return int(self.target_steps.size)


def join_on_axis(
    source_values: Sequence[float] | np.ndarray,
    source_index: Sequence[float] | np.ndarray,
    target_values: Sequence[float] | np.ndarray,
    target_steps: Sequence[float] | np.ndarray,
    *,
    alignment: AxisAlignment | None,
    instrument: str,
    allow_assumed: bool = False,
) -> JoinedSeries | Refusal:
    """Pair a source-axis series with a target-axis one, by step **value** and never by position.

    Two things here are load-bearing and both were wrong in the code this replaces.

    The first is that `alignment` has no default. A caller who has not established the offset gets a
    refusal rather than a zero, because an implicit join is the error that does not announce itself:
    both series have the right length, the arithmetic runs, and every number afterwards is off by
    however many steps nobody checked.

    The second is that the target is indexed by the value of its step field rather than by its
    position in the array. A trainer log with a gap in it, or one that starts at step 1 rather than
    0, silently shifts every pairing under positional indexing, and a run with a missing index is
    exactly the case where a step-axis question is being asked in the first place.
    """
    if alignment is None:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.UNIT_MISMATCH,
            detail=(
                "two series on different step axes were joined with no alignment. The offset "
                "between a rollout file index and a trainer log step is a property of the "
                "publication and is not recoverable from either series on its own."
            ),
            remedy=(
                "Establish the offset and pass an `AxisAlignment` carrying it, its evidence and "
                "its strength. `measure.ledger.reconstruct.shift_residual_curve` fits one; an "
                "order-statistic identity establishes one without a fit and does not degrade when "
                "the reconstruction is incomplete."
            ),
            statistics={"n_source": int(np.asarray(source_index).size)},
        )
    if alignment.is_assumed and not allow_assumed:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"the {alignment.source.value} to {alignment.target.value} alignment is assumed "
                f"rather than established: {alignment.evidence}"
            ),
            remedy=(
                "Establish the offset against something the publication records, or pass "
                "`allow_assumed=True` and state in the write-up that every step-indexed quantity "
                "downstream is denominated in an inference about file ordering."
            ),
            statistics={"offset": alignment.offset, "strength": alignment.strength.value},
        )

    source = np.asarray(source_values, dtype=np.float64).ravel()
    index = np.asarray(source_index, dtype=np.float64).ravel()
    target = np.asarray(target_values, dtype=np.float64).ravel()
    steps = np.asarray(target_steps, dtype=np.float64).ravel()
    if source.size != index.size:
        raise ValueError(
            f"{instrument}: {source.size} source values against {index.size} source index entries. "
            f"A series and its own index are the same length or one of them is not what it says."
        )
    if target.size != steps.size:
        raise ValueError(
            f"{instrument}: {target.size} target values against {steps.size} target steps."
        )

    wanted = alignment.to(index)
    position = {float(s): i for i, s in enumerate(steps)}
    keep_source: list[int] = []
    keep_target: list[int] = []
    for i, want in enumerate(wanted):
        hit = position.get(float(want))
        if hit is not None:
            keep_source.append(i)
            keep_target.append(hit)
    si = np.asarray(keep_source, dtype=np.intp)
    ti = np.asarray(keep_target, dtype=np.intp)
    return JoinedSeries(
        source_values=source[si],
        target_values=target[ti],
        target_steps=steps[ti],
        alignment=alignment,
        n_source_unmatched=int(source.size - si.size),
        n_target_unmatched=int(target.size - ti.size),
    )


def index_gaps(index: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Whether an integer index is contiguous, and exactly which values are absent.

    A published index is usually contiguous and the run where it is not is the run where every
    file-to-step mapping after the gap is a question rather than an assumption. Reported as the
    list of missing values rather than as a count, because which value is missing decides whether
    the gap displaces the tail or merely puts a hole in it.
    """
    arr = np.asarray(index, dtype=np.float64).ravel()
    arr = np.unique(arr[np.isfinite(arr)])
    if arr.size == 0:
        return {"n_distinct": 0, "contiguous": True, "missing": [], "low": None, "high": None}
    low, high = int(arr.min()), int(arr.max())
    present = set(int(v) for v in arr)
    missing = [v for v in range(low, high + 1) if v not in present]
    return {
        "n_distinct": int(arr.size),
        "contiguous": not missing,
        "missing": missing,
        "low": low,
        "high": high,
        "span": high - low + 1,
    }


# ---------------------------------------------------------------------------
# 2. The reward, as a composition of named components
# ---------------------------------------------------------------------------


class ComponentResolution(enum.Enum):
    """How faithfully a published column carries the component the trainer actually summed."""

    #: Published at the resolution the scorer computes. A reconstruction using it is exact.
    EXACT = "exact"
    #: Published on a coarser grid than the scorer's, but still numeric. Recoverable to within the
    #: grid, so a residual has a floor that can be computed rather than argued about.
    COARSENED = "coarsened"
    #: Published as an indicator of itself. This is **not** a coarse version of the component: a
    #: threshold discards the fractional part entirely, so the difference between the column and
    #: the component is unbounded by any grid and unrecoverable by any weighting.
    BINARISED = "binarised"
    #: Not published in any form.
    ABSENT = "absent"
    #: Nobody has checked.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RewardComponent:
    """One term of a trainer's reward, with the provenance of its weight and of its resolution.

    ``source`` is where the weight came from and it is a required sentence rather than an optional
    note. A weight read out of a configuration file, a weight fitted from a regression and a weight
    somebody assumed are three different kinds of number, and the distinction disappears the moment
    they are all stored as floats.
    """

    name: str
    weight: float
    column: str | None
    resolution: ComponentResolution
    source: str
    #: The grid the scorer's own source computes on, where it is known. A format score built from
    #: four independent quarter increments has a scorer grid of 0.25.
    scorer_grid: float | None = None
    #: The grid the published column lands on.
    published_grid: float | None = None
    #: The interval the scorer's source bounds the component to, where it states one.
    bounds: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError(
                f"component {self.name!r} carries no source for its weight of {self.weight}. A "
                f"weight with no provenance cannot be told apart from a fitted coefficient."
            )

    @property
    def is_recoverable(self) -> bool:
        """Whether this component can be read off the table at the resolution the reward used."""
        if self.weight == 0.0:
            return True
        if self.column is None:
            return False
        return self.resolution in (ComponentResolution.EXACT, ComponentResolution.COARSENED)

    @property
    def loses_downward(self) -> bool:
        """Whether substituting the published column can only ever undercount the component.

        True for an indicator of a full score: the indicator is 1 exactly when the score is at its
        ceiling and 0 otherwise, so a composition built on it is bounded above by the true one and
        the residual has a sign that a mis-explanation would not produce.
        """
        return self.resolution is ComponentResolution.BINARISED and self.weight > 0

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weight": self.weight,
            "column": self.column,
            "resolution": self.resolution.value,
            "source": self.source,
            "scorer_grid": self.scorer_grid,
            "published_grid": self.published_grid,
            "bounds": list(self.bounds) if self.bounds else None,
        }


@dataclass(frozen=True)
class ProxyReward:
    """The best composite the published columns support, under a name that says it is not the reward.

    Returned instead of an array so that a caller cannot lose track of what it is holding. Every
    consumer that renders a number takes ``detail`` with it, and `StepSample.detail` carries it
    into the ledger, which is the point: the sentence travels as far as the number does.
    """

    values: np.ndarray
    composed_from: tuple[str, ...]
    missing: tuple[str, ...]
    detail: str

    @property
    def is_exact(self) -> bool:
        return not self.missing


@dataclass(frozen=True)
class ComposedReward:
    """A trainer's reward as named components with weights, and what each is published at.

    The type exists to make one distinction expressible that a `{column: weight}` mapping cannot:
    a component that is present in the table and a component that is present **as a binarisation of
    itself** are different situations, and only the first supports an exact reconstruction.
    """

    components: tuple[RewardComponent, ...]
    intercept: float = 0.0
    source: str = ""

    def __post_init__(self) -> None:
        names = [c.name for c in self.components]
        if len(set(names)) != len(names):
            raise ValueError(f"a composition names a component twice: {sorted(names)}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.components)

    def component(self, name: str) -> RewardComponent:
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(f"no component {name!r}; this composition has {self.names}")

    def missing(self) -> tuple[RewardComponent, ...]:
        """Non-zero-weight components the table cannot supply at the resolution the reward used."""
        return tuple(c for c in self.components if not c.is_recoverable)

    def ceiling(self) -> float | None:
        """The largest reward a rollout can receive, or None when a component is unbounded.

        Every bounded component contributes `weight * upper`, so a configuration whose ceiling is
        below the largest per-step mean the trainer logged is refuted by arithmetic and needs no
        fit to refute it. That is a stronger refutation than any residual, which is why it is a
        method on the composition rather than a step in an analysis.
        """
        total = float(self.intercept)
        for c in self.components:
            if c.bounds is None:
                return None
            lo, hi = c.bounds
            total += c.weight * (hi if c.weight >= 0 else lo)
        return total

    def operational(
        self, table: Mapping[str, Any], *, instrument: str = "ComposedReward.operational"
    ) -> np.ndarray | Refusal:
        """The reward the optimiser actually paid, per rollout, or a refusal naming what is absent.

        This is the accessor a selection-side claim needs and it is the one that refuses. A caller
        that wants the best available composite asks for `proxy` and gets an object that says so.
        """
        gaps = self.missing()
        if gaps:
            worst = gaps[0]
            return refuse_incomplete(
                instrument,
                field=(
                    f"component {worst.name!r} at the resolution the scorer computed it"
                    + (
                        f" (column {worst.column!r} is {worst.resolution.value})"
                        if worst.column
                        else ""
                    )
                ),
                subject="this published rollout table",
                remedy=(
                    "Ask the publisher for the per-rollout "
                    + ", ".join(sorted(c.name for c in gaps))
                    + " value at the resolution its own scorer computes it, or for the final "
                    "per-rollout composite reward. Either one closes this exactly. No weighting "
                    "over the published columns recovers it, because the published column carries "
                    "the component's indicator rather than the component."
                ),
                missing=[c.name for c in gaps],
                resolutions={c.name: c.resolution.value for c in gaps},
                columns={c.name: c.column for c in gaps},
                scorer_grid={c.name: c.scorer_grid for c in gaps},
                published_grid={c.name: c.published_grid for c in gaps},
            )
        return self._combine(table)

    def proxy(
        self, table: Mapping[str, Any], *, instrument: str = "ComposedReward.proxy"
    ) -> ProxyReward | Refusal:
        """The best composite the published columns support, labelled with what it is missing.

        A component with no column at all cannot even be proxied, so that case still refuses. A
        component published as its own indicator is substituted and named in ``missing``.
        """
        absent = [c for c in self.components if c.weight != 0.0 and c.column is None]
        if absent:
            return refuse_incomplete(
                instrument,
                field=f"any column for component(s) {sorted(c.name for c in absent)}",
                subject="this published rollout table",
                remedy=(
                    "A component the table does not carry in any form cannot be proxied. Ask the "
                    "publisher for the column, or drop the component from the composition and "
                    "state that the composite is missing a term."
                ),
                missing=[c.name for c in absent],
            )
        gaps = self.missing()
        direction = (
            "can only undercount the reward, never overcount it"
            if all(c.loses_downward for c in gaps)
            else "differs from the reward in a direction this composition does not constrain"
        )
        detail = (
            "proxy composite over " + ", ".join(f"{c.weight:g}*{c.column}" for c in self.components)
            if not gaps
            else (
                "proxy composite over "
                + ", ".join(f"{c.weight:g}*{c.column}" for c in self.components)
                + "; "
                + ", ".join(f"{c.name} is published {c.resolution.value}" for c in gaps)
                + ", so this "
                + direction
                + " and it is not the operational reward"
            )
        )
        return ProxyReward(
            values=self._combine(table),
            composed_from=tuple(c.column for c in self.components if c.column),
            missing=tuple(c.name for c in gaps),
            detail=detail,
        )

    def _combine(self, table: Mapping[str, Any]) -> np.ndarray:
        n: int | None = None
        for c in self.components:
            if c.column is None:
                continue
            if c.column not in table:
                raise KeyError(
                    f"the table has no column {c.column!r} for component {c.name!r}; it carries "
                    f"{sorted(table)[:12]}"
                )
            n = np.asarray(table[c.column]).ravel().size
            break
        if n is None:
            raise ValueError("a composition with no columned component composes nothing")
        out = np.full(n, float(self.intercept), dtype=np.float64)
        for c in self.components:
            if c.column is None or c.weight == 0.0:
                continue
            column = np.asarray(table[c.column], dtype=np.float64).ravel()
            out = out + float(c.weight) * column
        return out

    def as_json(self) -> dict[str, Any]:
        return {
            "components": [c.as_json() for c in self.components],
            "intercept": self.intercept,
            "source": self.source,
            "ceiling": self.ceiling(),
            "missing": [c.name for c in self.missing()],
        }


# ---------------------------------------------------------------------------
# 3. The advantage convention
# ---------------------------------------------------------------------------


class AdvantageConvention(enum.Enum):
    """What the trainer did to a group's rewards to get its advantages."""

    #: `A = r - mean_g`. Dr. GRPO's recommendation and TRL's `scale_rewards: none`.
    CENTRED = "centred"
    #: `A = (r - mean_g) / (std_g + eps)`. TRL's default.
    STD_NORMALISED = "std_normalised"
    #: Nobody has established which. Every scale-dependent quantity refuses under this.
    UNKNOWN = "unknown"


class ConventionSource(enum.Enum):
    """Where the determination came from, worst to best.

    Kept because the three license different sentences. "The artifact records it" is a fact about
    this run; "every configuration in the publisher's repository says so" is a fact about a family
    the run is presumed to belong to; "we inferred it from the numbers" is a fit.
    """

    UNDETERMINED = "undetermined"
    INFERRED = "inferred"
    CONFIGURATION_FAMILY = "configuration_family"
    ARTIFACT = "artifact"


@dataclass(frozen=True)
class DeclaredConvention:
    """The advantage convention plus where the determination came from and what it needs.

    ``std_ddof`` is separate from the convention and is `None` when nobody has established it, on
    the same reasoning `EstimatorSpec.std_ddof` uses: the two conventions differ by
    `sqrt(K/(K-1))`, which is 3.3 per cent at K=16 and 15.5 per cent at K=4, and a near-certain
    assumption about a denominator is exactly the shape of confident wrong number this library
    exists to prevent.
    """

    convention: AdvantageConvention
    source: ConventionSource
    detail: str
    std_epsilon: float | None = None
    std_ddof: int | None = None

    def __post_init__(self) -> None:
        if not self.detail.strip():
            raise ValueError("a declared convention carries no detail saying how it was determined")
        if self.convention is AdvantageConvention.UNKNOWN:
            if self.source is not ConventionSource.UNDETERMINED:
                raise ValueError(
                    "an UNKNOWN convention cannot have a source; UNDETERMINED is the only "
                    "consistent pair and anything else claims a determination that was not made"
                )

    @property
    def std_normalised(self) -> bool:
        return self.convention is AdvantageConvention.STD_NORMALISED

    def requires(self, *, instrument: str) -> Refusal | None:
        """The refusal this convention forces on a scale-dependent quantity, or None.

        `None` is the pass. A caller computing anything whose magnitude depends on the scale of the
        advantage calls this first and returns whatever it gets back.
        """
        if self.convention is AdvantageConvention.UNKNOWN:
            return Refusal(
                instrument=instrument,
                reason=RefusalReason.RECORD_INCOMPLETE,
                detail=(
                    "the advantage convention is undetermined, so the scale of every reconstructed "
                    f"advantage is undetermined with it. {self.detail}"
                ),
                remedy=(
                    "Read `scale_rewards` (TRL) or the equivalent field from the run's own "
                    "configuration, or establish it from the artifact, and declare it. Ordinal "
                    "quantities are unaffected and can be computed under either convention."
                ),
                statistics={"source": self.source.value},
            )
        if self.std_normalised and self.std_ddof is None:
            return Refusal(
                instrument=instrument,
                reason=RefusalReason.RECORD_INCOMPLETE,
                detail=(
                    "the convention standardises and nobody has established which standard "
                    "deviation. The population and sample forms differ by sqrt(K/(K-1)), which is "
                    "3.3 per cent at a group of sixteen and 15.5 per cent at a group of four."
                ),
                remedy=(
                    "Declare `std_ddof`. Every framework in scope uses 1: TRL's `nanstd` applies "
                    "Bessel's correction explicitly and its older `Tensor.std` path takes torch's "
                    "default of `correction=1`."
                ),
                statistics={"convention": self.convention.value, "source": self.source.value},
            )
        if self.std_normalised and self.std_epsilon is None:
            return Refusal(
                instrument=instrument,
                reason=RefusalReason.RECORD_INCOMPLETE,
                detail="the convention standardises and no epsilon was declared for the divisor",
                remedy="Declare `std_epsilon`; TRL's literal is 1e-4.",
                statistics={"convention": self.convention.value},
            )
        return None

    def amplification(self, *, instrument: str) -> Refusal | None:
        """Refuses when the question does not apply, as against when it cannot be answered."""
        if self.convention is AdvantageConvention.CENTRED:
            return refuse_undefined(
                instrument,
                quantity="advantage amplification",
                subject="a mean-centred estimator",
                instead="the centred advantage's own spread, which is a scale and not a ratio",
                remedy=(
                    "This estimator does not divide by a group standard deviation, so there is no "
                    "amplification to measure at any access and from any record. Read the spread "
                    "of the centred advantages instead."
                ),
            )
        return None

    def as_json(self) -> dict[str, Any]:
        return {
            "convention": self.convention.value,
            "source": self.source.value,
            "detail": self.detail,
            "std_epsilon": self.std_epsilon,
            "std_ddof": self.std_ddof,
        }


#: The convention to declare when nothing is known. Named so the undetermined case is a value a
#: caller passes deliberately rather than a default it inherits.
UNDETERMINED_CONVENTION = DeclaredConvention(
    convention=AdvantageConvention.UNKNOWN,
    source=ConventionSource.UNDETERMINED,
    detail="no configuration, artifact field or inference has established the convention",
)


# ---------------------------------------------------------------------------
# 4. Optimiser group identity, separate from semantic problem identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptGroups:
    """The optimiser's own partition of a batch, and what recovering it cost.

    ``group_id`` is the optimiser's group and ``generation_index`` is the position inside it. The
    semantic problem the prompt came from is a **different** field and is deliberately not here:
    conflating the two is what merges two prompt groups whenever a batch draws the same problem
    twice, which is a group the trainer never normalised over.
    """

    step: np.ndarray
    group_id: np.ndarray
    generation_index: np.ndarray
    source: str
    #: `(step, ordinal)` for every block whose length is not the declared generation count.
    irregular_blocks: tuple[tuple[float, int, int], ...] = ()

    @property
    def n(self) -> int:
        return int(self.group_id.size)

    def keys(self) -> np.ndarray:
        """`(step, group)` as one opaque key per rollout, which is the unit GRPO normalises over."""
        return np.asarray(
            [f"{int(s)}:{int(g)}" for s, g in zip(self.step, self.group_id)], dtype=object
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "n_rollouts": self.n,
            "n_groups": int(np.unique(self.keys()).size),
            "n_irregular_blocks": len(self.irregular_blocks),
            "irregular_blocks": [list(b) for b in self.irregular_blocks[:32]],
        }


def recover_prompt_groups(
    step_index: Sequence[float] | np.ndarray,
    *,
    generations: int,
    instrument: str = "recover_prompt_groups",
    strict: bool = True,
) -> PromptGroups | Refusal:
    """The optimiser's groups from contiguous row blocks, with a size invariant that fires.

    A trainer emitting `num_generations` completions per prompt writes them consecutively, so a run
    of rows is a prompt group and the group identity is a property of the row order rather than of
    any column. That is worth recovering rather than reading off a key, because the semantic
    identifier a publisher exposes is not the optimiser's group and differs from it exactly when a
    batch draws one problem twice.

    The invariant is the part that makes this safe to rely on. If any block is not `generations`
    rows long, the row order is not what this function assumes and every group boundary after the
    short block is wrong. Under ``strict`` that refuses; otherwise the blocks are returned with the
    offenders listed, because a caller reporting the structure of a broken batch needs to see it.
    """
    index = np.asarray(step_index, dtype=np.float64).ravel()
    if generations < 1:
        raise ValueError(
            f"{instrument}: a prompt group holds at least one generation, not {generations}"
        )
    group = np.empty(index.size, dtype=np.int64)
    position = np.empty(index.size, dtype=np.int64)
    seen: dict[float, int] = {}
    for i, value in enumerate(index):
        n = seen.get(float(value), 0)
        group[i] = n // generations
        position[i] = n % generations
        seen[float(value)] = n + 1

    irregular: list[tuple[float, int, int]] = []
    for step in np.unique(index[np.isfinite(index)]):
        mask = index == step
        for g in np.unique(group[mask]):
            size = int(np.sum(mask & (group == g)))
            if size != generations:
                irregular.append((float(step), int(g), size))

    if irregular and strict:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"{len(irregular)} of {int(np.unique(np.asarray([f'{s}:{g}' for s, g in zip(index, group)], dtype=object)).size)} "
                f"recovered blocks are not {generations} rows long, the first being step "
                f"{irregular[0][0]:g} block {irregular[0][1]} at {irregular[0][2]} rows. Row-order "
                f"grouping assumes every prompt's generations are written consecutively and "
                f"completely; a short block means they are not, and every group boundary after it "
                f"is displaced."
            ),
            remedy=(
                "Check `num_generations` against the publisher's configuration and the rows per "
                "step against the batch size. If the table genuinely holds partial groups, pass "
                "`strict=False` and treat the irregular blocks as a reported defect rather than "
                "as groups."
            ),
            statistics={
                "generations": generations,
                "n_irregular": len(irregular),
                "irregular": [list(b) for b in irregular[:32]],
            },
        )
    return PromptGroups(
        step=index,
        group_id=group,
        generation_index=position,
        source=(
            f"contiguous blocks of {generations} rows within each step, recovered from row order "
            f"because the publisher names no optimiser group column"
        ),
        irregular_blocks=tuple(irregular),
    )


def semantic_collisions(
    step_index: Sequence[float] | np.ndarray,
    semantic_id: Sequence[Any] | np.ndarray,
    groups: PromptGroups,
) -> dict[str, Any]:
    """Where a semantic problem appears in more than one optimiser group at the same step.

    A warning about the semantic column and never a merge instruction. Two prompt groups that drew
    the same problem are still two groups: the trainer normalised each of them on its own, and
    keying on the problem would fuse them into a group of `2 * num_generations` that no optimiser
    step ever saw.
    """
    index = np.asarray(step_index, dtype=np.float64).ravel()
    ids = np.asarray(semantic_id, dtype=object).ravel().astype(str)
    collisions: list[dict[str, Any]] = []
    for step in np.unique(index[np.isfinite(index)]):
        mask = index == step
        for key in sorted(set(ids[mask].tolist())):
            sel = mask & (ids == key)
            spanned = sorted(set(int(g) for g in groups.group_id[sel]))
            if len(spanned) > 1:
                collisions.append(
                    {
                        "step": float(step),
                        "semantic_id": key,
                        "groups": spanned,
                        "n_rows": int(sel.sum()),
                    }
                )
    return {
        "n_collisions": len(collisions),
        "steps_affected": sorted({c["step"] for c in collisions}),
        "collisions": collisions[:64],
        "note": (
            "a semantic id spanning two optimiser groups at one step is a property of the batch "
            "sampler, not a reason to merge the groups"
        ),
    }


# ---------------------------------------------------------------------------
# The run, which is the four determinations bound to a table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RolloutColumns:
    """Which published column carries what, for everything that is not the reward.

    The reward is deliberately absent from this mapping. A reward is a composition of components at
    stated resolutions, and letting it be a column name here is what made "the reward is
    `training_passed`" a thing a caller could say without anybody having checked.
    """

    step: str
    file: str
    label: str
    text: str
    #: The publisher's semantic problem identifier, where one exists. Never the optimiser's group.
    semantic_id: str | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "file": self.file,
            "label": self.label,
            "text": self.text,
            "semantic_id": self.semantic_id,
        }


@dataclass(frozen=True)
class ImportedRun:
    """A published rollout table plus the four determinations needed to read it.

    Every field is required. That is the design: a run somebody else produced is not readable until
    four questions have been answered, and a type that let three of them default would be a type
    that let three of them be wrong without anybody writing anything down.
    """

    name: str
    columns: RolloutColumns
    reward: ComposedReward
    convention: DeclaredConvention
    generations: int
    axis: StepAxis
    #: The alignment from this run's own step axis onto the trainer's log. `None` is legal and means
    #: nobody has established it; every join through it then refuses.
    alignment: AxisAlignment | None = None
    note: str = ""

    def prompt_groups(
        self, table: Mapping[str, Any], *, strict: bool = True
    ) -> PromptGroups | Refusal:
        index = np.asarray(table[self.columns.step], dtype=np.float64).ravel()
        return recover_prompt_groups(
            index,
            generations=self.generations,
            instrument=f"{self.name}.prompt_groups",
            strict=strict,
        )

    def operational_reward(self, table: Mapping[str, Any]) -> np.ndarray | Refusal:
        return self.reward.operational(table, instrument=f"{self.name}.operational_reward")

    def proxy_reward(self, table: Mapping[str, Any]) -> ProxyReward | Refusal:
        return self.reward.proxy(table, instrument=f"{self.name}.proxy_reward")

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "columns": self.columns.as_json(),
            "reward": self.reward.as_json(),
            "convention": self.convention.as_json(),
            "generations": self.generations,
            "axis": self.axis.value,
            "alignment": self.alignment.as_json() if self.alignment else None,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# Declared floors, checked for reachability before the data is seen
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FloorReachability:
    """Whether a declared floor is attainable at all on the artifact it was declared for."""

    name: str
    floor: float
    maximum_attainable: float
    reachable: bool
    detail: str
    #: Whether the question was answerable at all. A non-finite maximum does not make a floor
    #: reachable; it makes the check impossible, and the two must not be reported as one.
    #: `reachable` is only meaningful when this is true. SPEC-ERRATA E67.
    established: bool = True

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "floor": self.floor,
            "maximum_attainable": self.maximum_attainable,
            "reachable": self.reachable,
            "established": self.established,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class DeclaredFloor:
    """A threshold an analysis declares in advance, with the rule that produced it.

    ``rule`` is required and it is the field that separates a floor from a number. A floor derived
    from a stated rule can be re-derived on a different artifact; a floor somebody picked can only
    be defended, and the defence is usually written after the data has been seen.
    """

    name: str
    value: float
    unit: str
    rule: str

    def __post_init__(self) -> None:
        if not self.rule.strip():
            raise ValueError(
                f"floor {self.name!r} carries no rule. A threshold with no derivation cannot be "
                f"re-derived on another artifact and cannot be checked for reachability."
            )

    def reachability(self, maximum_attainable: float) -> FloorReachability:
        """Whether this floor can be met at all, given the most the artifact can produce.

        Cheap, mechanical and worth running at freeze time rather than at analysis time. A floor
        above the maximum the artifact can produce is not a demanding threshold that the data
        failed to meet; it is a threshold that was never a test, and the difference is invisible
        once the analysis reports that the floor was not met.

        **A non-finite maximum is not a pass.** This used to return `reachable=True` whenever the
        maximum was `NaN` or infinite, so a floor whose maximum had never been computed sailed
        through the check that exists to catch exactly that. The failure is silent by construction:
        the caller asked whether a threshold was a test, and got back "yes" from an artifact that
        had not been measured. `established` carries the third state, and `check` refuses on it.
        SPEC-ERRATA E67.
        """
        established = bool(math.isfinite(maximum_attainable))
        reachable = bool(maximum_attainable >= self.value) if established else False
        if not established:
            detail = (
                f"the maximum attainable {self.unit} for {self.name!r} is "
                f"{maximum_attainable:g}, which is not a finite number, so whether the declared "
                f"floor of {self.value:g} is reachable was never established"
            )
        else:
            detail = (
                f"the artifact can produce at most {maximum_attainable:g} {self.unit} and the "
                f"declared floor is {self.value:g}"
                + ("" if reachable else ", so no placement of the window can meet it")
            )
        return FloorReachability(
            name=self.name,
            floor=self.value,
            maximum_attainable=float(maximum_attainable),
            reachable=reachable,
            detail=detail,
            established=established,
        )

    def check(self, maximum_attainable: float, *, instrument: str) -> Refusal | None:
        got = self.reachability(maximum_attainable)
        if got.reachable:
            return None
        if not got.established:
            return Refusal(
                instrument=instrument,
                reason=RefusalReason.RECORD_INCOMPLETE,
                detail=got.detail,
                remedy=(
                    f"Compute the maximum attainable {self.unit} for {self.name!r} on this "
                    f"artifact and pass it, so the floor can be checked. A floor whose "
                    f"reachability is unknown has not been checked, and reporting it as reachable "
                    f"is how an unreachable floor reaches a freeze."
                ),
                statistics=got.as_json(),
            )
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.PLAN_NOT_CLOSED,
            detail=(
                f"the declared floor {self.name!r} of {self.value:g} {self.unit} is unreachable on "
                f"this artifact, which produces at most {maximum_attainable:g}. The floor was "
                f"declared as: {self.rule}"
            ),
            remedy=(
                "Re-derive the floor from a rule that references the artifact's own geometry, or "
                "record that this artifact cannot answer the question the floor was gating. Do "
                "not lower the floor to the observed value: a threshold that moves toward the "
                "answer has stopped being a test."
            ),
            statistics=got.as_json(),
        )

    def as_json(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "unit": self.unit, "rule": self.rule}


def check_floors(
    floors: Sequence[DeclaredFloor],
    maxima: Mapping[str, float],
    *,
    instrument: str = "check_floors",
) -> list[Refusal]:
    """Every declared floor against the maximum its artifact can produce, at freeze time.

    Returns the refusals rather than raising on the first, because a specification with two
    unreachable floors should report two.
    """
    out: list[Refusal] = []
    for floor in floors:
        if floor.name not in maxima:
            out.append(
                Refusal(
                    instrument=instrument,
                    reason=RefusalReason.PLAN_NOT_CLOSED,
                    detail=(
                        f"floor {floor.name!r} was declared and no maximum attainable value was "
                        f"supplied for it, so its reachability cannot be checked"
                    ),
                    remedy=(
                        "Supply the largest value the artifact can produce for this quantity. If "
                        "that is not computable before the run, say so on the floor rather than "
                        "leaving the check unrun."
                    ),
                    statistics={"floor": floor.as_json(), "supplied": sorted(maxima)},
                )
            )
            continue
        got = floor.check(maxima[floor.name], instrument=instrument)
        if got is not None:
            out.append(got)
    return out


__all__ = [
    "AdvantageConvention",
    "AlignmentStrength",
    "AxisAlignment",
    "ComponentResolution",
    "ComposedReward",
    "ConventionSource",
    "DeclaredConvention",
    "DeclaredFloor",
    "FloorReachability",
    "ImportedRun",
    "JoinedSeries",
    "PromptGroups",
    "ProxyReward",
    "RewardComponent",
    "RolloutColumns",
    "StepAxis",
    "UNDETERMINED_CONVENTION",
    "check_floors",
    "identity_alignment",
    "index_gaps",
    "join_on_axis",
    "recover_prompt_groups",
    "semantic_collisions",
]
