"""Recovering a trainer's reward from a published rollout table, and checking whether it closed.

`labelled.py` reads a published per-rollout table as ledger steps, and its docstring names the
assumption the whole adapter rests on: the reconstruction is correct only if the column named as the
reward is the column the trainer normalised over. That is a testable claim and this module is the
test. It has no privileged knowledge of any publisher's configuration and does not need any, because
the trainer's own `log_history` carries the per-step reward moments and a composition either
reproduces them or it does not.

**The model.** For each step `t`, with per-rollout candidate columns `c_1 ... c_k`,

    R_t = sum_j ( w_j * mean_t(c_j) ) + b

is fitted against the trainer's logged mean reward alone. Everything else the trainer logged is then
checked without refitting. That ordering is the point: a composition that matches the mean because
it was fitted to the mean has demonstrated nothing, and a composition that also predicts the spread,
the degenerate-group fraction and the batch composition has demonstrated a great deal.

**Weights are recovered, not estimated.** A configured reward has weights that are round numbers. A
regression on collinear columns produces weights that are not, and the difference between those two
situations is the difference between recovering a configuration and describing a correlation. So a
fitted weight is snapped to a simple rational and the snap either succeeds within tolerance or the
fit is reported as a regression rather than a recovery.

**Columns expected to carry zero are the falsification test.** A published table usually contains
adjudication columns, recording what a rollout did rather than what the trainer paid for it. Those
must fit weights whose intervals contain zero. When one does not, the fit has found a collinearity
with a real reward term, not a hidden reward term, and reporting it as a discovery would be the
characteristic error of this kind of work.

**Generally useful beyond the run it was written for.** Nothing here is specific to one publisher.
Any rollout table with per-step grouping plus any Hugging Face checkpoint's `trainer_state.json` can
be put through it, and the shift-residual curve is a step-alignment test for any pair of series
where one is indexed by a true optimiser step and the other by an inferred order.

**Where this module refuses and where it raises.** The library's standing rule is that a `Refusal`
is for a condition the instrument anticipated and an exception is for one it did not. Applied here
that is a real distinction rather than a formality, and the two sites that changed are the ones
where a data condition was being reported as a bug. Too few usable steps to leave any residual
degrees of freedom is a fact about the record, and so is two series with no step in common; both
now return a refusal naming what would fix them. A design and a target of different lengths, a
composition with no weighted column, and an unknown convention string are all caller errors, and
they stay exceptions, because turning a bug into a refusal is how a wrong call site acquires a
polite explanation and survives review.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from reward_lens.core.reading import Refusal, RefusalReason

#: Denominators a configured weight is allowed to have. Wider than any real configuration needs, so
#: that a failure to snap is a fact about the fit rather than about this tuple being too narrow.
SIMPLE_DENOMINATORS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 16)

#: The largest numerator a snap will consider. A reward weight of 25 is possible and a fit that
#: needs one is telling you something; the bound exists so that "snapped" keeps meaning something.
MAX_SIMPLE_NUMERATOR = 24


# ---------------------------------------------------------------------------
# Per-step reduction
# ---------------------------------------------------------------------------


def steps_and_means(
    steps: Sequence[float] | np.ndarray,
    columns: Mapping[str, Sequence[float] | np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """`(distinct steps, column name -> per-step mean)`, with NaN rows excluded per column.

    Excluded per column rather than per row on purpose. A null in an adjudication column is an
    unscored rollout, and dropping that rollout's whole row would change the denominator of every
    other column at that step, which is a silent change to the quantity being fitted.
    """
    axis = np.asarray(steps, dtype=np.float64).ravel()
    order = np.unique(axis[np.isfinite(axis)])
    out: dict[str, np.ndarray] = {}
    for name, values in columns.items():
        arr = np.asarray(values, dtype=np.float64).ravel()
        means = np.full(order.size, np.nan, dtype=np.float64)
        for i, step in enumerate(order):
            block = arr[(axis == step) & np.isfinite(arr)]
            if block.size:
                means[i] = float(block.mean())
        out[name] = means
    return order, out


# ---------------------------------------------------------------------------
# Snapping a fitted weight to a configured constant
# ---------------------------------------------------------------------------


def snap_to_simple_ratio(
    value: float,
    *,
    tolerance: float = 1e-3,
    denominators: Sequence[int] = SIMPLE_DENOMINATORS,
    max_numerator: int = MAX_SIMPLE_NUMERATOR,
) -> tuple[float, str] | None:
    """The nearest simple rational within `tolerance`, as `(value, "p/q")`, or None.

    Returns the *closest* candidate rather than the first inside tolerance, so the answer does not
    depend on the order of `denominators`, and prefers the coarsest representation of an equal
    value: 2/1 rather than 4/2, because a configuration file would have written the first.
    """
    if not math.isfinite(value):
        return None
    best: tuple[float, int, int] | None = None
    for q in sorted(set(int(d) for d in denominators if int(d) > 0)):
        p = int(round(value * q))
        if abs(p) > max_numerator:
            continue
        candidate = p / q
        error = abs(candidate - value)
        if error > tolerance:
            continue
        if (
            best is None
            or error < best[0] - 1e-15
            or (abs(error - best[0]) <= 1e-15 and q < best[2])
        ):
            best = (error, p, q)
    if best is None:
        return None
    _, p, q = best
    return (p / q, f"{p}" if q == 1 else f"{p}/{q}")


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompositionFit:
    """A recovered reward composition, with everything needed to decide whether to believe it."""

    names: tuple[str, ...]
    weights: np.ndarray
    intercept: float
    ci_low: np.ndarray
    ci_high: np.ndarray
    intercept_ci: tuple[float, float]
    condition_number: float
    n_steps: int
    #: `p/q` per weight where the snap succeeded, None where it did not. Same order as `names`.
    snapped: tuple[str | None, ...] = ()
    snapped_values: tuple[float | None, ...] = ()
    intercept_snapped: str | None = None

    def weight(self, name: str) -> float:
        return float(self.weights[self.names.index(name)])

    def interval(self, name: str) -> tuple[float, float]:
        i = self.names.index(name)
        return (float(self.ci_low[i]), float(self.ci_high[i]))

    def contains_zero(self, name: str) -> bool:
        lo, hi = self.interval(name)
        return lo <= 0.0 <= hi

    def all_snapped(self) -> bool:
        return all(s is not None for s in self.snapped) and self.intercept_snapped is not None

    def as_mapping(self, *, snapped: bool = False) -> dict[str, float]:
        """The composition as `{column: weight}`, optionally using the snapped constants.

        `snapped=True` is what a downstream reconstruction should use once the snap has succeeded:
        the configured constant is the quantity of interest and the fitted decimal is an estimate of
        it, so carrying the estimate forward would put fitting noise into every later number.
        """
        if not snapped:
            return {n: float(w) for n, w in zip(self.names, self.weights)}
        out: dict[str, float] = {}
        for name, fitted, snap in zip(self.names, self.weights, self.snapped_values):
            out[name] = float(snap) if snap is not None else float(fitted)
        return out

    def snapped_intercept(self) -> float:
        if self.intercept_snapped is None:
            return float(self.intercept)
        p, _, q = self.intercept_snapped.partition("/")
        return float(p) / float(q) if q else float(p)


def fit_composition(
    means: Mapping[str, np.ndarray],
    target: np.ndarray,
    *,
    names: Sequence[str] | None = None,
    intercept: bool = True,
    n_bootstrap: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
    snap_tolerance: float = 1e-3,
) -> CompositionFit | Refusal:
    """Least squares of the logged per-step reward on per-step column means, with step bootstrap.

    The bootstrap resamples whole steps with replacement, which is the right unit: a step is one
    batch the trainer scored and the rollouts inside it are not independent of each other. Intervals
    from resampling rollouts would be narrower and wrong, and the adjudication-column check is read
    off these intervals, so their width is load-bearing rather than decorative.

    A design and a target of different lengths raises, because the caller has not aligned two step
    axes and that is a bug at the call site rather than a property of the record; `imported.
    join_on_axis` is the aligned join. Too few usable steps after non-finite rows are dropped
    refuses, because that is a property of the record and the remedy is upstream.
    """
    keys = tuple(names) if names is not None else tuple(sorted(means))
    design = np.column_stack([np.asarray(means[k], dtype=np.float64) for k in keys])
    y = np.asarray(target, dtype=np.float64).ravel()
    if design.shape[0] != y.size:
        raise ValueError(
            f"the design has {design.shape[0]} steps and the target has {y.size}. A composition "
            f"fitted across a length mismatch is fitting one run's rewards to another's batches. "
            f"Align the two axes first: `measure.ledger.imported.join_on_axis` takes an "
            f"`AxisAlignment` and returns the pair of series it matched."
        )
    good = np.isfinite(y) & np.all(np.isfinite(design), axis=1)
    n_dropped = int(good.size - good.sum())
    design, y = design[good], y[good]
    if design.shape[0] <= design.shape[1]:
        return Refusal(
            instrument="reconstruct.fit_composition",
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"{design.shape[0]} usable steps against {design.shape[1]} candidate columns, "
                f"after {n_dropped} steps were dropped for carrying a non-finite value in the "
                f"target or in a candidate column. A fit with no residual degrees of freedom "
                f"reproduces its target exactly and has therefore demonstrated nothing."
            ),
            remedy=(
                f"Fit over at least {design.shape[1] + 2} steps carrying a finite value in every "
                f"candidate column, or drop candidate columns until the design has residual "
                f"degrees of freedom. Which columns carry the nulls is worth checking first: a "
                f"column that is null on most steps is not a reward term this record can test."
            ),
            statistics={
                "n_usable_steps": int(design.shape[0]),
                "n_candidate_columns": int(design.shape[1]),
                "n_steps_dropped": n_dropped,
                "columns": list(keys),
            },
        )

    def solve(x: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, float]:
        mat = np.column_stack([x, np.ones(x.shape[0])]) if intercept else x
        beta, *_ = np.linalg.lstsq(mat, obs, rcond=None)
        return (beta[:-1], float(beta[-1])) if intercept else (beta, 0.0)

    weights, b = solve(design, y)

    rng = np.random.default_rng(seed)
    draws = np.empty((max(n_bootstrap, 0), len(keys)), dtype=np.float64)
    intercepts = np.empty(max(n_bootstrap, 0), dtype=np.float64)
    n = design.shape[0]
    for d in range(n_bootstrap):
        pick = rng.integers(0, n, size=n)
        try:
            w_d, b_d = solve(design[pick], y[pick])
        except np.linalg.LinAlgError:  # pragma: no cover - lstsq is very hard to break
            w_d, b_d = np.full(len(keys), np.nan), math.nan
        draws[d] = w_d
        intercepts[d] = b_d
    lo_q, hi_q = 100.0 * (1.0 - ci) / 2.0, 100.0 * (1.0 + ci) / 2.0
    if n_bootstrap > 0:
        lo = np.nanpercentile(draws, lo_q, axis=0)
        hi = np.nanpercentile(draws, hi_q, axis=0)
        b_ci = (
            float(np.nanpercentile(intercepts, lo_q)),
            float(np.nanpercentile(intercepts, hi_q)),
        )
    else:
        lo = hi = np.full(len(keys), np.nan)
        b_ci = (math.nan, math.nan)

    snaps = [snap_to_simple_ratio(float(w), tolerance=snap_tolerance) for w in weights]
    b_snap = snap_to_simple_ratio(float(b), tolerance=snap_tolerance)
    scaled = design / np.maximum(np.abs(design).max(axis=0), 1e-300)
    return CompositionFit(
        names=keys,
        weights=weights,
        intercept=b,
        ci_low=lo,
        ci_high=hi,
        intercept_ci=b_ci,
        condition_number=float(np.linalg.cond(scaled)),
        n_steps=int(n),
        snapped=tuple(s[1] if s else None for s in snaps),
        snapped_values=tuple(s[0] if s else None for s in snaps),
        intercept_snapped=b_snap[1] if b_snap else None,
    )


def compose(
    columns: Mapping[str, Sequence[float] | np.ndarray],
    weights: Mapping[str, float],
    intercept: float = 0.0,
) -> np.ndarray:
    """The per-rollout reward implied by a composition. NaN wherever a weighted column is null.

    NaN rather than a zero fill, because a null in a component the trainer summed is a rollout whose
    reward this composition cannot state, and averaging it in as zero would put a fabricated value
    into the very quantity the closure test is about.
    """
    first = next(iter(weights), None)
    if first is None:
        raise ValueError("a composition with no weighted column composes nothing")
    n = np.asarray(columns[first], dtype=np.float64).ravel().size
    out = np.full(n, float(intercept), dtype=np.float64)
    for name, w in weights.items():
        out = out + float(w) * np.asarray(columns[name], dtype=np.float64).ravel()
    return out


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClosureResidual:
    """A predicted series against a logged one, reported as a pattern and not only as a size.

    The size alone cannot separate a misreconstruction from sampling noise off a parallel stream.
    Sampling noise is symmetric, unbiased and independent across steps; a misreconstruction is
    typically none of the three. `lag1_autocorrelation` and `mean_z` are what make that operational.
    """

    target: str
    n: int
    absolute_rms: float
    relative_rms: float
    mean_absolute_relative: float
    max_absolute: float
    max_relative: float
    max_relative_at_step: float
    mean_residual: float
    mean_z: float
    lag1_autocorrelation: float
    residuals: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    steps: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))

    def at_floor(self, quantisation: float) -> bool:
        """Whether the residual is no larger than half the grid the target was logged on."""
        return quantisation > 0.0 and self.absolute_rms <= quantisation / 2.0

    def floor_ratio(self, quantisation: float) -> float:
        """The residual over the RMS a pure rounding onto that grid would give. 1.0 is exactly it."""
        floor = quantisation / math.sqrt(12.0) if quantisation > 0 else 0.0
        return self.absolute_rms / floor if floor > 0 else math.inf


def closure_residuals(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    target: str = "",
    steps: np.ndarray | None = None,
) -> ClosureResidual | Refusal:
    """Residual statistics for one closure target, over the steps where both series are finite.

    An empty overlap refuses rather than returning a residual of `n = 0` with every statistic NaN.
    The NaN object was honest and it was not safe: `at_floor` and `floor_ratio` both compare a NaN
    against a tolerance, every such comparison is False, and a closure that was never computed then
    reads in a report exactly like a closure that failed.
    """
    pred = np.asarray(predicted, dtype=np.float64).ravel()
    obs = np.asarray(observed, dtype=np.float64).ravel()
    if pred.size != obs.size:
        raise ValueError(
            f"target {target!r}: {pred.size} predicted values against {obs.size} observed. A "
            f"residual across a length mismatch is comparing two different step axes. Align them "
            f"first with `measure.ledger.imported.join_on_axis`."
        )
    axis = (
        np.asarray(steps, dtype=np.float64).ravel()
        if steps is not None
        else np.arange(pred.size, dtype=np.float64)
    )
    good = np.isfinite(pred) & np.isfinite(obs)
    pred, obs, axis = pred[good], obs[good], axis[good]
    if pred.size == 0:
        return Refusal(
            instrument="reconstruct.closure_residuals",
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"target {target!r}: no step carries a finite value in both the predicted and the "
                f"observed series, over {good.size} paired positions. There is no residual to "
                f"report, which is a different statement from a residual that is large."
            ),
            remedy=(
                "Check that the two series were aligned onto the same step axis and that the "
                "target key names a metric this trainer logged. A closure target present under a "
                "different key reads as absent here."
            ),
            statistics={"target": target, "n_paired": int(good.size), "n_finite": 0},
        )
    resid = pred - obs
    abs_rms = float(np.sqrt(np.mean(resid**2)))
    obs_rms = float(np.sqrt(np.mean(obs**2)))
    safe = np.where(np.abs(obs) > 0, np.abs(obs), np.nan)
    rel = np.abs(resid) / safe
    worst = int(np.nanargmax(rel)) if np.any(np.isfinite(rel)) else 0
    sd = float(resid.std(ddof=1)) if resid.size > 1 else 0.0
    mean_z = float(resid.mean() / (sd / math.sqrt(resid.size))) if sd > 0 else 0.0
    if resid.size > 2 and resid.std() > 0:
        centred = resid - resid.mean()
        lag1 = float(np.dot(centred[:-1], centred[1:]) / np.dot(centred, centred))
    else:
        lag1 = math.nan
    return ClosureResidual(
        target=target,
        n=int(resid.size),
        absolute_rms=abs_rms,
        relative_rms=abs_rms / obs_rms if obs_rms > 0 else math.inf,
        mean_absolute_relative=float(np.nanmean(rel)),
        max_absolute=float(np.max(np.abs(resid))),
        max_relative=float(np.nanmax(rel)) if np.any(np.isfinite(rel)) else math.nan,
        max_relative_at_step=float(axis[worst]),
        mean_residual=float(resid.mean()),
        mean_z=mean_z,
        lag1_autocorrelation=lag1,
        residuals=resid,
        steps=axis,
    )


# ---------------------------------------------------------------------------
# Step alignment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShiftCurve:
    """The closure residual as a function of integer shift between two step axes."""

    shifts: tuple[int, ...]
    residuals: tuple[float, ...]
    n_overlap: tuple[int, ...]

    @property
    def best_shift(self) -> int:
        finite = [(r, s) for s, r in zip(self.shifts, self.residuals) if math.isfinite(r)]
        return min(finite)[1] if finite else 0

    def residual_at(self, shift: int) -> float:
        return self.residuals[self.shifts.index(shift)] if shift in self.shifts else math.nan

    def overlap_at(self, shift: int) -> int:
        """How many step pairs the residual at `shift` was computed on.

        Zero for a shift outside the sweep, which reads the same as a shift whose windows do not
        meet, and both are the cases where no residual exists to defend.
        """
        return self.n_overlap[self.shifts.index(shift)] if shift in self.shifts else 0

    def best_shift_with_overlap(self, *, min_overlap: int = 0) -> int:
        """`best_shift`, restricted to shifts scored on at least `min_overlap` step pairs.

        `best_shift` takes the smallest residual in the sweep and says nothing about how many steps
        produced it. Over a window wide enough to reach the registered analysis lag the extreme
        shifts are scored on a handful of pairs, and the smallest residual in the sweep is then
        routinely one of those rather than the alignment. Falls back to `best_shift` when the floor
        excludes everything, so a floor set too high degrades to the shipped answer rather than to
        a silent zero.
        """
        finite = [
            (r, s)
            for s, r, n in zip(self.shifts, self.residuals, self.n_overlap)
            if math.isfinite(r) and n >= min_overlap
        ]
        return min(finite)[1] if finite else self.best_shift

    def has_clear_minimum_at(
        self, shift: int = 0, *, ratio: float = 1.5, min_overlap: int = 0
    ) -> bool:
        """Whether `shift` is the strict minimum and both neighbours are `ratio` times worse.

        A minimum that is merely the smallest of eleven near-equal numbers is not evidence about an
        alignment, so the neighbour ratio is the part of this test that does the work.

        A residual of exactly zero is the one case the ratio cannot express, and it is the strongest
        evidence available rather than a degenerate one: the series reproduce each other bit for bit
        at this shift and at no other. It short-circuits to True once the strict-minimum test has
        already passed. Treating it as a failure, which an earlier version of this method did by
        rejecting any non-positive residual, would have declared the cleanest possible alignment
        unclear.

        `min_overlap` is the floor on how many step pairs a shift must be scored on to take part.
        It defaults to zero, which is this method's shipped behaviour, and it is a parameter rather
        than a constant because the floor is a threshold and this package registers thresholds. The
        case it exists for: over a sweep wide enough to reach lag 33, two independent 36-step series
        put their smallest residual at -33, scored on three pairs, and the neighbour ratio calls it
        clear. The floor removes those shifts from both the candidate and the comparison set, so a
        shift is never declared clear by beating residuals nobody would have believed.
        """
        if self.overlap_at(shift) < min_overlap:
            return False
        here = self.residual_at(shift)
        if not math.isfinite(here) or here < 0:
            return False
        others = [
            r
            for s, r, n in zip(self.shifts, self.residuals, self.n_overlap)
            if s != shift and math.isfinite(r) and n >= min_overlap
        ]
        if not others or min(others) <= here:
            return False
        neighbours = [
            self.residual_at(s) for s in (shift - 1, shift + 1) if self.overlap_at(s) >= min_overlap
        ]
        present = [v for v in neighbours if math.isfinite(v)]
        if not present:
            return False
        if here == 0.0:
            return True
        return all(v / here >= ratio for v in present)


#: The registered analysis lag and its two secondaries. The default sweep is the smallest symmetric
#: contiguous window that reaches all three, because a sweep that stops short of a lag cannot
#: confirm it and cannot refute it either: it reports a property of the window.
REGISTERED_ANALYSIS_LAGS: tuple[int, ...] = (11, 22, 33)

#: The shipped default sweep. It was `range(-5, 6)`, which could not reach lag 11 at all, so the
#: gating check that reads at 11 was being answered by a window that never evaluated it (BLK-022,
#: `CHAIN_GAPS` `G21`). Widening it rather than pushing the range onto the caller is the choice the
#: row's own closure proof forces: the proof is a static check on the sweep's range, so a library
#: whose range is whatever the caller passed has no range to check.
DEFAULT_SHIFTS: range = range(-max(REGISTERED_ANALYSIS_LAGS), max(REGISTERED_ANALYSIS_LAGS) + 1)


def shift_residual_curve(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    shifts: Iterable[int] = DEFAULT_SHIFTS,
) -> ShiftCurve:
    """Residual RMS when the predicted series is read against `observed` displaced by `s` steps.

    Positive `s` compares predicted step `i` against observed step `i + s`. Only the overlapping
    span is used, so a shift is never scored against padding, and the overlap count is returned
    beside each residual because a curve whose ends are computed on fewer steps than its middle is
    not read the same way as one where they are not.

    The default sweep spans `DEFAULT_SHIFTS`, which reaches the registered analysis lag and both
    its secondaries. On a series shorter than about four times the widest shift the far ends of the
    sweep are scored on very few pairs; `ShiftCurve.has_clear_minimum_at` and
    `best_shift_with_overlap` both take a `min_overlap` floor for that case, and it is the caller
    who passes it, because the floor is a threshold.
    """
    pred = np.asarray(predicted, dtype=np.float64).ravel()
    obs = np.asarray(observed, dtype=np.float64).ravel()
    out_shifts: list[int] = []
    out_resid: list[float] = []
    out_n: list[int] = []
    for s in shifts:
        lo = max(0, -s)
        hi = min(pred.size, obs.size - s)
        if hi <= lo:
            out_shifts.append(int(s))
            out_resid.append(math.nan)
            out_n.append(0)
            continue
        a = pred[lo:hi]
        b = obs[lo + s : hi + s]
        good = np.isfinite(a) & np.isfinite(b)
        out_shifts.append(int(s))
        out_resid.append(
            float(np.sqrt(np.mean((a[good] - b[good]) ** 2))) if np.any(good) else math.nan
        )
        out_n.append(int(good.sum()))
    return ShiftCurve(tuple(out_shifts), tuple(out_resid), tuple(out_n))


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepGroups:
    """One step's group structure, as the trainer would have seen it."""

    step: float
    sizes: tuple[int, ...]
    stds: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))

    @property
    def n_groups(self) -> int:
        return len(self.sizes)

    @property
    def n_rollouts(self) -> int:
        return int(sum(self.sizes))

    def n_informative(self, *, atol: float = 0.0) -> int:
        """Groups whose reward spread is not zero, which are the only ones carrying an advantage."""
        return int(np.sum(self.stds > atol))

    def mean_std(self) -> float:
        return float(np.mean(self.stds)) if self.stds.size else math.nan

    def frac_zero_std_by_group(self, *, atol: float = 0.0) -> float:
        return float(np.mean(self.stds <= atol)) if self.stds.size else math.nan

    def frac_zero_std_by_rollout(self, *, atol: float = 0.0) -> float:
        """The same fraction weighted by group size, which is what a per-rollout mean produces.

        TRL has logged this quantity both ways across versions: over groups, and over rollouts
        after the per-group value is broadcast back. On an evenly sized batch the two agree exactly
        and on an uneven one they do not, so which is which is decided by the data rather than by
        remembering a version number.
        """
        if not self.sizes:
            return math.nan
        weights = np.asarray(self.sizes, dtype=np.float64)
        return float(np.sum(weights * (self.stds <= atol)) / weights.sum())


def group_statistics(
    steps: Sequence[float] | np.ndarray,
    groups: Sequence[object] | np.ndarray,
    rewards: Sequence[float] | np.ndarray,
    *,
    ddof: int = 1,
) -> list[StepGroups]:
    """Per-step group sizes and within-group reward spreads, keyed on `(step, group)`.

    Keyed on the pair and never on the group alone: a prompt group recurring at three steps is
    three groups, because the trainer normalised each batch separately. That is trap five of
    `labelled.py` and it is restated here because this module is where it would do the most damage.

    ``ddof`` is 1 rather than numpy's default of 0, and the two are not interchangeable on a group
    of sixteen: they differ by `sqrt(16/15)`, about 3.3 per cent. One is the library's verified
    framework table, which records that TRL's `nanstd` applies Bessel's correction explicitly and
    that its older `Tensor.std` path takes torch's default of `correction=1`. Whether a spread is
    zero is the same under either, so the degeneracy structure this module tests does not depend on
    it; the magnitude of `reward_std` does.
    """
    axis = np.asarray(steps, dtype=np.float64).ravel()
    keys = np.asarray(groups, dtype=object).ravel().astype(str)
    r = np.asarray(rewards, dtype=np.float64).ravel()
    out: list[StepGroups] = []
    for step in np.unique(axis[np.isfinite(axis)]):
        mask = axis == step
        sizes: list[int] = []
        stds: list[float] = []
        for key in sorted(set(keys[mask].tolist())):
            block = r[mask & (keys == key)]
            sizes.append(int(block.size))
            usable = block[np.isfinite(block)]
            stds.append(float(usable.std(ddof=ddof)) if usable.size > ddof else 0.0)
        out.append(
            StepGroups(step=float(step), sizes=tuple(sizes), stds=np.asarray(stds, dtype=float))
        )
    return out


@dataclass(frozen=True)
class InformativeGroups:
    """One step's count of groups that can carry an advantage, raw and clone-adjusted.

    An advantage is a within-group contrast, so a group whose rewards are all equal contributes
    exactly nothing to it regardless of how many rollouts it holds. The raw count is how many groups
    at this step are not in that state. The clone-adjusted count goes one step further and asks how
    many *independent* such groups there are: two groups whose rollouts are byte-identical to each
    other's are one observation wearing two labels, and a floor set on the raw count would be met by
    a step that contains one group copied eight times.
    """

    step: float
    n_groups: int
    n_informative: int
    n_informative_clone_adjusted: float
    duplicate_fraction: float


def informative_group_counts(
    steps: Sequence[float] | np.ndarray,
    groups: Sequence[object] | np.ndarray,
    rewards: Sequence[float] | np.ndarray,
    content_hashes: Sequence[str] | None = None,
    *,
    ddof: int = 1,
    atol: float = 0.0,
) -> list[InformativeGroups]:
    """Per-step informative-group counts, clone-adjusted through the Kish effective size.

    A group's identity for clone purposes is the sorted tuple of its rollouts' content hashes, so
    two groups collapse only when they hold the same responses, not merely the same rewards. With no
    hashes supplied every group is treated as distinct and the adjusted count equals the raw one,
    which is the correct degenerate case rather than a silent approximation.
    """
    from reward_lens.stats.ess import effective_sample_size

    axis = np.asarray(steps, dtype=np.float64).ravel()
    keys = np.asarray(groups, dtype=object).ravel().astype(str)
    r = np.asarray(rewards, dtype=np.float64).ravel()
    hashes = (
        np.asarray(content_hashes, dtype=object).ravel().astype(str)
        if content_hashes is not None
        else None
    )
    out: list[InformativeGroups] = []
    for step in np.unique(axis[np.isfinite(axis)]):
        mask = axis == step
        signatures: list[str] = []
        n_groups = 0
        n_informative = 0
        for key in sorted(set(keys[mask].tolist())):
            sel = mask & (keys == key)
            n_groups += 1
            block = r[sel]
            usable = block[np.isfinite(block)]
            spread = float(usable.std(ddof=ddof)) if usable.size > ddof else 0.0
            if spread <= atol:
                continue
            n_informative += 1
            if hashes is not None:
                signatures.append("|".join(sorted(hashes[sel].tolist())))
            else:
                signatures.append(f"{step}:{key}")
        adjusted = effective_sample_size(signatures) if signatures else 0.0
        distinct = len(set(signatures))
        out.append(
            InformativeGroups(
                step=float(step),
                n_groups=n_groups,
                n_informative=n_informative,
                n_informative_clone_adjusted=float(adjusted),
                duplicate_fraction=(
                    (n_informative - distinct) / n_informative if n_informative else 0.0
                ),
            )
        )
    return out


def predicted_reward_std(
    per_step: Sequence[StepGroups],
    *,
    convention: str,
    batch_rewards: Sequence[Sequence[float] | np.ndarray] | None = None,
    ddof: int = 1,
) -> np.ndarray:
    """`reward_std` under one of the conventions a trainer might have logged it under.

    `mean_group_std` is TRL's: the within-group standard deviation, averaged over groups.
    `batch_std` is the spread of the whole batch ignoring groups. They differ by exactly the
    between-group variance, so which one the log holds is itself evidence about whether the
    grouping is right, and computing both is the only way to find out which.

    `batch_std` used to validate its own argument and then raise unconditionally, which is a
    declared convention with no implementation: every caller that asked for it got an exception
    after passing the check that was supposed to admit it. It needs the rollout rewards rather than
    the group summaries, so it takes them, one array per step in the same order as `per_step`.

    ``ddof`` is 1 to match the library's verified framework table: TRL's `nanstd` applies Bessel's
    correction explicitly and its older `Tensor.std` path takes torch's default of `correction=1`.
    """
    if convention not in {"mean_group_std", "batch_std"}:
        raise ValueError(
            f"unknown reward_std convention {convention!r}; it is 'mean_group_std' or 'batch_std'"
        )
    if convention == "mean_group_std":
        return np.asarray([g.mean_std() for g in per_step], dtype=np.float64)
    if batch_rewards is None:
        raise ValueError(
            "the batch_std convention needs the rollout rewards, because the spread of a batch is "
            "not a function of its groups' spreads: the between-group variance is exactly what "
            "separates the two. Pass `batch_rewards` as one array per step."
        )
    if len(batch_rewards) != len(per_step):
        raise ValueError(
            f"{len(batch_rewards)} reward arrays against {len(per_step)} steps of group summaries"
        )
    out = np.full(len(per_step), np.nan, dtype=np.float64)
    for i, block in enumerate(batch_rewards):
        arr = np.asarray(block, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size > ddof:
            out[i] = float(arr.std(ddof=ddof))
    return out


# ---------------------------------------------------------------------------
# Subset search
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubsetResult:
    """One candidate subset of columns, fitted and then judged on its snapped constants."""

    names: tuple[str, ...]
    fit: CompositionFit
    residual_fitted: float
    residual_snapped: float
    all_snapped: bool


def search_subsets(
    means: Mapping[str, np.ndarray],
    target: np.ndarray,
    *,
    candidates: Sequence[str],
    max_terms: int = 4,
    intercept: bool = True,
    snap_tolerance: float = 1e-3,
    n_bootstrap: int = 0,
    seed: int = 0,
) -> list[SubsetResult]:
    """Every subset up to `max_terms`, fitted, ranked by the residual of its *snapped* weights.

    Ranking on the snapped residual rather than the fitted one is deliberate and it is the whole
    reason this function exists. Adding a collinear column always lowers the fitted residual and
    almost never lowers the snapped one, because a spurious term does not land on a round number.
    So a search that ranks on the fitted residual selects the largest subset every time, and one
    that ranks on the snapped residual selects the configuration.
    """
    results: list[SubsetResult] = []
    for size in range(1, min(max_terms, len(candidates)) + 1):
        for combo in itertools.combinations(candidates, size):
            try:
                fit = fit_composition(
                    means,
                    target,
                    names=combo,
                    intercept=intercept,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                    snap_tolerance=snap_tolerance,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            if isinstance(fit, Refusal):
                # A subset the record cannot support is skipped rather than scored. The search is
                # over candidates and a candidate that leaves no residual degrees of freedom has
                # not lost the comparison, it was never in it.
                continue
            design = np.column_stack([np.asarray(means[k], dtype=np.float64) for k in combo])
            fitted = design @ fit.weights + fit.intercept
            snapped_w = np.asarray(
                [
                    fit.snapped_values[i] if fit.snapped_values[i] is not None else fit.weights[i]
                    for i in range(len(combo))
                ],
                dtype=np.float64,
            )
            snapped = design @ snapped_w + fit.snapped_intercept()
            obs = np.asarray(target, dtype=np.float64).ravel()
            good = np.isfinite(obs) & np.all(np.isfinite(design), axis=1)
            results.append(
                SubsetResult(
                    names=combo,
                    fit=fit,
                    residual_fitted=float(np.sqrt(np.mean((fitted[good] - obs[good]) ** 2))),
                    residual_snapped=float(np.sqrt(np.mean((snapped[good] - obs[good]) ** 2))),
                    all_snapped=fit.all_snapped(),
                )
            )
    results.sort(key=lambda r: (r.residual_snapped, len(r.names)))
    return results


__all__ = [
    "ClosureResidual",
    "CompositionFit",
    "InformativeGroups",
    "MAX_SIMPLE_NUMERATOR",
    "SIMPLE_DENOMINATORS",
    "ShiftCurve",
    "StepGroups",
    "SubsetResult",
    "closure_residuals",
    "compose",
    "fit_composition",
    "group_statistics",
    "informative_group_counts",
    "predicted_reward_std",
    "search_subsets",
    "shift_residual_curve",
    "snap_to_simple_ratio",
    "steps_and_means",
]
