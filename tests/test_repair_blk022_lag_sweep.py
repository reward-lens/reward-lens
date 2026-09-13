"""BLK-022 — the shipped lag sweep could not reach the registered analysis lag.

The design reads the ledger at lag 11, with 22 and 33 registered as its secondaries, and
`shift_residual_curve` shipped a default sweep of `range(-5, 6)`. A sweep that stops at 5 cannot
confirm or refute a reading at 11, and a caller who takes the default and reports "no minimum at
the assumed alignment" has reported a property of the window rather than of the data.

The closure proof BLK-022 names is a static check on the default's range. It is written here as a
check on the live signature rather than on a copy of the literal, so a later edit that narrows the
default fails this test instead of passing it silently.

The second half of the file is not in BLK-022's closure proof and is here because widening the
default makes an existing hazard reachable by default: the residual at an extreme shift is computed
on a handful of overlapping steps, and an RMS over three points goes below the RMS at the true
alignment often enough to be declared a clear minimum. The counterexample is recorded, and the
guard that stops it is opt-in, because its default is a threshold and thresholds are registered
rather than chosen by a builder. See ERRATUM-overlap-guard.md in this row's proof directory.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from reward_lens.measure.ledger.reconstruct import shift_residual_curve

#: The registered analysis lag and its two secondaries, per BLK-022 and CHAIN_GAPS `G21`.
REGISTERED_LAGS = (11, 22, 33)


def _default_shifts() -> list[int]:
    """The shipped default, read off the live signature rather than restated."""
    default = inspect.signature(shift_residual_curve).parameters["shifts"].default
    return [int(s) for s in default]


# ---------------------------------------------------------------------------
# The closure proof: the sweep's range covers 11, 22 and 33
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lag", REGISTERED_LAGS)
def test_the_default_sweep_reaches_the_registered_analysis_lag(lag: int) -> None:
    """The default range covers the registered lag in both directions.

    Both signs, because the sweep's job is to locate an alignment whose sign is not known in
    advance: a design that reads at +11 and a record that is eleven steps early are the same
    question asked from the two ends.
    """
    shifts = _default_shifts()
    assert lag in shifts, f"the default sweep stops short of the registered lag {lag}: {shifts}"
    assert -lag in shifts, f"the default sweep does not reach -{lag}: {shifts}"


def test_the_default_sweep_is_contiguous_and_symmetric() -> None:
    """No gaps and no lopsided window, so `has_clear_minimum_at` always has both neighbours."""
    shifts = _default_shifts()
    assert shifts == list(range(min(shifts), max(shifts) + 1)), "the default sweep has gaps"
    assert min(shifts) == -max(shifts), (
        f"the default sweep is lopsided: {min(shifts)}..{max(shifts)}"
    )


def test_the_curve_actually_evaluates_every_registered_lag() -> None:
    """The range is covered by the returned curve, not only by the signature.

    A default that reads correctly and a function that silently drops the far shifts would pass the
    signature check above, so the shifts are read back off a real call.
    """
    rng = np.random.default_rng(22)
    a = rng.normal(0.0, 1.0, 400)
    curve = shift_residual_curve(a, a)
    for lag in REGISTERED_LAGS:
        assert lag in curve.shifts
        assert -lag in curve.shifts
        assert np.isfinite(curve.residual_at(lag)), f"lag {lag} evaluated to a non-finite residual"
        assert np.isfinite(curve.residual_at(-lag))


def test_a_planted_offset_at_the_registered_lag_is_found_and_called_clear() -> None:
    """The sweep locates an alignment planted at 11, which is the reading the design takes.

    This is the property the shipped default could not have: at `range(-5, 6)` the minimum is at an
    endpoint and the residual at 11 is not computed at all.
    """
    rng = np.random.default_rng(11)
    logged = rng.normal(3.0, 1.0, 300)
    reconstructed = logged[11:]
    curve = shift_residual_curve(reconstructed, logged)
    assert curve.best_shift == 11
    assert curve.has_clear_minimum_at(11)
    assert not curve.has_clear_minimum_at(0)
    assert curve.residual_at(11) < 1e-12


# ---------------------------------------------------------------------------
# The hazard the widening exposes, and the guard that stops it
# ---------------------------------------------------------------------------


def test_a_far_shift_on_a_short_series_is_scored_on_a_handful_of_steps() -> None:
    """The counterexample, recorded rather than asserted away.

    Two independent noise series of 36 steps. The widened sweep reaches -33, where exactly three
    steps overlap, and an RMS over three points is small often enough that the neighbour-ratio test
    declares it a clear minimum. Nothing about these two series is aligned.
    """
    rng = np.random.default_rng(7)
    a = rng.normal(0.0, 1.0, 36)
    b = rng.normal(0.0, 1.0, 36)
    curve = shift_residual_curve(a, b)
    overlap = dict(zip(curve.shifts, curve.n_overlap))
    assert curve.best_shift == -33
    assert overlap[-33] == 3, (
        "the fixture no longer reaches the three-step overlap it was built for"
    )
    assert curve.has_clear_minimum_at(-33), "the hazard this test records did not reproduce"


def test_the_overlap_guard_rejects_a_minimum_scored_on_too_few_steps() -> None:
    """With a floor on the overlap the same counterexample stops being an alignment.

    The floor is a parameter and its default is zero, which is exactly the shipped behaviour: this
    test supplies its own so that the guard has a caller, and the registered default stays with the
    integrator.
    """
    rng = np.random.default_rng(7)
    a = rng.normal(0.0, 1.0, 36)
    b = rng.normal(0.0, 1.0, 36)
    curve = shift_residual_curve(a, b)
    assert not curve.has_clear_minimum_at(-33, min_overlap=10)
    assert curve.best_shift_with_overlap(min_overlap=10) != -33


def test_the_guard_does_not_reject_a_real_alignment() -> None:
    """A guard that refuses everything would pass the test above and be useless."""
    rng = np.random.default_rng(11)
    logged = rng.normal(3.0, 1.0, 300)
    curve = shift_residual_curve(logged[11:], logged)
    assert curve.has_clear_minimum_at(11, min_overlap=10)
    assert curve.best_shift_with_overlap(min_overlap=10) == 11


def test_the_guard_default_leaves_shipped_behaviour_unchanged() -> None:
    """`min_overlap=0` is the shipped path, so the guard is opt-in rather than a silent retier."""
    rng = np.random.default_rng(7)
    a = rng.normal(0.0, 1.0, 36)
    b = rng.normal(0.0, 1.0, 36)
    curve = shift_residual_curve(a, b)
    assert curve.has_clear_minimum_at(-33, min_overlap=0) == curve.has_clear_minimum_at(-33)
    assert curve.best_shift_with_overlap(min_overlap=0) == curve.best_shift
