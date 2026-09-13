"""BLK-024 — no rank-based pooling existed, so Part 13.2's pooling factor was arithmetic by hand.

Part 13.2, verbatim:

    The participation ratio is measured on this run's own movement matrix, in the pre-transition
    window named in advance, and the pooling factor is its square root.

That factor multiplies the interval half-width on every registered row. It was computed nowhere:
the package's only pooling functions pool by study count (`stats/meta.py` `power_for_pooled_effect`)
and by laboratory (`measure/meta/interlab.py`), and `geometry.hessian.participation_ratio` takes a
spectrum rather than a movement matrix. A number that enters every interval in the ledger had no
library call behind it.

The size of the error the row is about, from Part 13.2's own figures: pooling by the naive feature
count `sqrt(12) = 3.464` where `sqrt(3.870) = 1.967` is correct is a factor of 1.76 of false
precision, and the project's trap register calls this the most likely source of an overstated
result.

**The window is not chosen here.** BLK-080 records that Part 13.2 and Phase 0's F5 both say the
window is named in advance and neither names it, and that the ratio is strongly non-stationary:
3.870 over the full run against 1.233 over steps 201 to 300. So `window` is a required keyword with
no default, and it travels on the result.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.stats.variance import PoolingFactor, rank_pooling_factor


def _matrix_with_known_participation_ratio(
    singular_values: np.ndarray, *, n_steps: int, seed: int = 0
) -> np.ndarray:
    """A movement matrix whose participation ratio is known in closed form.

    Built as `U diag(s) V^T` from Haar-random orthonormal factors, so the singular values are
    exactly `singular_values` and the participation ratio is exactly the moment ratio on their
    squares. Nothing about the fixture is estimated.
    """
    rng = np.random.default_rng(seed)
    k = singular_values.size
    u, _ = np.linalg.qr(rng.standard_normal((n_steps, k)))
    v, _ = np.linalg.qr(rng.standard_normal((k, k)))
    return u @ np.diag(singular_values) @ v.T


def _expected_pr(singular_values: np.ndarray) -> float:
    """`(sum s^2)^2 / sum s^4`, the package's stated convention on squared singular values."""
    lam = np.asarray(singular_values, dtype=np.float64) ** 2
    return float(lam.sum() ** 2 / (lam**2).sum())


# ---------------------------------------------------------------------------
# The closure proof: the factor is the square root of a known participation ratio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "singular_values",
    [
        np.array([1.0, 1.0, 1.0, 1.0]),  # isotropic rank 4, PR = 4 exactly
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),  # naive count 12
        np.array([1.0, 0.0, 0.0, 0.0]),  # rank 1, PR = 1 exactly
        np.array([3.0, 2.0, 1.0, 0.5]),  # anisotropic, PR from the closed form
        np.array([5.0, 0.1, 0.1, 0.1, 0.1]),  # one dominant mode
    ],
)
def test_the_pooling_factor_is_the_square_root_of_the_participation_ratio(
    singular_values: np.ndarray,
) -> None:
    """BLK-024's closure proof, verbatim: equals its square root to 1e-12."""
    movement = _matrix_with_known_participation_ratio(singular_values, n_steps=80)
    result = rank_pooling_factor(movement, window=None)
    expected_pr = _expected_pr(singular_values)
    assert result.participation_ratio == pytest.approx(expected_pr, abs=1e-12)
    assert result.factor == pytest.approx(np.sqrt(expected_pr), abs=1e-12)


def test_an_isotropic_rank_r_matrix_pools_by_the_square_root_of_r() -> None:
    """`n` equal singular values give `PR = n`, so the factor is `sqrt(n)` and matches the naive one.

    This is the one case where pooling by the feature count is right, and it is the case that makes
    the naive rule look reasonable.
    """
    for r in (1, 3, 8, 12):
        movement = _matrix_with_known_participation_ratio(np.ones(r), n_steps=60, seed=r)
        result = rank_pooling_factor(movement, window=None)
        assert result.participation_ratio == pytest.approx(float(r), abs=1e-12)
        assert result.factor == pytest.approx(np.sqrt(r), abs=1e-12)
        assert result.naive_factor == pytest.approx(np.sqrt(r), abs=1e-12)
        assert result.false_precision == pytest.approx(1.0, abs=1e-12)


def test_a_rank_one_movement_matrix_pools_by_one() -> None:
    """One direction carrying everything is one independent quantity, whatever the column count."""
    rng = np.random.default_rng(2)
    left = rng.standard_normal((50, 1))
    right = rng.standard_normal((1, 12))
    result = rank_pooling_factor(left @ right, window=None)
    assert result.participation_ratio == pytest.approx(1.0, abs=1e-12)
    assert result.factor == pytest.approx(1.0, abs=1e-12)
    assert result.n_features == 12
    assert result.naive_factor == pytest.approx(np.sqrt(12.0), abs=1e-12)


# ---------------------------------------------------------------------------
# Part 13.2's own worked numbers, recomputed from the inputs rather than quoted
# ---------------------------------------------------------------------------


def test_part_13_2_s_false_precision_figure_reproduces() -> None:
    """Part 13.2 states a factor of 1.76 between the naive and correct pooling. Recompute it.

    Constructed from Part 13.2's two numbers, a participation ratio of 3.870 against a naive feature
    count of 12, rather than by typing 1.76 in: the ratio of the two factors comes out of the
    function, and the check is that it lands on the design's figure.
    """
    # Singular values chosen so the participation ratio is exactly 3.870 over 12 features.
    target_pr = 3.870
    values = np.zeros(12)
    values[0] = 1.0
    # Two-level spectrum: one mode at 1 and the rest at t, solved for the stated PR.
    lo, hi = 1e-9, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        values[1:] = mid
        pr = _expected_pr(values)
        if pr < target_pr:
            lo = mid
        else:
            hi = mid
    values[1:] = (lo + hi) / 2
    movement = _matrix_with_known_participation_ratio(values, n_steps=100, seed=13)
    result = rank_pooling_factor(movement, window=None)
    assert result.participation_ratio == pytest.approx(3.870, abs=1e-3)
    assert result.factor == pytest.approx(1.967, abs=1e-3)
    assert result.naive_factor == pytest.approx(3.4641, abs=1e-3)
    assert result.false_precision == pytest.approx(1.76, abs=0.01)


def test_the_naive_factor_is_reported_beside_the_correct_one() -> None:
    """Both are on the result, so a reader can see the size of the correction that was applied."""
    values = np.array([4.0, 1.0, 1.0, 0.2, 0.2, 0.2])
    movement = _matrix_with_known_participation_ratio(values, n_steps=70, seed=5)
    result = rank_pooling_factor(movement, window=None)
    assert result.n_features == 6
    assert result.naive_factor == pytest.approx(np.sqrt(6.0), abs=1e-12)
    assert result.false_precision == pytest.approx(result.naive_factor / result.factor, abs=1e-12)
    assert result.factor < result.naive_factor, (
        "an anisotropic matrix must pool by less than sqrt(p)"
    )


# ---------------------------------------------------------------------------
# BLK-080: the window is not chosen here, and it has to be named
# ---------------------------------------------------------------------------


def test_the_window_is_a_required_argument() -> None:
    """BLK-080's rule is that the window is named first. A default would be a window nobody named."""
    movement = _matrix_with_known_participation_ratio(np.ones(4), n_steps=60)
    with pytest.raises(TypeError):
        rank_pooling_factor(movement)  # type: ignore[call-arg]


def test_the_window_is_applied_and_travels_on_the_result() -> None:
    """The non-stationarity BLK-080 is about: the same run gives different factors by window.

    Built so the first half is isotropic over four modes and the second half is dominated by one, so
    the two windows genuinely disagree and a result that ignored the window would be caught here.
    """
    rng = np.random.default_rng(80)
    early = _matrix_with_known_participation_ratio(np.ones(4), n_steps=100, seed=1)
    late = rng.standard_normal((100, 1)) @ rng.standard_normal((1, 4)) * 3.0
    movement = np.vstack([early, late])

    whole = rank_pooling_factor(movement, window=None)
    first = rank_pooling_factor(movement, window=(0, 100))
    second = rank_pooling_factor(movement, window=(100, 200))

    assert first.window == (0, 100)
    assert second.window == (100, 200)
    assert whole.window is None
    assert first.n_steps == 100 and second.n_steps == 100 and whole.n_steps == 200
    assert first.participation_ratio == pytest.approx(4.0, abs=1e-12)
    assert second.participation_ratio == pytest.approx(1.0, abs=1e-10)
    assert first.factor / second.factor == pytest.approx(2.0, abs=1e-6)


def test_a_window_outside_the_series_raises_rather_than_silently_clipping() -> None:
    """A window that runs off the end is a window nobody checked, and clipping hides it."""
    movement = _matrix_with_known_participation_ratio(np.ones(4), n_steps=60)
    with pytest.raises(ValueError):
        rank_pooling_factor(movement, window=(40, 200))
    with pytest.raises(ValueError):
        rank_pooling_factor(movement, window=(30, 30))
    with pytest.raises(ValueError):
        rank_pooling_factor(movement, window=(30, 10))


def test_a_window_with_fewer_steps_than_features_is_reported_as_underdetermined() -> None:
    """The participation ratio of a window shorter than its width is bounded by the step count.

    Not an error, because a short window is a legitimate thing to ask about, but the bound is the
    reason the number cannot be read as an effective dimension there, and it is on the result.
    """
    movement = _matrix_with_known_participation_ratio(np.ones(12), n_steps=200)
    result = rank_pooling_factor(movement, window=(0, 5))
    assert result.underdetermined is True
    assert result.participation_ratio <= 5.0 + 1e-9
    full = rank_pooling_factor(movement, window=None)
    assert full.underdetermined is False


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_an_all_zero_movement_matrix_raises_rather_than_returning_zero() -> None:
    """A pooling factor of zero would divide an interval half-width by nothing."""
    with pytest.raises(ValueError):
        rank_pooling_factor(np.zeros((40, 6)), window=None)


def test_non_finite_entries_raise() -> None:
    movement = _matrix_with_known_participation_ratio(np.ones(4), n_steps=40)
    movement[3, 2] = np.nan
    with pytest.raises(ValueError):
        rank_pooling_factor(movement, window=None)


def test_the_result_is_a_frozen_record() -> None:
    """The factor multiplies every interval in the ledger, so it is not mutable after the fact."""
    movement = _matrix_with_known_participation_ratio(np.ones(4), n_steps=40)
    result = rank_pooling_factor(movement, window=None)
    assert isinstance(result, PoolingFactor)
    with pytest.raises(Exception):
        result.factor = 1.0  # type: ignore[misc]


def test_it_is_exported_from_the_module_that_holds_it() -> None:
    """Reachable as `stats.variance.rank_pooling_factor`, which is this module's own convention.

    `variance` and `gtheory` are the two `stats` modules deliberately not re-exported from the
    package `__init__`; every caller in the package imports `reward_lens.stats.variance` by path.
    Adding a package-level re-export for this one function would make it the exception.
    """
    import reward_lens.stats.variance as variance

    assert variance.rank_pooling_factor is rank_pooling_factor
    assert "rank_pooling_factor" in variance.__all__
    assert "PoolingFactor" in variance.__all__
