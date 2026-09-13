"""BLK-013 — nothing drew a random direction from inside a supplied subspace.

`norm_matched_random` draws `rng.standard_normal(u.shape[0])` over the full ambient dimension, and
so does every other random-direction generator in the package. Part 9.2 rules that draw out in as
many words:

    A random direction from the full residual space is nearly orthogonal to everything and
    therefore nearly free, and it is not a control.

`C15` needs at least twenty norm-matched draws from the identified subspace and `C16` needs draws
inside the same subspace but orthogonal to the fitted direction. Neither existed. The three pieces
did: `_orthonormalize` in `geometry/subspace.py`, the Gaussian-QR basis in `stats/nulls.py`, and the
projector in `loops/recorder.py`. Only the draw was missing.

**The subspace's construction is not decided here.** BLK-073 records that "the identified subspace"
is used eight times in the design and defined at none, and that the corpus carries two constructions
that disagree. So `basis` is a required argument of both functions: this packet builds the draw and
leaves the construction to the row that owns it.

Tolerances are G8's, quoted rather than chosen: complement projection below `1e-10`, norms matching
the target to `1e-12`, and the target-orthogonal inner product below `1e-10`.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.interventions.rescue import (
    RescueError,
    SubspaceDraw,
    subspace_matched_random,
    target_orthogonal_random,
)

#: G8's fixture rank. BLK-073 records that this is a unit-test fixture and not a registration, so it
#: is used here as a test parameter and nowhere as a default.
FIXTURE_RANK = 8
AMBIENT = 64
N_DRAWS = 200


@pytest.fixture
def basis() -> np.ndarray:
    """A rank-8 subspace of a 64-dimensional ambient space, with a non-orthonormal generator.

    Deliberately not handed in already orthonormal: a caller's construction is a span, and a draw
    that is only correct for an orthonormal input is a draw that is silently wrong for the object
    the design actually registers.
    """
    rng = np.random.default_rng(8)
    return rng.standard_normal((AMBIENT, FIXTURE_RANK))


@pytest.fixture
def target(basis: np.ndarray) -> np.ndarray:
    """A fitted direction lying inside the subspace, which is the case both controls are for."""
    rng = np.random.default_rng(13)
    coefficients = rng.standard_normal(FIXTURE_RANK)
    v = basis @ coefficients
    return v * (2.5 / np.linalg.norm(v))


def _complement_projection(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """How much of each row lies outside the span of `basis`, computed independently of the code.

    The projector is rebuilt here from an SVD rather than reusing whatever the implementation used,
    so a bug shared between the draw and its check cannot cancel out.
    """
    u, s, _ = np.linalg.svd(np.asarray(basis, dtype=np.float64), full_matrices=False)
    q = u[:, s > 1e-12]
    inside = np.asarray(vectors, dtype=np.float64) @ q @ q.T
    return np.linalg.norm(np.asarray(vectors, dtype=np.float64) - inside, axis=1)


# ---------------------------------------------------------------------------
# G8's closure proof, first half: the subspace-matched draw
# ---------------------------------------------------------------------------


def test_two_hundred_draws_lie_inside_the_rank_eight_subspace(
    basis: np.ndarray, target: np.ndarray
) -> None:
    """G8, verbatim: projection onto the complement below 1e-10."""
    draw = subspace_matched_random(target, basis, n_draws=N_DRAWS, seed=0)
    assert draw.directions.shape == (N_DRAWS, AMBIENT)
    residual = _complement_projection(draw.directions, basis)
    assert residual.max() < 1e-10, f"largest complement projection {residual.max():.3e}"


def test_two_hundred_draws_match_the_targets_norm(basis: np.ndarray, target: np.ndarray) -> None:
    """G8, verbatim: norms match the target to 1e-12."""
    draw = subspace_matched_random(target, basis, n_draws=N_DRAWS, seed=0)
    wanted = float(np.linalg.norm(target))
    got = np.linalg.norm(draw.directions, axis=1)
    assert np.max(np.abs(got - wanted)) < 1e-12, (
        f"worst norm error {np.max(np.abs(got - wanted)):.3e}"
    )


def test_the_draws_are_not_all_the_same_direction(basis: np.ndarray, target: np.ndarray) -> None:
    """A constant vector satisfies both tests above and is not a control distribution."""
    draw = subspace_matched_random(target, basis, n_draws=N_DRAWS, seed=0)
    unit = draw.directions / np.linalg.norm(draw.directions, axis=1, keepdims=True)
    gram = unit @ unit.T
    off_diagonal = gram[~np.eye(N_DRAWS, dtype=bool)]
    assert np.max(np.abs(off_diagonal)) < 0.99, "the draws are collinear"
    assert np.std(off_diagonal) > 0.01, "the draws carry no spread"


def test_the_draw_is_uniform_over_the_subspace_rather_than_over_the_generator(
    basis: np.ndarray, target: np.ndarray
) -> None:
    """Uniform on the sphere inside the subspace, not Gaussian in the generator's coordinates.

    A draw written as `basis @ rng.standard_normal(k)` passes every test above and is not uniform:
    it inherits the generator's conditioning, so a badly conditioned construction concentrates the
    control family along the generator's dominant direction and the control gets easy to beat in a
    way nobody can see. Checked by projecting into an orthonormal basis of the same span, where a
    uniform draw has an isotropic covariance and the naive one does not.
    """
    draw = subspace_matched_random(target, basis, n_draws=4000, seed=1)
    u, s, _ = np.linalg.svd(basis, full_matrices=False)
    q = u[:, s > 1e-12]
    coordinates = draw.directions @ q
    eigenvalues = np.linalg.eigvalsh(np.cov(coordinates, rowvar=False))
    # Isotropic in the subspace: every eigenvalue equal. The generator here has a condition number
    # of order 5, so the naive draw would show a spread far outside this band.
    assert eigenvalues.max() / eigenvalues.min() < 1.6, (
        f"the control family is anisotropic inside the subspace: "
        f"eigenvalue ratio {eigenvalues.max() / eigenvalues.min():.2f}"
    )


# ---------------------------------------------------------------------------
# G8's closure proof, second half: the target-orthogonal draw
# ---------------------------------------------------------------------------


def test_the_target_orthogonal_draw_is_orthogonal_to_the_fitted_direction(
    basis: np.ndarray, target: np.ndarray
) -> None:
    """G8, verbatim: inner product with the fitted direction below 1e-10."""
    draw = target_orthogonal_random(target, basis, n_draws=N_DRAWS, seed=0)
    inner = np.abs(draw.directions @ np.asarray(target, dtype=np.float64))
    assert inner.max() < 1e-10, f"largest inner product with the target {inner.max():.3e}"


def test_the_target_orthogonal_draw_stays_inside_the_subspace(
    basis: np.ndarray, target: np.ndarray
) -> None:
    """Orthogonal to the target AND inside the subspace. Dropping either half is the whole point.

    A draw orthogonal to the target over the full ambient space would pass the test above and be the
    control Part 9.2 rules out, since almost every ambient direction is nearly orthogonal to any
    fixed one.
    """
    draw = target_orthogonal_random(target, basis, n_draws=N_DRAWS, seed=0)
    residual = _complement_projection(draw.directions, basis)
    assert residual.max() < 1e-10, f"largest complement projection {residual.max():.3e}"


def test_the_target_orthogonal_draw_also_matches_the_norm(
    basis: np.ndarray, target: np.ndarray
) -> None:
    draw = target_orthogonal_random(target, basis, n_draws=N_DRAWS, seed=0)
    wanted = float(np.linalg.norm(target))
    got = np.linalg.norm(draw.directions, axis=1)
    assert np.max(np.abs(got - wanted)) < 1e-12


def test_the_target_orthogonal_family_spans_the_rest_of_the_subspace(
    basis: np.ndarray, target: np.ndarray
) -> None:
    """Rank `k - 1`, not a single direction repeated with different signs."""
    draw = target_orthogonal_random(target, basis, n_draws=N_DRAWS, seed=0)
    rank = np.linalg.matrix_rank(draw.directions, tol=1e-8)
    assert rank == FIXTURE_RANK - 1, f"expected rank {FIXTURE_RANK - 1}, got {rank}"


# ---------------------------------------------------------------------------
# The rank cases BLK-183 says make C16 unresolvable, asserted as raises
# ---------------------------------------------------------------------------


def test_a_rank_one_subspace_refuses_the_target_orthogonal_draw() -> None:
    """BLK-183: at rank 1 the target-orthogonal set is empty. Empty has to raise, not return zeros.

    Returning a zero vector, or falling back to an ambient draw, would give `C16` twenty controls
    that are not the control it registered and nothing downstream could tell.
    """
    rng = np.random.default_rng(3)
    line = rng.standard_normal((AMBIENT, 1))
    direction = line[:, 0] * 2.0
    with pytest.raises(RescueError, match="rank"):
        target_orthogonal_random(direction, line, n_draws=20, seed=0)


def test_a_rank_two_subspace_reports_that_the_family_is_two_point_masses() -> None:
    """BLK-183: at rank 2 the family is a single direction up to sign.

    It is drawable, so this does not raise, but a caller reading a percentile off twenty draws from
    two point masses is reading a percentile off two numbers. The degeneracy is on the result.
    """
    rng = np.random.default_rng(4)
    plane = rng.standard_normal((AMBIENT, 2))
    direction = plane @ np.array([1.0, 0.5])
    draw = target_orthogonal_random(direction, plane, n_draws=20, seed=0)
    assert draw.degenerate is True
    assert draw.effective_rank == 1
    unit = draw.directions / np.linalg.norm(draw.directions, axis=1, keepdims=True)
    assert np.allclose(np.abs(unit @ unit[0]), 1.0, atol=1e-10), "expected two point masses"


def test_the_subspace_matched_draw_is_fine_at_rank_one() -> None:
    """The first control family is well defined on a line; only the second one is not."""
    rng = np.random.default_rng(3)
    line = rng.standard_normal((AMBIENT, 1))
    direction = line[:, 0] * 2.0
    draw = subspace_matched_random(direction, line, n_draws=20, seed=0)
    assert draw.effective_rank == 1
    assert draw.degenerate is True
    assert np.max(_complement_projection(draw.directions, line)) < 1e-10


# ---------------------------------------------------------------------------
# The result object, the family strings, and reproducibility
# ---------------------------------------------------------------------------


def test_the_family_string_is_the_one_the_prereg_filters_on(
    basis: np.ndarray, target: np.ndarray
) -> None:
    """`C16` filters `control_family = "target_orthogonal"`, verbatim in CHAIN_PREREG."""
    assert target_orthogonal_random(target, basis, n_draws=4).family == "target_orthogonal"
    assert subspace_matched_random(target, basis, n_draws=4).family == "subspace_matched"


def test_the_draw_records_what_it_was_given(basis: np.ndarray, target: np.ndarray) -> None:
    """A control distribution whose rank and seed are not on it cannot be reproduced from a record."""
    draw = subspace_matched_random(target, basis, n_draws=20, seed=7)
    assert isinstance(draw, SubspaceDraw)
    assert draw.effective_rank == FIXTURE_RANK
    assert draw.seed == 7
    assert draw.target_norm == pytest.approx(float(np.linalg.norm(target)), abs=1e-12)


def test_the_same_seed_gives_the_same_draws(basis: np.ndarray, target: np.ndarray) -> None:
    a = subspace_matched_random(target, basis, n_draws=20, seed=42)
    b = subspace_matched_random(target, basis, n_draws=20, seed=42)
    c = subspace_matched_random(target, basis, n_draws=20, seed=43)
    assert np.array_equal(a.directions, b.directions)
    assert not np.array_equal(a.directions, c.directions)


def test_an_explicit_norm_overrides_the_targets_own(basis: np.ndarray, target: np.ndarray) -> None:
    """Part 9.2 rescales each control to the intervention's norm, which need not be the target's."""
    draw = subspace_matched_random(target, basis, n_draws=20, seed=0, norm=0.75)
    assert np.max(np.abs(np.linalg.norm(draw.directions, axis=1) - 0.75)) < 1e-12


def test_a_target_outside_the_subspace_is_reported_rather_than_silently_projected(
    basis: np.ndarray,
) -> None:
    """If the fitted direction is not in the registered subspace, both controls change meaning.

    The draw still happens, because projecting is the defensible thing to do, but the fraction of
    the target that survives the projection travels on the result so a reader can see that "inside
    the subspace and orthogonal to the target" was answered about a projected target.
    """
    rng = np.random.default_rng(99)
    outside = rng.standard_normal(AMBIENT)
    draw = target_orthogonal_random(outside, basis, n_draws=20, seed=0)
    assert 0.0 < draw.target_in_subspace_fraction < 0.9
    inner = np.abs(draw.directions @ (basis @ np.linalg.lstsq(basis, outside, rcond=None)[0]))
    assert inner.max() < 1e-10, "not orthogonal to the projected target"


def test_a_zero_target_raises_rather_than_dividing_by_a_vanishing_norm(
    basis: np.ndarray,
) -> None:
    with pytest.raises((RescueError, ValueError)):
        subspace_matched_random(np.zeros(AMBIENT), basis, n_draws=4)


def test_a_basis_whose_ambient_dimension_disagrees_with_the_target_raises(
    basis: np.ndarray, target: np.ndarray
) -> None:
    with pytest.raises(RescueError):
        subspace_matched_random(target[: AMBIENT - 1], basis, n_draws=4)


def test_the_shipped_full_dimension_draw_is_left_alone(target: np.ndarray) -> None:
    """`norm_matched_random` keeps its unit convention and its ambient draw.

    It is still the right object for the ambient control family; the defect was that it was the only
    object. Regressing it to close this row would be a second defect.
    """
    from reward_lens.interventions.rescue import norm_matched_random

    v = norm_matched_random(target, seed=0)
    assert v.shape == (AMBIENT,)
    assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-6)
