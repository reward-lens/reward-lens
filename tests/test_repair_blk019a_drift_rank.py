"""BLK-019-a — Procrustes drift read `pi` radians for a bank compared against itself.

`representation_drift` solved the alignment in the full ambient width. The objective only ever sees
the rotation through `X^T Y`, whose rank is at most the number of items, so with a bank of 256 rows
in a 2,560-wide residual stream the great majority of a `2560 x 2560` rotation was unconstrained and
`numpy.linalg.svd` filled it with whatever its factorisation produced. Read back as principal
angles, that filling is not small: at the run's own geometry a bank compared **against itself**
returned 3.1408 radians, and planted rotations of 0.1, 0.5 and 1.0 radians all returned 3.14 as
well. The statistic carried no information about the quantity it names.

**The geometry here is the run's own, not a corner case.** `chain/parts/part_13.md:208` registers
one frozen bank of 256 items; `chain/parts/part_04.md:7` fixes the policy at Qwen3-4B, whose
residual width is 2,560. The BLK-019 fixture is 300 x 24, the inverse shape, and it is the one
regime in which the ambient solve is correct.

Raw before-and-after at `chain/repair/proofs/BLK-019-a/`.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.core.errors import NumericsError
from reward_lens.geometry.drift import representation_drift
from reward_lens.geometry.subspace import orthogonal_fit_on_span, procrustes

FRAME = "frame:blk019a"

# The registered geometry: 256 items (part_13.md:208) at width 2560 (part_04.md:7, Qwen3-4B).
N_REGISTERED = 256
D_REGISTERED = 2560

# A smaller shape in the same failing regime (n < d), for the cases that do not need the exact
# registered numbers. Every one of these also returns pi before the repair.
N_SMALL = 40
D_SMALL = 300

LATENT_RANK = 6


def _bank(n: int, d: int, seed: int) -> np.ndarray:
    """A bank with genuine low-rank structure over a noise floor, as a residual stream has."""
    rng = np.random.default_rng(seed)
    scale = np.array([5.0, 4.0, 3.0, 2.0, 1.5, 1.2])[:LATENT_RANK]
    latent = rng.standard_normal((n, LATENT_RANK)) * scale
    loading, _ = np.linalg.qr(rng.standard_normal((d, LATENT_RANK)))
    return latent @ loading.T + rng.standard_normal((n, d)) * 0.15


def _plane_in_span(bank: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors the bank's own rows span.

    Planted inside the row space deliberately. A rotation of a plane the bank does not span is not
    observable from the bank at all, so asking the estimator to report it would be a broken test
    rather than a broken estimator. The defect under repair is the opposite case: a rotation that
    **is** observable, reported as pi.
    """
    q, _ = np.linalg.qr(bank.T)
    rng = np.random.default_rng(seed)
    i, j = rng.choice(q.shape[1], size=2, replace=False)
    return np.ascontiguousarray(q[:, i]), np.ascontiguousarray(q[:, j])


def _rotate(bank: np.ndarray, u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """`bank @ R` for the rotation by `theta` in the (u, v) plane, without forming `R`.

    Forming a 2,560-square rotation costs 52 MB and this is the same arithmetic. The equivalence is
    asserted in `test_the_plane_rotation_helper_is_a_rotation`, because a fixture nobody checked is
    how a test ends up measuring itself.
    """
    c, s = float(np.cos(theta)), float(np.sin(theta))
    p = bank @ u
    q = bank @ v
    return (
        bank + (c - 1.0) * (np.outer(p, u) + np.outer(q, v)) + s * (np.outer(p, v) - np.outer(q, u))
    )


# ---------------------------------------------------------------------------
# The fixture's own correctness, first
# ---------------------------------------------------------------------------


def test_the_plane_rotation_helper_is_a_rotation() -> None:
    """`_rotate` equals multiplication by an orthogonal matrix, and preserves every norm."""
    bank = _bank(N_SMALL, D_SMALL, seed=11)
    u, v = _plane_in_span(bank, seed=3)
    theta = 0.7

    explicit = np.eye(D_SMALL)
    outer = np.outer(u, u) + np.outer(v, v)
    explicit = (
        explicit - outer + np.cos(theta) * outer + np.sin(theta) * (np.outer(u, v) - np.outer(v, u))
    )
    assert np.allclose(explicit @ explicit.T, np.eye(D_SMALL), atol=1e-12)
    assert np.allclose(_rotate(bank, u, v, theta), bank @ explicit, atol=1e-10)
    assert np.allclose(
        np.linalg.norm(_rotate(bank, u, v, theta), axis=1), np.linalg.norm(bank, axis=1)
    )


# ---------------------------------------------------------------------------
# The closure proof the child blocker states, at the registered geometry
# ---------------------------------------------------------------------------


def test_self_drift_is_zero_at_the_registered_geometry() -> None:
    """256 x 2560, a bank against itself. Before the repair this read 3.1408.

    `G17`'s tolerance is `1e-12` on the similarities; `is_zero` compares the angle at `sqrt(tol)`
    for the reason its docstring gives, so this asserts the raw angle separately and tightly.
    """
    bank = _bank(N_REGISTERED, D_REGISTERED, seed=19)
    d = representation_drift(bank, bank, FRAME, rank=LATENT_RANK, n_null=4)

    assert d.identical_inputs is True
    assert d.n_items == N_REGISTERED
    assert d.d_ambient == D_REGISTERED
    assert d.procrustes_angle < 1e-8, f"self-drift angle is {d.procrustes_angle}"
    assert d.procrustes_geodesic < 1e-6, f"self-drift geodesic is {d.procrustes_geodesic}"
    assert d.is_zero(tol=1e-12)


@pytest.mark.parametrize("theta", [0.1, 0.5, 1.0])
def test_planted_rotation_is_recovered_at_the_registered_geometry(theta: float) -> None:
    """256 x 2560. Before the repair all three of these read 3.14.

    `G17`'s tolerance, verbatim: a known rotation recovered to 0.01 radians.
    """
    bank = _bank(N_REGISTERED, D_REGISTERED, seed=19)
    u, v = _plane_in_span(bank, seed=3)
    rotated = _rotate(bank, u, v, theta)

    d = representation_drift(bank, rotated, FRAME, rank=LATENT_RANK, n_null=4)
    assert abs(d.procrustes_angle - theta) < 0.01, f"planted {theta} rad, read {d.procrustes_angle}"
    assert abs(d.procrustes_geodesic - theta) < 0.01
    # An exact rotation, so the residual is numerically zero and CKA is unchanged: the
    # representation is intact and only the coordinates moved. Three numbers, never one.
    assert d.procrustes_disparity < 1e-12
    assert abs(d.cka_drift) < 1e-12


# ---------------------------------------------------------------------------
# The same, at the small shape, so the fast suite covers the property too
# ---------------------------------------------------------------------------


def test_self_drift_is_zero_in_the_wide_regime() -> None:
    bank = _bank(N_SMALL, D_SMALL, seed=11)
    d = representation_drift(bank, bank, FRAME, rank=LATENT_RANK, n_null=8)
    assert d.procrustes_angle < 1e-8
    assert d.is_zero(tol=1e-12)


@pytest.mark.parametrize("theta", [0.05, 0.1, 0.5, 1.0, 2.0])
def test_the_angle_tracks_the_planted_rotation_in_the_wide_regime(theta: float) -> None:
    """Five magnitudes, so no constant and no clamp can pass this file.

    A repair that returned zero would fail every case but the first; a repair that kept returning pi
    fails all five; a repair that refused rather than measured returns no number at all.
    """
    bank = _bank(N_SMALL, D_SMALL, seed=11)
    u, v = _plane_in_span(bank, seed=3)
    d = representation_drift(bank, _rotate(bank, u, v, theta), FRAME, rank=LATENT_RANK, n_null=8)
    assert abs(d.procrustes_angle - theta) < 0.01, f"planted {theta}, read {d.procrustes_angle}"


def test_two_planes_give_the_max_and_the_root_sum_of_squares() -> None:
    """The geodesic is not the angle. One number cannot carry a rotation in two planes.

    This is the case the module's own docstring says a single-plane fixture hides.
    """
    bank = _bank(N_SMALL, D_SMALL, seed=11)
    q, _ = np.linalg.qr(bank.T)
    u1, v1, u2, v2 = (np.ascontiguousarray(q[:, i]) for i in (0, 1, 2, 3))
    rotated = _rotate(_rotate(bank, u1, v1, 0.3), u2, v2, 0.7)

    d = representation_drift(bank, rotated, FRAME, rank=LATENT_RANK, n_null=8)
    assert abs(d.procrustes_angle - 0.7) < 0.01
    assert abs(d.procrustes_geodesic - float(np.hypot(0.3, 0.7))) < 0.01
    assert d.procrustes_geodesic > d.procrustes_angle


# ---------------------------------------------------------------------------
# The observability limit is on the record rather than implied
# ---------------------------------------------------------------------------


def test_rotation_support_records_how_much_of_the_width_was_measured() -> None:
    """A bank against itself constrains its own span; a rotated pair constrains the union."""
    bank = _bank(N_SMALL, D_SMALL, seed=11)
    same = representation_drift(bank, bank, FRAME, rank=LATENT_RANK, n_null=8)
    assert same.rotation_support == N_SMALL < D_SMALL

    u, v = _plane_in_span(bank, seed=3)
    turned = representation_drift(bank, _rotate(bank, u, v, 0.5), FRAME, rank=LATENT_RANK, n_null=8)
    # The plane sits inside the bank's own span, so the rotated bank spans the same subspace.
    assert turned.rotation_support == N_SMALL


def test_a_tall_bank_is_measured_in_the_full_width() -> None:
    """When the items outnumber the width there is nothing unconstrained, and support is `d`."""
    bank = _bank(120, 24, seed=5)
    d = representation_drift(bank, bank, FRAME, rank=LATENT_RANK, n_null=8)
    assert d.rotation_support == 24
    assert d.procrustes_angle < 1e-8


# ---------------------------------------------------------------------------
# The primitive underneath, and the regime where the old solve was right
# ---------------------------------------------------------------------------


def test_the_span_fit_agrees_with_the_classical_solve_where_the_classical_solve_is_valid() -> None:
    """`n >= d`: nothing is unconstrained, so the constrained fit must be the classical answer.

    Asserted against a rotation solved the textbook way, in this test, so the equivalence does not
    rest on the same code being right on both sides.
    """
    rng = np.random.default_rng(7)
    x = rng.standard_normal((150, 12))
    rot, _ = np.linalg.qr(rng.standard_normal((12, 12)))
    y = x @ rot

    classical_u, _, classical_vt = np.linalg.svd(x.T @ y)
    classical = classical_u @ classical_vt

    fit = orthogonal_fit_on_span(x, y, FRAME)
    assert fit.support == 12
    ambient = fit.basis @ fit.rotation @ fit.basis.T
    assert np.allclose(ambient, classical, atol=1e-10)
    assert np.allclose(procrustes(x, y, FRAME).rotation, classical, atol=1e-5)
    assert fit.disparity < 1e-20


def test_the_span_fit_reproduces_the_data_it_was_fitted_to() -> None:
    """`X R = Y` on the span, in the wide regime, which is what the disparity claims."""
    bank = _bank(N_SMALL, D_SMALL, seed=11)
    u, v = _plane_in_span(bank, seed=3)
    rotated = _rotate(bank, u, v, 0.8)

    fit = orthogonal_fit_on_span(bank, rotated, FRAME)
    reduced_a = bank @ fit.basis
    reduced_b = rotated @ fit.basis
    assert np.allclose(reduced_a @ fit.rotation, reduced_b, atol=1e-9)
    assert np.allclose(fit.rotation @ fit.rotation.T, np.eye(fit.support), atol=1e-10)
    assert fit.disparity < 1e-20


def test_the_ambient_rotation_is_the_identity_off_the_span() -> None:
    """What the support field promises, checked rather than described."""
    bank = _bank(N_SMALL, D_SMALL, seed=11)
    u, v = _plane_in_span(bank, seed=3)
    res = procrustes(bank, _rotate(bank, u, v, 0.4), FRAME)

    assert res.support == N_SMALL
    r = np.asarray(res.rotation, dtype=np.float64)
    assert np.allclose(r @ r.T, np.eye(D_SMALL), atol=1e-5)
    # Off the span the map does nothing, so the eigenvalues there are 1 and the trace is nearly `d`.
    assert D_SMALL - float(np.trace(r)) < 1.0


# ---------------------------------------------------------------------------
# The rank guard the row also names: shape is not rank
# ---------------------------------------------------------------------------


def test_the_rank_guard_tests_the_centred_rank_and_not_the_shape() -> None:
    """A 256-item bank supports 255 centred directions, not 256, and the guard now says so."""
    bank = _bank(60, D_SMALL, seed=11)
    representation_drift(bank, bank, FRAME, rank=59, n_null=4)
    with pytest.raises(NumericsError, match="centred rank"):
        representation_drift(bank, bank, FRAME, rank=60, n_null=4)


def test_the_rank_guard_follows_a_rank_deficient_bank_down() -> None:
    """A bank of 40 items that only spans 6 directions supports 6, whatever its shape says."""
    rng = np.random.default_rng(2)
    latent = rng.standard_normal((40, LATENT_RANK))
    loading, _ = np.linalg.qr(rng.standard_normal((D_SMALL, LATENT_RANK)))
    degenerate = latent @ loading.T  # exactly rank 6, no noise floor

    representation_drift(degenerate, degenerate, FRAME, rank=6, n_null=4)
    with pytest.raises(NumericsError, match="centred rank"):
        representation_drift(degenerate, degenerate, FRAME, rank=7, n_null=4)
