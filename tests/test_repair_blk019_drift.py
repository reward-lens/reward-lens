"""BLK-019 — no instrument measured representation drift between two checkpoints on a bank.

`cka`, `procrustes` and `subspace_alignment` all existed and, in `G17`'s words, "each takes two bare
matrices and a frame and none takes a checkpoint or a bank". The only checkpoint-pair instrument
measured the rotation of a fitted weight rather than representation drift.

The contract this file tests is Part 7.4's, not this builder's: see
`chain/repair/proofs/BLK-019/CONTRACT.md`, which records that the dispatch brief was wrong to say
the triple is written down nowhere. Part 7.4 registers three drift quantities and forbids collapsing
them into one.

Tolerances are G17's, quoted: self-drift zero to `1e-12`, a planted rotation recovered to 0.01
radians.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.geometry.drift import (
    DriftRefused,
    ReadingDrift,
    RepresentationDrift,
    reading_drift,
    representation_drift,
)

FRAME = "frame:test-bank"
N_ITEMS = 300
D = 24
RANK = 6


@pytest.fixture
def bank() -> np.ndarray:
    """An activation bank with genuine structure, so CKA and the subspace test have work to do.

    Six strong directions carrying most of the variance and a low-variance floor under them, which
    is what a residual stream looks like and is the case where a drift measure that only tracks the
    top singular vector reads stable while the bank moves.
    """
    rng = np.random.default_rng(19)
    latent = rng.standard_normal((N_ITEMS, RANK)) * np.array([5.0, 4.0, 3.0, 2.0, 1.5, 1.2])
    loading, _ = np.linalg.qr(rng.standard_normal((D, RANK)))
    return latent @ loading.T + rng.standard_normal((N_ITEMS, D)) * 0.15


def _givens(d: int, i: int, j: int, theta: float) -> np.ndarray:
    """A rotation by `theta` in the (i, j) plane and the identity everywhere else."""
    r = np.eye(d)
    c, s = np.cos(theta), np.sin(theta)
    r[i, i] = c
    r[j, j] = c
    r[i, j] = -s
    r[j, i] = s
    return r


# ---------------------------------------------------------------------------
# G17's closure proof, first half: a checkpoint against itself is zero drift
# ---------------------------------------------------------------------------


def test_a_checkpoint_against_itself_has_zero_drift(bank: np.ndarray) -> None:
    """G17, verbatim: drift between a checkpoint and itself is zero to 1e-12.

    All four numbers, in the one convention where zero means no drift.
    """
    d = representation_drift(bank, bank, FRAME, rank=RANK)
    assert abs(d.cka_drift) < 1e-12, f"cka_drift {d.cka_drift:.3e}"
    assert abs(d.subspace_drift) < 1e-12, f"subspace_drift {d.subspace_drift:.3e}"
    assert abs(d.procrustes_angle) < 1e-6, f"procrustes_angle {d.procrustes_angle:.3e}"
    assert abs(d.procrustes_disparity) < 1e-12, f"disparity {d.procrustes_disparity:.3e}"
    assert d.is_zero(tol=1e-12) is True


def test_a_rescaled_checkpoint_is_not_reported_as_drift(bank: np.ndarray) -> None:
    """CKA is scale-invariant, so a uniformly rescaled bank is the same representation.

    Included because a drift measure that fires on a rescaling would fire on every checkpoint of a
    run whose activation norms grow, which they do.
    """
    d = representation_drift(bank, bank * 3.0, FRAME, rank=RANK)
    assert abs(d.cka_drift) < 1e-12
    assert abs(d.subspace_drift) < 1e-12


# ---------------------------------------------------------------------------
# G17's closure proof, second half: a known rotation is recovered
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta", [0.05, 0.3, 1.0, np.pi / 2, 2.5])
def test_a_planted_rotation_is_recovered_to_a_hundredth_of_a_radian(
    bank: np.ndarray, theta: float
) -> None:
    """G17, verbatim: recovers the rotation angle to 0.01 radians."""
    rotated = bank @ _givens(D, 0, 1, theta).T
    d = representation_drift(bank, rotated, FRAME, rank=RANK)
    assert d.procrustes_angle == pytest.approx(theta, abs=0.01)
    assert d.procrustes_geodesic == pytest.approx(theta, abs=0.01)


def test_a_rotation_in_two_planes_separates_the_largest_angle_from_the_total(
    bank: np.ndarray,
) -> None:
    """A `d x d` rotation has several principal angles, so one scalar cannot be the whole answer.

    `procrustes_angle` is the largest and `procrustes_geodesic` is the root sum of squares. On a
    single-plane fixture the two agree, which is why reporting only one would look correct until a
    real checkpoint pair, where it never is.
    """
    a, b = 0.6, 0.4
    rotated = bank @ (_givens(D, 0, 1, a) @ _givens(D, 2, 3, b)).T
    d = representation_drift(bank, rotated, FRAME, rank=RANK)
    assert d.procrustes_angle == pytest.approx(max(a, b), abs=0.01)
    assert d.procrustes_geodesic == pytest.approx(np.hypot(a, b), abs=0.01)
    assert d.procrustes_geodesic > d.procrustes_angle


def test_the_three_numbers_disagree_under_a_rotation_and_all_three_are_right(
    bank: np.ndarray,
) -> None:
    """A pure rotation is the case that separates the three quantities, and it separates them.

    CKA is invariant to an orthogonal transform, so it reads no drift: the representation is intact.
    Procrustes recovers the full angle: the coordinates moved by a radian. Subspace alignment moves
    a little, because the dominant subspace physically rotates with the bank and no longer coincides
    with where it was.

    All three are correct answers to three different questions, and a caller reading only CKA would
    conclude a frozen frame while every direction fitted in it had rotated out from under them.
    That is Part 7.4's argument for reporting three numbers, on a fixture where it is checkable.
    """
    rotated = bank @ _givens(D, 0, 1, 1.0).T
    d = representation_drift(bank, rotated, FRAME, rank=RANK)
    assert abs(d.cka_drift) < 1e-10, "CKA is rotation-invariant and should read no drift"
    assert d.procrustes_angle == pytest.approx(1.0, abs=0.01)
    assert 0.0 < d.subspace_drift < 0.5, (
        f"the dominant subspace should move under a rotation but not vanish: {d.subspace_drift:.4f}"
    )
    assert d.is_zero(tol=1e-12) is False, "a rotated bank is not zero drift"


def test_rotating_inside_the_dominant_subspace_leaves_the_subspace_alignment_alone(
    bank: np.ndarray,
) -> None:
    """The distinction the test above rests on, isolated.

    A rotation that acts only within the dominant subspace moves the coordinates and leaves the
    subspace where it was, so Procrustes fires and the subspace test does not. A rotation that
    takes the subspace out of itself moves both. If subspace alignment were rotation-invariant, as a
    first reading of it suggests, neither of these would be distinguishable.
    """
    basis = np.linalg.svd(bank - bank.mean(axis=0), full_matrices=False)[2][:RANK].T
    inner = _givens(RANK, 0, 1, 0.7)
    within = np.eye(D) + basis @ (inner - np.eye(RANK)) @ basis.T
    d = representation_drift(bank, bank @ within.T, FRAME, rank=RANK)
    assert d.procrustes_angle == pytest.approx(0.7, abs=0.01)
    assert abs(d.subspace_drift) < 1e-8, (
        f"a rotation inside the subspace must leave the subspace alone: {d.subspace_drift:.3e}"
    )


def test_a_genuinely_different_bank_moves_every_number(bank: np.ndarray) -> None:
    """A drift measure that cannot be made to fire is decoration."""
    rng = np.random.default_rng(77)
    other = rng.standard_normal((N_ITEMS, D))
    d = representation_drift(bank, other, FRAME, rank=RANK)
    assert d.cka_drift > 0.5
    assert d.subspace_drift > 0.2
    assert d.procrustes_disparity > 0.1


# ---------------------------------------------------------------------------
# The contract's structural commitments
# ---------------------------------------------------------------------------


def test_there_is_no_collapsed_scalar_drift(bank: np.ndarray) -> None:
    """Part 7.4 reports three quantities "separately rather than one" and gives the reason.

    A single `.drift` float would be exactly the thing that paragraph forbids, so its absence is a
    contract term rather than an oversight and is asserted as one.
    """
    d = representation_drift(bank, bank, FRAME, rank=RANK)
    assert not hasattr(d, "drift")
    assert not hasattr(d, "overall")


def test_the_rank_is_required(bank: np.ndarray) -> None:
    """No rank is registered anywhere, so a default here would be this module choosing one."""
    with pytest.raises(TypeError):
        representation_drift(bank, bank, FRAME)  # type: ignore[call-arg]


def test_a_frameless_comparison_raises(bank: np.ndarray) -> None:
    """Gate 2 and invariant I3: a COVARIANT comparison with no shared frame is a coordinate artifact."""
    from reward_lens.core.errors import GaugeError

    with pytest.raises(GaugeError):
        representation_drift(bank, bank, None, rank=RANK)


def test_mismatched_banks_raise_rather_than_comparing_different_inputs(bank: np.ndarray) -> None:
    """Row `i` of each bank has to be the same input, or the number is about the inputs."""
    from reward_lens.core.errors import NumericsError

    with pytest.raises((NumericsError, ValueError)):
        representation_drift(bank, bank[:100], FRAME, rank=RANK)


def test_identical_banks_are_flagged_because_that_is_what_blk_002_produces(
    bank: np.ndarray,
) -> None:
    """Until BLK-002 lands the cache serves one capture to every checkpoint, so both banks are one.

    A flag rather than a refusal: a genuinely frozen checkpoint pair produces the same thing, and
    this function cannot tell those apart. What it can do is not stay silent about it.
    """
    same = representation_drift(bank, bank.copy(), FRAME, rank=RANK)
    assert same.identical_inputs is True
    moved = representation_drift(bank, bank @ _givens(D, 0, 1, 0.3).T, FRAME, rank=RANK)
    assert moved.identical_inputs is False


# ---------------------------------------------------------------------------
# Part 7.4's three quantities and C9's three fields
# ---------------------------------------------------------------------------


@pytest.fixture
def labelled_bank() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A bank at checkpoint B, its labels, and the direction that separates them there."""
    rng = np.random.default_rng(9)
    truth = np.zeros(D)
    truth[0] = 1.0
    labels = (rng.random(N_ITEMS) < 0.5).astype(int)
    b = rng.standard_normal((N_ITEMS, D))
    b[:, 0] += labels * 2.0
    return b, labels, truth


def test_the_reading_reports_all_three_of_part_7_4_s_quantities(
    labelled_bank: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    b, labels, truth = labelled_bank
    stale = np.zeros(D)
    stale[0] = np.cos(0.4)
    stale[1] = np.sin(0.4)
    out = reading_drift(stale, truth, FRAME, bank=b, labels=labels, cos_floor=0.5)
    assert isinstance(out, ReadingDrift)
    assert out.cos_consecutive == pytest.approx(np.cos(0.4), abs=1e-6)
    assert 0.5 <= out.auc_frozen <= 1.0
    assert 0.5 <= out.auc_refit <= 1.0
    assert out.auc_refit > out.auc_frozen, "the refit direction should read at least as well"
    assert out.auc_decay == pytest.approx(out.auc_refit - out.auc_frozen, abs=1e-12)


def test_the_cosine_can_stay_high_while_the_auc_collapses(
    labelled_bank: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """`C9`'s registered prediction, as a fixture: the axis holds while its content decays.

    Part 7.4's reason for reporting three numbers, made to happen: a direction 0.2 radians off the
    fitted one keeps a cosine of 0.98, and on a bank whose separating direction is narrow it still
    loses most of its AUC. Printing only the cosine would call this instrument stable.
    """
    rng = np.random.default_rng(91)
    labels = (rng.random(N_ITEMS) < 0.5).astype(int)
    b = rng.standard_normal((N_ITEMS, D)) * 6.0
    truth = np.zeros(D)
    truth[0] = 1.0
    b[:, 0] = rng.standard_normal(N_ITEMS) * 0.05 + labels * 0.6

    stale = np.zeros(D)
    stale[0] = np.cos(0.2)
    stale[1] = np.sin(0.2)
    out = reading_drift(stale, truth, FRAME, bank=b, labels=labels, cos_floor=0.5)
    assert out.cos_consecutive > 0.95, "the cosine is meant to stay high in this fixture"
    assert out.auc_refit > 0.99
    assert out.auc_frozen < 0.75, f"the frozen AUC did not decay: {out.auc_frozen:.3f}"


def test_the_raw_and_canonical_cosines_are_both_returned(
    labelled_bank: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Part 7.4 does not say which cosine `C9`'s 0.95 is on, so neither is dropped."""
    b, labels, truth = labelled_bank
    stale = np.zeros(D)
    stale[0] = np.cos(0.3)
    stale[2] = np.sin(0.3)
    out = reading_drift(stale, truth, FRAME, bank=b, labels=labels, cos_floor=0.5)
    assert np.isfinite(out.cos_raw)
    assert np.isfinite(out.cos_consecutive)


# ---------------------------------------------------------------------------
# The refusal Part 7.4 asks for. Assert the raise, never read a number.
# ---------------------------------------------------------------------------


def test_a_cosine_below_the_floor_refuses_rather_than_reporting(
    labelled_bank: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Part 7.4: "a declared threshold on cosine similarity below which the reading refuses".

    G16's standard: the test asserts the raise rather than rendering a reading.
    """
    b, labels, truth = labelled_bank
    orthogonal = np.zeros(D)
    orthogonal[5] = 1.0
    with pytest.raises(DriftRefused) as caught:
        reading_drift(orthogonal, truth, FRAME, bank=b, labels=labels, cos_floor=0.95)
    assert caught.value.cos_floor == 0.95
    assert caught.value.cos_consecutive < 0.95


def test_the_floor_is_required_because_no_number_is_registered(
    labelled_bank: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    b, labels, truth = labelled_bank
    with pytest.raises(TypeError):
        reading_drift(truth, truth, FRAME, bank=b, labels=labels)  # type: ignore[call-arg]


def test_the_refusal_does_not_compute_the_aucs_it_refused_to_report(
    labelled_bank: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """A refusal that carries the number it refused is a value with a warning attached."""
    b, labels, truth = labelled_bank
    orthogonal = np.zeros(D)
    orthogonal[5] = 1.0
    with pytest.raises(DriftRefused) as caught:
        reading_drift(orthogonal, truth, FRAME, bank=b, labels=labels, cos_floor=0.95)
    assert not hasattr(caught.value, "auc_frozen")
    assert not hasattr(caught.value, "auc_refit")


def test_a_floor_that_passes_returns_a_reading(
    labelled_bank: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """The refusal must not have eaten the working path."""
    b, labels, truth = labelled_bank
    out = reading_drift(truth, truth, FRAME, bank=b, labels=labels, cos_floor=0.95)
    assert out.cos_consecutive == pytest.approx(1.0, abs=1e-6)
    assert out.auc_frozen == pytest.approx(out.auc_refit, abs=1e-12)


def test_the_reading_is_frame_gated_too(
    labelled_bank: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    from reward_lens.core.errors import GaugeError

    b, labels, truth = labelled_bank
    with pytest.raises(GaugeError):
        reading_drift(truth, truth, None, bank=b, labels=labels, cos_floor=0.5)


def test_the_types_are_frozen_records(bank: np.ndarray) -> None:
    d = representation_drift(bank, bank, FRAME, rank=RANK)
    assert isinstance(d, RepresentationDrift)
    with pytest.raises(Exception):
        d.cka = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# The same closure proof at the geometry the run actually produces
#
# The fixture above is 300 items in 24 dimensions. That is the inverse of the registered shape and
# it is the one regime in which an unguarded Procrustes fit works, so a file that only used it
# would pass with the defect still in place: `chain/boundary/evidence/BLK-019-a.baseline.txt`
# records the unguarded fit returning 3.1416 for all four cases at 256 x 2560 while the same fit is
# correct at 300 x 24. The row's assertion names 256 x 2560, so the cases that discriminate belong
# in the file the row's own acceptance command runs, not only in a sibling.
#
# `n_null` is 4 rather than the default 1000: the null draw is per-permutation work at the full
# width and the numbers asserted here are the fitted angles, which the null does not enter.
# ---------------------------------------------------------------------------

#: 256 items (`chain/parts/part_13.md:208`) at width 2560 (`chain/parts/part_04.md:7`, Qwen3-4B).
N_REGISTERED = 256
D_REGISTERED = 2560


def _registered_bank(seed: int = 19) -> np.ndarray:
    """The same low-rank-over-noise structure as `bank`, at the registered shape."""
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((N_REGISTERED, RANK)) * np.array([5.0, 4.0, 3.0, 2.0, 1.5, 1.2])
    loading, _ = np.linalg.qr(rng.standard_normal((D_REGISTERED, RANK)))
    return latent @ loading.T + rng.standard_normal((N_REGISTERED, D_REGISTERED)) * 0.15


def _plane_in_span(bank_: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors the bank's own rows span.

    Inside the row space deliberately. A rotation of a plane the bank does not span is not
    observable from the bank at all, so asking the estimator to report it would be a broken test.
    """
    q, _ = np.linalg.qr(bank_.T)
    rng = np.random.default_rng(seed)
    i, j = rng.choice(q.shape[1], size=2, replace=False)
    return np.ascontiguousarray(q[:, i]), np.ascontiguousarray(q[:, j])


def _rotate_in_plane(bank_: np.ndarray, u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """`bank @ R` for the rotation by `theta` in the (u, v) plane, without forming `R`.

    Forming a 2,560-square rotation costs 52 MB and this is the same arithmetic; the equivalence is
    asserted below, because a fixture nobody checked is how a test ends up measuring itself.
    """
    c, s = float(np.cos(theta)), float(np.sin(theta))
    p = bank_ @ u
    q = bank_ @ v
    return (
        bank_
        + (c - 1.0) * (np.outer(p, u) + np.outer(q, v))
        + s * (np.outer(p, v) - np.outer(q, u))
    )


def test_the_plane_rotation_helper_is_an_orthogonal_map() -> None:
    """Check the fixture before asserting anything with it, at a shape where `R` is affordable."""
    small = np.random.default_rng(11).standard_normal((40, 60))
    u, v = _plane_in_span(small, seed=3)
    theta = 0.7
    outer = np.outer(u, u) + np.outer(v, v)
    explicit = (
        np.eye(60)
        - outer
        + np.cos(theta) * outer
        + np.sin(theta) * (np.outer(u, v) - np.outer(v, u))
    )
    assert np.allclose(explicit @ explicit.T, np.eye(60), atol=1e-12)
    assert np.allclose(_rotate_in_plane(small, u, v, theta), small @ explicit, atol=1e-10)


def test_self_drift_is_zero_at_the_registered_geometry() -> None:
    """256 x 2560, a bank against itself. G17's first half, at the shape the run produces."""
    registered = _registered_bank()
    d = representation_drift(registered, registered, FRAME, rank=RANK, n_null=4)

    assert d.n_items == N_REGISTERED
    assert d.d_ambient == D_REGISTERED
    assert d.identical_inputs is True
    assert d.procrustes_angle < 1e-8, f"self-drift angle is {d.procrustes_angle}"
    assert d.procrustes_geodesic < 1e-6, f"self-drift geodesic is {d.procrustes_geodesic}"
    assert d.is_zero(tol=1e-12)
    # How many directions the rotation was measured in at all, which is the number that makes the
    # 256-in-2560 regime legible rather than silently unconstrained.
    assert d.rotation_support == N_REGISTERED


@pytest.mark.parametrize("theta", [0.1, 0.5, 1.0])
def test_a_planted_rotation_is_recovered_at_the_registered_geometry(theta: float) -> None:
    """G17's second half, tolerance verbatim: a known rotation recovered to 0.01 radians."""
    registered = _registered_bank()
    u, v = _plane_in_span(registered, seed=3)
    rotated = _rotate_in_plane(registered, u, v, theta)

    d = representation_drift(registered, rotated, FRAME, rank=RANK, n_null=4)
    assert abs(d.procrustes_angle - theta) < 0.01, f"planted {theta}, read {d.procrustes_angle}"
    assert abs(d.procrustes_geodesic - theta) < 0.01
    # An exact rotation: the residual is numerically zero and CKA has not moved, so the
    # representation is intact and only the coordinates turned. Three numbers, never one.
    assert d.procrustes_disparity < 1e-12
    assert abs(d.cka_drift) < 1e-12
