"""Drift between two checkpoints on a fixed bank, in the cosine unit and the AUC unit (Part 7.4).

`cka`, `procrustes` and `subspace_alignment` all lived next door and, in the gap ledger's words,
"each takes two bare matrices and a frame and none takes a checkpoint or a bank". The only
checkpoint-pair instrument in the package measured the rotation of a fitted weight, which is a
different question, and captured activations once on the last checkpoint only to fit a gauge. So the
three drift quantities Part 7.4 registers were assembled by hand wherever anybody needed them, which
is what this module is for.

**Two objects, because the design asks two questions.** Part 7.4's three quantities are about a
fitted direction: does the instrument still point the same way, and does it still predict.
`CHAIN_GAPS` `G17`'s closure proof is about the representation: does a bank rotated by a known angle
give that angle back. A rotation angle is not a property of an AUC, so `representation_drift`
answers the second and `reading_drift` answers the first. Both take a bank; that is the whole of
what was missing.

**Three numbers, never one.** Part 7.4, verbatim:

    So this design measures and reports three drift quantities separately rather than one: the
    cosine similarity between directions fitted at consecutive checkpoints, the AUC of a frozen
    direction read at later checkpoints, and the AUC of a direction refit at each checkpoint.
    Printing only the first would say the instrument is stable when its predictive content had
    decayed.

There is deliberately no collapsed scalar on either record. The design's own paragraph exists to
forbid it, and the reason is in the two published numbers behind it: over 120 GRPO steps a probe's
off-domain AUC falls from 0.991 to 0.376 while independently fitted directions keep a cosine at or
above 0.99. Both are true. One number cannot say both.

**What this module does not decide.** The subspace rank, and the cosine floor below which a reading
refuses. Neither has a registered value anywhere in the corpus, so both are required arguments with
no default. The full contract, including the question about which cosine `C9`'s 0.95 threshold is
measured on, is in `chain/repair/proofs/BLK-019/CONTRACT.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from reward_lens.core.errors import NumericsError
from reward_lens.core.gates import require_frame_for_comparison
from reward_lens.core.types import FrameID, GaugeStatus
from reward_lens.geometry.canonical import canonicalize
from reward_lens.geometry.frame import Frame
from reward_lens.geometry.subspace import (
    cka,
    orthogonal_fit_on_span,
    subspace_alignment,
)


class DriftRefused(NumericsError):
    """A reading refused because the fitted direction has drifted past the declared floor.

    Part 7.4 asks for "a declared threshold on cosine similarity below which the reading refuses
    rather than reports". This is that refusal. It carries the two cosines and the floor and
    **deliberately does not carry the AUCs**, which are not computed in this branch: an AUC read
    along a direction that has drifted past the floor is the confident wrong number the refusal
    exists to prevent, and attaching it to the refusal would put it one attribute access away.

    Raised rather than returned because that is how this layer signals. Gate 2 raises `GaugeError`
    and the subspace primitives raise `NumericsError`. A `core.reading.Refusal` is the right carrier
    one layer up, at an instrument; these are geometry functions and instruments are their callers.
    """

    def __init__(self, cos_consecutive: float, cos_raw: float, cos_floor: float) -> None:
        self.cos_consecutive = float(cos_consecutive)
        self.cos_raw = float(cos_raw)
        self.cos_floor = float(cos_floor)
        super().__init__(
            f"the canonical cosine between the two fitted directions is "
            f"{self.cos_consecutive:.4f}, below the declared floor of {self.cos_floor:.4f}, so a "
            f"reading taken along the earlier direction is not a reading of the same quantity. The "
            f"raw cosine is {self.cos_raw:.4f}. Refit at this checkpoint, or report the row as "
            f"unreadable across this pair, or lower the floor and record that you did"
        )


def _frame_id(frame: Any) -> FrameID | None:
    return None if frame is None else (frame.id if isinstance(frame, Frame) else frame)


def _require(frame: Any) -> FrameID:
    """Gate 2, in the same form `geometry.subspace` applies it."""
    fid = _frame_id(frame)
    require_frame_for_comparison(GaugeStatus.COVARIANT, fid)
    return fid  # type: ignore[return-value]


def _bank(x: Any, name: str) -> np.ndarray:
    m = np.asarray(x, dtype=np.float64)
    if m.ndim != 2:
        raise NumericsError(f"{name} is an (n_items, d) activation bank; got shape {m.shape}")
    if m.size == 0:
        raise NumericsError(f"{name} is empty")
    if not np.all(np.isfinite(m)):
        raise NumericsError(f"{name} carries non-finite entries")
    return m


def _principal_angles(rotation: np.ndarray) -> np.ndarray:
    """The principal rotation angles of an orthogonal matrix, in radians.

    An orthogonal `R` has eigenvalues on the unit circle: conjugate pairs `exp(+/- i theta)` for
    each plane it rotates, and real +/-1 for the directions it fixes or reflects. The arguments of
    those eigenvalues are the rotation angles, so this reads them off directly and is exact for any
    number of planes rather than only for one.

    Taken from the eigenvalues rather than from `arccos((tr R - (d - 2)) / 2)`, which is the closed
    form for a single-plane rotation and is silently wrong for every real checkpoint pair, where the
    representation moves in many planes at once.
    """
    eigenvalues = np.linalg.eigvals(np.asarray(rotation, dtype=np.float64))
    angles = np.abs(np.angle(eigenvalues))
    # Each rotated plane contributes a conjugate pair, so every angle appears twice. Sorting and
    # taking every second entry recovers one per plane, which is what the geodesic norm sums over.
    return np.sort(angles)[::-1][::2]


@dataclass(frozen=True)
class RepresentationDrift:
    """How far the activation bank itself moved between two checkpoints.

    Three quantities, in `G17`'s naming order, which is also increasing specificity.

    ``cka`` asks whether this is the same representation at all, up to an orthogonal transform and a
    rescaling. It is invariant to both by construction, so a purely rotated bank reads `cka = 1`.
    That is the right answer and it is the reason three numbers are reported: the representation is
    intact and the coordinates have moved out from under every direction fitted in them.

    ``procrustes_angle`` and ``procrustes_geodesic`` ask how far the coordinates moved. The best
    aligning rotation is a `d x d` orthogonal matrix with one angle per plane it turns, so one
    scalar cannot be the whole answer: ``procrustes_angle`` is the largest of them and
    ``procrustes_geodesic`` is the root sum of squares, the geodesic distance from the identity on
    `SO(d)`. On a single-plane fixture the two coincide, which is exactly why reporting only one
    looks correct in a test and never is on a real pair. ``procrustes_disparity`` is the residual
    that survives the best alignment, in the primitive's own normalisation.

    ``subspace_alignment`` asks whether the dominant subspace survived, read against the
    identifiability null the primitive already computes, with ``subspace_excess`` the amount above
    chance.

    ``cka_drift`` and ``subspace_drift`` are `1 - similarity`, so that "the drift is zero" means the
    same thing across all four numbers. There is no collapsed scalar and its absence is a contract
    term.

    ``identical_inputs`` is set when the two banks are bit for bit equal. Until BLK-002 lands, the
    activation cache serves the first capture ever taken to every checkpoint of every seed, and that
    is what a collided pair looks like from here. A genuinely frozen pair looks the same, so this is
    a flag and not a refusal; what it is not is silent.

    ``rotation_support`` is how many of the ``d_ambient`` directions the two banks jointly span, and
    therefore how many the rotation is measured in at all. A bank of 256 items in a 2,560-wide
    residual stream constrains at most 512, and the remaining directions are reported as unrotated
    because nothing observed them, not because they are known to have held still. It is on the
    record so a reader can see which of the two it is (BLK-019-a).
    """

    cka: float
    procrustes_angle: float
    procrustes_geodesic: float
    procrustes_disparity: float
    subspace_alignment: float
    subspace_excess: float
    rank: int
    n_items: int
    d_ambient: int
    rotation_support: int
    frame: FrameID
    identical_inputs: bool

    @property
    def cka_drift(self) -> float:
        return 1.0 - self.cka

    @property
    def subspace_drift(self) -> float:
        return 1.0 - self.subspace_alignment

    def is_zero(self, *, tol: float) -> bool:
        """Whether every quantity reads no drift at `tol`.

        The angle is compared at `sqrt(tol)` rather than at `tol`. A rotation by `theta` changes an
        inner product by order `theta**2` near the identity, so holding an angle to `1e-12` while
        holding a similarity to `1e-12` asks the angle for twice the precision the arithmetic has.
        """
        return bool(
            abs(self.cka_drift) < tol
            and abs(self.subspace_drift) < tol
            and abs(self.procrustes_disparity) < tol
            and abs(self.procrustes_angle) < max(float(np.sqrt(tol)), 1e-9)
        )


def representation_drift(
    bank_a: Any,
    bank_b: Any,
    frame: Any,
    *,
    rank: int,
    n_null: int = 1000,
    seed: int = 0,
) -> RepresentationDrift:
    """Drift between two checkpoints, read on one fixed bank of inputs.

    ``bank_a`` and ``bank_b`` are `(n_items, d)` activation matrices captured on **the same inputs
    in the same order** at the two checkpoints. Row `i` of each has to be the same input; if it is
    not, every number below is about the inputs rather than about the checkpoints, and the row-count
    check is the only part of that this function can verify for you.

    ``rank`` is the dimension of the dominant subspace the third quantity compares, taken as the top
    `rank` right singular vectors of each centred bank. It is required and has no default because no
    rank is registered anywhere in the corpus.

    ``frame`` is required and has no default, per gate 2 and invariant I3: a cosine or a subspace
    overlap between two checkpoints is COVARIANT, and comparing them without a shared frame is how a
    coordinate change gets published as a functional one.
    """
    fid = _require(frame)
    a = _bank(bank_a, "bank_a")
    b = _bank(bank_b, "bank_b")
    if a.shape[0] != b.shape[0]:
        raise NumericsError(
            f"the two banks hold {a.shape[0]} and {b.shape[0]} items. Drift is read on a fixed "
            f"bank, so row i of each has to be the same input"
        )
    if a.shape[1] != b.shape[1]:
        raise NumericsError(
            f"the two banks have widths {a.shape[1]} and {b.shape[1]}; Procrustes needs equal shapes"
        )
    k = int(rank)
    if k < 1:
        raise NumericsError(f"the compared subspace needs rank at least 1; got {rank}")
    available = min(_centred_rank(a), _centred_rank(b))
    if k > available:
        raise NumericsError(
            f"rank {k} exceeds the banks' own centred rank {available}; the comparison would be "
            f"between subspaces that are partly numerical noise. The shape bound this guard used to "
            f"test, min{a.shape}, is not the same number: centring costs a dimension and a bank can "
            f"be short of full rank besides"
        )

    similarity = float(cka(a, b, fid))
    fit = orthogonal_fit_on_span(a, b, fid)
    # The angles come from the reduced fit. The ambient rotation is the identity outside the span,
    # so it contributes only zeros, and materialising a `d x d` matrix to read them off would cost
    # a 2,560-square eigendecomposition to learn nothing.
    angles = _principal_angles(fit.rotation)
    alignment = subspace_alignment(
        _dominant_basis(a, k), _dominant_basis(b, k), fid, n_null=n_null, seed=seed
    )
    return RepresentationDrift(
        cka=similarity,
        procrustes_angle=float(angles.max()) if angles.size else 0.0,
        procrustes_geodesic=float(np.sqrt(float(np.sum(angles**2)))),
        procrustes_disparity=float(fit.disparity),
        subspace_alignment=float(alignment.alignment),
        subspace_excess=float(alignment.excess),
        rank=k,
        n_items=int(a.shape[0]),
        d_ambient=int(a.shape[1]),
        rotation_support=int(fit.support),
        frame=fid,
        identical_inputs=bool(a.shape == b.shape and np.array_equal(a, b)),
    )


def _centred_rank(bank: np.ndarray) -> int:
    """The numerical rank of the centred bank, which is what `_dominant_basis` can draw from.

    Not `min(bank.shape)`. Centring removes one dimension, so a 256-item bank supports 255 and not
    256, and a bank whose items are not in general position supports fewer still. The guard that
    tested the shape accepted ranks the data cannot carry and reported the top singular vectors of
    the noise floor as a dominant subspace.
    """
    centred = bank - bank.mean(axis=0)
    return int(np.linalg.matrix_rank(centred))


def _dominant_basis(bank: np.ndarray, rank: int) -> np.ndarray:
    """The top-`rank` right singular vectors of a centred bank, as a `d x rank` basis.

    Centred first, because an uncentred bank's leading singular vector is its mean and two
    checkpoints of one model share a mean far more closely than they share their structure. An
    alignment computed on the uncentred bank would therefore read high on every pair and be a
    measurement of the mean.
    """
    centred = bank - bank.mean(axis=0)
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    return np.ascontiguousarray(vh[:rank].T)


@dataclass(frozen=True)
class ReadingDrift:
    """Part 7.4's three quantities, which are also `C9`'s three recorded fields.

    ``cos_consecutive`` is the canonical cosine between the direction fitted at the earlier
    checkpoint and the one fitted at the later one, whitened in the shared frame. ``cos_raw`` is the
    un-whitened cosine, kept beside it for the same reason `AngleResult` keeps it: it is the number
    that made E19's cross-model comparison meaningless, and Part 7.4 does not say which of the two
    `C9`'s 0.95 threshold is measured on.

    ``auc_frozen`` is the later bank read along the **earlier** direction and ``auc_refit`` is the
    same bank read along the direction refit there. ``auc_decay`` is the difference. `C9` predicts
    the cosine holds above 0.95 while the frozen AUC falls by more than 0.05, and the two published
    sources behind that row point opposite ways, which is the whole reason both units are reported.
    """

    cos_consecutive: float
    cos_raw: float
    auc_frozen: float
    auc_refit: float
    n_items: int
    n_positive: int
    frame: FrameID

    @property
    def auc_decay(self) -> float:
        return self.auc_refit - self.auc_frozen


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        raise NumericsError("a fitted direction has (near) zero norm; there is no cosine to take")
    return float(np.dot(a, b) / (na * nb))


def reading_drift(
    direction_a: Any,
    direction_b: Any,
    frame: Any,
    *,
    bank: Any,
    labels: Any,
    cos_floor: float,
    n_null: int = 0,
) -> ReadingDrift:
    """Part 7.4's three drift quantities for a direction fitted at A and read at B.

    ``direction_a`` is the direction fitted at the earlier checkpoint and ``direction_b`` the one
    refit at the later one. ``bank`` and ``labels`` are the later checkpoint's, since both AUCs are
    read there: the question is what the earlier direction is worth *now*.

    ``cos_floor`` is required and has no default. Part 7.4 asks for "a declared threshold on cosine
    similarity below which the reading refuses rather than reports" and no number is declared
    anywhere in the corpus. `C9`'s 0.95 is a different object: a reading below it refutes `C9`, it
    does not make the reading unreportable, and using one as the other would let a refuted row
    silently become an unread one.

    Raises `DriftRefused` when the canonical cosine is below the floor, **before** either AUC is
    computed. The AUCs both come from `stats.roc.roc_pr`, which is this package's definition of an
    AUC; no second one is written here.
    """
    fid = _require(frame)
    if not np.isfinite(cos_floor):
        raise NumericsError(f"the cosine floor must be finite; got {cos_floor}")

    wa = np.asarray(direction_a, dtype=np.float64).ravel()
    wb = np.asarray(direction_b, dtype=np.float64).ravel()
    if wa.shape != wb.shape:
        raise NumericsError(f"the two directions have shapes {wa.shape} and {wb.shape}")

    cos_raw = _cosine(wa, wb)
    if isinstance(frame, Frame):
        cos_canonical = _cosine(
            np.asarray(canonicalize(wa, frame), dtype=np.float64),
            np.asarray(canonicalize(wb, frame), dtype=np.float64),
        )
    else:
        # Only a frame id was supplied, so the whitening matrix is not available here and the raw
        # cosine is what there is. Recorded as equal rather than silently different, so a caller
        # comparing the two fields can see that no whitening was applied.
        cos_canonical = cos_raw

    if cos_canonical < cos_floor:
        raise DriftRefused(cos_canonical, cos_raw, float(cos_floor))

    from reward_lens.stats.roc import roc_pr

    x = _bank(bank, "bank")
    y = np.asarray(labels).ravel()
    if x.shape[0] != y.size:
        raise NumericsError(f"the bank holds {x.shape[0]} items and {y.size} labels")
    if x.shape[1] != wa.size:
        raise NumericsError(
            f"the bank is {x.shape[1]}-dimensional and the directions are {wa.size}-dimensional"
        )
    positives = int(np.count_nonzero(y))
    if positives == 0 or positives == y.size:
        raise NumericsError(
            "the labels are all one class, so an AUC is undefined and a drift in AUC units is a "
            "statement about a quantity that does not exist on this bank"
        )
    return ReadingDrift(
        cos_consecutive=cos_canonical,
        cos_raw=cos_raw,
        auc_frozen=float(roc_pr(x @ wa, y).auc),
        auc_refit=float(roc_pr(x @ wb, y).auc),
        n_items=int(x.shape[0]),
        n_positive=positives,
        frame=fid,
    )


__all__ = [
    "DriftRefused",
    "ReadingDrift",
    "RepresentationDrift",
    "reading_drift",
    "representation_drift",
]
