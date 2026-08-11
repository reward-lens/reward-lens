"""Subspace comparison: CKA, Procrustes, and null-anchored feature alignment.

The measurement layer for universality and transfer (DESIGN section 2.7.3). Every number here is an
alignment between two signals' representations, and every alignment is COVARIANT: it is meaningless
without a shared `Frame`, because a raw-coordinate overlap conflates a coordinate change with a
functional one (the E19 lesson). So each public function takes a ``frame`` argument with no default
and calls ``require_frame_for_comparison``; passing ``frame=None`` raises (gate 2, I3). The caller is
responsible for having read both representations at the frame's site and, where directions are
compared, for letting the frame whiten them into the shared canonical coordinate system.

Every alignment is also read against a null: CKA and Procrustes against the identity/rotation
baselines they define, and the subspace and feature alignments against ``stats.nulls`` (random
direction and RUM identifiability, A14). An overlap is only reported as real when it beats the
alignment two unrelated representations would show by chance.

Pure numpy/scipy; the null modules are imported lazily and degrade gracefully if a specific
function is momentarily unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from reward_lens.core.errors import NumericsError
from reward_lens.core.gates import require_frame_for_comparison
from reward_lens.core.types import FrameID, GaugeStatus
from reward_lens.geometry.canonical import canonicalize
from reward_lens.geometry.frame import Frame


def _frame_id(frame: Any) -> FrameID | None:
    """Extract the FrameID from a `Frame` or accept a bare FrameID string (or None)."""
    if frame is None:
        return None
    return frame.id if isinstance(frame, Frame) else frame


def _require(frame: Any) -> FrameID:
    """Enforce gate 2 and return the frame id, so every alignment records its shared frame."""
    fid = _frame_id(frame)
    require_frame_for_comparison(GaugeStatus.COVARIANT, fid)
    return fid  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# CKA
# ---------------------------------------------------------------------------


def cka(x: Any, y: Any, frame: Any) -> float:
    """Linear centered kernel alignment between two representation matrices (DESIGN section 2.7.3).

    ``x`` and ``y`` are ``n x d_x`` and ``n x d_y`` activation matrices for the same ``n`` items,
    read in the shared ``frame``. Linear CKA is ``||Y_c^T X_c||_F^2 / (||X_c^T X_c||_F ||Y_c^T
    Y_c||_F)`` on column-centered matrices, in ``[0, 1]``: 1 for representations identical up to an
    orthogonal transform and rescaling, ~0 for independent high-dimensional ones. Requires a shared
    frame (gate 2).
    """
    _require(frame)
    xc = np.asarray(x, dtype=np.float64)
    yc = np.asarray(y, dtype=np.float64)
    if xc.shape[0] != yc.shape[0]:
        raise NumericsError(f"CKA needs matched rows; got {xc.shape[0]} and {yc.shape[0]}")
    xc = xc - xc.mean(axis=0)
    yc = yc - yc.mean(axis=0)
    hsic = float(np.sum((yc.T @ xc) ** 2))
    norm_x = float(np.linalg.norm(xc.T @ xc))
    norm_y = float(np.linalg.norm(yc.T @ yc))
    if norm_x == 0 or norm_y == 0:
        return float("nan")
    return hsic / (norm_x * norm_y)


# ---------------------------------------------------------------------------
# Orthogonal Procrustes
# ---------------------------------------------------------------------------


@dataclass
class ProcrustesResult:
    """The result of an orthogonal Procrustes alignment (DESIGN section 2.7.3).

    ``rotation`` is the ``d x d`` orthogonal ``R`` minimizing ``||X R - Y||_F``; ``disparity`` is the
    residual ``||X R - Y||_F^2 / ||Y||_F^2`` (0 when ``Y`` is an exact rotation of ``X``); ``scale``
    is the optimal isotropic scale had one been fit. A shared frame is required (gate 2).

    ``support`` is the dimension the two inputs' rows jointly span, which is the dimension in which
    the rotation is determined at all. When ``support < d`` the rest of ``rotation`` is the identity,
    for the reason `orthogonal_fit_on_span` gives.
    """

    rotation: np.ndarray
    disparity: float
    scale: float
    support: int = -1


@dataclass(frozen=True)
class SpanFit:
    """An orthogonal Procrustes fit expressed in the subspace the data actually constrains.

    ``basis`` is ``d x r`` with orthonormal columns spanning the two inputs' joint row space, and
    ``rotation`` is the ``r x r`` orthogonal fit in that basis, in float64. ``disparity`` and
    ``scale`` are the same numbers `ProcrustesResult` carries, which are determined whatever ``r``
    is. ``support`` is ``r``.
    """

    basis: np.ndarray
    rotation: np.ndarray
    disparity: float
    scale: float
    support: int


def _numeric_rank(s: np.ndarray, n_rows: int, n_cols: int) -> int:
    """Numerical rank from singular values, at numpy's own scale-relative cut.

    Relative to the largest singular value on purpose. An absolute cut such as ``1e-10`` makes the
    reported rank depend on the units the activations happen to be in, which is how a bank scaled by
    a thousand acquires full rank and a bank scaled by a thousandth loses all of it.
    """
    if s.size == 0:
        return 0
    cut = float(max(n_rows, n_cols)) * float(np.finfo(np.float64).eps) * float(s[0])
    return int(np.count_nonzero(s > cut))


def _minimal_maximiser(m: np.ndarray) -> np.ndarray:
    """``argmax`` over orthogonal ``R`` of ``tr(R^T m)``, taking the representative nearest ``I``.

    With ``m = U S V^T`` of rank ``p``, **every** ``R = U diag(I_p, Z) V^T`` with ``Z`` orthogonal
    attains the same maximum. The directions in that free block are exactly the ones the data does
    not constrain, and `numpy.linalg.svd` fills them with whatever its factorisation happened to
    produce, which is why a rank-deficient fit returns a rotation full of reflections. Choosing the
    representative closest to the identity is the only choice that does not report a rotation nobody
    measured; every other choice invents one.
    """
    u, s, vt = np.linalg.svd(m)
    r = int(m.shape[0])
    p = _numeric_rank(s, r, r)
    if p >= r:
        return u @ vt
    d = np.eye(r)
    if p < r:
        block = (vt @ u)[p:, p:]
        ub, _, vtb = np.linalg.svd(block)
        d[p:, p:] = vtb.T @ ub.T
    return u @ d @ vt


def orthogonal_fit_on_span(x: Any, y: Any, frame: Any) -> SpanFit:
    """Orthogonal Procrustes, solved in the subspace the two inputs' rows actually span.

    ``X`` and ``Y`` are ``(n, d)``. The objective ``||X R - Y||_F`` only ever sees ``R`` through
    ``X^T Y``, whose rank is at most ``n``. So when ``n < d`` most of a ``d x d`` rotation is
    unconstrained by the data, and solving in the ambient width returns a matrix whose determined
    part is correct and whose remaining ``d - n`` directions are an artefact of the SVD. Read as
    rotation angles that artefact is not small: at ``256 x 2560``, the geometry this project's own
    run uses, a bank compared **against itself** returns a largest principal angle of ``pi``
    (BLK-019-a).

    This solves the same problem in an orthonormal basis of the joint row space, dimension ``r``,
    and leaves the fit as the identity on the orthogonal complement. The residual and the scale are
    unchanged, because both are determined; the rotation becomes the one the data supports.

    When ``n >= d`` and the inputs have full column rank this is the classical solution, exactly.
    """
    _require(frame)
    xm = np.asarray(x, dtype=np.float64)
    ym = np.asarray(y, dtype=np.float64)
    if xm.ndim != 2 or ym.ndim != 2:
        raise NumericsError(f"Procrustes needs two matrices; got {xm.ndim}d and {ym.ndim}d")
    if xm.shape != ym.shape:
        raise NumericsError(f"Procrustes needs equal shapes; got {xm.shape} and {ym.shape}")

    stacked = np.vstack((xm, ym))
    _, sv, vh = np.linalg.svd(stacked, full_matrices=False)
    r = _numeric_rank(sv, stacked.shape[0], stacked.shape[1])
    basis = np.ascontiguousarray(vh[:r].T)  # d x r

    a = xm @ basis
    b = ym @ basis
    rotation = _minimal_maximiser(a.T @ b)

    denom = float(np.sum(ym**2))
    disparity = float(np.sum((a @ rotation - b) ** 2) / denom) if denom > 0 else float("nan")
    norm_x = float(np.sum(xm**2))
    # The nonzero singular values of `X^T Y` and of `A^T B` coincide, because `X = A basis^T` and
    # `Y = B basis^T` exactly, so the scale is the same number the ambient solve reported.
    s_reduced = np.linalg.svd(a.T @ b, compute_uv=False)
    scale = float(np.sum(s_reduced) / norm_x) if norm_x > 0 else float("nan")
    return SpanFit(basis=basis, rotation=rotation, disparity=disparity, scale=scale, support=int(r))


def procrustes(x: Any, y: Any, frame: Any) -> ProcrustesResult:
    """Orthogonal Procrustes alignment of ``X`` onto ``Y`` in a shared frame (DESIGN section 2.7.3).

    Solves ``R = argmin_{R^T R = I} ||X R - Y||_F`` and reports the residual disparity. Recovering a
    known rotation to numerical zero disparity is the property test. Requires a shared frame
    (gate 2).

    The solve runs in the subspace the rows span (`orthogonal_fit_on_span`) and the returned ``R`` is
    the identity outside it. For ``n >= d`` that is the classical answer unchanged; for ``n < d`` it
    is the difference between a rotation angle and an artefact (BLK-019-a).
    """
    fit = orthogonal_fit_on_span(x, y, frame)
    d = int(np.asarray(x, dtype=np.float64).shape[1])
    k = fit.basis.shape[1]
    rotation = np.eye(d) + fit.basis @ (fit.rotation - np.eye(k)) @ fit.basis.T
    return ProcrustesResult(
        rotation=rotation.astype(np.float32),
        disparity=fit.disparity,
        scale=fit.scale,
        support=fit.support,
    )


# ---------------------------------------------------------------------------
# Subspace and feature alignment against nulls
# ---------------------------------------------------------------------------


@dataclass
class AlignmentResult:
    """An alignment score read against its null (DESIGN section 2.7.3, Appendix A14).

    ``alignment`` is the measured overlap; ``null_mean`` and ``null_p95`` summarize the chance
    distribution; ``p_value`` is the fraction of null draws at least as aligned; ``excess`` is
    ``alignment - null_mean``, the convergence beyond chance the VCE index reports. ``null`` names the
    baseline used.
    """

    alignment: float
    null_mean: float
    null_p95: float
    p_value: float
    excess: float
    null: str


def _orthonormalize(basis: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis for the column space of ``basis`` (QR)."""
    q, _ = np.linalg.qr(basis)
    return q[:, : basis.shape[1]]


def subspace_alignment(
    a_basis: Any,
    b_basis: Any,
    frame: Any,
    *,
    d_ambient: int | None = None,
    n_null: int = 1000,
    seed: int = 0,
) -> AlignmentResult:
    """Projection overlap between two subspaces, against the RUM-identifiability null (A14).

    ``a_basis`` and ``b_basis`` are ``d x k`` bases (columns span each subspace) in the shared frame.
    The alignment is the mean squared cosine of the principal angles, ``(1/k) ||Q_A^T Q_B||_F^2`` in
    ``[0, 1]``, the same statistic ``stats.nulls.rum_identifiability_null`` reports for two random
    ``k``-subspaces. Alignment above that null is convergence beyond what identifiability freedom
    alone forces, which is exactly the value-convergence-excess construction. Requires a shared frame
    (gate 2).
    """
    _require(frame)
    qa = _orthonormalize(np.asarray(a_basis, dtype=np.float64))
    qb = _orthonormalize(np.asarray(b_basis, dtype=np.float64))
    if qa.shape[0] != qb.shape[0]:
        raise NumericsError(f"subspaces need equal ambient dim; got {qa.shape[0]}, {qb.shape[0]}")
    d = int(d_ambient or qa.shape[0])
    k = qa.shape[1]
    cos_angles = np.linalg.svd(qa.T @ qb, compute_uv=False)
    cos_angles = np.clip(cos_angles, 0.0, 1.0)
    alignment = float(np.mean(cos_angles**2))

    null_mean = float("nan")
    null_p95 = float("nan")
    p_value = float("nan")
    try:
        from reward_lens.stats.nulls import rum_identifiability_null

        null = rum_identifiability_null(d=d, k=k, n=n_null, seed=seed)
        null_mean = float(np.mean(null))
        null_p95 = float(np.quantile(null, 0.95))
        p_value = float((np.sum(null >= alignment) + 1) / (null.size + 1))
    except Exception:
        pass

    return AlignmentResult(
        alignment=alignment,
        null_mean=null_mean,
        null_p95=null_p95,
        p_value=p_value,
        excess=alignment - null_mean if np.isfinite(null_mean) else float("nan"),
        null="rum_identifiability",
    )


def hungarian_feature_alignment(
    a_dirs: Any,
    b_dirs: Any,
    frame: Frame,
    *,
    weights_a: Any | None = None,
    weights_b: Any | None = None,
    top_frac: float = 1.0,
    n_null: int = 10_000,
    seed: int = 0,
) -> AlignmentResult:
    """Hungarian-matched feature alignment on high-reward-loading directions (DESIGN section 2.7.3).

    ``a_dirs`` and ``b_dirs`` are ``(m_a, d)`` and ``(m_b, d)`` sets of feature/decoder directions.
    When per-direction reward-loading ``weights`` (``|d r / d feature|``) are supplied, the match is
    restricted to the top ``top_frac`` of each set, so the alignment measures whether the directions
    that actually price reward line up. Directions are canonicalized in the shared ``frame`` before
    matching, then the Hungarian algorithm finds the maximum-similarity assignment, and the mean
    matched canonical cosine is read against the random-direction null (``stats.nulls``). Requires a
    full `Frame` (canonicalization needs its whitening); gate 2 is enforced.
    """
    _require(frame)
    if not isinstance(frame, Frame):
        raise NumericsError(
            "feature alignment needs a full Frame (canonicalization uses whitening)"
        )
    a = np.asarray(a_dirs, dtype=np.float64)
    b = np.asarray(b_dirs, dtype=np.float64)
    a = _top_by_weight(a, weights_a, top_frac)
    b = _top_by_weight(b, weights_b, top_frac)

    ca = np.stack([canonicalize(row, frame) for row in a]).astype(np.float64)
    cb = np.stack([canonicalize(row, frame) for row in b]).astype(np.float64)
    sim = ca @ cb.T  # canonical cosines (rows already unit-norm)
    row_ind, col_ind = linear_sum_assignment(-sim)  # maximize total matched similarity
    matched = sim[row_ind, col_ind]
    alignment = float(np.mean(np.abs(matched)))

    null_mean = float("nan")
    null_p95 = float("nan")
    p_value = float("nan")
    try:
        from reward_lens.stats.nulls import random_direction_null

        res = random_direction_null(observed_cos=alignment, d=frame.d, n=n_null, seed=seed)
        null_mean = float(res["null_mean"])
        null_p95 = float(res["null_p95"])
        p_value = float(res["p_value"])
    except Exception:
        pass

    return AlignmentResult(
        alignment=alignment,
        null_mean=null_mean,
        null_p95=null_p95,
        p_value=p_value,
        excess=alignment - null_mean if np.isfinite(null_mean) else float("nan"),
        null="random_direction",
    )


def _top_by_weight(dirs: np.ndarray, weights: Any | None, top_frac: float) -> np.ndarray:
    """Keep the top ``top_frac`` directions by reward-loading weight (all of them when no weights)."""
    if weights is None or top_frac >= 1.0:
        return dirs
    w = np.abs(np.asarray(weights, dtype=np.float64).ravel())
    keep = max(1, int(round(top_frac * dirs.shape[0])))
    idx = np.argsort(w)[::-1][:keep]
    return dirs[idx]


__all__ = [
    "cka",
    "orthogonal_fit_on_span",
    "procrustes",
    "ProcrustesResult",
    "SpanFit",
    "subspace_alignment",
    "hungarian_feature_alignment",
    "AlignmentResult",
]
