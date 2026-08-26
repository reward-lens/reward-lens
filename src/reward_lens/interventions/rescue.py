"""Record what an ablation removed, and put it back (C6, and the mechanics C5 and C7 share).

Interpretability's standard causal claim is a knockout with no rescue: remove X, observe a change,
conclude X mattered. Genetics stopped accepting that decades ago, because a knockout confounds the
loss of X with every downstream consequence of the perturbation itself, and the fix is to restore X
and check that the phenotype comes back. It costs one extra forward pass over the ablation you have
already run, and as far as the field scan behind this catalogue established, essentially nobody in
interpretability does it.

**What is here and what is deliberately not.** `AblationIntervention` already removes a direction and
it is not reimplemented. Two small interventions are added beside it:

`RecordRemoved` is a pure observer. It reads the coordinate `h·u` at a site, stores it, and returns
the activation untouched. Composed *before* an ablation at the same site it captures exactly what
the ablation is about to remove, because `ComposedIntervention` chains hooks at one site in
declaration order.

`Reinject` adds a recorded coordinate back, at the same site or at a later one, along the direction
it came from or along a substitute. That substitute is the control, and it is norm-matched by
construction rather than by arithmetic: re-injecting `c·v` for any unit `v` has the same magnitude
as re-injecting `c·u`, so a random re-injection differs from the real one only in direction.

**Same site or a different one, and why the answer is both.** Restoring at the site the ablation
acted on is close to a no-op and is the sanity check: it should recover the behaviour almost
exactly, and if it does not, something in the apparatus is wrong rather than something in the model.
Restoring at a *later* site is the informative version, because it asks whether the direction is
carrying the behaviour or merely correlated with a pathway that is. Both are supported and the
`RescueSpec` records which was run, because reporting the first as though it were the second is the
way this control becomes decorative.

**What it cannot do.** The coordinate is recorded and replayed within one forward pass, so this
rescues an ablation applied during that pass and cannot rescue a weight edit. And re-injecting at a
later layer puts the coordinate into a residual stream that the intervening layers have already
written to under the ablated condition, so a rescue fraction below 1 confounds "the direction was
not sufficient" with "the intervening computation had already gone somewhere else". That is the
honest limit of a within-pass rescue and it is why the number is a fraction rather than a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from reward_lens.core.types import Site, content_hash, hash_bytes
from reward_lens.interventions.base import CompiledIntervention, MountHook
from reward_lens.interventions.steer import unit_direction

if TYPE_CHECKING:
    import torch

    from reward_lens.signals.base import RewardSignal


class RescueError(RuntimeError):
    """A rescue whose recorded coordinate does not match the site it is being replayed into.

    Raised rather than returned. A shape mismatch between the recording site and the re-injection
    site is a programming error in how the pass was assembled, not an anticipated condition of the
    subject, and turning it into a `Refusal` would hide a wiring bug behind a remedy string.
    """


@dataclass
class Mounted:
    """One site's hook behind the shape `runtime/hooks.py` actually mounts.

    **There are two intervention contracts in this library and they do not meet.**
    `interventions/base.py` defines `Intervention.compile(signal) -> CompiledIntervention`, whose
    `mounts` is a `{Site: hook}` mapping, and that is what the causal algebra composes over.
    `runtime/hooks.py::mounted_interventions` mounts objects exposing `site: Site` and
    `apply(hidden) -> hidden`, one site each, and its own docstring says
    "``interventions.base.CompiledIntervention`` will satisfy it in M6", which it does not yet. So a
    `ComposedIntervention` has no `.site` and cannot be mounted at all: it raises
    `AttributeError: 'CompiledIntervention' object has no attribute 'site'` inside the runtime.

    This adapter is the seam until that is reconciled. It is deliberately tiny and it does not
    change either contract: it takes the `{Site: hook}` a compiled intervention already produces and
    returns one mountable object per site, in a stable order, so an arm built out of the algebra can
    be handed to the shipped runtime. The reconciliation itself is a request in this package's
    report rather than an edit, because both files are outside its path set.
    """

    site: Site
    hook: MountHook
    label: str = ""

    def apply(self, hidden: "torch.Tensor") -> "torch.Tensor":
        return self.hook(hidden, {})


def mountable(intervention: Any, *, signal: Any = None) -> list[Mounted]:
    """Compile an `Intervention` and return one mountable object per site it touches.

    Ordering is by the site's layer then its point, so a recorder at layer 3 and a re-injection at
    layer 7 mount in the order the forward pass reaches them. Within one site the compiled hooks are
    already chained by `ComposedIntervention` in declaration order, so a recorder composed before an
    ablation still sees the activation the ablation is about to change.
    """
    compiled = intervention.compile(signal)
    ordered = sorted(compiled.mounts.items(), key=lambda kv: (kv[0].layer, str(kv[0].point)))
    return [
        Mounted(site=site, hook=hook, label=str(compiled.meta.get("kind", "")))
        for site, hook in ordered
    ]


@dataclass
class RemovedCoordinate:
    """The scalar coordinate an ablation took out, per batch row and position.

    A mutable box shared by the recorder and the re-injector, because they are two hooks in one
    forward pass and the value has to travel between them. One box per rescue: reusing a box across
    two passes replays the first pass's coordinates into the second, which is a silent wrong answer
    rather than a crash, so `clear` is called by the recorder on every entry.
    """

    #: `(batch, positions, 1)`, in the activation's own dtype and device. None before the recorder
    #: has run, which is what `Reinject` refuses on.
    value: Any = None
    site: Site | None = None
    n_calls: int = 0
    label: str = ""

    def clear(self) -> None:
        self.value = None
        self.n_calls = 0

    @property
    def recorded(self) -> bool:
        return self.value is not None

    def magnitude(self) -> float:
        """The mean absolute coordinate, for the record. NaN before anything was recorded."""
        if self.value is None:
            return float("nan")
        return float(self.value.abs().mean().item())


@dataclass
class RecordRemoved:
    """Read `h·u` at a site into a box and pass the activation through unchanged.

    A pure observer with an `Intervention`'s shape, so it mounts through the same hook path as
    everything else and composes with an ablation at the same site rather than needing a second
    capture pass. Its fingerprint is distinct from the ablation's, so a recorded run and a plain
    ablated run do not share a cache key even though they produce identical activations.
    """

    site: Site
    direction: Any
    into: RemovedCoordinate = field(default_factory=RemovedCoordinate)
    id: str = "record_removed"

    def fingerprint(self) -> str:
        return content_hash(
            {
                "kind": "record_removed",
                "site": str(self.site),
                "direction": hash_bytes(unit_direction(self.direction).tobytes(), "dir"),
            },
            "iv",
        )

    def _hook(self) -> MountHook:
        unit = unit_direction(self.direction)
        box = self.into
        site = self.site

        def apply(hidden: "torch.Tensor", _ctx: dict) -> "torch.Tensor":
            import torch

            vec = torch.as_tensor(unit, device=hidden.device, dtype=hidden.dtype)
            coord = (hidden * vec).sum(dim=-1, keepdim=True)
            box.value = coord.detach().clone()
            box.site = site
            box.n_calls += 1
            return hidden

        return apply

    def compile(self, signal: "RewardSignal | None" = None) -> CompiledIntervention:
        del signal  # reading a coordinate is signal-independent; the site is resolved at mount
        self.into.clear()
        return CompiledIntervention(
            fingerprint=self.fingerprint(),
            mounts={self.site: self._hook()},
            meta={"kind": "record_removed", "site": str(self.site)},
        )


@dataclass
class Reinject:
    """Add a recorded coordinate back into the residual, along `u` or along a substitute.

    ``direction`` is the direction the coordinate is replayed along. Passing the direction it was
    recorded from is the rescue; passing any other unit vector is the norm-matched control, and it
    is matched by construction because the magnitude comes from the recorded coordinate rather than
    from the direction.

    ``scale`` exists for the dose sweep rather than for tuning. Re-injecting a fraction of what was
    removed turns a rescue into a dose-response curve on the same axis C4 sweeps, which is how a
    partial rescue becomes a number instead of an anecdote.
    """

    site: Site
    direction: Any
    source: RemovedCoordinate
    scale: float = 1.0
    id: str = "reinject"

    def fingerprint(self) -> str:
        return content_hash(
            {
                "kind": "reinject",
                "site": str(self.site),
                "direction": hash_bytes(unit_direction(self.direction).tobytes(), "dir"),
                "scale": float(self.scale),
                "from": str(self.source.site) if self.source.site else None,
            },
            "iv",
        )

    def _hook(self) -> MountHook:
        unit = unit_direction(self.direction)
        box = self.source
        scale = float(self.scale)
        site = self.site

        def apply(hidden: "torch.Tensor", _ctx: dict) -> "torch.Tensor":
            import torch

            if not box.recorded:
                raise RescueError(
                    f"nothing was recorded before the re-injection at {site}. A `RecordRemoved` "
                    f"has to be mounted at a site the forward pass reaches *before* this one, and "
                    f"the two have to be composed into one intervention so they run in the same "
                    f"pass. Re-injecting a coordinate captured in an earlier pass would replay the "
                    f"wrong items."
                )
            coord = box.value
            if coord.shape[:-1] != hidden.shape[:-1]:
                raise RescueError(
                    f"the coordinate recorded at {box.site} has shape {tuple(coord.shape)} and the "
                    f"activation at {site} has shape {tuple(hidden.shape)}. The two sites see "
                    f"different batch or position axes, so there is no row-for-row correspondence "
                    f"to replay along."
                )
            vec = torch.as_tensor(unit, device=hidden.device, dtype=hidden.dtype)
            return hidden + scale * coord.to(hidden.dtype).to(hidden.device) * vec

        return apply

    def compile(self, signal: "RewardSignal | None" = None) -> CompiledIntervention:
        del signal
        return CompiledIntervention(
            fingerprint=self.fingerprint(),
            mounts={self.site: self._hook()},
            meta={
                "kind": "reinject",
                "site": str(self.site),
                "scale": float(self.scale),
                "from": str(self.source.site) if self.source.site else None,
            },
        )


@dataclass(frozen=True)
class RescueSpec:
    """Which rescue was run: the ablated site, the restoring site, and along what.

    Carried onto the reading because "restored at the same site" and "restored three layers later"
    are different experiments and the first is close to a no-op. A rescue fraction reported without
    saying which one it was is the way this control stops being one.
    """

    ablate_at: Site
    restore_at: Site
    direction_id: str
    substitute_id: str | None = None
    scale: float = 1.0

    @property
    def is_same_site(self) -> bool:
        """The sanity check rather than the informative version."""
        return self.ablate_at == self.restore_at

    @property
    def is_control(self) -> bool:
        return self.substitute_id is not None

    def render(self) -> str:
        where = (
            f"restored at the ablated site {self.ablate_at}"
            if self.is_same_site
            else f"ablated at {self.ablate_at}, restored at {self.restore_at}"
        )
        along = (
            f"along the substitute {self.substitute_id}"
            if self.substitute_id
            else f"along {self.direction_id}"
        )
        scale = "" if self.scale == 1.0 else f" at {self.scale:g}x"
        return f"{where}, {along}{scale}"


def knockout_and_rescue(
    *,
    ablate_at: Site,
    direction: Any,
    restore_at: Site | None = None,
    substitute: Any = None,
    scale: float = 1.0,
    direction_id: str = "direction",
    substitute_id: str | None = None,
) -> tuple[list[Mounted], list[Mounted], RescueSpec]:
    """Build the ablated arm and the rescued arm, each as a list of mountable single-site objects.

    Returns `(ablated, rescued, spec)`. The ablated arm is a plain `AblationIntervention` with a
    recorder in front of it; the rescued arm is that same pair plus a `Reinject`. The recorder is
    shared between them by construction: the rescued arm records and replays inside one forward
    pass, which is the only way the coordinate replayed is the one that was removed.

    ``substitute`` is the norm-matched control direction. Absent, the coordinate is replayed along
    the direction it came from, which is the rescue itself.

    Each arm is a list because that is what the shipped runtime mounts. Pass it straight to
    `subject.with_interventions(*arm)` or to `policy.selection.behaviour_under(..., intervention=arm)`.
    """
    from reward_lens.interventions.ablate import AblationIntervention
    from reward_lens.interventions.base import compose

    restore = restore_at if restore_at is not None else ablate_at
    box = RemovedCoordinate(label=direction_id)
    recorder = RecordRemoved(site=ablate_at, direction=direction, into=box)
    ablation = AblationIntervention(site=ablate_at, direction=direction, mode="directional")
    along = substitute if substitute is not None else direction
    reinject = Reinject(site=restore, direction=along, source=box, scale=scale)

    # Returned as mountable single-site objects rather than as `ComposedIntervention`s, because the
    # shipped runtime mounts `site` + `apply(hidden)` and cannot mount a composed one. See
    # `Mounted` for why the two contracts differ and what the fix is.
    ablated = mountable(compose([recorder, ablation]))
    rescued = mountable(compose([recorder, ablation, reinject]))
    spec = RescueSpec(
        ablate_at=ablate_at,
        restore_at=restore,
        direction_id=direction_id,
        substitute_id=substitute_id if substitute is not None else None,
        scale=float(scale),
    )
    return ablated, rescued, spec


def norm_matched_random(direction: Any, *, seed: int = 0) -> np.ndarray:
    """A random unit direction of the same dimension, for the rescue control.

    Unit rather than norm-matched to the input, because the magnitude replayed comes from the
    recorded coordinate and not from this vector: `c·v` and `c·u` have the same norm for any two
    unit vectors. Making that structural rather than arithmetic is what stops the control being
    accidentally weaker than the thing it controls for.
    """
    u = unit_direction(direction)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(u.shape[0])
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else u


# ---------------------------------------------------------------------------
# The two control families that are drawn from inside a supplied subspace
# ---------------------------------------------------------------------------
#
# `norm_matched_random` above draws over the full ambient dimension, and so did every other
# random-direction generator in this package. Part 9.2 rules that draw out of the control family in
# as many words: "A random direction from the full residual space is nearly orthogonal to everything
# and therefore nearly free, and it is not a control." The ambient draw is still the right object for
# the ambient control family and is left exactly as it was; what was missing was any way to draw
# from inside a subspace somebody supplied.
#
# Two families are needed and they are different controls. The first asks "does any direction of
# this norm inside the identified subspace do what the fitted one does". The second asks the sharper
# question, "does any direction inside the subspace but orthogonal to the fitted one do it", and it
# exists because if the whole subspace is behaviourally relevant then the first control loses by
# construction and a control that cannot fail is decoration.
#
# **The subspace's construction is not decided here and must not be.** The design uses the phrase
# "the identified subspace" eight times and defines it at none of them, and the wider corpus carries
# two constructions that disagree with each other. `basis` is therefore a required argument: this
# module knows how to draw from a subspace and does not know which subspace.
#
# Everything runs in float64. `unit_direction`, which the ambient draw uses, casts to float32, and
# float32 cannot express the tolerances these controls are held to: a complement projection below
# 1e-10 and a norm matched to 1e-12 are both under float32's resolution at these magnitudes.


@dataclass(frozen=True)
class SubspaceDraw:
    """A control family drawn from inside a supplied subspace, with what it was drawn from.

    ``directions`` is ``(n_draws, d_ambient)``, one control per row, each already rescaled to
    ``target_norm``. The whole matrix is returned rather than a summary because Part 9.2 asks for
    the distribution to be reported rather than summarised, and a percentile computed from a mean
    and a spread is not that.

    ``effective_rank`` is the rank of the space the draw actually explores, which is the rank of the
    supplied basis for the subspace-matched family and one less for the target-orthogonal family.
    ``degenerate`` is set when that space has rank one, which is the case where the family collapses
    to a single direction up to sign: twenty draws from it are two point masses, and a percentile
    read off them is a percentile read off two numbers. It is a flag rather than a refusal because
    the draw is well defined; what is not well defined is the statistic somebody computes from it.

    ``target_in_subspace_fraction`` is how much of the supplied direction survived projection into
    the subspace, in norm. It is 1 when the fitted direction lies inside the registered subspace,
    which is the case both controls are written for. Below 1 the target-orthogonal family is
    orthogonal to the *projected* target rather than to the one that was passed, and a reader has to
    know that before reading a percentile.
    """

    directions: np.ndarray
    family: str
    effective_rank: int
    target_norm: float
    seed: int
    degenerate: bool
    target_in_subspace_fraction: float

    @property
    def n_draws(self) -> int:
        return int(self.directions.shape[0])


def _orthonormal_span(basis: Any, *, d_expected: int) -> np.ndarray:
    """An orthonormal basis for the column space of `basis`, with its numerical rank taken.

    `geometry.subspace._orthonormalize` is the package's definition of this and is used rather than
    a second QR written here, per the one-canonical-definition rule. It is imported at call time so
    that `interventions` does not acquire an import-time dependency on the `white-box` extra, and
    deliberately without a `try`/`except`: `ExtraRequiredError` subclasses `ImportError`, so
    catching `ImportError` here would swallow the message that names the missing extra.

    The QR result is then trimmed to the numerical rank, because a caller's construction is a span
    and a span given by four vectors may have rank three. Keeping a null column would put draws
    outside the subspace at the level of the QR's own rounding, which is the one error this whole
    function exists to make impossible.
    """
    from reward_lens.geometry.subspace import _orthonormalize

    b = np.asarray(basis, dtype=np.float64)
    if b.ndim == 1:
        b = b[:, None]
    if b.ndim != 2:
        raise RescueError(f"a subspace basis is a 2-D array of columns; got shape {b.shape}")
    if b.shape[0] != d_expected:
        raise RescueError(
            f"the basis spans dimension {b.shape[0]} and the direction has dimension "
            f"{d_expected}; a draw from a subspace of a different space is not a control"
        )
    rank = int(np.linalg.matrix_rank(b, tol=1e-10))
    if rank == 0:
        raise RescueError("the supplied basis has rank zero; there is no subspace to draw from")
    q = _orthonormalize(b)
    if q.shape[1] != rank:
        # Rank-deficient generator: take the span from an SVD, which orders by singular value and
        # lets the deficient directions be dropped rather than kept at rounding level.
        u, s, _ = np.linalg.svd(b, full_matrices=False)
        q = u[:, :rank]
    return np.ascontiguousarray(q, dtype=np.float64)


def _draw_in_coordinates(rank: int, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """`n_draws` directions uniform on the unit sphere of a `rank`-dimensional space.

    Gaussian then normalise, which is the same construction `stats.nulls._random_orthonormal_basis`
    relies on for its Haar property and is uniform for the same reason: a spherical Gaussian has no
    preferred direction, so dividing out the length leaves the uniform law on the sphere.

    Drawn in the subspace's *own* coordinates and mapped out through an orthonormal basis, not as
    `basis @ standard_normal(k)`. The second is the obvious way to write this and it is wrong in a
    way that hides: it inherits the generator's conditioning, so a construction whose columns are
    badly scaled concentrates the control family along the generator's dominant direction and the
    control becomes easier to beat than it looks. Because an orthonormal basis is an isometry, the
    uniform law in coordinates is the uniform law in the subspace.
    """
    g = rng.standard_normal((n_draws, rank))
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    # A Gaussian draw hits the origin with probability zero, but a redraw is cheaper than a
    # documented impossibility that turns into a NaN once every few billion draws.
    while np.any(norms < 1e-300):
        bad = (norms < 1e-300).ravel()
        g[bad] = rng.standard_normal((int(bad.sum()), rank))
        norms = np.linalg.norm(g, axis=1, keepdims=True)
    return g / norms


def _target_norm(direction: np.ndarray, norm: float | None) -> float:
    if norm is not None:
        if not np.isfinite(norm) or norm <= 0:
            raise RescueError(f"an explicit control norm must be positive and finite; got {norm}")
        return float(norm)
    value = float(np.linalg.norm(direction))
    if value < 1e-12:
        raise RescueError(
            "the direction has (near) zero norm, so there is no norm to match and no orientation "
            "to be orthogonal to"
        )
    return value


def _as_direction(direction: Any) -> np.ndarray:
    d = np.asarray(direction, dtype=np.float64).reshape(-1)
    if d.size == 0:
        raise RescueError("the direction is empty")
    if not np.all(np.isfinite(d)):
        raise RescueError("the direction carries non-finite entries")
    return d


def subspace_matched_random(
    direction: Any,
    basis: Any,
    *,
    n_draws: int = 1,
    seed: int = 0,
    norm: float | None = None,
) -> SubspaceDraw:
    """`n_draws` norm-matched random directions drawn from inside the span of `basis`.

    This is `C15`'s control family and Part 9.2 calls it the control that carries link 4's whole
    argument. Each draw is uniform on the sphere of the subspace and rescaled to `norm`, which
    defaults to the supplied direction's own norm; Part 9.2 rescales to the intervention's norm, so
    pass it explicitly when the intervention's norm is not the fitted direction's.

    `basis` is the construction of the identified subspace and it is required, not defaulted. The
    design does not define it and two constructions in the corpus disagree, so a default here would
    be this module choosing the rank that decides `C15`'s power. Columns need not be orthonormal or
    independent; the span is taken and its numerical rank is used.

    Rank one is allowed and flagged. A line is a perfectly good subspace to draw from, and twenty
    draws from it are twenty rescaled copies of one direction up to sign, so `degenerate` is set and
    the caller decides whether a percentile over that means anything.
    """
    u = _as_direction(direction)
    q = _orthonormal_span(basis, d_expected=u.size)
    rank = int(q.shape[1])
    wanted = _target_norm(u, norm)
    n = int(n_draws)
    if n < 1:
        raise RescueError(f"a control family needs at least one draw; got {n_draws}")
    rng = np.random.default_rng(seed)
    coordinates = _draw_in_coordinates(rank, n, rng)
    directions = (coordinates @ q.T) * wanted
    inside = q @ (q.T @ u)
    return SubspaceDraw(
        directions=directions,
        family="subspace_matched",
        effective_rank=rank,
        target_norm=wanted,
        seed=int(seed),
        degenerate=rank < 2,
        target_in_subspace_fraction=float(np.linalg.norm(inside) / np.linalg.norm(u)),
    )


def target_orthogonal_random(
    direction: Any,
    basis: Any,
    *,
    n_draws: int = 1,
    seed: int = 0,
    norm: float | None = None,
) -> SubspaceDraw:
    """`n_draws` norm-matched directions inside `basis`'s span and orthogonal to `direction`.

    This is `C16`'s control family, and the design's note on it is that a search of the literature
    found no published instance of it. It is the sharper of the two controls because it is the one
    with a live failure mode: if the whole identified subspace is behaviourally relevant then draws
    from it are also causal, and only the orthogonal ones can still lose.

    Both halves matter and dropping either gives a different control. Orthogonal to the target over
    the full ambient space is the nearly-free draw Part 9.2 rules out. Inside the subspace without
    the orthogonality is the other family, `subspace_matched_random`.

    **Raises at rank one.** The set of directions inside a line and orthogonal to a direction in it
    is empty, and there is no honest vector to return: a zero vector, or a fallback to an ambient
    draw, would hand `C16` twenty controls that are not the control it registered, and nothing
    downstream could tell. At rank two the set is a single direction up to sign, which is drawable
    and is flagged as `degenerate` rather than raised, because the draw is well defined even though
    a percentile over it is not.

    When `direction` does not lie inside the span, the orthogonality is to its projection, and
    `target_in_subspace_fraction` on the result says how much of it survived.
    """
    u = _as_direction(direction)
    q = _orthonormal_span(basis, d_expected=u.size)
    rank = int(q.shape[1])
    wanted = _target_norm(u, norm)
    n = int(n_draws)
    if n < 1:
        raise RescueError(f"a control family needs at least one draw; got {n_draws}")
    if rank < 2:
        raise RescueError(
            f"the supplied subspace has rank {rank}, and the set of directions inside a rank-1 "
            f"subspace orthogonal to a direction in it is empty. There is no target-orthogonal "
            f"control at this rank; register a construction of rank 2 or more, or report that this "
            f"control is unavailable rather than substituting an ambient draw for it"
        )

    coefficients = q.T @ u
    c_norm = float(np.linalg.norm(coefficients))
    if c_norm < 1e-12:
        raise RescueError(
            "the direction has no component inside the supplied subspace, so 'orthogonal to the "
            "target, inside the subspace' is the whole subspace and this is not the control it "
            "claims to be. Check the construction the basis came from"
        )
    c_hat = coefficients / c_norm

    rng = np.random.default_rng(seed)
    coords = _draw_in_coordinates(rank, n, rng)
    # Remove the target's component in the subspace's own coordinates, then renormalise. Drawing in
    # the full subspace and projecting out is uniform on the orthogonal sphere: the Gaussian is
    # isotropic, so its projection onto any hyperplane through the origin is an isotropic Gaussian
    # on that hyperplane.
    coords = coords - np.outer(coords @ c_hat, c_hat)
    residual = np.linalg.norm(coords, axis=1, keepdims=True)
    while np.any(residual < 1e-12):
        bad = (residual < 1e-12).ravel()
        fresh = _draw_in_coordinates(rank, int(bad.sum()), rng)
        coords[bad] = fresh - np.outer(fresh @ c_hat, c_hat)
        residual = np.linalg.norm(coords, axis=1, keepdims=True)
    coords = coords / residual
    # One more removal after the renormalisation, so the orthogonality holds to float64 rather than
    # to whatever the division left behind. This is what buys the 1e-10 the closure proof asks for.
    coords = coords - np.outer(coords @ c_hat, c_hat)
    coords = coords / np.linalg.norm(coords, axis=1, keepdims=True)

    directions = (coords @ q.T) * wanted
    inside = q @ coefficients
    return SubspaceDraw(
        directions=directions,
        family="target_orthogonal",
        effective_rank=rank - 1,
        target_norm=wanted,
        seed=int(seed),
        degenerate=(rank - 1) < 2,
        target_in_subspace_fraction=float(np.linalg.norm(inside) / np.linalg.norm(u)),
    )


__all__ = [
    "Mounted",
    "RecordRemoved",
    "Reinject",
    "RemovedCoordinate",
    "RescueError",
    "RescueSpec",
    "SubspaceDraw",
    "knockout_and_rescue",
    "subspace_matched_random",
    "target_orthogonal_random",
    "mountable",
    "norm_matched_random",
]
