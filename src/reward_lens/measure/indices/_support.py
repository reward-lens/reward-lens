"""Shared plumbing for the index library (DESIGN section 2.8.3, Appendix A).

The indices are the vocabulary of the cards and the scoreboard, so they all reach for the same few
operations: read a reward direction ``w_r`` off a signal, read final-token activations at a site,
and turn activations into named feature values through a feature bank. Those live here so each index
module is just its own definition, and so every index reads the substrate the same way.

The feature bank is the one interface the corpus's concept layer will implement. It is deliberately
tiny: a bank names a set of properties and turns an ``(n, d)`` activation matrix into an ``(n, k)``
matrix of feature values, optionally exposing the ``(k, d)`` decoder directions. ``concepts`` (built
concurrently) will provide production banks; ``LinearFeatureBank`` is the synthetic bank of known
directions that makes an index like ``chi`` provable without waiting for it. Indices lazy-import
concepts and degrade gracefully when a bank is absent, so importing this module pulls no torch and no
concept machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from reward_lens.core.types import Site

if TYPE_CHECKING:
    from reward_lens.signals.base import Readout, RewardSignal


# ---------------------------------------------------------------------------
# Reading the substrate: w_r and activations
# ---------------------------------------------------------------------------


def find_readout(signal: "RewardSignal", name: str = "reward") -> "Readout":
    """Look up a readout by name through the frozen ``RewardSignal`` protocol surface.

    Uses ``readouts()`` (the protocol method every adapter implements) rather than a signal-specific
    ``readout`` accessor, so an index runs against any substrate. Falls back to the first readout
    when the name is absent and there is only one, which is the common single-head case.
    """
    readouts = list(signal.readouts())
    for r in readouts:
        if r.name == name:
            return r
    accessor = getattr(signal, "readout", None)
    if callable(accessor):
        try:
            return accessor(name)
        except Exception:  # noqa: BLE001 - fall through to the single-readout convenience
            pass
    if len(readouts) == 1:
        return readouts[0]
    raise KeyError(f"unknown readout {name!r}; available: {[r.name for r in readouts]}")


def reward_vector(signal: "RewardSignal", readout: str = "reward") -> np.ndarray:
    """The reward direction ``w_r`` for a readout, as a float64 numpy vector.

    ``readouts()[0].vector`` is the head weight (a torch tensor for linear readouts); this coerces it
    to a detached fp64 numpy array so the pure index math never touches torch. Non-linear readouts
    (simplex, token-value) carry no vector and raise, which is the honest failure for an index that
    needs a linear reward direction.
    """
    read = find_readout(signal, readout)
    vec = read.vector
    if vec is None:
        raise ValueError(
            f"readout {readout!r} is a {read.kind!r} readout with no reward vector; "
            "the linear index math needs a linear or logit_diff readout"
        )
    arr = np.asarray(_to_numpy(vec), dtype=np.float64).ravel()
    return arr


def readout_site(signal: "RewardSignal", readout: str = "reward") -> Site:
    """The site the readout reads at (the final residual for a classifier head)."""
    return find_readout(signal, readout).site


def final_activations(
    signal: "RewardSignal",
    view: Any,
    site: Site | None = None,
    *,
    readout: str = "reward",
) -> np.ndarray:
    """Capture final-token activations at a site for every item, as an ``(n, d)`` float64 matrix.

    Captures in fp32 (frames and covariances refuse fp16) at the resolved final position, then coerces
    to fp64 numpy. When ``site`` is None the readout's own site is used, which is where ``w_r`` acts, so
    ``activations @ w_r`` reproduces the signal's score up to the head bias. This is the production
    path; the index math is tested directly on synthetic activation matrices.
    """
    from reward_lens.runtime.backend import CaptureSpec
    from reward_lens.signals.base import PositionSpec

    if site is None:
        site = readout_site(signal, readout)
    spec = CaptureSpec(
        sites=(site,),
        position=PositionSpec("final"),
        full_sequence=False,
        dtype="float32",
    )
    capture = next(iter(signal.capture(view, spec)))
    tensor = capture.tensors[site]
    return np.asarray(_to_numpy(tensor), dtype=np.float64)


def reward_scores(signal: "RewardSignal", view: Any, readout: str = "reward") -> np.ndarray:
    """The per-item reward scores under a readout, as a float64 vector.

    Thin wrapper over ``signal.score`` that unwraps the ``Evidence[Scores]`` to the raw values. Base
    policy samples fed here give the ``r`` that ``chi`` and ``tail`` are functionals of.
    """
    evidence = signal.score(view, readout)
    return np.asarray(_to_numpy(evidence.value.values), dtype=np.float64).ravel()


def _to_numpy(x: Any) -> np.ndarray:
    """Coerce a torch tensor or array-like to a detached CPU numpy array without importing torch."""
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().to("cpu").numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# The feature bank interface (the concept layer's contract)
# ---------------------------------------------------------------------------


@runtime_checkable
class FeatureBank(Protocol):
    """The minimal contract a concept-feature bank satisfies for the indices (DESIGN section 2.5).

    ``names`` labels the ``k`` features; ``featurize`` turns an ``(n, d)`` activation matrix into an
    ``(n, k)`` matrix of feature values; ``directions`` optionally exposes the ``(k, d)`` decoder
    directions (None when the bank is not linear). The concept subsystem provides production banks
    (SAE features, probes, difference dictionaries); ``LinearFeatureBank`` is the synthetic bank that
    makes the susceptibility and knowledge-utilization indices provable on planted structure.
    """

    names: tuple[str, ...]

    def featurize(self, activations: np.ndarray) -> np.ndarray: ...

    def directions(self) -> np.ndarray | None: ...


@dataclass
class LinearFeatureBank:
    """A synthetic feature bank of known linear directions (the test-time and default bank).

    Features are linear readouts ``f = activations @ D^T`` for decoder directions ``D`` (``k, d``).
    This is exactly the object an index needs to recover a planted ``Cov(feature, reward)``: pick the
    directions, plant a reward that loads on one of them, and ``chi`` must light up that feature. A
    production concept bank implements the same ``FeatureBank`` protocol with learned features.
    """

    directions_: np.ndarray  # (k, d)
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.directions_ = np.asarray(self.directions_, dtype=np.float64)
        if self.directions_.ndim != 2:
            raise ValueError(f"directions must be (k, d); got shape {self.directions_.shape}")
        if not self.names:
            self.names = tuple(f"f{i}" for i in range(self.directions_.shape[0]))
        if len(self.names) != self.directions_.shape[0]:
            raise ValueError(
                f"names has {len(self.names)} entries but there are {self.directions_.shape[0]} "
                "directions"
            )

    def featurize(self, activations: np.ndarray) -> np.ndarray:
        a = np.asarray(activations, dtype=np.float64)
        return a @ self.directions_.T

    def directions(self) -> np.ndarray | None:
        return self.directions_


def load_default_bank(signal: "RewardSignal") -> FeatureBank | None:
    """Try to obtain a production feature bank from the concept layer, else None (graceful degrade).

    The concept subsystem is built concurrently, so this lazy-imports it and returns None on any
    failure. An index that gets None falls back to an injected bank or reports that no feature bank
    was available, rather than fabricating features. This is the seam that keeps the index library
    provable now and upgradeable later without a code change here.
    """
    try:  # pragma: no cover - exercised only once concepts lands
        import reward_lens.concepts as concepts  # noqa: F401

        factory = getattr(concepts, "default_feature_bank", None)
        if callable(factory):
            bank = factory(signal)
            if isinstance(bank, FeatureBank):
                return bank
    except Exception:  # noqa: BLE001 - concepts absent or incompatible: degrade to None
        return None
    return None


def percentile_within_battery(values: np.ndarray) -> np.ndarray:
    """Map a battery of raw values to their percentile ranks in ``[0, 1]`` (average-rank, ties shared).

    This is the standardization that fixes the KUI unit bug: decodability and mediation live on
    incommensurable raw scales, so both are pushed to their rank-within-battery before they are ever
    combined. With ``m`` values the ``i``-th smallest gets ``(rank + 0.5) / m`` so the percentiles are
    symmetric in ``(0, 1)`` and a singleton battery maps to ``0.5`` rather than an undefined ``0/0``.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    m = v.size
    if m == 0:
        return v
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(m, dtype=np.float64)
    ranks[order] = np.arange(m, dtype=np.float64)
    # average tied ranks so equal raw values share a percentile
    _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    avg_rank = starts[inv] + (counts[inv] - 1) / 2.0
    return (avg_rank + 0.5) / m


__all__ = [
    "find_readout",
    "reward_vector",
    "readout_site",
    "final_activations",
    "reward_scores",
    "FeatureBank",
    "LinearFeatureBank",
    "load_default_bank",
    "percentile_within_battery",
]
