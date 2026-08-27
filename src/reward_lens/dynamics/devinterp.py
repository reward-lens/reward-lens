"""Optional bridge to LLC / essential-dynamics tooling for phase-transition detection (DESIGN 2.12).

Developmental interpretability reads training as a sequence of phase transitions and measures them
with the local learning coefficient (LLC, a scalable estimate of the RLCT / effective dimension) and
essential-dynamics analyses (the low-rank structure of how internals move over training). Those live
in the external ``devinterp`` package, which is not a dependency of this library and is not installed
in this environment.

What this module carries is the half of the bridge that can be honest without the package:
`is_available` reports whether ``devinterp`` is importable, and `LLCTrajectory` is the typed record
an estimator's output lands in, defined unconditionally so a caller can name the return type without
the package present. The adapters that would drive the estimators are not here. They need
per-checkpoint gradient sampling over a data loader, a GPU-scale computation on the real RM-Pythia
run (DESIGN 4.5), and nothing stands in their place: a caller that wants an LLC curve drives
``devinterp`` itself and builds an `LLCTrajectory` from what comes back. The stabilization detector
in `curves.py` is the CPU-provable, dependency-free developmental signal this library ships on its
own; this module is the seam to the specialist tooling, not a copy of it.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any

_PACKAGE = "devinterp"


def is_available() -> bool:
    """Whether the external ``devinterp`` package is importable in this environment."""
    return importlib.util.find_spec(_PACKAGE) is not None


@dataclass
class LLCTrajectory:
    """A local-learning-coefficient trajectory over training (DESIGN 2.12). Populated only via ``devinterp``.

    ``steps`` is the training-time covariate and ``llc`` the estimated local learning coefficient at
    each checkpoint (a scalable RLCT estimate; a jump marks a phase transition). ``estimator`` records
    which ``devinterp`` estimator produced it and ``meta`` its configuration, so a trajectory carries
    its own provenance. This dataclass is defined even without the package so the return type is
    importable; nothing in this library fills it, because no estimator here runs.
    """

    steps: list[int]
    llc: list[float]
    estimator: str = "devinterp.sgld"
    meta: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "is_available",
    "LLCTrajectory",
]
