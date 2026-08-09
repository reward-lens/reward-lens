"""The reach panel: what the graded process can touch, established by running it (section 7.3).

Six surfaces, and one entry per surface carrying the attempts that were made. Nothing here reports
a reach from source: every claim rests on a response that was handed to the grader and executed in
the sandbox at the tier that held. An unreachable surface is reported too, as a finding of kind
`pass`, because that is how the record says what the reward correctly refuses.
"""

from __future__ import annotations

from .probes import (
    FINDING_IDS,
    METHOD_VERSION,
    PANEL,
    ReachRequiresTaskSet,
    ReachSurfaces,
    findings_for,
    probe_surfaces,
    sandbox_for,
)
from .surfaces import ATTEMPT_NAMES, MEASURANDS, SURFACES, Surface, entry_id_for, finding_id_for
from .witness import (
    CANARY_BYTES,
    CANARY_NAME,
    OUTSIDE_BYTES,
    OUTSIDE_NAME,
    WITNESS_VERSION,
    Attempt,
    ReachRun,
    SurfaceReport,
)

from . import probes  # noqa: F401  the audit's own tests reach for it by name

__all__ = [
    "ATTEMPT_NAMES",
    "Attempt",
    "CANARY_BYTES",
    "CANARY_NAME",
    "FINDING_IDS",
    "OUTSIDE_BYTES",
    "OUTSIDE_NAME",
    "MEASURANDS",
    "METHOD_VERSION",
    "PANEL",
    "ReachRequiresTaskSet",
    "ReachRun",
    "ReachSurfaces",
    "SURFACES",
    "Surface",
    "SurfaceReport",
    "WITNESS_VERSION",
    "entry_id_for",
    "finding_id_for",
    "findings_for",
    "probe_surfaces",
    "sandbox_for",
]
