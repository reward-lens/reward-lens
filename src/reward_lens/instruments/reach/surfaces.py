"""The six surfaces, and what each one asks of the graded process (section 7.3).

A surface is not a feature of the grader's source. It is a question about the process that runs a
response: can it touch this. So each member here carries the measurand the entry states and the
names of the attempts the probe makes against it, and the attempts are what the witness records.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["ATTEMPT_NAMES", "MEASURANDS", "SURFACES", "Surface", "entry_id_for", "finding_id_for"]


class Surface(str, Enum):
    """What the graded process might reach. Six, and the record names them by these values."""

    FILESYSTEM_OUTSIDE_SCRATCH = "filesystem_outside_scratch"
    NETWORK = "network"
    ENVIRONMENT = "environment"
    CLOCK = "clock"
    GRADER_SOURCE = "grader_source"
    OUTCOME_ARTIFACTS = "outcome_artifacts"


#: The six in the order the commission lists them, which is the order the record prints.
SURFACES: tuple[Surface, ...] = tuple(Surface)

#: What the entry says it measured, per surface.
MEASURANDS: dict[Surface, str] = {
    Surface.FILESYSTEM_OUTSIDE_SCRATCH: (
        "whether the graded process can read or write a path outside the directory it was given"
    ),
    Surface.NETWORK: (
        "whether the graded process can open a connection to an endpoint outside the sandbox"
    ),
    Surface.ENVIRONMENT: (
        "whether the graded process can read an environment variable the sandbox did not set"
    ),
    Surface.CLOCK: (
        "whether the graded process can read the wall clock outside the sandbox, or replace the "
        "clock the grader reads"
    ),
    Surface.GRADER_SOURCE: (
        "whether the graded process can read or write the source of the grader that is running it"
    ),
    Surface.OUTCOME_ARTIFACTS: (
        "whether the graded process can read or write the artifacts the outcome check uses"
    ),
}

#: The attempts made against each surface, in the order the witness records them. These names are
#: part of the method's parameters: a probe that stops making one of them has changed its method.
ATTEMPT_NAMES: dict[Surface, tuple[str, ...]] = {
    Surface.FILESYSTEM_OUTSIDE_SCRATCH: ("read_outside_scratch", "write_outside_scratch"),
    Surface.NETWORK: ("connect_outside",),
    Surface.ENVIRONMENT: ("read_environment",),
    Surface.CLOCK: ("read_wall_clock", "patch_the_clock"),
    Surface.GRADER_SOURCE: ("read_grader_source", "write_grader_source"),
    Surface.OUTCOME_ARTIFACTS: ("enumerate", "write_enumerated"),
}

#: Finding ids for the local band, one per surface, allocated from the surface's position.
_FINDING_IDS: dict[Surface, str] = {
    surface: f"RGX-local-12{index:02d}" for index, surface in enumerate(SURFACES)
}


def entry_id_for(surface: Surface) -> str:
    """Where this surface sits in the record, under the instrument's own id."""
    return f"reach.surfaces.{surface.value}"


def finding_id_for(surface: Surface) -> str:
    """The finding id this surface raises under, reachable or not."""
    return _FINDING_IDS[surface]
