"""The five partition kinds and the capabilities allowed to read each one (D-40).

A kind is not a label on a directory: it decides who may open the bytes. No field here is named for
concealment, because nothing here conceals anything. A partition on a filesystem owned by one
operating-system user is readable by that user; what these names govern is which capability the
product will open them for, and the boundary that holds against a candidate process is measured in
`reward_lens.outcome.separation`, not asserted here.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "ACCEPTANCE_EVALUATOR",
    "ALLOWED_CAPABILITIES",
    "CANDIDATE_AUTHOR",
    "CANDIDATE_PROCESS",
    "PANEL",
    "SEEKER",
    "PartitionKind",
]

#: The capability an acceptance evaluator runs under. It is the only one that opens a protected
#: suite, and it runs after the candidate digest is frozen.
ACCEPTANCE_EVALUATOR = "acceptance-evaluator"

#: The capability a candidate author (human or agent) runs under.
CANDIDATE_AUTHOR = "candidate-author"

#: The capability the candidate's own graded process runs under, inside the sandbox.
CANDIDATE_PROCESS = "candidate-process"

#: The capability an exploit seeker runs under: attack development, never acceptance.
#:
#: It appears in no row of `ALLOWED_CAPABILITIES`, and that is the design rather than a gap. A
#: seeker spends a sealed round against the material and reads what came back; it never opens a
#: partition. Reading the bytes of an acceptance set would let a seeker write an attack against
#: those particular items, which measures the items and not the reward system, and it would spend
#: the set without a round ever being sealed. So `read(..., capability=SEEKER)` refuses on every
#: kind, reference material included.
SEEKER = "seeker"

#: The capability a registered panel evaluator runs under.
PANEL = "panel"


class PartitionKind(StrEnum):
    """The five kinds. Their order is the order of the interface and does not change."""

    PROTECTED_SUITE = "protected_suite"
    HELD_OUT_TASKS = "held_out_tasks"
    PANEL_LABELS = "panel_labels"
    REFERENCE_MATERIAL = "reference_material"
    SEALED_ROUND = "sealed_round"

    @property
    def candidate_visible(self) -> bool:
        """Whether a candidate author may read this kind at all.

        Reference material is the one kind that is meant to be read by whoever is being measured:
        it is the certified known answer an instrument is checked against, and keeping it back
        would make the instrument uncheckable. The other four decide an outcome, and an author who
        has seen them has already spent them.
        """
        return self is PartitionKind.REFERENCE_MATERIAL


#: Which capabilities may read which kind. A kind that is candidate-visible admits any capability
#: this table governs; the rest admit the acceptance evaluator, and panel labels also admit a
#: registered panel. `SEEKER` is in no row and in no `None` row either: `Partition.read` refuses it
#: ahead of this table, because the seeker spends a sealed round instead of reading a partition.
ALLOWED_CAPABILITIES: dict[PartitionKind, frozenset[str] | None] = {
    PartitionKind.PROTECTED_SUITE: frozenset({ACCEPTANCE_EVALUATOR}),
    PartitionKind.HELD_OUT_TASKS: frozenset({ACCEPTANCE_EVALUATOR}),
    PartitionKind.PANEL_LABELS: frozenset({ACCEPTANCE_EVALUATOR, PANEL}),
    PartitionKind.SEALED_ROUND: frozenset({ACCEPTANCE_EVALUATOR}),
    PartitionKind.REFERENCE_MATERIAL: None,
}
