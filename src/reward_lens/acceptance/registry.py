"""Declared producer bindings for D-72 policy checks.

This registry validates a policy's reference to a producer.  It does *not*
declare required evidence: each policy file remains the sole authority for the
checks it requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import PolicyCheck


@dataclass(frozen=True)
class ProducerBinding:
    """One exact canonical producer or a deliberately frozen future seam."""

    key: str
    source: str
    section: str | None = None
    entry_id: str | None = None
    alternate_entry_ids: tuple[str, ...] = ()
    result_field: str | None = None
    status: str = "canonical"
    owner: str | None = None


# Frozen entries are intentionally not treated as measured evidence.  Their
# names reserve a reviewable handoff seam until their owning producer emits an
# Assay Entry with the exact binding below.
PRODUCERS: dict[str, ProducerBinding] = {
    "validity.complete": ProducerBinding("validity.complete", "all_checks_complete", "validity"),
    "validity.owner_declared_input_handling": ProducerBinding(
        "validity.owner_declared_input_handling",
        "entry_required_check",
        "validity",
        "validity.input_handling",
    ),
    "outcome.qualified": ProducerBinding("outcome.qualified", "outcome_qualified"),
    "reach.open_findings": ProducerBinding("reach.open_findings", "finding_guard", "reach"),
    "exploits.reproducible_severity": ProducerBinding(
        "exploits.reproducible_severity", "finding_guard", "exploits"
    ),
    "signal.measured": ProducerBinding("signal.measured", "section_measured", "signal"),
    "rung2.complete": ProducerBinding(
        "rung2.complete",
        "entry_complete",
        "reward_statistics",
        "reward_statistics.selection_stress",
    ),
    "susceptibility.calibration_conditions": ProducerBinding(
        "susceptibility.calibration_conditions",
        "entry_result_field",
        "reward_statistics",
        "reward_statistics.selection_stress",
        result_field="calibration",
    ),
    "capability_floor.held": ProducerBinding(
        "capability_floor.held",
        "entry_required_check",
        "soundness",
        "soundness.capability_floor",
    ),
    "rung4_or_5.complete": ProducerBinding(
        "rung4_or_5.complete",
        "entry_any_complete",
        "reward_statistics",
        "reward_statistics.controlled_update_response",
        alternate_entry_ids=("reward_statistics.short_fork_response",),
    ),
    "null_replay.passed": ProducerBinding(
        "null_replay.passed",
        "entry_any_control_field",
        "reward_statistics",
        "reward_statistics.controlled_update_response",
        alternate_entry_ids=("reward_statistics.short_fork_response",),
        result_field="null_replay",
    ),
    "placebo.passed": ProducerBinding(
        "placebo.passed",
        "entry_any_control_field",
        "reward_statistics",
        "reward_statistics.controlled_update_response",
        alternate_entry_ids=("reward_statistics.short_fork_response",),
        result_field="placebo",
    ),
}


def validate_producer(check: "PolicyCheck") -> None:
    """Fail closed unless the check exactly names its declared producer."""
    from .types import PolicyInvalid

    binding = PRODUCERS.get(check.key)
    if binding is None:
        raise PolicyInvalid(f"evidence key {check.key!r} has no registered producer")
    observed = (
        check.source,
        check.section,
        check.entry_id,
        check.alternate_entry_ids,
        check.result_field,
    )
    expected = (
        binding.source,
        binding.section,
        binding.entry_id,
        binding.alternate_entry_ids,
        binding.result_field,
    )
    if observed != expected:
        raise PolicyInvalid(
            f"evidence key {check.key!r} does not match its registered producer {expected!r}"
        )
