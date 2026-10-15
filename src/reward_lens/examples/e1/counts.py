"""Counts derived from the E1 rollout rows, never copied into this module."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from .build import Assay


@dataclass(frozen=True)
class Counts:
    groups: int
    groups_with_contrast: int
    accepted_hacks: int
    solutions: int
    functions_summed: int
    advantage_components: int
    registered_analysis_support: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def counts(assay: "Assay") -> Counts:
    """Compute the seven section-10 fields from the parsed analysis and rollout rows."""
    rows = assay.rollouts.rows
    group_rows: dict[tuple[object, object], list[object]] = {}
    for row in rows:
        group_rows.setdefault((row["source_eval_file"], row["sample_id"]), []).append(row)
    composite = tuple(
        float(row["thinking_format"]) * assay.optimized_weights["thinking_format"]
        + float(row["training_passed"]) * assay.optimized_weights["training_passed"]
        for row in rows
    )
    top_cell = max(composite)
    binary_cells = [
        {
            (
                float(row["thinking_format"]) == float(assay.optimized_weights["thinking_format"]),
                float(row["training_passed"]) > float(),
            )
            for row in group
        }
        for group in group_rows.values()
    ]
    return Counts(
        groups=len(group_rows),
        groups_with_contrast=sum(
            len(cells) >= len(assay.contrast_threshold) for cells in binary_cells
        ),
        accepted_hacks=sum(
            bool(row["reward_hacked"]) and value == top_cell for row, value in zip(rows, composite)
        ),
        solutions=sum(bool(row["actually_solved"]) for row in rows),
        functions_summed=len(assay.reward_functions),
        advantage_components=sum(weight != float() for weight in assay.optimized_weights.values()),
        registered_analysis_support=sum(
            len(cells) >= len(assay.registered_threshold) for cells in binary_cells
        ),
    )


def assert_matches_golden(measured: Counts, golden: Mapping[str, int] | None = None) -> None:
    """Fail on a controller-golden mismatch; this function never writes the golden."""
    if golden is None:
        from .build import load_golden

        golden = load_golden()
    actual = measured.as_dict()
    disagreements = {
        key: (actual.get(key), expected)
        for key, expected in golden.items()
        if actual.get(key) != expected
    }
    if disagreements:
        raise AssertionError(f"E1 counts disagree with the controller golden: {disagreements}")
