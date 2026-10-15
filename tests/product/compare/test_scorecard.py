"""The common compare scorecard has six fixed rows at every rung."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from reward_lens import contracts
from reward_lens.contracts import EntryProvenance, ProjectConfig
from reward_lens.outcome import (
    OutcomeCheck,
    SuiteResult,
    TaskValidity,
    UnqualifiedReason,
    qualified_protocol,
    qualify,
    resolved_fail,
    resolved_pass,
    unqualified,
)
from reward_lens.partitions import Partition, PartitionKind
from reward_lens.product.compare.scorecard import (
    ROWS,
    Absence,
    CapabilityFloor,
    Estimate,
    Scorecard,
    ScorecardRow,
    Tradeoff,
    capability_floor_assay_entry,
)

SCOPE = "evaluator_comparison"
_PROVENANCE = EntryProvenance(
    started="2026-09-26T00:00:00Z", duration_s=0.25, sandbox_tier="L3", offline=True
)


def _floor() -> CapabilityFloor:
    measured = Estimate(Decimal("0.750000"), (Decimal("0.5"), Decimal("0.9")), n=4)
    return CapabilityFloor(
        value=measured.value,
        interval=measured.interval,
        n=measured.n,
        partition_id="heldout",
        estimand="held-out legitimate-solution success",
        floor=Decimal("0.700000"),
        measured=measured,
        held=True,
    )


def _rows() -> tuple[ScorecardRow, ...]:
    rows = []
    for row_id in ROWS:
        if row_id == "legitimate_solutions_kept":
            value: Estimate | Absence = _floor()
        elif row_id == "signal":
            value = Estimate(
                Decimal("0.750000"), (Decimal("0.5"), Decimal("0.9")), n=4, n_eff=Decimal("3.5")
            )
        elif row_id == "cost":
            value = Estimate(
                Decimal("0.750000"), (Decimal("0.5"), Decimal("0.9")), n=4, unit="USD", samples=4
            )
        else:
            value = Estimate(Decimal("0.750000"), (Decimal("0.5"), Decimal("0.9")), n=4)
        rows.append(
            ScorecardRow(id=row_id, label=row_id.replace("_", " "), value=value, scope=SCOPE)
        )
    return tuple(rows)


def _held_out_partition(tmp_path: Path, *, empty: bool = False) -> Partition:
    root = tmp_path / "heldout"
    root.mkdir()
    if not empty:
        for index in range(4):
            (root / f"task-{index}.txt").write_text("independent task", encoding="utf-8")
    return Partition.from_dir(
        "heldout",
        PartitionKind.HELD_OUT_TASKS,
        root,
        log_path=tmp_path / "heldout-access.jsonl",
    )


def _qualification_controls(tmp_path: Path) -> Partition:
    root = tmp_path / "controls"
    root.mkdir()
    (root / "control.py").write_text("independent qualification control", encoding="utf-8")
    return Partition.from_dir(
        "qualification-controls",
        PartitionKind.PROTECTED_SUITE,
        root,
        log_path=tmp_path / "controls-access.jsonl",
    )


def _declared_project() -> ProjectConfig:
    return ProjectConfig.model_validate(
        {
            "reward": {"entry": "grader.py"},
            "tasks": {"path": "tasks.jsonl", "prompt": "prompt"},
            "outcome": {
                "kind": "protected_test_suite",
                "path": "outcome-check",
                "held_out_capability": {"partition_id": "heldout", "path": "heldout"},
            },
        }
    )


def _resolved_outcomes(
    partition: Partition,
    controls: Partition,
    *,
    check_source: str = "c",
    evaluation_partition: Partition | None = None,
    capture_evaluation: bool = True,
    item_ids: tuple[str, ...] = ("task-0.txt", "task-1.txt", "task-2.txt", "task-3.txt"),
):
    qualification = qualify(
        check=OutcomeCheck(
            owner="independent evaluator",
            source_digest="sha256:" + check_source * 64,
            interface="pytest",
            partition_id=controls.id,
        ),
        suite=controls,
        grader_owner="audited grader",
        grader_source_digest="sha256:" + "b" * 64,
        subject_ref="sha256:" + "a" * 64,
        provenance=_PROVENANCE,
        result=SuiteResult(
            known_good_passed=1, known_good_total=1, known_wrong_failed=1, known_wrong_total=1
        ),
    )
    protocol = qualified_protocol(qualification)
    target = (
        partition if evaluation_partition is None and capture_evaluation else evaluation_partition
    )
    return tuple(
        (resolved_pass if index < len(item_ids) - 1 else resolved_fail)(
            item_id=item_id,
            reason="accepted" if index < len(item_ids) - 1 else "rejected",
            protocol=protocol,
            task_validity=TaskValidity.VALID,
            evaluation_partition=target,
        )
        for index, item_id in enumerate(item_ids)
    )


@pytest.mark.parametrize(
    "rung",
    (
        "evaluator_comparison",
        "selection_stress",
        "reconstructed_training_pressure",
        "controlled_update_response",
        "short_fork_behavior",
        "held_out_transfer",
    ),
)
def test_every_rung_keeps_the_same_six_ordered_rows(rung: str) -> None:
    scorecard = Scorecard.render(_rows(), rung=rung)
    assert tuple(row.id for row in scorecard.rows) == ROWS
    assert all(
        isinstance(row.value, Estimate) and row.value.interval is not None for row in scorecard.rows
    )


def test_capability_floor_is_measured_from_declared_held_out_partition(tmp_path: Path) -> None:
    partition = _held_out_partition(tmp_path)
    controls = _qualification_controls(tmp_path)
    floor = CapabilityFloor.from_outcomes(
        project=_declared_project(),
        project_root=tmp_path,
        partition=partition,
        outcomes=_resolved_outcomes(partition, controls),
        floor=Decimal("0.700000"),
    )
    assert isinstance(floor, CapabilityFloor)
    assert floor.partition_id == partition.id
    assert floor.measured.value == Decimal("0.750000")
    assert floor.held is True
    assert controls.id != partition.id


def _capability_entry(
    *,
    tmp_path: Path,
    partition: Partition,
    controls: Partition,
    outcomes: tuple,
    floor: Decimal,
):
    protocol = outcomes[0]._protocol
    assert protocol is not None
    return capability_floor_assay_entry(
        project=_declared_project(),
        project_root=tmp_path,
        partition=partition,
        outcomes=outcomes,
        protocol=protocol,
        floor=floor,
        subject_ref="sha256:" + "e" * 64,
        provenance=_PROVENANCE,
        scope=SCOPE,
    )


def test_capability_floor_assay_entry_is_canonical_typed_target_evidence(tmp_path: Path) -> None:
    partition = _held_out_partition(tmp_path)
    controls = _qualification_controls(tmp_path)
    outcomes = _resolved_outcomes(partition, controls)

    entry = _capability_entry(
        tmp_path=tmp_path,
        partition=partition,
        controls=controls,
        outcomes=outcomes,
        floor=Decimal("0.700000"),
    )

    assert entry.entry_id == "soundness.capability_floor"
    assert entry.section == "soundness"
    assert entry.kind == "estimate"
    assert entry.subject_ref == "sha256:" + "e" * 64
    assert entry.provenance == _PROVENANCE
    assert entry.n == 4
    assert entry.uncertainty.method == "wilson"
    assert entry.uncertainty.level == 0.95
    assert entry.check is not None and entry.check.passed is True
    assert entry.result == {
        "type": "held_out_capability_floor/v1",
        "target": {
            "partition_id": partition.id,
            "partition_digest": partition.digest,
            "active_item_digests": dict(partition.manifest()),
        },
        "qualification": {
            "protocol_digest": outcomes[0].protocol_digest,
            "control_partition_id": controls.id,
            "control_partition_digest": controls.digest,
            "control_distinct_from_target": True,
        },
        "estimate": {
            "value": "0.750000",
            "interval": [str(entry.uncertainty.interval[0]), str(entry.uncertainty.interval[1])],
            "n": 4,
            "pass_count": 3,
            "fail_count": 1,
            "unqualified_count": 0,
            "confidence": "0.950",
            "sidedness": "two-sided",
        },
        "threshold": {"operator": ">=", "floor": "0.700000"},
        "held": True,
    }
    assert contracts.Entry.model_validate_json(contracts.canonical_bytes(entry)) == entry


def test_capability_floor_assay_entry_records_a_false_check_for_policy_rejection(
    tmp_path: Path,
) -> None:
    partition = _held_out_partition(tmp_path)
    controls = _qualification_controls(tmp_path)
    entry = _capability_entry(
        tmp_path=tmp_path,
        partition=partition,
        controls=controls,
        outcomes=_resolved_outcomes(partition, controls),
        floor=Decimal("0.800000"),
    )

    assert entry.kind == "estimate"
    assert entry.result["held"] is False
    assert entry.check is not None and entry.check.passed is False


def test_capability_floor_assay_entry_keeps_unqualified_or_wrong_target_evidence_absent(
    tmp_path: Path,
) -> None:
    partition = _held_out_partition(tmp_path)
    controls = _qualification_controls(tmp_path)
    qualified_rows = _resolved_outcomes(partition, controls)
    unqualified_rows = tuple(
        unqualified(
            item_id=row.item_id,
            reason=UnqualifiedReason.TIMEOUT,
            protocol=row._protocol,
            task_validity=TaskValidity.UNKNOWN,
            evaluation_partition=partition,
        )
        for row in qualified_rows
    )
    foreign_root = tmp_path / "wrong-target"
    foreign_root.mkdir()
    for index in range(4):
        (foreign_root / f"task-{index}.txt").write_text("independent task", encoding="utf-8")
    wrong_target = Partition.from_dir(
        "wrong-target",
        PartitionKind.HELD_OUT_TASKS,
        foreign_root,
        log_path=tmp_path / "wrong-target-access.jsonl",
    )
    wrong_target_rows = _resolved_outcomes(partition, controls, evaluation_partition=wrong_target)

    for rows, reason in (
        (unqualified_rows, "not measured: no resolved held-out outcomes"),
        (
            wrong_target_rows,
            "not measured: evaluated-on partition id does not match declared target",
        ),
    ):
        entry = _capability_entry(
            tmp_path=tmp_path,
            partition=partition,
            controls=controls,
            outcomes=rows,
            floor=Decimal("0.700000"),
        )
        assert entry.kind == "absence"
        assert entry.check is None
        assert entry.absence is not None and entry.absence.missing_access == reason


def test_no_declared_held_out_set_is_an_absence_never_default_membership(tmp_path: Path) -> None:
    no_outcome = ProjectConfig.model_validate(
        {
            "reward": {"entry": "grader.py"},
            "tasks": {"path": "tasks.jsonl", "prompt": "prompt"},
            "outcome": {"kind": "protected_test_suite", "path": "outcome-check"},
        }
    )
    floor = CapabilityFloor.from_outcomes(
        project=no_outcome,
        project_root=tmp_path,
        partition=_held_out_partition(tmp_path),
        outcomes=(),
        floor=Decimal("0.700000"),
    )
    assert isinstance(floor, Absence)
    assert floor.reason == "not measured: no held-out capability set declared"


def test_declared_capability_path_cannot_substitute_a_different_held_out_partition(
    tmp_path: Path,
) -> None:
    partition = _held_out_partition(tmp_path)
    different = ProjectConfig.model_validate(
        {
            "reward": {"entry": "grader.py"},
            "tasks": {"path": "tasks.jsonl", "prompt": "prompt"},
            "outcome": {
                "kind": "protected_test_suite",
                "path": "outcome-check",
                "held_out_capability": {"partition_id": "heldout", "path": "another-partition"},
            },
        }
    )
    floor = CapabilityFloor.from_outcomes(
        project=different,
        project_root=tmp_path,
        partition=partition,
        outcomes=_resolved_outcomes(partition, _qualification_controls(tmp_path)),
        floor=Decimal("0.700000"),
    )
    assert isinstance(floor, Absence)
    assert "path does not resolve" in floor.reason


def test_empty_declared_capability_partition_cannot_produce_a_floor(tmp_path: Path) -> None:
    partition = _held_out_partition(tmp_path, empty=True)
    floor = CapabilityFloor.from_outcomes(
        project=_declared_project(),
        project_root=tmp_path,
        partition=partition,
        outcomes=(),
        floor=Decimal("0.700000"),
    )
    assert isinstance(floor, Absence)
    assert floor.reason == "not measured: declared held-out capability set is empty"


def test_missing_active_item_cannot_produce_a_floor(tmp_path: Path) -> None:
    partition = _held_out_partition(tmp_path)
    controls = _qualification_controls(tmp_path)
    floor = CapabilityFloor.from_outcomes(
        project=_declared_project(),
        project_root=tmp_path,
        partition=partition,
        outcomes=_resolved_outcomes(partition, controls)[:-1],
        floor=Decimal("0.700000"),
    )
    assert isinstance(floor, Absence)
    assert floor.reason == "not measured: held-out outcome coverage is incomplete"


def test_duplicate_or_foreign_item_cannot_produce_a_floor(tmp_path: Path) -> None:
    partition = _held_out_partition(tmp_path)
    controls = _qualification_controls(tmp_path)
    outcomes = _resolved_outcomes(partition, controls)
    duplicate = (outcomes[0], outcomes[0], outcomes[2], outcomes[3])
    foreign_root = tmp_path / "foreign-items"
    foreign_root.mkdir()
    for item_id in ("foreign-task", "task-1.txt", "task-2.txt", "task-3.txt"):
        (foreign_root / item_id).write_text("foreign target item", encoding="utf-8")
    foreign_target = Partition.from_dir(
        "foreign-items",
        PartitionKind.HELD_OUT_TASKS,
        foreign_root,
        log_path=tmp_path / "foreign-items-access.jsonl",
    )
    foreign = _resolved_outcomes(
        foreign_target,
        controls,
        item_ids=("foreign-task", "task-1.txt", "task-2.txt", "task-3.txt"),
    )
    for supplied, expected in (
        (duplicate, "not measured: held-out outcome coverage contains duplicate item ids"),
        (foreign, "not measured: held-out outcome coverage contains foreign item ids"),
    ):
        floor = CapabilityFloor.from_outcomes(
            project=_declared_project(),
            project_root=tmp_path,
            partition=partition,
            outcomes=supplied,
            floor=Decimal("0.700000"),
        )
        assert isinstance(floor, Absence)
        assert floor.reason == expected


def test_all_unqualified_complete_coverage_cannot_produce_a_floor(tmp_path: Path) -> None:
    partition = _held_out_partition(tmp_path)
    outcomes = _resolved_outcomes(partition, _qualification_controls(tmp_path))
    unqualified_rows = tuple(
        unqualified(
            item_id=row.item_id,
            reason=UnqualifiedReason.TIMEOUT,
            protocol=row._protocol,
            task_validity=TaskValidity.UNKNOWN,
            evaluation_partition=partition,
        )
        for row in outcomes
    )
    floor = CapabilityFloor.from_outcomes(
        project=_declared_project(),
        project_root=tmp_path,
        partition=partition,
        outcomes=unqualified_rows,
        floor=Decimal("0.700000"),
    )
    assert isinstance(floor, Absence)
    assert floor.reason == "not measured: no resolved held-out outcomes"


def test_mixed_qualified_protocol_or_scope_cannot_produce_a_floor(tmp_path: Path) -> None:
    partition = _held_out_partition(tmp_path)
    controls = _qualification_controls(tmp_path)
    outcomes = _resolved_outcomes(partition, controls)
    other_protocol = _resolved_outcomes(partition, controls, check_source="d")
    cases = (
        (
            (other_protocol[0], *outcomes[1:]),
            "not measured: held-out outcomes do not share one qualified protocol",
        ),
        (
            (replace(outcomes[0], scope="selection_stress"), *outcomes[1:]),
            "not measured: held-out outcomes do not share one scope",
        ),
    )
    for supplied, expected in cases:
        floor = CapabilityFloor.from_outcomes(
            project=_declared_project(),
            project_root=tmp_path,
            partition=partition,
            outcomes=supplied,
            floor=Decimal("0.700000"),
        )
        assert isinstance(floor, Absence)
        assert floor.reason == expected


def test_generic_outcome_without_target_witness_cannot_produce_a_floor(tmp_path: Path) -> None:
    partition = _held_out_partition(tmp_path)
    generic = _resolved_outcomes(
        partition, _qualification_controls(tmp_path), capture_evaluation=False
    )
    floor = CapabilityFloor.from_outcomes(
        project=_declared_project(),
        project_root=tmp_path,
        partition=partition,
        outcomes=generic,
        floor=Decimal("0.700000"),
    )
    assert isinstance(floor, Absence)
    assert floor.reason == "not measured: held-out outcome lacks an evaluated-on partition witness"


def test_foreign_target_witness_or_changed_declared_file_cannot_produce_a_floor(
    tmp_path: Path,
) -> None:
    partition = _held_out_partition(tmp_path)
    controls = _qualification_controls(tmp_path)
    foreign_root = tmp_path / "foreign-target"
    foreign_root.mkdir()
    for index in range(4):
        (foreign_root / f"task-{index}.txt").write_text("independent task", encoding="utf-8")
    foreign = Partition.from_dir(
        "foreign-target",
        PartitionKind.HELD_OUT_TASKS,
        foreign_root,
        log_path=tmp_path / "foreign-access.jsonl",
    )
    foreign_outcomes = _resolved_outcomes(partition, controls, evaluation_partition=foreign)
    floor = CapabilityFloor.from_outcomes(
        project=_declared_project(),
        project_root=tmp_path,
        partition=partition,
        outcomes=foreign_outcomes,
        floor=Decimal("0.700000"),
    )
    assert isinstance(floor, Absence)
    assert floor.reason == "not measured: evaluated-on partition id does not match declared target"

    current_outcomes = _resolved_outcomes(partition, controls)
    (partition.root / "task-0.txt").write_text("changed after evaluation", encoding="utf-8")
    stale = CapabilityFloor.from_outcomes(
        project=_declared_project(),
        project_root=tmp_path,
        partition=partition,
        outcomes=current_outcomes,
        floor=Decimal("0.700000"),
    )
    assert isinstance(stale, Absence)
    assert (
        stale.reason
        == "not measured: evaluated-on partition digest does not match current declared target"
    )


def test_clean_comparison_remains_a_result_and_tradeoff_is_not_a_score() -> None:
    clean = Scorecard.render(_rows(), rung=SCOPE)
    tradeoff = Tradeoff(
        option_a="hardened reward",
        option_b="baseline reward",
        competing_consequences=("lower exploit pressure", "higher sampling cost"),
    )
    compared = Scorecard.render(_rows(), rung=SCOPE, tradeoff=tradeoff)
    assert clean.tradeoff is None
    assert clean.is_result is True
    assert clean.capability_floor == _rows()[1].value
    assert compared.tradeoff is tradeoff
    assert compared.is_result is False


def test_render_refuses_a_rung_specific_or_reordered_row_set() -> None:
    with pytest.raises(ValueError, match="exactly the fixed ROWS"):
        Scorecard.render(_rows()[::-1], rung=SCOPE)


def test_estimate_requires_its_interval_to_stay_on_its_row() -> None:
    with pytest.raises(ValueError, match="interval"):
        Estimate(Decimal("0.5"), None, n=2)
