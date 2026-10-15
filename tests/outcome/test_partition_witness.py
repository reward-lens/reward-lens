"""Qualification controls and evaluated target partitions are separate, inspectable witnesses."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from reward_lens.contracts import EntryProvenance
from reward_lens.outcome import (
    OutcomeCheck,
    SuiteResult,
    TaskValidity,
    qualified_protocol,
    qualify,
    resolved_pass,
)
from reward_lens.partitions import Partition, PartitionKind

SUBJECT = "sha256:" + "a" * 64
GRADER = "sha256:" + "b" * 64
CHECK_SOURCE = "sha256:" + "c" * 64


@pytest.fixture
def provenance() -> EntryProvenance:
    return EntryProvenance(
        started="2026-09-26T00:00:00Z", duration_s=0.25, sandbox_tier="L3", offline=True
    )


def partition(
    tmp_path: Path, name: str, item_id: str, body: str = "assert output == 1"
) -> Partition:
    root = tmp_path / name
    root.mkdir()
    (root / item_id).write_text(body, encoding="utf-8")
    return Partition.from_dir(
        name,
        PartitionKind.PROTECTED_SUITE,
        root,
        log_path=tmp_path / f"{name}.jsonl",
    )


def qualification(controls: Partition, provenance: EntryProvenance, *, check_id: str | None = None):
    return qualify(
        check=OutcomeCheck(
            owner="independent evaluator",
            source_digest=CHECK_SOURCE,
            interface="pytest",
            partition_id=check_id or controls.id,
        ),
        suite=controls,
        grader_owner="audited grader",
        grader_source_digest=GRADER,
        subject_ref=SUBJECT,
        provenance=provenance,
        result=SuiteResult(
            known_good_passed=1, known_good_total=1, known_wrong_failed=1, known_wrong_total=1
        ),
    )


def test_check_partition_mismatch_is_the_normal_unqualified_absence(
    tmp_path: Path, provenance: EntryProvenance
) -> None:
    controls = partition(tmp_path, "protected-controls", "control.py")
    result = qualification(controls, provenance, check_id="wrong-controls")
    assert result.qualified is False
    assert result.entry.kind == "absence"
    assert "does not name protected-controls" in result.reason


def test_qualified_controls_can_evaluate_a_different_heldout_target(
    tmp_path: Path, provenance: EntryProvenance
) -> None:
    controls = partition(tmp_path, "protected-controls", "control.py")
    target = partition(tmp_path, "heldout-target", "target.py")
    protocol = qualified_protocol(qualification(controls, provenance))
    outcome = resolved_pass(
        item_id="target.py",
        reason="the qualified evaluator accepted this target item",
        protocol=protocol,
        task_validity=TaskValidity.VALID,
        evaluation_partition=target,
    )
    assert protocol.qualified_on_partition_id == controls.id
    assert protocol.qualified_on_partition_digest == controls.digest
    assert outcome.evaluated_on_partition_id == target.id
    assert outcome.evaluated_on_partition_digest == target.digest
    assert outcome.evaluated_item_digest == dict(target.manifest())["target.py"]


def test_evaluation_partition_rejects_an_item_outside_its_active_manifest(
    tmp_path: Path, provenance: EntryProvenance
) -> None:
    controls = partition(tmp_path, "protected-controls", "control.py")
    target = partition(tmp_path, "heldout-target", "target.py")
    protocol = qualified_protocol(qualification(controls, provenance))
    with pytest.raises(ValueError, match="active item"):
        resolved_pass(
            item_id="foreign.py",
            reason="forged target membership",
            protocol=protocol,
            task_validity=TaskValidity.VALID,
            evaluation_partition=target,
        )


def test_replacing_captured_target_metadata_cannot_relabel_the_evaluation(
    tmp_path: Path, provenance: EntryProvenance
) -> None:
    controls = partition(tmp_path, "protected-controls", "control.py")
    target = partition(tmp_path, "heldout-target", "target.py")
    protocol = qualified_protocol(qualification(controls, provenance))
    outcome = resolved_pass(
        item_id="target.py",
        reason="accepted",
        protocol=protocol,
        task_validity=TaskValidity.VALID,
        evaluation_partition=target,
    )
    changed = partition(tmp_path, "changed-target", "target.py", "assert output == 2")
    with pytest.raises(ValueError, match="evaluation partition witness"):
        replace(outcome, evaluated_on_partition_digest=changed.digest)
    with pytest.raises(ValueError, match="evaluation partition witness"):
        replace(outcome, evaluated_item_digest=dict(changed.manifest())["target.py"])
