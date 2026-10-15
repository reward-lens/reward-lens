"""Rung 6 transfer comparison uses qualified outcome evidence, not caller labels."""

from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

import pytest

from reward_lens import contracts
from reward_lens.execution import Limits, default_sandbox
from reward_lens.graders.python_grader import PythonGrader
from reward_lens.outcome import (
    OutcomeCheck,
    SuiteResult,
    TaskValidity,
    qualified_protocol,
    qualify,
    resolved_fail,
    resolved_pass,
)
from reward_lens.outcome.supervised import SupervisedCase
from reward_lens.partitions import Partition, PartitionKind
from reward_lens.product.compare.registry import CompareInputs, Rung, get
from reward_lens.product.compare.rung6 import (
    R6_ENTRY_ID,
    Rung6Input,
    SupervisedRung6Config,
    SupervisedRung6Original,
    SupervisedRung6Target,
    TransferPartition,
    TransferPartitionAbsent,
    TransferPartitionClass,
    append_transfer_assay,
    build_supervised_rung6,
    evaluate_transfer,
    load_supervised_rung6_execution,
    run,
)


def _partition(
    tmp_path: Path, name: str, kind: PartitionKind, *, items: tuple[str, ...] = ("case.py",)
) -> Partition:
    root = tmp_path / name
    root.mkdir(parents=True)
    for item in items:
        (root / item).write_text(f"{name}:{item}", encoding="utf-8")
    return Partition.from_dir(name, kind, root, log_path=tmp_path / f"{name}.access.jsonl")


def _qualified(tmp_path: Path):
    controls = _partition(tmp_path, "qualification-controls", PartitionKind.PROTECTED_SUITE)
    provenance = contracts.EntryProvenance(
        started="2026-09-27T00:00:00Z", duration_s=0.1, sandbox_tier="L3", offline=True
    )
    qualification = qualify(
        check=OutcomeCheck("independent evaluator", "sha256:" + "c" * 64, "pytest", controls.id),
        suite=controls,
        grader_owner="audited grader",
        grader_source_digest="sha256:" + "d" * 64,
        subject_ref="sha256:" + "e" * 64,
        provenance=provenance,
        result=SuiteResult(
            known_good_passed=1, known_good_total=1, known_wrong_failed=1, known_wrong_total=1
        ),
    )
    return controls, provenance, qualification, qualified_protocol(qualification)


def _input(
    tmp_path: Path, *, fail_partition: str | None = None, original_blocked: bool = True
) -> Rung6Input:
    controls, provenance, qualification, protocol = _qualified(tmp_path)
    partitions = tuple(
        _partition(tmp_path, name, PartitionKind.HELD_OUT_TASKS)
        for name in ("heldout-family", "different-exploit", "different-checkpoint")
    )
    declarations = tuple(
        TransferPartition(
            partition_id=partition.id,
            partition_class=partition_class,
            family=family,
            exploit=exploit,
            checkpoint=checkpoint,
            declared_at="2026-09-27T00:00:00Z",
            digest=partition.digest,
        )
        for partition, partition_class, family, exploit, checkpoint in zip(
            partitions,
            (
                TransferPartitionClass.FAMILY,
                TransferPartitionClass.EXPLOIT,
                TransferPartitionClass.CHECKPOINT,
            ),
            ("family-b", "family-a", "family-a"),
            ("exploit-a", "exploit-b", "exploit-a"),
            ("checkpoint-a", "checkpoint-a", "checkpoint-b"),
            strict=True,
        )
    )
    outcomes = {
        partition.id: (
            (resolved_fail if partition.id == fail_partition else resolved_pass)(
                item_id="case.py",
                reason="resolved",
                protocol=protocol,
                task_validity=TaskValidity.VALID,
                evaluation_partition=partition,
            ),
        )
        for partition in partitions
    }
    original = _partition(tmp_path, "original-example", PartitionKind.HELD_OUT_TASKS)
    original_outcome = (resolved_fail if original_blocked else resolved_pass)(
        item_id="case.py",
        reason="original observed",
        protocol=protocol,
        task_validity=TaskValidity.VALID,
        evaluation_partition=original,
    )
    return Rung6Input(
        partitions=declarations,
        evaluation_partitions={partition.id: partition for partition in partitions},
        outcomes=outcomes,
        qualification=qualification,
        protected_controls=controls,
        original_partition=original,
        original_outcome=original_outcome,
        original_family="family-a",
        original_exploit="exploit-a",
        original_checkpoint="checkpoint-a",
        subject_ref="sha256:" + "f" * 64,
        provenance=provenance,
    )


def test_transfer_calculates_from_three_live_declared_partitions_and_qualified_outcomes(
    tmp_path: Path,
) -> None:
    comparison = evaluate_transfer(_input(tmp_path))

    assert comparison.scope == "held_out_transfer"
    assert comparison.entries[0].entry_id == R6_ENTRY_ID
    assert comparison.entries[0].check.passed is True
    evidence = comparison.entries[0].result
    assert evidence["qualification"]["state"] == "qualified"
    assert tuple(evidence["partitions"]) == (
        "heldout-family",
        "different-exploit",
        "different-checkpoint",
    )
    assert [row["partition_class"] for row in evidence["partitions"].values()] == [
        "FAMILY",
        "EXPLOIT",
        "CHECKPOINT",
    ]
    assert evidence["original_example"]["partition_id"] == "original-example"
    assert evidence["original_example"]["item_digest"].startswith("sha256:")
    assert evidence["original_example_only"] is False
    assert all(row.value.n == 3 for row in comparison.scorecard.rows if hasattr(row.value, "n"))


def test_original_block_without_transfer_is_reported_as_original_example_only(
    tmp_path: Path,
) -> None:
    comparison = evaluate_transfer(_input(tmp_path, fail_partition="different-exploit"))

    assert comparison.entries[0].check.passed is False
    assert comparison.entries[0].result["original_example_only"] is True
    assert "original example" in comparison.does_not_establish


def test_missing_partition_class_refuses_with_the_precise_missing_class(tmp_path: Path) -> None:
    input = _input(tmp_path)
    missing = Rung6Input(
        partitions=input.partitions[:-1],
        evaluation_partitions=input.evaluation_partitions,
        outcomes=input.outcomes,
        qualification=input.qualification,
        protected_controls=input.protected_controls,
        original_partition=input.original_partition,
        original_outcome=input.original_outcome,
        original_family=input.original_family,
        original_exploit=input.original_exploit,
        original_checkpoint=input.original_checkpoint,
        subject_ref=input.subject_ref,
        provenance=input.provenance,
    )
    with pytest.raises(TransferPartitionAbsent, match="TRANSFER_PARTITION_ABSENT.*checkpoint"):
        evaluate_transfer(missing)


def test_unqualified_outcome_is_a_named_canonical_absence(tmp_path: Path) -> None:
    input = _input(tmp_path)
    unqualified = qualify(
        check=None,
        suite=input.protected_controls,
        grader_owner="audited grader",
        grader_source_digest="sha256:" + "d" * 64,
        subject_ref=input.subject_ref,
        provenance=input.provenance,
    )
    comparison = evaluate_transfer(
        Rung6Input(
            partitions=input.partitions,
            evaluation_partitions=input.evaluation_partitions,
            outcomes={},
            qualification=unqualified,
            protected_controls=input.protected_controls,
            original_partition=None,
            original_outcome=None,
            original_family=None,
            original_exploit=None,
            original_checkpoint=None,
            subject_ref=input.subject_ref,
            provenance=input.provenance,
        )
    )
    entry = comparison.entries[0]
    assert entry.kind == "absence"
    assert "OUTCOME_UNQUALIFIED" in entry.absence.missing_access
    assert "qualification.state" in entry.absence.missing_access


def test_protected_controls_cannot_be_one_of_the_transfer_partitions(tmp_path: Path) -> None:
    input = _input(tmp_path)
    collision = Rung6Input(
        partitions=input.partitions,
        evaluation_partitions=input.evaluation_partitions,
        outcomes=input.outcomes,
        qualification=input.qualification,
        protected_controls=input.evaluation_partitions["heldout-family"],
        original_partition=input.original_partition,
        original_outcome=input.original_outcome,
        original_family=input.original_family,
        original_exploit=input.original_exploit,
        original_checkpoint=input.original_checkpoint,
        subject_ref=input.subject_ref,
        provenance=input.provenance,
    )
    with pytest.raises((TypeError, ValueError), match="protected_controls"):
        evaluate_transfer(collision)


def test_registry_uses_the_strict_rung_six_input_seam(tmp_path: Path) -> None:
    input = _input(tmp_path)
    assert get(Rung.TRANSFER) is run
    with pytest.raises(TypeError, match="Rung6Input"):
        run(CompareInputs(run_record=object()))
    assert run(CompareInputs(run_record=input)).entries[0].check.passed is True


def test_axis_collapsed_declarations_are_refused_even_with_three_live_targets(
    tmp_path: Path,
) -> None:
    input = _input(tmp_path)
    collapsed = (
        replace(
            input.partitions[0],
            partition_class=TransferPartitionClass.FAMILY,
            family="family-b",
            exploit="exploit-b",
            checkpoint="checkpoint-b",
        ),
        replace(input.partitions[1], partition_class=TransferPartitionClass.EXPLOIT),
        replace(input.partitions[2], partition_class=TransferPartitionClass.CHECKPOINT),
    )
    with pytest.raises(ValueError, match="FAMILY.*exploit"):
        evaluate_transfer(replace(input, partitions=collapsed))


def test_target_outcome_cannot_be_relabelled_as_the_original_example(tmp_path: Path) -> None:
    input = _input(tmp_path)
    target = input.evaluation_partitions["heldout-family"]
    protocol = qualified_protocol(input.qualification)
    misattributed = resolved_fail(
        item_id="case.py",
        reason="misattributed target result",
        protocol=protocol,
        task_validity=TaskValidity.VALID,
        evaluation_partition=target,
    )
    with pytest.raises(ValueError, match="original_partition.*disjoint"):
        evaluate_transfer(replace(input, original_partition=target, original_outcome=misattributed))


def test_canonical_transfer_entry_persists_and_reopens(tmp_path: Path) -> None:
    from reward_lens.store import Project
    from tests.store.conftest import build_record, write_project
    from tests.store.test_project import align

    project_root = write_project(tmp_path / "project")
    project = Project.open(project_root)
    base = contracts.Assay.model_validate(align(build_record(), project))
    input = _input(tmp_path / "transfer")
    input = Rung6Input(
        partitions=input.partitions,
        evaluation_partitions=input.evaluation_partitions,
        outcomes=input.outcomes,
        qualification=input.qualification,
        protected_controls=input.protected_controls,
        original_partition=input.original_partition,
        original_outcome=input.original_outcome,
        original_family=input.original_family,
        original_exploit=input.original_exploit,
        original_checkpoint=input.original_checkpoint,
        subject_ref=base.subject.version.digest,
        provenance=input.provenance,
    )
    saved = append_transfer_assay(base, evaluate_transfer(input))
    path = project.record(
        contracts.Assay.model_validate(align(saved.to_dict(), project)), name="rung-six"
    )
    reopened = project.open_record("rung-six")
    entry = next(row for row in reopened.entries() if row.entry_id == R6_ENTRY_ID)
    assert path.is_file()
    assert entry.result["original_example_only"] is False


def _case(item_id: str, response: str) -> SupervisedCase:
    return SupervisedCase(
        item_id=item_id,
        task={"id": item_id, "prompt": "return reference", "reference": "correct"},
        response=response,
    )


def _supervised_config(tmp_path: Path) -> SupervisedRung6Config:
    source = tmp_path / "owner_evaluator.py"
    source.write_text(
        "def score(task, response):\n    return 1.0 if response == task['reference'] else 0.0\n",
        encoding="utf-8",
    )
    evaluator = PythonGrader.from_path(source)
    controls = _partition(
        tmp_path,
        "supervised-controls",
        PartitionKind.PROTECTED_SUITE,
        items=("good.case", "wrong.case"),
    )
    original = _partition(tmp_path, "supervised-original", PartitionKind.HELD_OUT_TASKS)
    partitions = tuple(
        _partition(tmp_path, name, PartitionKind.HELD_OUT_TASKS)
        for name in ("supervised-family", "supervised-exploit", "supervised-checkpoint")
    )
    targets = tuple(
        SupervisedRung6Target(
            partition_class=partition_class,
            partition=partition,
            family=family,
            exploit=exploit,
            checkpoint=checkpoint,
            declared_at="2026-09-27T00:00:00Z",
            cases=(_case("case.py", "correct"),),
        )
        for partition, partition_class, family, exploit, checkpoint in zip(
            partitions,
            TransferPartitionClass,
            ("family-b", "family-a", "family-a"),
            ("exploit-a", "exploit-b", "exploit-a"),
            ("checkpoint-a", "checkpoint-a", "checkpoint-b"),
            strict=True,
        )
    )
    return SupervisedRung6Config(
        evaluator=evaluator,
        check=OutcomeCheck(
            owner="assay-owner",
            source_digest=evaluator.describe().source_digest,
            interface="reward_lens.graders.python_grader.PythonGrader.score/v1",
            partition_id=controls.id,
        ),
        owner="assay-owner",
        candidate_owner="candidate",
        protected_controls=controls,
        known_good=(_case("good.case", "correct"),),
        known_wrong=(_case("wrong.case", "wrong"),),
        audited_grader_owner="candidate",
        audited_grader_source_digest="sha256:" + "a" * 64,
        original=SupervisedRung6Original(
            partition=original,
            family="family-a",
            exploit="exploit-a",
            checkpoint="checkpoint-a",
            case=_case("case.py", "wrong"),
        ),
        targets=targets,
        subject_ref="sha256:" + "b" * 64,
        sandbox=default_sandbox(),
        limits=Limits(wall_s=15.0, cpu_s=15.0, network=False),
    )


def test_supervised_factory_constructs_rung_six_from_actual_outcome_receipts(
    tmp_path: Path,
) -> None:
    produced = build_supervised_rung6(_supervised_config(tmp_path))

    assert produced.comparison.entries[0].check.passed is True
    assert produced.input.original_outcome is produced.original.outcome
    receipts = produced.comparison.entries[0].result["supervised_receipts"]
    assert len(receipts["controls"]) == 2
    assert len(receipts["targets"]) == 3
    assert receipts["original"]["target_partition_id"] == "supervised-original"
    assert all(
        row[0]["receipt_digest"].startswith("sha256:") for row in receipts["targets"].values()
    )


def test_supervised_factory_refuses_axis_source_partition_and_receipt_shortcuts(
    tmp_path: Path,
) -> None:
    config = _supervised_config(tmp_path)
    collapsed = replace(config.targets[0], exploit="exploit-b", checkpoint="checkpoint-b")
    with pytest.raises(ValueError, match="FAMILY.*exploit"):
        build_supervised_rung6(replace(config, targets=(collapsed, *config.targets[1:])))
    with pytest.raises(TypeError, match="PythonGrader"):
        build_supervised_rung6(replace(config, evaluator=object()))
    with pytest.raises(ValueError, match="disjoint"):
        build_supervised_rung6(
            replace(config, original=replace(config.original, partition=config.protected_controls))
        )
    names = {field.name for field in fields(SupervisedRung6Config)}
    assert {"outcomes", "receipts", "suite_result", "runner"}.isdisjoint(names)


def test_supervised_factory_refuses_same_target_membership_before_any_evaluation(
    tmp_path: Path,
) -> None:
    config = _supervised_config(tmp_path)
    shared = _partition(tmp_path, "shared-target-membership", PartitionKind.HELD_OUT_TASKS)
    aliases = tuple(
        replace(
            target,
            partition=Partition.from_dir(
                f"alias-{target.partition_class.value.lower()}",
                PartitionKind.HELD_OUT_TASKS,
                shared.root,
                log_path=tmp_path / f"alias-{target.partition_class.value.lower()}.access.jsonl",
            ),
        )
        for target in config.targets
    )
    invalid = replace(config, targets=aliases)

    with pytest.raises(ValueError, match="physical.*separation"):
        build_supervised_rung6(invalid)

    assert not config.protected_controls.access_log.events
    assert not config.original.partition.access_log.events
    assert all(not target.partition.access_log.events for target in aliases)


def _supervised_execution_mapping(tmp_path: Path) -> dict[str, object]:
    config = _supervised_config(tmp_path)
    root = tmp_path.resolve()

    def relative(path: Path) -> str:
        return path.resolve().relative_to(root).as_posix()

    def case(value: SupervisedCase) -> dict[str, object]:
        return {"item_id": value.item_id, "task": dict(value.task), "response": value.response}

    def partition(value: Partition) -> dict[str, str]:
        return {
            "partition_id": value.id,
            "path": relative(value.root),
            "digest": value.digest,
            "tree_version": "tree-v1",
        }

    return {
        "schema_version": 1,
        "subject_ref": config.subject_ref,
        "owners": {
            "evaluator": config.owner,
            "candidate": config.candidate_owner,
            "audited_grader": config.audited_grader_owner,
            "audited_grader_source_digest": config.audited_grader_source_digest,
        },
        "grader": {
            "version": "python-grader-v1",
            "path": relative(config.evaluator.path),
            "source_digest": config.evaluator.describe().source_digest,
        },
        "limits": {
            "wall_s": 15.0,
            "cpu_s": 15.0,
            "memory_bytes": 2 * 2**30,
            "processes": 64,
            "network": False,
            "stdout_bytes": 1 << 20,
            "artifact_bytes": 64 << 20,
        },
        "controls": {
            **partition(config.protected_controls),
            "known_good": [case(value) for value in config.known_good],
            "known_wrong": [case(value) for value in config.known_wrong],
        },
        "original": {
            **partition(config.original.partition),
            "family": config.original.family,
            "exploit": config.original.exploit,
            "checkpoint": config.original.checkpoint,
            "case": case(config.original.case),
        },
        "targets": [
            {
                **partition(target.partition),
                "partition_class": target.partition_class.value,
                "family": target.family,
                "exploit": target.exploit,
                "checkpoint": target.checkpoint,
                "declared_at": target.declared_at,
                "cases": [case(value) for value in target.cases],
            }
            for target in config.targets
        ],
    }


def test_supervised_execution_json_loader_rebuilds_local_inputs_and_executes(
    tmp_path: Path,
) -> None:
    raw = _supervised_execution_mapping(tmp_path)
    path = tmp_path / "rung6-execution.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = load_supervised_rung6_execution(path)
    produced = build_supervised_rung6(loaded)

    assert type(loaded.evaluator) is PythonGrader
    assert loaded.protected_controls.root == tmp_path / "supervised-controls"
    assert produced.comparison.entries[0].check.passed is True
    assert (
        produced.comparison.entries[0]
        .result["supervised_receipts"]["original"]["receipt_digest"]
        .startswith("sha256:")
    )


def test_supervised_execution_json_loader_refuses_tampered_sources_partitions_and_shortcuts(
    tmp_path: Path,
) -> None:
    raw = _supervised_execution_mapping(tmp_path)
    bad_source = dict(raw)
    bad_source["grader"] = {**raw["grader"], "source_digest": "sha256:" + "0" * 64}
    with pytest.raises(ValueError, match="grader.source_digest"):
        SupervisedRung6Config.from_mapping(bad_source, root=tmp_path)

    bad_partition = dict(raw)
    targets = [dict(value) for value in raw["targets"]]
    targets[0]["digest"] = "sha256:" + "0" * 64
    bad_partition["targets"] = targets
    with pytest.raises(ValueError, match=r"targets\[0\].digest"):
        SupervisedRung6Config.from_mapping(bad_partition, root=tmp_path)

    bad_shortcut = dict(raw)
    bad_shortcut["outcomes"] = []
    with pytest.raises(ValueError, match="config has missing=.*extra=.*outcomes"):
        SupervisedRung6Config.from_mapping(bad_shortcut, root=tmp_path)

    bad_control = dict(raw)
    bad_control["controls"] = {**raw["controls"], "suite_result": {"known_good_passed": 1}}
    with pytest.raises(ValueError, match="controls has missing=.*extra=.*suite_result"):
        SupervisedRung6Config.from_mapping(bad_control, root=tmp_path)
