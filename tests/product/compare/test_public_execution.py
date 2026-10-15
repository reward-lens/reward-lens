"""Shared executable-Compare configuration boundary tests."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from reward_lens import contracts
from reward_lens.contracts import UsageError
from reward_lens.product.compare.execution import load_execution_config
from reward_lens.product.compare.rung4 import CpuControlledUpdateConfig
from reward_lens.product.compare.rung5 import R5_ENTRY_ID, default_cpu_short_fork_config
from reward_lens.product.compare.rung6 import R6_ENTRY_ID

_EXECUTION_EXTENSION = "https://reward-lens.github.io/extensions/compare-execution/1"
_R4_ENTRY_ID = "reward_statistics.controlled_update_response"
_R6_METHOD = "reward_statistics.repair_transfer@3.1.0"


def _unread_compare_subject(path: Path) -> str:
    """The projectless Compare subject is the configured source path it reads."""
    return contracts.digest({"unread_subject": {"command": "compare", "path": str(path)}})


def _write_short_fork_config(
    path: Path, *, target_value: float = 0.5, subject_ref: str | None = None
) -> Path:
    config = default_cpu_short_fork_config(
        subject_ref=subject_ref or _unread_compare_subject(path)
    ).to_mapping()
    config["target"]["value"] = target_value
    path.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_controlled_update_config(path: Path, *, subject_ref: str | None = None) -> Path:
    config = CpuControlledUpdateConfig(
        seed=37,
        max_completion_length=4,
        candidate_offsets=(0.5, -0.25),
        subject_ref=subject_ref or _unread_compare_subject(path),
    ).canonical_mapping()
    path.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _project(root: Path) -> Path:
    from reward_lens.examples import REGISTRY

    module_name, _, attribute = REGISTRY["code-reward"].partition(":")
    getattr(importlib.import_module(module_name), attribute)(root)
    return root


def _project_rung_five_subject(root: Path) -> str:
    from reward_lens.api._record import absent_record
    from reward_lens.store import Project
    from reward_lens.store.digests import method_set

    provisional = absent_record(command="compare", reproduce=[], path=root / "short-fork.json")
    methods = set(method_set(provisional.to_dict()))
    absence = provisional.measurement.reward_statistics[0].method
    methods.discard(f"{absence.id}@{absence.version}")
    methods.add(f"{R5_ENTRY_ID}@3.1.0")
    return Project.open(root).version(methods=tuple(sorted(methods))).digest()


def _project_rung_four_subject(root: Path) -> str:
    from reward_lens.api._record import absent_record
    from reward_lens.store import Project
    from reward_lens.store.digests import method_set

    provisional = absent_record(
        command="compare", reproduce=[], path=root / "controlled-update.json"
    )
    methods = set(method_set(provisional.to_dict()))
    absence = provisional.measurement.reward_statistics[0].method
    methods.discard(f"{absence.id}@{absence.version}")
    methods.add(f"{_R4_ENTRY_ID}@1.0.0")
    return Project.open(root).version(methods=tuple(sorted(methods))).digest()


def _project_rung_six_subject(root: Path) -> str:
    from reward_lens.api._record import absent_record
    from reward_lens.store import Project
    from reward_lens.store.digests import method_set

    provisional = absent_record(command="compare", reproduce=[], path=root / "transfer.json")
    methods = set(method_set(provisional.to_dict()))
    absence = provisional.measurement.reward_statistics[0].method
    methods.discard(f"{absence.id}@{absence.version}")
    methods.add(_R6_METHOD)
    return Project.open(root).version(methods=tuple(sorted(methods))).digest()


def _write_transfer_config(
    root: Path, *, subject_ref: str | None = None, qualified: bool = True
) -> Path:
    """Build one strict local R6 declaration with actual separate partitions."""
    from reward_lens.partitions import Partition, PartitionKind

    for name in ("r6-controls", "r6-original", "r6-family", "r6-exploit", "r6-checkpoint"):
        directory = root / name
        directory.mkdir(exist_ok=True)
        (directory / "case.py").write_text(name, encoding="utf-8")
    (root / "r6-controls" / "known-good.py").write_text("good", encoding="utf-8")
    (root / "r6-controls" / "known-wrong.py").write_text("wrong", encoding="utf-8")
    evaluator = root / "r6-evaluator.py"
    evaluator.write_text(
        "def score(task, response):\n    return 1.0 if response == task['reference'] else 0.0\n"
        if qualified
        else "def score(task, response):\n    return 1.0\n",
        encoding="utf-8",
    )

    def part(name: str, kind: PartitionKind) -> dict[str, str]:
        value = Partition.from_dir(name, kind, root / name, log_path=root / f"{name}.access.jsonl")
        return {
            "partition_id": name,
            "path": name,
            "digest": value.digest,
            "tree_version": "tree-v1",
        }

    source = evaluator.read_text(encoding="utf-8")

    def case(item: str, response: str) -> dict[str, object]:
        return {
            "item_id": item,
            "task": {"id": item, "prompt": "return reference", "reference": "correct"},
            "response": response,
        }

    targets = []
    for name, partition_class, family, exploit, checkpoint in (
        ("r6-family", "FAMILY", "family-b", "exploit-a", "checkpoint-a"),
        ("r6-exploit", "EXPLOIT", "family-a", "exploit-b", "checkpoint-a"),
        ("r6-checkpoint", "CHECKPOINT", "family-a", "exploit-a", "checkpoint-b"),
    ):
        targets.append(
            {
                **part(name, PartitionKind.HELD_OUT_TASKS),
                "partition_class": partition_class,
                "family": family,
                "exploit": exploit,
                "checkpoint": checkpoint,
                "declared_at": "2026-09-29T00:00:00Z",
                "cases": [case("case.py", "correct")],
            }
        )
    config = {
        "schema_version": 1,
        "subject_ref": subject_ref or _project_rung_six_subject(root),
        "owners": {
            "evaluator": "independent",
            "candidate": "candidate",
            "audited_grader": "reward",
            "audited_grader_source_digest": "sha256:" + "b" * 64,
        },
        "grader": {
            "version": "python-grader-v1",
            "path": "r6-evaluator.py",
            "source_digest": contracts.digest({"source": source, "entrypoint": "score"}),
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
            **part("r6-controls", PartitionKind.PROTECTED_SUITE),
            "known_good": [case("known-good.py", "correct")],
            "known_wrong": [case("known-wrong.py", "wrong")],
        },
        "original": {
            **part("r6-original", PartitionKind.HELD_OUT_TASKS),
            "family": "family-a",
            "exploit": "exploit-a",
            "checkpoint": "checkpoint-a",
            "case": case("case.py", "wrong"),
        },
        "targets": targets,
    }
    path = root / "transfer.json"
    path.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_sdk_rung_six_executes_strict_transfer_config_and_reopens_bundle(tmp_path: Path) -> None:
    from reward_lens import api

    root = _project(tmp_path / "transfer-project")
    config_path = _write_transfer_config(root)
    record = api.compare(
        api.CompareRequest(
            baseline="before",
            candidate="after",
            project=root,
            rungs=(6,),
            transfer_config=config_path,
            name="sdk-transfer",
        )
    )
    entry = next(entry for entry in record.entries() if entry.entry_id == R6_ENTRY_ID)
    identity = record.extensions[_EXECUTION_EXTENSION]["rung_6"]
    assert entry.kind == "check" and entry.check.passed is True
    assert entry.subject_ref == record.subject.version.digest == _project_rung_six_subject(root)
    assert entry.method.id == "reward_statistics.repair_transfer"
    assert entry.result["supervised_receipts"]["original"]["receipt_digest"].startswith("sha256:")
    assert identity["source_digest"].startswith("sha256:")
    reopened = contracts.Assay.model_validate_json(
        (root / "assays" / "sdk-transfer.assay.json").read_bytes()
    )
    assert reopened.assay_id == record.assay_id
    assert (root / "assays" / "sdk-transfer.assay.html").is_file()


def test_sdk_rung_six_persists_an_honest_unqualified_absence(tmp_path: Path) -> None:
    """A failed supervised control is retained as an R6 absence, never relabelled transfer."""
    from reward_lens import api

    root = _project(tmp_path / "unqualified-transfer-project")
    config_path = _write_transfer_config(root, qualified=False)
    record = api.compare(
        api.CompareRequest(
            baseline="before",
            candidate="after",
            project=root,
            rungs=(6,),
            transfer_config=config_path,
            name="sdk-transfer-unqualified",
        )
    )
    entry = next(entry for entry in record.entries() if entry.entry_id == R6_ENTRY_ID)
    assert entry.kind == "absence"
    assert entry.method.id == "reward_statistics.repair_transfer"
    assert entry.subject_ref == record.subject.version.digest == _project_rung_six_subject(root)
    assert "OUTCOME_UNQUALIFIED" in entry.absence.missing_access
    identity = record.extensions[_EXECUTION_EXTENSION]["rung_6"]
    assert identity["source_digest"].startswith("sha256:")
    qualification = identity["supervised_qualification"]
    assert qualification["state"] == "OUTCOME_UNQUALIFIED"
    assert len(qualification["control_receipts"]) == 2
    assert all(
        item["receipt_digest"].startswith("sha256:") for item in qualification["control_receipts"]
    )
    reopened = contracts.Assay.model_validate_json(
        (root / "assays" / "sdk-transfer-unqualified.assay.json").read_bytes()
    )
    reopened_entry = next(item for item in reopened.entries() if item.entry_id == R6_ENTRY_ID)
    assert reopened_entry.kind == "absence"
    assert reopened_entry.method.id == entry.method.id


def test_sdk_rung_six_refuses_wrong_subject_before_supervisor_access(tmp_path: Path) -> None:
    from reward_lens import api

    root = _project(tmp_path / "transfer-project")
    config_path = _write_transfer_config(root, subject_ref="sha256:" + "a" * 64)
    with pytest.raises(UsageError) as caught:
        api.compare(
            api.CompareRequest(
                baseline="before",
                candidate="after",
                project=root,
                rungs=(6,),
                transfer_config=config_path,
            )
        )
    assert caught.value.code == "RL0131"
    assert caught.value.context["field"] == "transfer_config.subject_ref"
    assert not (root / "r6-controls.access.jsonl").exists()
    assert not (root / "assays").exists()


def test_console_rung_six_reopens_project_bundle_from_a_different_cwd(tmp_path: Path) -> None:
    root = _project(tmp_path / "transfer-project")
    config_path = _write_transfer_config(root)
    command_cwd = tmp_path / "different-working-directory"
    command_cwd.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from reward_lens.cli.main import main; main()",
            "compare",
            "before",
            "after",
            "--rung",
            "6",
            "--project",
            str(root),
            "--transfer-config",
            str(config_path),
            "--name",
            "cli-transfer",
            "--format",
            "json",
        ],
        cwd=command_cwd,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 2, completed.stderr
    envelope = json.loads(completed.stdout)
    reopened = contracts.Assay.model_validate_json(
        (root / envelope["artifacts"]["assay"]).read_bytes()
    )
    assert any(entry.entry_id == R6_ENTRY_ID for entry in reopened.entries())
    assert (root / envelope["artifacts"]["report"]).is_file()


def _write_selection_stress_inputs(root: Path) -> tuple[Path, Path]:
    from reward_lens.bank import load_bank
    from reward_lens.product.compare.registry import Rung
    from reward_lens.product.compare.verb import _authoritative_factory_subject
    from reward_lens.store import Project

    bank_path = root / "selection-bank.jsonl"
    bank_path.write_text(
        "\n".join(
            (
                json.dumps({"name": "selection-bank"}),
                json.dumps(
                    {
                        "task_id": "task",
                        "responses": [
                            {"text": "correct-low-a"},
                            {"text": "correct-low-b"},
                            {"text": "wrong-high-a"},
                            {"text": "wrong-high-b"},
                        ],
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    bank = load_bank(bank_path, name="selection-bank")
    responses = bank.responses_for(bank.tasks[0])
    (root / "controls").mkdir()
    (root / "controls" / "known-good").write_text("control", encoding="utf-8")
    (root / "controls" / "known-wrong").write_text("control", encoding="utf-8")
    (root / "targets").mkdir()
    for response in responses:
        (root / "targets" / response.digest).write_text(response.digest, encoding="utf-8")
    (root / "selection-selector.py").write_text(
        "def score(task, response):\n    return {'correct-low-a': 1, 'correct-low-b': 2, 'wrong-high-a': 3, 'wrong-high-b': 4}[response]\n",
        encoding="utf-8",
    )
    evaluator = root / "selection-outcome.py"
    evaluator.write_text(
        "def score(task, response):\n    return 1.0 if response in ('control-good', 'correct-low-a', 'correct-low-b') else 0.0\n",
        encoding="utf-8",
    )
    source_digest = contracts.digest(
        {"source": evaluator.read_text(encoding="utf-8"), "entrypoint": "score"}
    )
    subject = _authoritative_factory_subject(
        source=bank_path, project=Project.open(root), rungs=(Rung.SELECTION_STRESS,)
    ).version.digest
    config_path = root / "selection-stress.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "selector": {"path": "selection-selector.py", "entrypoint": "score"},
                "outcome": {
                    "check": {
                        "owner": "independent",
                        "source_digest": source_digest,
                        "interface": "reward_lens.graders.python_grader.PythonGrader.score/v1",
                        "partition_id": "control-suite",
                    },
                    "evaluator": {"path": "selection-outcome.py", "entrypoint": "score"},
                    "controls": {
                        "partition_path": "controls",
                        "partition_id": "control-suite",
                        "known_good": [
                            {
                                "item_id": "known-good",
                                "task": {"id": "control-good", "prompt": "control"},
                                "response": "control-good",
                            }
                        ],
                        "known_wrong": [
                            {
                                "item_id": "known-wrong",
                                "task": {"id": "control-wrong", "prompt": "control"},
                                "response": "control-wrong",
                            }
                        ],
                    },
                    "target": {
                        "partition_path": "targets",
                        "partition_id": "target-suite",
                        "response_items": {
                            response.digest: response.digest for response in responses
                        },
                        "tasks": {"task": {"id": "task", "prompt": "fixture task"}},
                    },
                    "candidate_owner": "candidate",
                    "audited_grader": {"owner": "reward", "source_digest": "sha256:" + "b" * 64},
                    "subject_ref": subject,
                },
                "ladder": [2],
                "bootstrap_resamples": 10,
                "sandbox_tier": "L1",
                "limits": {
                    "wall_s": 5,
                    "cpu_s": 5,
                    "memory_bytes": 1073741824,
                    "processes": 1,
                    "stdout_bytes": 4096,
                    "artifact_bytes": 4096,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return bank_path, config_path


def test_sdk_rung_two_executes_supervised_inputs_and_reopens_bundle(tmp_path: Path) -> None:
    from reward_lens import api

    root = _project(tmp_path / "selection-project")
    bank_path, config_path = _write_selection_stress_inputs(root)
    record = api.compare(
        api.CompareRequest(
            baseline="before",
            candidate="after",
            bank=bank_path,
            project=root,
            rungs=(2,),
            selection_config=config_path,
            name="sdk-selection",
        )
    )
    entry = next(
        entry
        for entry in record.entries()
        if entry.entry_id == "reward_statistics.selection_stress"
    )
    identity = record.extensions[_EXECUTION_EXTENSION]["rung_2"]
    assert (
        entry.subject_ref
        == record.subject.version.digest
        == identity["declaration"]["validated_subject"]
    )
    assert entry.result["supervised_target_receipts"]
    reopened = contracts.Assay.model_validate(
        json.loads((root / "assays" / "sdk-selection.assay.json").read_text(encoding="utf-8"))
    )
    assert reopened.assay_id == record.assay_id
    assert (root / "assays" / "sdk-selection.assay.html").is_file()


def test_console_rung_two_reopens_project_bundle_from_a_different_cwd(tmp_path: Path) -> None:
    root = _project(tmp_path / "selection-project")
    bank_path, config_path = _write_selection_stress_inputs(root)
    command_cwd = tmp_path / "different-working-directory"
    command_cwd.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from reward_lens.cli.main import main; main()",
            "compare",
            "before",
            "after",
            "--rung",
            "2",
            "--bank",
            str(bank_path),
            "--project",
            str(root),
            "--selection-config",
            str(config_path),
            "--name",
            "cli-selection",
            "--format",
            "json",
        ],
        cwd=command_cwd,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert completed.returncode == 2, completed.stderr
    envelope = json.loads(completed.stdout)
    assert envelope["artifacts"] == {
        "assay": "assays/cli-selection.assay.json",
        "report": "assays/cli-selection.assay.html",
    }
    reopened = contracts.Assay.model_validate(
        json.loads((root / envelope["artifacts"]["assay"]).read_text(encoding="utf-8"))
    )
    assert any(
        entry.entry_id == "reward_statistics.selection_stress" for entry in reopened.entries()
    )
    assert (root / envelope["artifacts"]["report"]).is_file()


def _replace_selection_evaluator(config_path: Path, source: str) -> None:
    evaluator = config_path.parent / "selection-outcome.py"
    evaluator.write_text(source, encoding="utf-8")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["outcome"]["check"]["source_digest"] = contracts.digest(
        {"source": source, "entrypoint": "score"}
    )
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")


def test_sdk_rung_two_persists_an_unqualified_control_absence(tmp_path: Path) -> None:
    from reward_lens import api

    root = _project(tmp_path / "selection-project")
    bank_path, config_path = _write_selection_stress_inputs(root)
    _replace_selection_evaluator(config_path, "def score(task, response):\n    return 1.0\n")
    record = api.compare(
        api.CompareRequest(
            baseline="before",
            candidate="after",
            bank=bank_path,
            project=root,
            rungs=(2,),
            selection_config=config_path,
            name="sdk-unqualified",
        )
    )
    entry = next(
        entry
        for entry in record.entries()
        if entry.entry_id == "reward_statistics.selection_stress"
    )
    assert entry.kind == "absence"
    assert entry.method.id == "reward_statistics.selection_stress"
    assert (
        contracts.Assay.model_validate_json(
            (root / "assays" / "sdk-unqualified.assay.json").read_bytes()
        ).assay_id
        == record.assay_id
    )


def test_console_rung_two_persists_all_ambiguous_absence_from_a_different_cwd(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path / "selection-project")
    bank_path, config_path = _write_selection_stress_inputs(root)
    _replace_selection_evaluator(
        config_path,
        "def score(task, response):\n    return 1.0 if response == 'control-good' else 0.0 if response == 'control-wrong' else 0.5\n",
    )
    command_cwd = tmp_path / "different-working-directory"
    command_cwd.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from reward_lens.cli.main import main; main()",
            "compare",
            "before",
            "after",
            "--rung",
            "2",
            "--bank",
            str(bank_path),
            "--project",
            str(root),
            "--selection-config",
            str(config_path),
            "--name",
            "cli-ambiguous",
            "--format",
            "json",
        ],
        cwd=command_cwd,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 2, completed.stderr
    reopened = contracts.Assay.model_validate(
        json.loads((root / "assays" / "cli-ambiguous.assay.json").read_text(encoding="utf-8"))
    )
    entry = next(
        entry
        for entry in reopened.entries()
        if entry.entry_id == "reward_statistics.selection_stress"
    )
    assert entry.kind == "absence"
    assert entry.method.id == "reward_statistics.selection_stress"


def test_execution_config_binds_raw_bytes_and_canonical_object_identity(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"seed":31,"limits":{"cpu_threads":1}}\n', encoding="utf-8")

    first = load_execution_config(path, field="short_fork_config")
    path.write_text('{\n  "limits": {"cpu_threads": 1}, "seed": 31\n}\n', encoding="utf-8")
    second = load_execution_config(path, field="short_fork_config")

    assert first.payload == second.payload == {"seed": 31, "limits": {"cpu_threads": 1}}
    assert first.source_digest != second.source_digest
    assert first.canonical_digest == second.canonical_digest
    assert second.identity()["source_path"] == str(path)


@pytest.mark.parametrize(
    "contents",
    (
        '["not", "an object"]',
        '{"seed": 1, "seed": 2}',
        '{"limit": NaN}',
    ),
)
def test_execution_config_refuses_ambiguous_or_non_object_json(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(UsageError) as caught:
        load_execution_config(path, field="controlled_update_config")
    assert caught.value.code == "RL0131"
    assert caught.value.context["field"] == "controlled_update_config"


def test_execution_config_refuses_a_missing_path() -> None:
    with pytest.raises(UsageError) as caught:
        load_execution_config(None, field="selection_config")
    assert caught.value.code == "RL0131"


def test_sdk_compare_executes_rung_five_from_its_strict_cpu_config_and_reopens_bundle(
    tmp_path: Path,
) -> None:
    from reward_lens import api

    config_path = _write_short_fork_config(tmp_path / "short-fork.json")
    record = api.compare(
        api.CompareRequest(
            baseline="before",
            candidate="after",
            rungs=(5,),
            short_fork_config=config_path,
            name="sdk-short-fork",
        )
    )

    entry = next(entry for entry in record.entries() if entry.entry_id == R5_ENTRY_ID)
    identity = record.extensions[_EXECUTION_EXTENSION]["rung_5"]
    assert entry.scope == "short_fork_behavior"
    assert (
        entry.subject_ref == record.subject.version.digest == _unread_compare_subject(config_path)
    )
    assert entry.check.passed is True
    assert entry.result["execution_config"] == identity
    assert identity["source_path"] == str(config_path)
    assert identity["source_digest"].startswith("sha256:")
    assert identity["canonical_config_digest"].startswith("sha256:")
    assert Path(identity["artifact_dir"]).is_dir()
    assert (Path(identity["artifact_dir"]) / "cpu-conformance.json").is_file()

    assay_path = tmp_path / "assays" / "sdk-short-fork.assay.json"
    reopened = contracts.Assay.model_validate(json.loads(assay_path.read_text(encoding="utf-8")))
    reopened_entry = next(entry for entry in reopened.entries() if entry.entry_id == R5_ENTRY_ID)
    assert reopened.assay_id == record.assay_id
    assert reopened_entry.method.params_digest == entry.method.params_digest
    assert (tmp_path / "assays" / "sdk-short-fork.assay.html").is_file()


def test_sdk_rung_five_refuses_a_valid_but_wrong_factory_subject_before_artifacts(
    tmp_path: Path,
) -> None:
    from reward_lens import api

    config_path = _write_short_fork_config(
        tmp_path / "wrong-subject.json", subject_ref="sha256:" + "a" * 64
    )
    with pytest.raises(UsageError) as caught:
        api.compare(
            api.CompareRequest(
                baseline="before",
                candidate="after",
                rungs=(5,),
                short_fork_config=config_path,
            )
        )

    assert caught.value.code == "RL0131"
    assert caught.value.context["field"] == "short_fork_config.subject_ref"
    assert not (tmp_path / "assays").exists()


def test_sdk_rung_five_preserves_the_project_bound_factory_subject(tmp_path: Path) -> None:
    from reward_lens import api
    from reward_lens.store import Project

    root = _project(tmp_path / "project")
    config_path = _write_short_fork_config(
        root / "short-fork.json", subject_ref=_project_rung_five_subject(root)
    )
    record = api.compare(
        api.CompareRequest(
            baseline="before",
            candidate="after",
            project=root,
            rungs=(5,),
            short_fork_config=config_path,
            name="project-short-fork",
        )
    )

    entry = next(entry for entry in record.entries() if entry.entry_id == R5_ENTRY_ID)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fixture = json.loads(
        (
            Path(entry.result["execution_config"]["artifact_dir"]) / "runtime-fixture" / "fork.json"
        ).read_text(encoding="utf-8")
    )
    reopened = Project.open(root).open_record("project-short-fork")
    reopened_entry = next(entry for entry in reopened.entries() if entry.entry_id == R5_ENTRY_ID)
    assert (
        config["subject_ref"]
        == fixture["subject_ref"]
        == entry.subject_ref
        == reopened_entry.subject_ref
        == record.subject.version.digest
        == reopened.subject.version.digest
    )


def test_console_rung_five_refuses_wrong_factory_subject_before_artifacts(tmp_path: Path) -> None:
    config_path = _write_short_fork_config(
        tmp_path / "cli-wrong-subject.json", subject_ref="sha256:" + "a" * 64
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from reward_lens.cli.main import main; main()",
            "compare",
            "before",
            "after",
            "--rung",
            "5",
            "--short-fork-config",
            str(config_path),
            "--format",
            "json",
        ],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert completed.returncode == 4
    assert (
        json.loads(completed.stdout)["error"]["context"]["field"] == "short_fork_config.subject_ref"
    )
    assert not (tmp_path / "assays").exists()


def test_sdk_rung_five_config_changes_measurement_and_sealed_method_identity(
    tmp_path: Path,
) -> None:
    from reward_lens import api

    first_config = _write_short_fork_config(tmp_path / "first.json", target_value=0.5)
    second_config = _write_short_fork_config(tmp_path / "second.json", target_value=0.25)
    first = api.compare(
        api.CompareRequest(
            baseline="before",
            candidate="after",
            rungs=(5,),
            short_fork_config=first_config,
            name="first",
        )
    )
    second = api.compare(
        api.CompareRequest(
            baseline="before",
            candidate="after",
            rungs=(5,),
            short_fork_config=second_config,
            name="second",
        )
    )
    first_entry = next(entry for entry in first.entries() if entry.entry_id == R5_ENTRY_ID)
    second_entry = next(entry for entry in second.entries() if entry.entry_id == R5_ENTRY_ID)

    assert first.assay_id != second.assay_id
    assert first_entry.method.params_digest != second_entry.method.params_digest
    assert (
        first_entry.result["execution_config"]["source_digest"]
        != second_entry.result["execution_config"]["source_digest"]
    )
    assert (
        first_entry.result["cpu_conformance"]["target_measure"]["values"]
        != second_entry.result["cpu_conformance"]["target_measure"]["values"]
    )


def test_console_compare_executes_rung_five_and_reports_the_saved_bundle(tmp_path: Path) -> None:
    config_path = _write_short_fork_config(tmp_path / "cli-short-fork.json")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from reward_lens.cli.main import main; main()",
            "compare",
            "before",
            "after",
            "--rung",
            "5",
            "--short-fork-config",
            str(config_path),
            "--name",
            "cli-short-fork",
            "--format",
            "json",
        ],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert completed.returncode == 2, completed.stderr
    envelope = json.loads(completed.stdout)
    assert envelope["artifacts"]["assay"] == "assays/cli-short-fork.assay.json"
    saved = contracts.Assay.model_validate(
        json.loads((tmp_path / "assays" / "cli-short-fork.assay.json").read_text(encoding="utf-8"))
    )
    assert any(entry.entry_id == R5_ENTRY_ID for entry in saved.entries())


def test_sdk_rung_five_refuses_an_unbounded_execution_config_before_running(tmp_path: Path) -> None:
    from reward_lens import api

    config_path = _write_short_fork_config(tmp_path / "invalid-short-fork.json")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["limits"]["cpu_threads"] = 2
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UsageError) as caught:
        api.compare(
            api.CompareRequest(
                baseline="before", candidate="after", rungs=(5,), short_fork_config=config_path
            )
        )
    assert caught.value.code == "RL0131"
    assert caught.value.context["field"] == "short_fork_config"
    assert not (tmp_path / "assays").exists()


def test_sdk_compare_executes_rung_four_from_its_declared_cpu_config_and_reopens_bundle(
    tmp_path: Path,
) -> None:
    from reward_lens import api

    config_path = _write_controlled_update_config(tmp_path / "controlled-update.json")
    record = api.compare(
        api.CompareRequest(
            baseline="before",
            candidate="after",
            rungs=(4,),
            controlled_update_config=config_path,
            name="sdk-controlled-update",
        )
    )

    entry = next(entry for entry in record.entries() if entry.entry_id == _R4_ENTRY_ID)
    identity = record.extensions[_EXECUTION_EXTENSION]["rung_4"]
    assert entry.scope == "controlled_update_response"
    assert (
        entry.subject_ref == record.subject.version.digest == _unread_compare_subject(config_path)
    )
    assert entry.check.passed is True
    assert entry.result["execution_config"] == identity
    assert entry.result["trace_record"]["subject_ref"] == entry.subject_ref
    assert identity["source_path"] == str(config_path)
    assert identity["source_digest"].startswith("sha256:")
    assert identity["canonical_config_digest"].startswith("sha256:")
    assert (Path(identity["artifact_dir"]) / "controlled-update-execution.json").is_file()

    assay_path = tmp_path / "assays" / "sdk-controlled-update.assay.json"
    reopened = contracts.Assay.model_validate(json.loads(assay_path.read_text(encoding="utf-8")))
    reopened_entry = next(entry for entry in reopened.entries() if entry.entry_id == _R4_ENTRY_ID)
    assert reopened.assay_id == record.assay_id
    assert reopened_entry.method.params_digest == entry.method.params_digest
    assert reopened_entry.result["trace_record"] == entry.result["trace_record"]


def test_sdk_rung_four_refuses_wrong_declared_subject_before_artifacts(tmp_path: Path) -> None:
    from reward_lens import api

    config_path = _write_controlled_update_config(
        tmp_path / "wrong-controlled-update.json", subject_ref="sha256:" + "a" * 64
    )
    with pytest.raises(UsageError) as caught:
        api.compare(
            api.CompareRequest(
                baseline="before",
                candidate="after",
                rungs=(4,),
                controlled_update_config=config_path,
            )
        )

    assert caught.value.code == "RL0131"
    assert caught.value.context["field"] == "controlled_update_config.subject_ref"
    assert not (tmp_path / "assays").exists()


def test_sdk_rung_four_preserves_project_bound_subject_through_factory_and_reopen(
    tmp_path: Path,
) -> None:
    from reward_lens import api
    from reward_lens.store import Project

    root = _project(tmp_path / "project")
    config_path = _write_controlled_update_config(
        root / "controlled-update.json", subject_ref=_project_rung_four_subject(root)
    )
    record = api.compare(
        api.CompareRequest(
            baseline="before",
            candidate="after",
            project=root,
            rungs=(4,),
            controlled_update_config=config_path,
            name="project-controlled-update",
        )
    )

    entry = next(entry for entry in record.entries() if entry.entry_id == _R4_ENTRY_ID)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    artifact = json.loads(
        (
            Path(entry.result["execution_config"]["artifact_dir"])
            / "controlled-update-execution.json"
        ).read_text(encoding="utf-8")
    )
    reopened = Project.open(root).open_record("project-controlled-update")
    reopened_entry = next(entry for entry in reopened.entries() if entry.entry_id == _R4_ENTRY_ID)
    assert (
        config["subject_ref"]
        == artifact["declared_subject_ref"]
        == entry.result["trace_record"]["subject_ref"]
        == entry.subject_ref
        == reopened_entry.subject_ref
        == record.subject.version.digest
        == reopened.subject.version.digest
    )


def test_console_compare_executes_rung_four_and_reports_the_saved_bundle(tmp_path: Path) -> None:
    config_path = _write_controlled_update_config(tmp_path / "cli-controlled-update.json")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from reward_lens.cli.main import main; main()",
            "compare",
            "before",
            "after",
            "--rung",
            "4",
            "--controlled-update-config",
            str(config_path),
            "--name",
            "cli-controlled-update",
            "--format",
            "json",
        ],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert completed.returncode == 2, completed.stderr
    envelope = json.loads(completed.stdout)
    assert envelope["artifacts"]["assay"] == "assays/cli-controlled-update.assay.json"
    saved = contracts.Assay.model_validate(
        json.loads(
            (tmp_path / "assays" / "cli-controlled-update.assay.json").read_text(encoding="utf-8")
        )
    )
    assert any(entry.entry_id == _R4_ENTRY_ID for entry in saved.entries())


def test_console_rung_four_reports_bundle_beside_config_from_a_different_cwd(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "declared-config"
    config_root.mkdir()
    config_path = _write_controlled_update_config(config_root / "controlled-update.json")
    command_cwd = tmp_path / "different-working-directory"
    command_cwd.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from reward_lens.cli.main import main; main()",
            "compare",
            "before",
            "after",
            "--rung",
            "4",
            "--controlled-update-config",
            str(config_path),
            "--name",
            "cli-controlled-update-different-cwd",
            "--format",
            "json",
        ],
        cwd=command_cwd,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert completed.returncode == 2, completed.stderr
    envelope = json.loads(completed.stdout)
    assert envelope["artifacts"] == {
        "assay": "assays/cli-controlled-update-different-cwd.assay.json",
        "report": "assays/cli-controlled-update-different-cwd.assay.html",
    }
    assay_path = config_root / envelope["artifacts"]["assay"]
    report_path = config_root / envelope["artifacts"]["report"]
    saved = contracts.Assay.model_validate(json.loads(assay_path.read_text(encoding="utf-8")))
    assert any(entry.entry_id == _R4_ENTRY_ID for entry in saved.entries())
    assert report_path.is_file()
    assert not (command_cwd / "assays").exists()


def test_console_rung_four_refuses_wrong_declared_subject_before_artifacts(tmp_path: Path) -> None:
    config_path = _write_controlled_update_config(
        tmp_path / "cli-wrong-controlled-update.json", subject_ref="sha256:" + "a" * 64
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from reward_lens.cli.main import main; main()",
            "compare",
            "before",
            "after",
            "--rung",
            "4",
            "--controlled-update-config",
            str(config_path),
            "--format",
            "json",
        ],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert completed.returncode == 4
    assert (
        json.loads(completed.stdout)["error"]["context"]["field"]
        == "controlled_update_config.subject_ref"
    )
    assert not (tmp_path / "assays").exists()


def _prepared_coding_config(tmp_path: Path) -> Path:
    """Use the real prepared checkpoint and copied coding inputs, without loading a model."""
    import shutil

    from reward_lens.product.compare import rung4

    prepared = os.environ.get("REWARD_LENS_CODING_PREPARATION")
    if not prepared:
        pytest.skip("requires the separately prepared, byte-pinned actual coding input fixture")
    original = Path(prepared)
    manifest = json.loads(original.read_text())
    shutil.copytree(
        original.parent / "inputs",
        tmp_path / "inputs",
        ignore=shutil.ignore_patterns("assays", "__pycache__"),
    )
    # Generated interpreter caches are neither copied nor sealed source inputs.
    # Retain every evaluator body, task/test fixture and all other declared bytes.
    manifest["input_files"] = {
        relative: wanted
        for relative, wanted in manifest["input_files"].items()
        if "__pycache__" not in Path(relative).parts and Path(relative).suffix != ".pyc"
    }
    source = Path(rung4.__file__).resolve().parents[4]
    manifest["source_root"] = str(source)
    manifest["source_files"] = {
        relative: rung4._file_digest(source / "src/reward_lens" / relative)
        for relative in manifest["source_files"]
    }
    config_path = tmp_path / "controlled-update.json"
    manifest["candidate_subject_ref"] = _project_rung_four_subject(tmp_path / "inputs/text-penalty")
    path = tmp_path / "PREPARED_TRIAL.json"
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    config = {
        "profile": rung4.CODING_UPDATE_PROFILE,
        "prepared_manifest": str(path),
        "prepared_manifest_digest": rung4._file_digest(path),
        "subject_ref": manifest["candidate_subject_ref"],
    }
    config_path.write_text(json.dumps(config) + "\n")
    return config_path


def _controlled_coding_loader(config, manifest, artifact_dir):
    """A consumer fixture with explicit test provenance; no CUDA/training claim."""
    from reward_lens.learning.contracts import EIGHT_OBJECTS, TrainerProfile
    from reward_lens.learning.intervention.swap import ControlledUpdateResult, Intervention
    from reward_lens.product.compare.rung4 import CODING_UPDATE_PROFILE

    base = (0.0, 1.0, 0.0, 1.0)
    offsets = (-0.5, 0.0, -0.5, 0.0)
    observations = [
        {
            "task_id": ident,
            "completion": "def solution(): return 0",
            "baseline": b,
            "candidate": b + offset,
            "envelopes": {
                "baseline": {"verdict": "scored", "score": b},
                "text-penalty": {"verdict": "scored", "score": b + offset},
            },
            "independent_frozen_response_outcome": {"verdict": "scored", "score": 0.0},
        }
        for ident, b, offset in zip(
            ("task-01", "task-01", "task-02", "task-02"), base, offsets, strict=True
        )
    ]
    arms = []
    for name, factor in (("null", 0), ("candidate", 1), ("placebo", -1)):
        applied = tuple(factor * offset for offset in offsets)
        value = {
            "batch_digest": "batch",
            "batch_bytes": b"frozen-coding-batch",
            "scored_replay_digest": "replay",
            "loss_input_digest": "loss-input",
            "intervention": Intervention(name, applied),
            "reward_witness": (
                {
                    "source_values": base,
                    "returned_values": tuple(
                        b + offset for b, offset in zip(base, applied, strict=True)
                    ),
                },
            ),
            "loss": 0.1,
            "gradient_norm": 1.0,
            "optimizer_update_norm": 1.0,
            "advantages": (0.5, -0.5, 0.5, -0.5),
            "changed_private_fields": ("_step",),
        }
        arms.append(
            (
                name,
                {
                    "value": value,
                    "during_digest": "changed",
                    "restored_digest": "same",
                    "changed_objects": (),
                },
            )
        )
    return (
        ControlledUpdateResult(
            "controlled_update_response",
            "mechanism conformance; not a behavioural result",
            "same",
            "same",
            ("null", "candidate", "placebo"),
            tuple(arms),
            TrainerProfile("1.10.0", EIGHT_OBJECTS, CODING_UPDATE_PROFILE),
        ),
        observations,
        "controlled-test-loader",
    )


def test_public_coding_profile_retains_bound_inputs_and_scoring_receipts(tmp_path, monkeypatch):
    from reward_lens import api
    from reward_lens.product.compare import rung4

    config = _prepared_coding_config(tmp_path)
    monkeypatch.setattr(rung4, "_run_coding_update", _controlled_coding_loader)
    record = api.compare(
        api.CompareRequest(
            baseline="baseline",
            candidate="text-penalty",
            rungs=(4,),
            project=tmp_path / "inputs/text-penalty",
            controlled_update_config=config,
            name="coding-consumer",
        )
    )
    entry = next(entry for entry in record.entries() if entry.entry_id == _R4_ENTRY_ID)
    coding = entry.result["coding_execution"]
    assert coding["execution_class"] == "controlled_test_loader"
    assert coding["device"] == "controlled-test-loader"
    assert len(coding["reward_observations"]) == 4
    assert coding["domain"]["task_count"] == 40
    assert coding["domain"]["reference_response_count"] == 120
    assert entry.check.passed and entry.scope == "controlled_update_response"
    assert "not a behavioural result" in entry.result["claim_boundary"]
    reopened = contracts.Assay.model_validate_json(
        (tmp_path / "inputs/text-penalty/assays/coding-consumer.assay.json").read_text()
    )
    persisted = next(entry for entry in reopened.entries() if entry.entry_id == _R4_ENTRY_ID)
    assert persisted.result["coding_execution"] == coding
    assert reopened.assay_id == record.assay_id
    artifact = json.loads(
        Path(entry.result["trace_record"]["source_binding"]["artifact_path"]).read_text()
    )
    assert artifact["coding_execution"] == coding
    assert record.extensions[_EXECUTION_EXTENSION]["rung_4"]["source_path"] == str(config)


def test_public_coding_profile_refuses_unavailable_cuda_without_assay(tmp_path, monkeypatch):
    import torch

    from reward_lens import api

    config = _prepared_coding_config(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(contracts.CapabilityUnavailable, match="CUDA is unavailable"):
        api.compare(
            api.CompareRequest(
                baseline="baseline",
                candidate="text-penalty",
                rungs=(4,),
                project=tmp_path / "inputs/text-penalty",
                controlled_update_config=config,
                name="coding-no-cuda",
            )
        )
    assert not (tmp_path / "inputs/text-penalty/assays").exists()


@pytest.mark.parametrize(
    "fault",
    (
        "missing_checkpoint",
        "protected_grading",
        "protected_control",
        "protected_outcome",
        "null",
        "restoration",
        "replay",
    ),
)
def test_public_coding_profile_refuses_broken_bindings_and_controls(tmp_path, monkeypatch, fault):
    from reward_lens import api
    from reward_lens.learning.intervention.swap import ControlledUpdateResult, Intervention
    from reward_lens.product.compare import rung4

    config_path = _prepared_coding_config(tmp_path)
    config = json.loads(config_path.read_text())
    path = Path(config["prepared_manifest"])
    manifest = json.loads(path.read_text())
    if fault == "missing_checkpoint":
        manifest["checkpoint"] = str(tmp_path / "missing" / Path(manifest["checkpoint"]).name)
    elif fault == "protected_grading":
        (tmp_path / "inputs/baseline/outcome/test_solution.py").write_text(
            "def test_wrong(): pass\n"
        )
    elif fault == "protected_control":
        control = tmp_path / "inputs/baseline/outcome/known_wrong/task-01.py"
        control.write_text("def wrong_control(): return 0\n")
        manifest["input_files"]["inputs/baseline/outcome/known_wrong/task-01.py"] = (
            rung4._file_digest(control)
        )
    else:

        def broken(config, manifest, artifact_dir):
            result, observations, device = _controlled_coding_loader(config, manifest, artifact_dir)
            if fault == "protected_outcome":
                observations[0]["independent_frozen_response_outcome"] = {
                    "verdict": "timeout",
                    "score": None,
                }
            arms = list(result.arms)
            name, witness = arms[0 if fault == "null" else 2]
            witness = dict(witness)
            if fault == "null":
                witness["value"] = {
                    **witness["value"],
                    "intervention": Intervention("null", (1.0, 1.0, 1.0, 1.0)),
                }
            elif fault == "restoration":
                witness["restored_digest"] = "not-the-start"
            elif fault == "replay":
                witness["value"] = {**witness["value"], "scored_replay_digest": "another-replay"}
            arms[0 if fault == "null" else 2] = (name, witness)
            return (
                ControlledUpdateResult(
                    result.scope,
                    result.statement,
                    result.start_digest,
                    result.final_digest,
                    result.arm_order,
                    tuple(arms),
                    result.profile,
                ),
                observations,
                device,
            )

        monkeypatch.setattr(rung4, "_run_coding_update", broken)
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    config["prepared_manifest_digest"] = rung4._file_digest(path)
    config_path.write_text(json.dumps(config))
    with pytest.raises(UsageError, match="coding (preparation|update evidence) refused"):
        api.compare(
            api.CompareRequest(
                baseline="baseline",
                candidate="text-penalty",
                rungs=(4,),
                project=tmp_path / "inputs/text-penalty",
                controlled_update_config=config_path,
                name="broken-coding",
            )
        )
    assert not (tmp_path / "inputs/text-penalty/assays").exists()


def _prepared_coding_short_fork_config(tmp_path):
    from reward_lens.product.compare import rung5

    config = _prepared_coding_config(tmp_path)
    declaration = json.loads(config.read_text())
    project = tmp_path / "inputs/text-penalty"
    path = project / "short-fork.json"
    raw = {
        "profile": rung5.CODING_SHORT_FORK_PROFILE,
        "prepared_manifest": declaration["prepared_manifest"],
        "prepared_manifest_digest": declaration["prepared_manifest_digest"],
        "subject_ref": _project_rung_five_subject(project),
    }
    path.write_text(json.dumps(raw))
    return path, project


def _controlled_coding_short_fork_loader(config, manifest, project, artifact_dir):
    from tests.product.compare.test_rung5 import _coding_payload

    return _coding_payload()


def test_public_coding_short_fork_retains_canonical_result_and_missing_floor(tmp_path, monkeypatch):
    from reward_lens import api
    from reward_lens.product.compare import rung5

    config, project = _prepared_coding_short_fork_config(tmp_path)
    monkeypatch.setattr(rung5, "_run_coding_short_fork", _controlled_coding_short_fork_loader)
    record = api.compare(
        api.CompareRequest(
            baseline="baseline",
            candidate="text-penalty",
            project=project,
            rungs=(5,),
            short_fork_config=config,
            name="coding-short-fork-consumer",
        )
    )
    entry = next(entry for entry in record.entries() if entry.entry_id == R5_ENTRY_ID)
    assert entry.result["coding_short_fork"]["execution_class"] == "controlled_test_loader"
    assert entry.result["capability_floor"]["state"] == "NOT_MEASURED"
    assert entry.result["capability_floor"]["held"] is None
    assert entry.subject_ref == record.subject.version.digest
    reopened = contracts.Assay.model_validate_json(
        (project / "assays/coding-short-fork-consumer.assay.json").read_text()
    )
    assert reopened.assay_id == record.assay_id
    reopened_entry = next(entry for entry in reopened.entries() if entry.entry_id == R5_ENTRY_ID)
    assert reopened_entry.result["coding_short_fork"] == entry.result["coding_short_fork"]
    identity = record.extensions[_EXECUTION_EXTENSION]["rung_5"]
    assert identity["source_path"] == str(config)
    assert (Path(identity["artifact_dir"]) / "coding-short-fork-execution.json").is_file()


def test_public_coding_short_fork_refuses_unavailable_cuda_before_artifacts(tmp_path, monkeypatch):
    import torch

    from reward_lens import api

    config, project = _prepared_coding_short_fork_config(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(contracts.CapabilityUnavailable, match="CUDA is unavailable") as caught:
        api.compare(
            api.CompareRequest(
                baseline="baseline",
                candidate="text-penalty",
                project=project,
                rungs=(5,),
                short_fork_config=config,
                name="coding-short-fork-no-cuda",
            )
        )
    assert caught.value.context["profile"] == "m3-qwen2-text-penalty-short-fork-v1"
    assert not (project / "assays").exists()


def test_public_coding_short_fork_refuses_changed_outcome_without_assay(tmp_path, monkeypatch):
    from reward_lens import api
    from reward_lens.product.compare import rung5

    config, project = _prepared_coding_short_fork_config(tmp_path)

    def broken(*args):
        payload = _controlled_coding_short_fork_loader(*args)
        payload["arms"][2]["outcomes"][0]["envelope"]["verdict"] = "timeout"
        return payload

    monkeypatch.setattr(rung5, "_run_coding_short_fork", broken)
    with pytest.raises(UsageError, match="independent outcomes are unresolved"):
        api.compare(
            api.CompareRequest(
                baseline="baseline",
                candidate="text-penalty",
                project=project,
                rungs=(5,),
                short_fork_config=config,
                name="coding-short-fork-broken",
            )
        )
    assert not (project / "assays").exists()
