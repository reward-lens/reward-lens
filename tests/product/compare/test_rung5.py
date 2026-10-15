"""Record integrity tests for compare rung five's CPU conformance route."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from reward_lens.learning.contracts import EIGHT_OBJECTS, TrainerProfile
from reward_lens.learning.intervention.clamp import (
    ClampSpecification,
    ShortForkArm,
    build_clamp_callback,
    fixture_digest,
    run_short_fork,
)
from reward_lens.product.compare.registry import Rung, get
from reward_lens.product.compare.rung5 import (
    R5_ENTRY_ID,
    CPUConformanceRecordInvalid,
    CPUShortForkConfig,
    Rung5Input,
    append_short_fork_assay,
    build_cpu_short_fork,
    compare_verified_short_fork,
    default_cpu_short_fork_config,
    load_cpu_conformance_record,
    run,
    write_cpu_conformance_record,
)

FIXTURE = Path(__file__).parents[2] / "fixtures" / "learning" / "cpu_toy" / "fork"
FIXTURE_INPUTS = json.loads((FIXTURE / "fork.json").read_text())


def test_rung5_public_factory_executes_the_explicit_cpu_config_and_returns_a_retained_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reward_lens.contracts import Assay
    from reward_lens.store import Project
    from reward_lens.store.digests import method_set
    from tests.store.conftest import build_record, write_project
    from tests.store.test_project import align

    project = Project.open(write_project(tmp_path / "factory-project"))
    methods = tuple(sorted((*method_set(build_record()), f"{R5_ENTRY_ID}@3.1.0")))
    base = Assay.model_validate(align(build_record(), project, methods=methods))
    config = default_cpu_short_fork_config(subject_ref=base.subject.version.digest)
    assert CPUShortForkConfig.from_mapping(config.to_mapping()) == config

    # The test package intentionally contains a `packaging` fixture package;
    # production has no test-path dependency, so make the third-party imports
    # resolve as they do for an installed reward-lens distribution.
    tests_root = Path(__file__).parents[2].resolve()
    monkeypatch.setattr(
        sys, "path", [entry for entry in sys.path if Path(entry or ".").resolve() != tests_root]
    )
    input = build_cpu_short_fork(config, tmp_path / "actual-factory")
    comparison = compare_verified_short_fork(input)

    assert input.record_path == tmp_path / "actual-factory" / "cpu-conformance.json"
    assert (tmp_path / "actual-factory" / "runtime-config.json").is_file()
    assert (tmp_path / "actual-factory" / "runtime-fixture" / "fork.json").is_file()
    assert input.result.arm_names == ("null", "null", "candidate", "placebo")
    assert comparison.entries[0].check.passed is True
    assert (
        comparison.entries[0].result["cpu_conformance"]["limits"] == config.to_mapping()["limits"]
    )
    produced = append_short_fork_assay(base, comparison)
    project.record(produced, name="factory-rung-five")
    reopened = project.open_record("factory-rung-five")
    entry = next(row for row in reopened.entries() if row.entry_id == R5_ENTRY_ID)
    fixture = json.loads(
        (tmp_path / "actual-factory" / "runtime-fixture" / "fork.json").read_text()
    )
    assert entry.subject_ref == input.subject_ref == reopened.subject.version.digest
    assert fixture["subject_ref"] == input.subject_ref
    assert entry.result["cpu_conformance"]["fixture_digest"] == fixture_digest(input.fixture_root)
    with pytest.raises(CPUConformanceRecordInvalid, match="subject_ref"):
        compare_verified_short_fork(replace(input, subject_ref="sha256:" + "b" * 64))


def test_rung5_public_factory_config_refuses_unbounded_cpu_execution() -> None:
    raw = default_cpu_short_fork_config(subject_ref="sha256:" + "2" * 64).to_mapping()
    raw["limits"]["cpu_threads"] = 2
    with pytest.raises(ValueError, match="limits.cpu_threads"):
        CPUShortForkConfig.from_mapping(raw)


def _result(fixture_root: Path = FIXTURE):
    profile = TrainerProfile("1.10.0", EIGHT_OBJECTS, "cpu-toy-fork-v2")
    _model = torch.nn.Linear(1, 1, bias=False)
    callback = build_clamp_callback(
        profile,
        ClampSpecification(
            target_name=FIXTURE_INPUTS["target"]["name"],
            target=lambda current: current.weight.mean(),
            target_value=0.5,
            tolerance=1e-6,
            max_correction=2.0,
            max_iterations=4,
        ),
    )
    return run_short_fork(
        (
            ShortForkArm(
                "null",
                lambda: None,
                lambda: 0.5,
                {FIXTURE_INPUTS["useful_task"]["name"]: lambda: 0.8},
            ),
            ShortForkArm(
                "null",
                lambda: None,
                lambda: 0.5,
                {FIXTURE_INPUTS["useful_task"]["name"]: lambda: 0.7},
            ),
            ShortForkArm(
                "candidate",
                lambda: None,
                lambda: 0.5,
                {FIXTURE_INPUTS["useful_task"]["name"]: lambda: 0.75},
            ),
            ShortForkArm(
                "placebo",
                lambda: None,
                lambda: 0.5,
                {FIXTURE_INPUTS["useful_task"]["name"]: lambda: 0.76},
            ),
        ),
        profile=profile,
        clamp=callback,
        fixture_root=fixture_root,
        limits=FIXTURE_INPUTS["limits"],
    )


def _record(tmp_path: Path, *, fixture_root: Path = FIXTURE, result=None) -> Path:
    path = tmp_path / "cpu-conformance.json"
    write_cpu_conformance_record(
        _result(fixture_root) if result is None else result, fixture_root=fixture_root, path=path
    )
    return path


def _bound_fixture(tmp_path: Path, subject_ref: str) -> Path:
    root = tmp_path / "bound-fixture"
    root.mkdir(exist_ok=True)
    payload = dict(FIXTURE_INPUTS, subject_ref=subject_ref)
    (root / "fork.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_rung5_generates_and_rechecks_an_integrity_checked_cpu_conformance_record(
    tmp_path: Path,
) -> None:
    path = _record(tmp_path)
    record = load_cpu_conformance_record(path, fixture_root=FIXTURE)
    assert record["scope"] == "short_fork_behavior"
    assert record["profile"]["version"] == "1.10.0"
    assert record["profile"]["mutable_objects"] == list(EIGHT_OBJECTS)
    assert record["profile"]["fingerprint"] == "cpu-toy-fork-v2"
    assert record["clamp"]["method"] == "optimizer_step_projection"
    assert record["target_measure"]["name"] == FIXTURE_INPUTS["target"]["name"]
    assert record["useful_task_controls"][FIXTURE_INPUTS["useful_task"]["name"]] == [
        0.8,
        0.7,
        0.75,
        0.76,
    ]
    assert record["null_spread"]["method"] == "sample_standard_deviation"
    assert record["limits"] == FIXTURE_INPUTS["limits"]


def test_rung5_registers_a_real_registry_entrypoint() -> None:
    assert get(Rung.SHORT_FORK) is run


def _canonical_input(tmp_path: Path, *, subject_ref: str = "sha256:" + "2" * 64) -> Rung5Input:
    from reward_lens.contracts import EntryProvenance

    fixture_root = _bound_fixture(tmp_path, subject_ref)
    result = _result(fixture_root)
    return Rung5Input(
        result=result,
        record_path=_record(tmp_path, fixture_root=fixture_root, result=result),
        fixture_root=fixture_root,
        subject_ref=subject_ref,
        provenance=EntryProvenance(
            started="2026-09-26T00:00:00Z", duration_s=0.0, sandbox_tier="T0", offline=True
        ),
    )


def test_rung5_makes_a_canonical_compare_entry_from_the_verified_record(tmp_path: Path) -> None:
    comparison = compare_verified_short_fork(_canonical_input(tmp_path))
    assert comparison.scope == "short_fork_behavior"
    assert comparison.entries[0].entry_id == R5_ENTRY_ID
    assert comparison.entries[0].check.passed is True
    retained = comparison.entries[0].result["cpu_conformance"]
    assert retained["arm_names"] == ["null", "null", "candidate", "placebo"]
    assert retained["null_spread"]["value"] == 0.0
    assert retained["profile"]["fingerprint"] == "cpu-toy-fork-v2"
    assert retained["limits"] == FIXTURE_INPUTS["limits"]


def test_rung5_derives_a_failed_control_check_from_retained_post_step_witnesses(
    tmp_path: Path,
) -> None:
    from reward_lens.contracts import EntryProvenance

    subject_ref = "sha256:" + "2" * 64
    fixture_root = _bound_fixture(tmp_path, subject_ref)
    failed = replace(_result(fixture_root), target_values=(0.4, 0.4, 0.5, 0.5))
    path = tmp_path / "failed-control.json"
    write_cpu_conformance_record(failed, fixture_root=fixture_root, path=path)
    comparison = compare_verified_short_fork(
        Rung5Input(
            result=failed,
            record_path=path,
            fixture_root=fixture_root,
            subject_ref=subject_ref,
            provenance=EntryProvenance(
                started="2026-09-26T00:00:00Z", duration_s=0.0, sandbox_tier="T0", offline=True
            ),
        )
    )
    evidence = comparison.entries[0].result
    assert comparison.entries[0].check.passed is False
    assert evidence["null_replay"]["passed"] is False
    assert evidence["candidate"]["passed"] is True
    assert evidence["placebo"]["passed"] is True


def test_rung5_appends_a_canonical_entry_then_the_store_reopens_it(tmp_path: Path) -> None:
    from reward_lens.contracts import Assay
    from reward_lens.store import Project
    from tests.store.conftest import build_record, write_project
    from tests.store.test_project import align

    root = write_project(tmp_path / "project")
    project = Project.open(root)
    base = Assay.model_validate(align(build_record(), project))
    input = _canonical_input(tmp_path, subject_ref=base.subject.version.digest)
    produced = append_short_fork_assay(base, compare_verified_short_fork(input))
    saved = Assay.model_validate(align(produced.to_dict(), project))
    path = project.record(saved, name="rung-five")
    reopened = project.open_record("rung-five")
    entry = next(row for row in reopened.entries() if row.entry_id == R5_ENTRY_ID)
    assert path.is_file()
    assert entry.result["cpu_conformance"]["arm_names"] == ["null", "null", "candidate", "placebo"]
    assert entry.check.passed is True


def test_rung5_legacy_unbound_fixture_cannot_claim_a_canonical_subject(tmp_path: Path) -> None:
    from reward_lens.contracts import EntryProvenance

    result = _result()
    with pytest.raises(CPUConformanceRecordInvalid, match="legacy unbound"):
        compare_verified_short_fork(
            Rung5Input(
                result=result,
                record_path=_record(tmp_path, result=result),
                fixture_root=FIXTURE,
                subject_ref="sha256:" + "2" * 64,
                provenance=EntryProvenance(
                    started="2026-09-26T00:00:00Z", duration_s=0.0, sandbox_tier="T0", offline=True
                ),
            )
        )


def test_rung5_refuses_a_record_missing_scope_with_the_precise_field(tmp_path: Path) -> None:
    path = _record(tmp_path)
    payload = json.loads(path.read_text())
    del payload["scope"]
    path.write_text(json.dumps(payload))
    with pytest.raises(CPUConformanceRecordInvalid, match="missing field 'scope'"):
        load_cpu_conformance_record(path, fixture_root=FIXTURE)


@pytest.mark.parametrize(
    "field",
    [
        "profile",
        "clamp",
        "target_measure",
        "useful_task_controls",
        "null_spread",
        "limits",
        "claim_boundary",
        "arm_names",
    ],
)
def test_rung5_refuses_every_other_missing_required_field_with_the_precise_field(
    tmp_path: Path, field: str
) -> None:
    path = _record(tmp_path)
    payload = json.loads(path.read_text())
    del payload[field]
    path.write_text(json.dumps(payload))
    with pytest.raises(CPUConformanceRecordInvalid, match=f"missing field '{field}'"):
        load_cpu_conformance_record(path, fixture_root=FIXTURE)


def test_rung5_refuses_a_changed_fixture_digest_with_the_precise_field(tmp_path: Path) -> None:
    path = _record(tmp_path)
    payload = json.loads(path.read_text())
    payload["fixture_digest"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(CPUConformanceRecordInvalid, match="fixture_digest"):
        load_cpu_conformance_record(path, fixture_root=FIXTURE)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("mutable_objects", list(reversed(EIGHT_OBJECTS))),
        ("fingerprint", "tampered-fingerprint"),
    ],
)
def test_rung5_refuses_tampered_profile_content_with_the_precise_field(
    tmp_path: Path, field: str, replacement: object
) -> None:
    path = _record(tmp_path)
    payload = json.loads(path.read_text())
    payload["profile"][field] = replacement
    path.write_text(json.dumps(payload))
    with pytest.raises(CPUConformanceRecordInvalid, match=f"profile.{field}"):
        load_cpu_conformance_record(path, fixture_root=FIXTURE)


def test_rung5_refuses_a_tampered_null_spread_with_the_precise_field(tmp_path: Path) -> None:
    path = _record(tmp_path)
    payload = json.loads(path.read_text())
    payload["null_spread"]["value"] = -1.0
    path.write_text(json.dumps(payload))
    with pytest.raises(CPUConformanceRecordInvalid, match="null_spread.value"):
        load_cpu_conformance_record(path, fixture_root=FIXTURE)


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (
            lambda payload: payload["target_measure"].update(name="renamed_target"),
            "target_measure.name",
        ),
        (lambda payload: payload["target_measure"]["values"].append(0.5), "target_measure.values"),
        (
            lambda payload: payload["target_measure"].update(values=[float("inf")] * 3),
            "target_measure.values",
        ),
        (lambda payload: payload["useful_task_controls"].clear(), "useful_task_controls"),
        (
            lambda payload: payload["useful_task_controls"].update(extra=[0.1, 0.2, 0.3]),
            "useful_task_controls",
        ),
        (
            lambda payload: payload["useful_task_controls"].__setitem__(
                FIXTURE_INPUTS["useful_task"]["name"], [0.1, 0.2]
            ),
            "useful_task_controls",
        ),
        (
            lambda payload: payload["useful_task_controls"].__setitem__(
                FIXTURE_INPUTS["useful_task"]["name"], [float("nan")] * 3
            ),
            "useful_task_controls",
        ),
    ],
)
def test_rung5_refuses_tampered_measurement_identity_or_shape(
    tmp_path: Path, mutate, field: str
) -> None:
    path = _record(tmp_path)
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(CPUConformanceRecordInvalid, match=field):
        load_cpu_conformance_record(path, fixture_root=FIXTURE)


def _coding_payload():
    """Explicit consumer fixtures; never presented as CUDA/training evidence."""
    from reward_lens.product.compare.rung5 import CODING_SHORT_FORK_PROFILE

    base = [0.0, 1.0, 0.0, 1.0]
    offsets = [-0.5, 0.0, -0.5, 0.0]
    receipts = [
        {
            "task_id": ident,
            "completion": "def solution(): return 0",
            "baseline": score,
            "candidate": score + offset,
            "envelopes": {
                "baseline": {"verdict": "scored", "score": score},
                "text-penalty": {"verdict": "scored", "score": score + offset},
            },
        }
        for ident, score, offset in zip(
            ("task-01", "task-01", "task-02", "task-02"), base, offsets, strict=True
        )
    ]
    arms = []
    for name, factor in (("null", 0), ("null", 0), ("candidate", 1), ("placebo", -1)):
        applied = [factor * value for value in offsets]
        update = {
            "batch_digest": "batch",
            "batch_bytes_hex": "6162",
            "scored_replay_digest": "replay",
            "loss_input_digest": "loss",
            "loss": 0.1,
            "gradient_norm": 1.0,
            "optimizer_update_norm": 1.0,
            "reward_offsets": applied,
            "reward_witness": [
                {
                    "source_values": base,
                    "returned_values": [
                        score + offset for score, offset in zip(base, applied, strict=True)
                    ],
                }
            ],
        }
        outcomes = [
            {
                "task_id": name,
                "response": "def solution(): return 0",
                "envelope": {"verdict": "scored", "score": 0.0},
            }
            for name in ("task-01", "task-02")
        ]
        arms.append(
            {
                "name": name,
                "start_digest": "start",
                "during_digest": "changed",
                "restored_digest": "start",
                "update": update,
                "outcomes": outcomes,
                "capability_floor": {
                    "state": "NOT_MEASURED",
                    "reason": "no held-out capability set declared",
                    "held": None,
                    "repair_claim_qualified": False,
                },
            }
        )
    return {
        "schema": "reward-lens.coding-short-fork-execution/v1",
        "profile": CODING_SHORT_FORK_PROFILE,
        "subject_ref": "sha256:" + "d" * 64,
        "horizon_steps": 1,
        "device": "controlled-test-loader",
        "execution_class": "controlled_test_loader",
        "training_receipts": receipts,
        "arms": arms,
        "null_spread": 0.0,
        "null_spread_method": "sample_standard_deviation",
        "claim_boundary": "controlled test loader only; no actual training or behavioral repair result",
    }


def test_coding_short_fork_declares_only_the_finite_profile(tmp_path):
    from reward_lens.product.compare.rung5 import CODING_SHORT_FORK_PROFILE, CodingShortForkConfig

    declaration = {
        "profile": CODING_SHORT_FORK_PROFILE,
        "prepared_manifest": str(tmp_path / "prepared.json"),
        "prepared_manifest_digest": "sha256:" + "a" * 64,
        "subject_ref": "sha256:" + "b" * 64,
    }
    assert CodingShortForkConfig.from_mapping(declaration).to_mapping() == declaration
    with pytest.raises(ValueError, match="exact prepared declaration"):
        CodingShortForkConfig.from_mapping({**declaration, "horizon": 1000})
    with pytest.raises(ValueError, match="exact profile"):
        CodingShortForkConfig.from_mapping({**declaration, "profile": "generic-trainer"})


def test_coding_short_fork_consumer_retains_capability_absence_and_rechecks_artifact(tmp_path):
    from reward_lens import contracts
    from reward_lens.product.compare import rung4, rung5

    payload = dict(_coding_payload(), module_source_digest=rung4._file_digest(Path(rung5.__file__)))
    artifact = tmp_path / "coding-short-fork.json"
    artifact.write_bytes(contracts.canonical_bytes(payload))
    value = rung5.CodingShortForkInput(
        payload,
        artifact,
        rung4._file_digest(artifact),
        payload["subject_ref"],
        contracts.EntryProvenance(
            started="2026-10-04T00:00:00Z", duration_s=0.1, sandbox_tier="T0", offline=True
        ),
    )
    comparison = rung5.compare_coding_short_fork(value)
    entry = comparison.entries[0]
    assert entry.check.passed
    assert entry.result["coding_short_fork"]["execution_class"] == "controlled_test_loader"
    assert entry.result["capability_floor"]["state"] == "NOT_MEASURED"
    assert entry.result["capability_floor"]["held"] is None
    assert "no held-out capability set declared" in comparison.scorecard.rows[1].value.reason
    artifact.write_text("{}")
    with pytest.raises(ValueError, match="artifact or source binding"):
        rung5.compare_coding_short_fork(value)


@pytest.mark.parametrize(
    "fault", ("restore", "replay", "null", "reward", "outcome", "floor", "spread", "classification")
)
def test_coding_short_fork_fails_closed_on_changed_controls(fault):
    from reward_lens.product.compare.rung5 import _validate_coding_short_fork_payload

    payload = _coding_payload()
    if fault == "restore":
        payload["arms"][3]["restored_digest"] = "different"
    if fault == "replay":
        payload["arms"][3]["update"]["scored_replay_digest"] = "different"
    if fault == "null":
        payload["arms"][0]["update"]["reward_offsets"][0] = 1.0
    if fault == "reward":
        payload["arms"][2]["update"]["reward_witness"][0]["returned_values"][0] = 10.0
    if fault == "outcome":
        payload["arms"][2]["outcomes"][0]["envelope"]["verdict"] = "timeout"
    if fault == "floor":
        payload["arms"][2]["capability_floor"]["held"] = True
    if fault == "spread":
        payload["null_spread"] = 1.0
    if fault == "classification":
        payload["execution_class"] = "actual_cuda_short_fork"
    with pytest.raises(ValueError):
        _validate_coding_short_fork_payload(payload)


def test_coding_short_fork_declared_floor_runs_existing_supervised_resolver(tmp_path):
    """A controlled project declaration, not a floor invented for the actual M3 project."""
    from reward_lens.contracts import ProjectConfig
    from reward_lens.execution import Limits, default_sandbox
    from reward_lens.product.compare.rung5 import _coding_floor_context, _coding_floor_payload
    from reward_lens.store import Project
    from tests.outcome.test_project_execution import SUBJECT, _declaration, _write_project

    project = _write_project(tmp_path)
    path, raw = _declaration(project)
    raw["capability_floor"] = {
        "partition_path": "outcome/targets",
        "partition_id": "heldout-v1",
        "candidates": [
            {
                "item_id": row["item_id"],
                "task": row["task"],
                "response": "declaration-reference-not-used-for-generated-response",
            }
            for row in raw["target"]["tasks"]
        ],
    }
    path.write_text(json.dumps(raw))
    config = project.config.to_dict()
    config["outcome"]["held_out_capability"] = {
        "partition_id": "heldout-v1",
        "path": "outcome/targets",
        "floor": 0.75,
    }
    project = Project(project.root, ProjectConfig.model_validate(config))
    execution = _coding_floor_context(
        project,
        subject_ref=SUBJECT,
        sandbox=default_sandbox(),
        limits=Limits(wall_s=15, cpu_s=15, network=False),
    )
    measured = _coding_floor_payload(
        execution,
        generate=lambda task: "correct" if task["id"] == "pass.case" else "wrong",
        device="controlled-test-loader",
    )
    assert measured["value"] == 0.5 and measured["held"] is False
    assert measured["n"] == 2 and len(measured["rows"]) == 2
    assert measured["repair_claim_qualified"] is False
    assert all(row["supervisor_receipt"]["verdict"] == "scored" for row in measured["rows"])
    absent = _coding_floor_payload(
        None,
        generate=lambda task: pytest.fail("absence cannot execute a default target"),
        device="controlled-test-loader",
    )
    assert absent["state"] == "NOT_MEASURED" and absent["held"] is None
