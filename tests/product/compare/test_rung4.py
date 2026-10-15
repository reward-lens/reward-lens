"""Rung-four producer checks using retained, non-placeholder arm evidence."""

from __future__ import annotations

import importlib.metadata
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from reward_lens import contracts
from reward_lens.learning.contracts import EIGHT_OBJECTS, TrainerProfile
from reward_lens.learning.intervention.swap import ControlledUpdateResult, Intervention
from reward_lens.product.compare.rung4 import (
    ControlledUpdateEvidence,
    CpuControlledUpdateConfig,
    build_cpu_controlled_update,
    controlled_update_compare,
    controlled_update_record,
)
from reward_lens.product.trace.record import TraceRecord

_PUBLIC_SUBJECT = "sha256:" + "d" * 64


def _trace() -> TraceRecord:
    return TraceRecord(
        "run_record",
        "fixture/run",
        "fixture-run",
        (),
        {},
        {},
        "local",
        "fixture",
        "strict",
        True,
        "complete fixture",
        "sha256:" + "1" * 64,
        ("digest:training_semantics", "digest:samples", "digest:policy"),
    )


def _evidence() -> ControlledUpdateEvidence:
    return ControlledUpdateEvidence(
        _trace(),
        contracts.EntryProvenance(
            started="2026-09-26T00:00:00Z", duration_s=0.1, sandbox_tier="T0", offline=True
        ),
    )


def _value(name: str) -> dict[str, object]:
    return {
        "batch_digest": "batch",
        "batch_bytes": b"batch",
        "scored_replay_digest": "replay",
        "loss_input_digest": "loss-input",
        "intervention": Intervention(name, (0.0, 0.0)),
        "reward_witness": ({"source_values": (0.0, 1.0), "returned_values": (0.0, 1.0)},),
        "loss": 0.1,
        "gradient_norm": 1.0,
        "optimizer_update_norm": 1.0,
        "advantages": (0.5, -0.5),
        "changed_private_fields": ("_step",),
    }


def _result(
    *, names: tuple[str, ...] = ("null", "candidate", "placebo"), mutate: str | None = None
) -> ControlledUpdateResult:
    arms = []
    for name in names:
        witness = {
            "value": _value(name),
            "during_digest": "changed",
            "restored_digest": "same",
            "changed_objects": (),
        }
        if mutate == "restored" and name == "placebo":
            witness["restored_digest"] = "not-baseline"
        if mutate == "replay" and name == "placebo":
            witness["value"] = {**witness["value"], "scored_replay_digest": "other-replay"}
        if mutate == "identity" and name == "null":
            witness["value"] = {
                **witness["value"],
                "intervention": Intervention("not-null", (0.0, 0.0)),
            }
        arms.append((name, witness))
    return ControlledUpdateResult(
        "controlled_update_response",
        "mechanism conformance; not a behavioural result",
        "same",
        "same",
        names,
        tuple(arms),
        TrainerProfile("1.10.0", EIGHT_OBJECTS, "cpu-fixture"),
    )


def test_rung4_produces_strict_base_null_candidate_placebo_records() -> None:
    record = controlled_update_record(_result(), _evidence())
    assert tuple(record.arm_results) == ("base", "null", "candidate", "placebo")
    assert record.arm_results["candidate"].telemetry["optimizer_update_norm"] == 1.0
    compared = controlled_update_compare(_result(), _evidence())
    entry = compared.entries[0]
    assert entry.entry_id == "reward_statistics.controlled_update_response"
    assert entry.result["null_replay"]["passed"] is True
    assert entry.result["placebo"]["passed"] is True
    assert tuple(entry.result["arm_results"]) == ("base", "null", "candidate", "placebo")
    assert entry.subject_ref == _trace().subject_ref
    assert entry.depends_on == list(_trace().depends_on)
    assert all(row.value.reason.startswith("not measured") for row in compared.scorecard.rows)


def test_rung4_missing_candidate_is_a_canonical_absence_not_a_placeholder() -> None:
    result = _result(names=("null", "placebo"))
    compared = controlled_update_compare(result, _evidence())
    entry = compared.entries[0]
    assert entry.kind == "absence"
    assert entry.state == "absent"
    assert entry.result is None
    assert "candidate" in entry.absence.missing_access


@pytest.mark.parametrize("mutation", ("restored", "replay", "identity"))
def test_rung4_mutated_runtime_witness_is_a_canonical_absence(mutation: str) -> None:
    entry = controlled_update_compare(_result(mutate=mutation), _evidence()).entries[0]
    assert entry.kind == "absence"
    assert entry.state == "absent"
    assert entry.result is None


@pytest.mark.parametrize("during", ("missing", None, "", "same", 7))
def test_rung4_missing_or_invalid_during_digest_is_a_canonical_absence(during: object) -> None:
    result = _result()
    arms = list(result.arms)
    name, witness = arms[0]
    altered = dict(witness)
    if during == "missing":
        altered.pop("during_digest")
    else:
        altered["during_digest"] = during
    arms[0] = (name, altered)
    malformed = ControlledUpdateResult(
        result.scope,
        result.statement,
        result.start_digest,
        result.final_digest,
        result.arm_order,
        tuple(arms),
        result.profile,
    )
    entry = controlled_update_compare(malformed, _evidence()).entries[0]
    assert entry.kind == "absence"
    assert entry.result is None


def test_result_refuses_a_behavioural_or_unspecified_claim_boundary() -> None:
    with pytest.raises(ValueError, match="non-behavioural"):
        ControlledUpdateResult(
            "controlled_update_response",
            "the toy is a behavioural finding",
            "same",
            "same",
            ("null", "placebo"),
            (("null", {}), ("placebo", {})),
            TrainerProfile("1.10.0", EIGHT_OBJECTS, "cpu-fixture"),
        )


def test_cpu_factory_runs_actual_configured_update_and_persists_canonical_assay(tmp_path) -> None:
    """The public factory must retain the actual configured update to reopenable evidence."""
    from reward_lens.api._record import absent_record
    from reward_lens.product.audit.run import persist_assay_bundle

    config = CpuControlledUpdateConfig.from_mapping(
        {
            "seed": 37,
            "max_completion_length": 4,
            "candidate_offsets": [0.5, -0.25],
            "subject_ref": _PUBLIC_SUBJECT,
        }
    )
    result, evidence = build_cpu_controlled_update(config, tmp_path / "factory-run")
    compared = controlled_update_compare(result, evidence)
    entry = compared.entries[0]
    candidate = dict(result.arms)["candidate"]["value"]

    assert entry.kind == "check"
    assert evidence.provenance.duration_s > 0
    assert evidence.trace.reader == "controlled_update_execution"
    assert evidence.trace.complete is False
    assert (
        evidence.trace.subject_ref
        == config.subject_ref
        == evidence.identities["declared_subject_ref"]
    )
    assert evidence.trace.depends_on == tuple(
        evidence.identities[name]
        for name in (
            "batch_digest",
            "scored_replay_digest",
            "loss_input_digest",
            "config_digest",
        )
    )
    assert len(evidence.trace.rows) == len(candidate["advantages"])
    artifact_path = Path(evidence.source_binding["artifact_path"])
    assert artifact_path.is_file()
    assert artifact_path.read_bytes()
    assert evidence.source_binding["artifact_digest"].startswith("sha256:")
    assert evidence.source_binding["trace_projection_digest"].startswith("sha256:")
    assert entry.result["trace_record"]["subject_ref"] == config.subject_ref
    assert entry.result["arm_results"]["candidate"]["parameters"]["intervention"] == {
        "name": "candidate",
        "reward_offsets": [0.5, -0.25],
    }
    assert candidate["reward_witness"][0]["returned_values"] == (0.5, 0.75)
    assert candidate["optimizer_update_norm"] > 0.0

    assay = absent_record(
        command="compare", reproduce=["bounded local rung four factory"], path=Path(__file__)
    )
    assay.measurement.reward_statistics = [entry]
    assay.holes_from_entries()
    assay.assay_id = contracts.digest(assay)
    contracts.validate_record(assay.to_dict())
    persisted, paths = persist_assay_bundle(
        assay, project=None, name="factory-rung4", fallback_dir=tmp_path
    )
    reopened = contracts.Assay.model_validate(
        json.loads(paths["assay"].read_text(encoding="utf-8"))
    )
    contracts.validate_record(reopened.to_dict(), raw=paths["assay"].read_bytes())
    survived = reopened.measurement.reward_statistics[0]
    assert persisted.assay_id == reopened.assay_id
    assert survived.result["trace_record"]["subject_ref"] == config.subject_ref
    assert survived.result["arm_results"]["candidate"]["parameters"]["intervention"][
        "reward_offsets"
    ] == [0.5, -0.25]


def test_cpu_factory_candidate_config_changes_the_retained_update(tmp_path) -> None:
    base_result, base_evidence = build_cpu_controlled_update(
        CpuControlledUpdateConfig(seed=41), tmp_path / "base"
    )
    changed = CpuControlledUpdateConfig(seed=41, candidate_offsets=(0.5, -0.25))
    changed_result, changed_evidence = build_cpu_controlled_update(changed, tmp_path / "changed")

    base_candidate = dict(base_result.arms)["candidate"]["value"]
    changed_candidate = dict(changed_result.arms)["candidate"]["value"]
    assert (
        base_evidence.identities["execution_digest"]
        != changed_evidence.identities["execution_digest"]
    )
    assert (
        base_candidate["reward_witness"][0]["returned_values"]
        != changed_candidate["reward_witness"][0]["returned_values"]
    )
    assert base_candidate["advantages"] != changed_candidate["advantages"]
    assert base_candidate["loss"] != changed_candidate["loss"]


@pytest.mark.parametrize(
    "value, message",
    [
        ({"unexpected": 1}, "undeclared fields"),
        ({"batch_size": 3}, "bounded two-by-two"),
        ({"candidate_offsets": [1, 0.0]}, "two floats"),
        ({"candidate_offsets": [float("nan"), 0.0]}, "finite"),
        ({"candidate_offsets": [float("inf"), 0.0]}, "finite"),
        ({"candidate_offsets": [float("-inf"), 0.0]}, "finite"),
        ({"subject_ref": "not-a-digest"}, "authoritative sha256"),
    ],
)
def test_cpu_factory_config_refuses_undeclared_or_noncanonical_inputs(value, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        CpuControlledUpdateConfig.from_mapping(
            value if "subject_ref" in value else {**value, "subject_ref": _PUBLIC_SUBJECT}
        )


def test_cpu_factory_refuses_unpinned_runtime_before_constructing_trainer(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda distribution: "1.13.0")
    with pytest.raises(ValueError, match=r"TRAINER_UNSUPPORTED.*trl==1.10.0"):
        build_cpu_controlled_update(CpuControlledUpdateConfig(), tmp_path / "unsupported")


def test_cpu_factory_public_config_requires_predeclared_authoritative_subject() -> None:
    with pytest.raises(ValueError, match="requires subject_ref"):
        CpuControlledUpdateConfig.from_mapping({"seed": 37})


def test_cpu_factory_rejects_declared_subject_substitution_after_execution(tmp_path) -> None:
    config = CpuControlledUpdateConfig(subject_ref=_PUBLIC_SUBJECT)
    result, evidence = build_cpu_controlled_update(config, tmp_path / "run")
    identities = dict(evidence.identities)
    identities["declared_subject_ref"] = "sha256:" + "e" * 64
    substituted_trace = replace(evidence.trace, subject_ref=identities["declared_subject_ref"])
    substituted = ControlledUpdateEvidence(
        substituted_trace,
        evidence.provenance,
        identities=identities,
        source_binding=evidence.source_binding,
    )
    entry = controlled_update_compare(result, substituted).entries[0]
    assert entry.kind == "absence"
    assert "identities" in entry.absence.missing_access


def test_cpu_factory_rejects_post_execution_identity_tampering(tmp_path) -> None:
    result, evidence = build_cpu_controlled_update(
        CpuControlledUpdateConfig(seed=43), tmp_path / "run"
    )
    arms = list(result.arms)
    name, witness = arms[1]
    altered_value = {**witness["value"], "loss_input_digest": "tampered"}
    arms[1] = (name, {**witness, "value": altered_value})
    tampered = ControlledUpdateResult(
        result.scope,
        result.statement,
        result.start_digest,
        result.final_digest,
        result.arm_order,
        tuple(arms),
        result.profile,
    )
    entry = controlled_update_compare(tampered, evidence).entries[0]
    assert entry.kind == "absence"
    assert "execution identities" in entry.absence.missing_access


def test_cpu_factory_does_not_contain_test_import_state_surgery(tmp_path) -> None:
    import reward_lens.product.compare.rung4 as rung4
    import tests.packaging as test_packaging

    packaging_before = sys.modules.get("packaging")
    path_before = tuple(sys.path)
    tests_packaging_before = sys.modules["tests.packaging"]
    result, _ = build_cpu_controlled_update(CpuControlledUpdateConfig(seed=47), tmp_path / "run")

    assert result.arm_order == ("null", "candidate", "placebo")
    assert "_pinned_cpu_dependencies" not in vars(rung4)
    assert tuple(sys.path) == path_before
    assert sys.modules.get("packaging") is packaging_before
    assert sys.modules["tests.packaging"] is tests_packaging_before is test_packaging


@pytest.mark.parametrize(
    "mutation", ("row", "advantage", "series", "source", "source_binding", "artifact")
)
def test_cpu_factory_rejects_tampered_execution_trace_facts(tmp_path, mutation) -> None:
    result, evidence = build_cpu_controlled_update(
        CpuControlledUpdateConfig(seed=59), tmp_path / "run"
    )
    trace = evidence.trace
    source_binding = dict(evidence.source_binding)
    if mutation == "row":
        rows = list(trace.rows)
        rows[0] = replace(rows[0], reward_applied=999.0)
        trace = replace(trace, rows=tuple(rows))
    elif mutation == "advantage":
        rows = list(trace.rows)
        rows[0] = replace(rows[0], advantage=999.0)
        trace = replace(trace, rows=tuple(rows))
    elif mutation == "series":
        series = dict(trace.series)
        reward_applied = series["reward_applied"]
        series["reward_applied"] = replace(
            reward_applied, values=(999.0, *reward_applied.values[1:])
        )
        trace = replace(trace, series=series)
    elif mutation == "source":
        trace = replace(trace, source="fabricated source")
    elif mutation == "source_binding":
        source_binding["execution"] = "fabricated source"
    else:
        Path(source_binding["artifact_path"]).write_bytes(b"tampered")
    tampered = ControlledUpdateEvidence(
        trace,
        evidence.provenance,
        identities=evidence.identities,
        source_binding=source_binding,
    )
    entry = controlled_update_compare(result, tampered).entries[0]
    assert entry.kind == "absence"
    assert any(word in entry.absence.missing_access for word in ("trace", "source", "artifact"))


def test_cpu_factory_refuses_complete_execution_trace(tmp_path) -> None:
    result, evidence = build_cpu_controlled_update(
        CpuControlledUpdateConfig(seed=61), tmp_path / "run"
    )
    with pytest.raises(ValueError, match="incomplete execution record"):
        ControlledUpdateEvidence(
            replace(evidence.trace, complete=True),
            evidence.provenance,
            identities=evidence.identities,
            source_binding=evidence.source_binding,
        )


def test_coding_profile_accepts_only_its_explicit_manifest_declaration(tmp_path):
    from reward_lens.product.compare.rung4 import (
        CODING_UPDATE_PROFILE,
        CodingControlledUpdateConfig,
    )

    declaration = {
        "profile": CODING_UPDATE_PROFILE,
        "prepared_manifest": str(tmp_path / "prepared.json"),
        "prepared_manifest_digest": "sha256:" + "a" * 64,
        "subject_ref": _PUBLIC_SUBJECT,
    }
    config = CodingControlledUpdateConfig.from_mapping(declaration)
    assert config.canonical_mapping() == declaration
    assert config.digest == contracts.digest(declaration)
    with pytest.raises(ValueError, match="exact declared profile fields"):
        CodingControlledUpdateConfig.from_mapping({**declaration, "trainer": "arbitrary"})
    with pytest.raises(ValueError, match="absolute local path"):
        CodingControlledUpdateConfig.from_mapping(
            {**declaration, "prepared_manifest": "relative.json"}
        )
    with pytest.raises(ValueError, match="unsupported"):
        CodingControlledUpdateConfig.from_mapping({**declaration, "profile": "generic-trainer"})


def _fake_cuda_coding_adapter(monkeypatch):
    """Exercise the actual eight-object adapter with two simulated CUDA generators."""
    import copy
    import random
    from types import SimpleNamespace

    import numpy as np
    import torch

    from reward_lens.product.compare import rung4

    cuda = {
        "states": [
            torch.tensor([1, 2, 3], dtype=torch.uint8),
            torch.tensor([7, 8, 9], dtype=torch.uint8),
        ],
        "count": 2,
        "uuids": ["device-A", "device-B"],
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: cuda["count"])
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(
            name="simulated CUDA", uuid=cuda["uuids"][index], major=8, minor=0, total_memory=1024
        ),
    )
    monkeypatch.setattr(
        torch.cuda, "get_rng_state_all", lambda: [state.clone() for state in cuda["states"]]
    )

    def set_all(states):
        cuda["states"] = [state.clone() for state in states]

    monkeypatch.setattr(torch.cuda, "set_rng_state_all", set_all)

    def advance():
        cuda["states"] = [state + index + 1 for index, state in enumerate(cuda["states"])]
        random.random()
        np.random.random()
        torch.rand(1)

    class StateStore:
        def __init__(self, values):
            self.values = values
            self.param_groups = [{"lr": 0.1}]

        def state_dict(self):
            return copy.deepcopy(self.values)

        def load_state_dict(self, values):
            self.values = copy.deepcopy(values)

    adapter_type = getattr(rung4, "_CodingTRL110StateAdapter", rung4.TRL110StateAdapter)

    class SimulatedCodingAdapter(adapter_type):
        def freeze_scored_replay(self, batch):
            advance()
            return batch

    trainer = SimpleNamespace(
        model=torch.nn.Linear(1, 1),
        optimizer=StateStore({"step": 0}),
        lr_scheduler=StateStore({"counter": 0}),
        args=SimpleNamespace(gradient_accumulation_steps=1),
        reward_funcs=[],
        _buffered_inputs={},
        _step=0,
        _current_train_step_time=0.0,
        _metrics={},
        _logs={},
        _pending_extra_logs={},
        _pending_metrics={},
        state=SimpleNamespace(global_step=0),
    )
    adapter = SimulatedCodingAdapter(trainer)
    return adapter, cuda, advance


def test_coding_cuda_rng_generation_arm_starts_and_final_restore(monkeypatch):
    from reward_lens.learning.contracts import EIGHT_OBJECTS, TrainerProfile
    from reward_lens.learning.intervention.swap import (
        MaterializedBatch,
        SwapArm,
        capture_state,
        run_swap,
    )

    adapter, cuda, advance = _fake_cuda_coding_adapter(monkeypatch)
    original_cuda = [state.clone() for state in cuda["states"]]
    profile = TrainerProfile("1.10.0", EIGHT_OBJECTS, "simulated-coding-cuda-generators")
    baseline = capture_state(adapter, profile)
    starts = []

    def arm(name):
        def apply(host, batch):
            starts.append([state.clone() for state in cuda["states"]])
            assert all(
                state.equal(expected)
                for state, expected in zip(cuda["states"], original_cuda, strict=True)
            ), "generation advanced CUDA RNG but arm did not restore its original start"
            advance()
            host.trainer.model.weight.data.add_(1)
            host.trainer.optimizer.values["step"] += 1
            host.trainer.lr_scheduler.values["counter"] += 1
            host._materialized_position += 1
            host.trainer._buffered_inputs = {"mutated": True}
            host.trainer._step += 1
            host.trainer.state.global_step += 1
            return name

        return SwapArm(name, apply)

    result = run_swap(
        MaterializedBatch(({"prompt": "task"},)),
        [arm("candidate"), arm("placebo")],
        profile,
        trainer=adapter,
        null=arm("null"),
    )
    assert len(starts) == 3
    assert all(
        state.equal(expected) for state, expected in zip(cuda["states"], original_cuda, strict=True)
    )
    assert (
        capture_state(adapter, profile).digest()
        == baseline.digest()
        == result.start_digest
        == result.final_digest
    )
    assert set(baseline.values) == set(EIGHT_OBJECTS)
    assert all(set(witness["changed_objects"]) == set(EIGHT_OBJECTS) for _, witness in result.arms)


@pytest.mark.parametrize(
    "fault",
    (
        "missing_device",
        "extra_device",
        "wrong_inventory",
        "reordered_inventory",
        "live_device_count",
        "live_device_uuid",
        "missing_device_uuid",
        "wrong_dtype",
        "empty_state",
        "missing_key",
    ),
)
def test_coding_cuda_rng_refuses_incomplete_or_changed_device_snapshot(monkeypatch, fault):
    import copy

    import torch

    adapter, cuda, advance = _fake_cuda_coding_adapter(monkeypatch)
    snapshot = copy.deepcopy(adapter.rng_states)
    if fault == "missing_device":
        snapshot["torch_cuda"] = snapshot["torch_cuda"][:1]
    elif fault == "extra_device":
        snapshot["torch_cuda"] += (snapshot["torch_cuda"][0],)
    elif fault == "wrong_inventory":
        snapshot["cuda_inventory"]["devices"][0]["uuid"] = "wrong-device"
    elif fault == "reordered_inventory":
        snapshot["cuda_inventory"]["devices"] = tuple(
            reversed(snapshot["cuda_inventory"]["devices"])
        )
    elif fault == "live_device_count":
        cuda["count"] = 1
    elif fault == "live_device_uuid":
        cuda["uuids"][1] = "replacement-device"
    elif fault == "missing_device_uuid":
        cuda["uuids"][1] = None
    elif fault == "wrong_dtype":
        snapshot["torch_cuda"] = (torch.tensor([1.0]), snapshot["torch_cuda"][1])
    elif fault == "empty_state":
        snapshot["torch_cuda"] = (torch.tensor([], dtype=torch.uint8), snapshot["torch_cuda"][1])
    else:
        del snapshot["torch_cuda"]
    before = [state.clone() for state in cuda["states"]]
    before_cpu = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="CUDA RNG"):
        adapter.rng_states = snapshot
    assert all(
        state.equal(expected) for state, expected in zip(cuda["states"], before, strict=True)
    )
    assert torch.get_rng_state().equal(before_cpu)
