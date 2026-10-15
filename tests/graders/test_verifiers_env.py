"""P-ADAPT-VERIFIERS: probes and writer execution receipts are machine-observed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reward_lens.graders import verifiers_env as verifier_module
from reward_lens.graders.base import AdapterCapabilityUnproven
from reward_lens.graders.verifiers_env import (
    DeterministicProbe,
    EvidencePlane,
    JudgeProbe,
    RubricComponent,
    StatefulProbe,
    VerifiedWriterArtifact,
    VerifiersEnvGrader,
)


def deterministic(completion: str) -> float:
    return float(completion == "correct")


def judge(prompt: str, completion: str) -> float:
    return 0.75 if prompt == "check" and completion == "correct" else 0.0


def observed_judge(prompt: str, completion: str, judge_score: float) -> float:
    assert prompt == "check"
    assert completion == "correct"
    return judge_score


def stateful_task(state: dict[str, object], task: dict[str, object]) -> float:
    state["terminal"] = bool(task["isolated"])
    return float(state["terminal"])


def _offline_judge_adapter() -> object:
    """Exercise the integrated offline adapter without copying its provider protocol."""
    from reward_lens.graders.judge import JudgeAdapter

    class OfflineTransport:
        def __call__(self, prompt: str, completion: str, **kwargs: object) -> object:
            return {"verdict": "left", "score": 0.75}

    return JudgeAdapter.from_transport(
        provider="local-protocol-fixture",
        model="fixture-judge-v1",
        prompt="Choose the more correct response and return JSON.",
        settings={"temperature": 0},
        transport=OfflineTransport(),
        fixture_conformance=True,
        passing_probes={"fixture_conformance": True},
    )


def _mixed_components() -> tuple[RubricComponent, ...]:
    return (
        RubricComponent(
            "deterministic",
            deterministic,
            DeterministicProbe(inputs={"completion": "correct"}, expected_score=1.0),
        ),
        RubricComponent(
            "judge",
            observed_judge,
            JudgeProbe(
                inputs={"prompt": "check", "completion": "correct"},
                expected_score=0.75,
                judge_adapter=_offline_judge_adapter(),
                task={"id": "judge-1", "prompt": "What is two plus two?"},
            ),
        ),
        RubricComponent(
            "stateful",
            stateful_task,
            StatefulProbe(
                inputs={"state": {"terminal": False}, "task": {"isolated": True}},
                expected_score=1.0,
                terminal_key="terminal",
            ),
        ),
    )


def _writer_row() -> dict[str, object]:
    return {
        "example_id": "mixed-1",
        "prompt": [{"role": "user", "content": "check"}],
        "completion": [{"role": "assistant", "content": "correct"}],
        "answer": "",
        "info": {},
        "reward": 0.75,
        "error": None,
        "timing": {},
        "is_completed": True,
        "is_truncated": False,
        "stop_condition": "terminal",
        "metrics": {"deterministic": 1.0, "judge": 0.75, "stateful": 1.0},
    }


def test_mixed_rubric_runs_plane_specific_probes_and_preserves_three_capability_sets() -> None:
    grader = VerifiersEnvGrader.from_rubric(_mixed_components())

    assert grader.describe().family == "VERIFIERS_ENV"
    assert [item.evidence_plane for item in grader.capability_sets] == [
        EvidencePlane.DETERMINISTIC,
        EvidencePlane.JUDGE,
        EvidencePlane.STATEFUL,
    ]
    assert grader.capability_sets[0].bound_inputs == frozenset({"completion"})
    assert grader.capability_sets[2].bound_inputs == frozenset({"state", "task"})
    assert [item.probe.observed_score for item in grader.capability_sets] == [1.0, 0.75, 1.0]
    assert grader.capability_sets[1].probe.evaluator_id is not None
    assert grader.capability_sets[1].probe.evaluator_id.startswith("sha256:")
    assert (
        grader.capability_sets[2].probe.state_before_digest
        != grader.capability_sets[2].probe.state_after_digest
    )


def test_nonempty_fabricated_judge_metadata_is_not_an_actual_adapter_execution() -> None:
    class FabricatedJudgeAdapter:
        evaluator_id = "invented"
        request = {"arbitrary": "request"}
        response = {"arbitrary": "response"}

    component = RubricComponent(
        "caller_claimed",
        lambda prompt, completion: 1.0,
        JudgeProbe(
            inputs={"prompt": "p", "completion": "c"},
            expected_score=1.0,
            judge_adapter=FabricatedJudgeAdapter(),
            task={"id": "fabricated", "prompt": "p"},
        ),
    )

    with pytest.raises(AdapterCapabilityUnproven) as caught:
        VerifiersEnvGrader.from_rubric((component,))

    assert caught.value.code == "RL0702"
    assert caught.value.exit_code == 5
    assert caught.value.capability == "component:caller_claimed.judge.adapter"


def test_a_static_local_callable_is_truthfully_deterministic_without_a_judge_exchange() -> None:
    grader = VerifiersEnvGrader.from_rubric(
        (
            RubricComponent(
                "local_static",
                lambda completion: float(completion == "correct"),
                DeterministicProbe(inputs={"completion": "correct"}, expected_score=1.0),
            ),
        )
    )

    assert grader.capability_sets[0].evidence_plane is EvidencePlane.DETERMINISTIC
    assert grader.capability_sets[0].probe.evaluator_id is None


def test_stateful_plane_refuses_when_the_executed_probe_has_no_terminal_transition() -> None:
    component = RubricComponent(
        "false_stateful",
        lambda state, task: 1.0,
        StatefulProbe(
            inputs={"state": {"terminal": False}, "task": {"isolated": True}},
            expected_score=1.0,
            terminal_key="terminal",
        ),
    )

    with pytest.raises(AdapterCapabilityUnproven) as caught:
        VerifiersEnvGrader.from_rubric((component,))

    assert caught.value.code == "RL0702"
    assert caught.value.exit_code == 5
    assert caught.value.capability == "component:false_stateful.state_transition"


def test_result_is_read_from_machine_observed_pinned_writer_execution(tmp_path: Path) -> None:
    grader = VerifiersEnvGrader.from_rubric(_mixed_components())
    artifact = grader.produce_verified_output(tmp_path / "verifiers-output", rows=[_writer_row()])

    envelope = grader.read_output(artifact)

    assert envelope.score == 0.75
    assert envelope.components == {"deterministic": 1.0, "judge": 0.75, "stateful": 1.0}
    assert envelope.provenance["writer"] == "verifiers.utils.save_utils.save_outputs"
    assert envelope.provenance["receipt_scope"] == "local_process_execution"
    assert envelope.observations["tap_converter"]["trajectory_id"] == "example:mixed-1"
    assert artifact.trap_dependency_calls == 0


def test_pinned_writer_source_refuses_drift_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert verifier_module._VENDORED_WRITER.is_relative_to(
        Path(verifier_module.__file__).resolve().parents[1]
    )
    drifted = tmp_path / "save_utils.py"
    drifted.write_bytes(verifier_module._VENDORED_WRITER.read_bytes() + b"\n# drift\n")
    monkeypatch.setattr(verifier_module, "_VENDORED_WRITER", drifted)
    grader = VerifiersEnvGrader.from_rubric(_mixed_components())

    with pytest.raises(AdapterCapabilityUnproven) as caught:
        grader.produce_verified_output(tmp_path / "output", rows=[_writer_row()])

    assert caught.value.code == "RL0702"
    assert caught.value.capability == "writer.source_digest"


def test_hand_authored_row_with_forged_metadata_is_not_an_observed_writer_receipt(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "forged-output"
    output_dir.mkdir()
    output = output_dir / "results.jsonl"
    output.write_text(json.dumps(_writer_row()) + "\n", encoding="utf-8")
    forged = VerifiedWriterArtifact(
        results_path=output_dir,
        framework_revision="edafab857aabf830237068193150694cfae50c3b",
        writer="verifiers.utils.save_utils.save_outputs",
        command=("save_outputs", "claimed"),
        output_sha256="claimed-by-caller",
        writer_source_sha256="claimed-by-caller",
        trap_dependency_calls=0,
    )
    grader = VerifiersEnvGrader.from_rubric(_mixed_components())

    with pytest.raises(AdapterCapabilityUnproven) as caught:
        grader.read_output(forged)

    assert caught.value.code == "RL0702"
    assert caught.value.exit_code == 5
    assert caught.value.capability == "provenance.execution_receipt"
