from __future__ import annotations

from dataclasses import fields

import pytest

from reward_lens.learning.contracts import (
    EIGHT_OBJECTS,
    ArmResult,
    ContractUndeclaredKey,
    RolloutInputs,
    SamplingProvenance,
    SamplingProvenanceIncomplete,
    TrainerProfile,
    TrainerStateIncomplete,
    TrainerUnsupported,
)


def test_rollout_inputs_preserves_the_snapshot_twenty_field_schema() -> None:
    assert tuple(field.name for field in fields(RolloutInputs)) == (
        "step",
        "group",
        "gen",
        "prompt",
        "completion",
        "token_ids",
        "logprobs_sampling",
        "logprobs_train",
        "logprobs_ref",
        "loss_mask",
        "component_scores",
        "truncated",
        "advantage",
        "training_passed",
        "oracle_stdout",
        "oracle_expected_n",
        "oracle_completed",
        "oracle_case_passed",
        "solution_code",
        "extra_files",
    )


@pytest.mark.parametrize("key", ["applied_weight", "oracle_fraction", "unknown"])
def test_rollout_inputs_refuses_undeclared_and_derived_keys(key: str) -> None:
    with pytest.raises(ContractUndeclaredKey, match=key):
        RolloutInputs.from_mapping({"step": 0, "group": 0, "gen": 0, key: 1})


def test_arm_result_preserves_the_actual_snapshot_schema_and_refuses_unknown_key() -> None:
    assert tuple(field.name for field in fields(ArmResult)) == (
        "adapter_path",
        "rollouts",
        "container_ids",
        "t_step_s",
        "t_gen_s",
        "arm",
        "arm_kind",
        "grader",
        "step_start",
        "step_stop",
        "steps_run",
        "resumed_from",
        "resume_restored",
        "group_witness",
        "a2_step0",
        "step_metrics_path",
        "instance_path",
        "checkpoints",
        "parameters",
        "instance",
        "telemetry",
    )
    with pytest.raises(ContractUndeclaredKey, match="future_field"):
        ArmResult.from_mapping({"future_field": True})


def test_profile_requires_trl_1_10_0_and_all_eight_objects() -> None:
    assert EIGHT_OBJECTS == (
        "adapter_weights",
        "adam_moments",
        "scheduler_counter",
        "rng_states",
        "dataloader_position",
        "buffered_inputs",
        "trl_step",
        "state_global_step",
    )
    with pytest.raises(TrainerUnsupported, match="1.9.0"):
        TrainerProfile(version="1.9.0", mutable_objects=EIGHT_OBJECTS, fingerprint="abc")
    with pytest.raises(TrainerStateIncomplete, match="rng_states"):
        TrainerProfile(version="1.10.0", mutable_objects=EIGHT_OBJECTS[:-5], fingerprint="abc")


def test_sampling_provenance_requires_every_decoding_field_and_field_provenance() -> None:
    values = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "min_p": None,
        "repetition_penalty": 1.0,
    }
    provenance = {name: "registered.row" for name in values}
    sampling = SamplingProvenance(values=values, provenance=provenance)
    assert sampling.values == values
    with pytest.raises(SamplingProvenanceIncomplete, match="top_k"):
        SamplingProvenance(
            values={k: v for k, v in values.items() if k != "top_k"}, provenance=provenance
        )
