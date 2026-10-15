"""Controlled same-batch swap checks on a CPU-only tiny trainer protocol."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import pytest

from reward_lens.learning.contracts import EIGHT_OBJECTS, TrainerProfile
from reward_lens.learning.contracts.errors import TrainerStateIncomplete, TrainerUnsupported
from reward_lens.learning.intervention.swap import (
    Intervention,
    MaterializedBatch,
    ScoredReplayBundle,
    SwapArm,
    TrainerState,
    TRL110StateAdapter,
    capture_state,
    restore_state,
    run_swap,
    same_start,
)


@dataclass
class _State:
    global_step: int


class _TinyTrainer:
    """A local CPU trainer protocol with each object a controlled step mutates."""

    def __init__(self) -> None:
        self.adapter_weights = [[1.0, 2.0], [3.0, 4.0]]
        self.adam_moments = {"step": 0, "exp_avg": [0.0, 0.0], "exp_avg_sq": [0.0, 0.0]}
        self.scheduler_counter = 0
        self.rng_states = {"python": 7, "numpy": 11, "torch": 13, "cuda": 17}
        self.dataloader_position = 0
        self.buffered_inputs = {"token_ids": [1, 2]}
        self.trl_step = 0
        self.state = _State(global_step=0)
        self.learning_rate = 0.125
        self.fused_loss = False

    def update(self, batch: dict[str, int], delta: int) -> int:
        self.adapter_weights[0][0] += delta
        self.adam_moments["step"] += 1
        self.adam_moments["exp_avg"][0] += delta
        self.adam_moments["exp_avg_sq"][0] += delta * delta
        self.scheduler_counter += 1
        self.rng_states["python"] += 1
        self.dataloader_position += 1
        self.buffered_inputs = {"token_ids": [batch["token"]]}
        self.trl_step += 1
        self.state.global_step += 1
        return delta


def _profile() -> TrainerProfile:
    return TrainerProfile("1.10.0", EIGHT_OBJECTS, "cpu-toy-trainer-v1")


def _state_with(start: TrainerState, **changes: object) -> TrainerState:
    fields = {
        "values": start.values,
        "learning_rate": start.learning_rate,
        "parameter_shape": start.parameter_shape,
        "optimizer_step_count": start.optimizer_step_count,
        "fused_loss": start.fused_loss,
    }
    fields.update(changes)
    return TrainerState(**fields)


def test_swap_adapter_has_a_public_controlled_update_surface() -> None:
    from reward_lens.learning.intervention.swap import run_swap

    assert callable(run_swap)


def test_capture_and_restore_all_eight_objects_proves_digest_equality() -> None:
    trainer = _TinyTrainer()
    before = capture_state(trainer, _profile())
    trainer.update({"token": 9}, 4)
    assert capture_state(trainer, _profile()).digest() != before.digest()
    restore_state(trainer, before, _profile())
    assert capture_state(trainer, _profile()).digest() == before.digest()


def test_capture_refuses_trainer_missing_one_of_eight_objects() -> None:
    trainer = _TinyTrainer()
    del trainer.buffered_inputs
    with pytest.raises(TrainerStateIncomplete, match="buffered_inputs"):
        capture_state(trainer, _profile())


def test_profile_refuses_an_unpinned_trainer_version() -> None:
    with pytest.raises(TrainerUnsupported, match="1.13.0"):
        TrainerProfile("1.13.0", EIGHT_OBJECTS, "wrong-version")


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"learning_rate": 0.5}, "learning rate"),
        ({"parameter_shape": (99,)}, "parameter shape"),
        ({"optimizer_step_count": 4}, "optimizer step count"),
        ({"fused_loss": True}, "fused loss"),
        ({"values": {**_TinyTrainer().__dict__, "adapter_weights": [[999]]}}, "digest"),
    ],
)
def test_same_start_names_every_divergent_control(change: dict[str, object], field: str) -> None:
    start = capture_state(_TinyTrainer(), _profile())
    changed = _state_with(start, **change)
    verdict = same_start(changed, start)
    assert verdict.field == field
    with pytest.raises(ValueError, match=field):
        verdict.require()


def test_null_runs_first_placebo_is_first_class_and_each_candidate_runs_once() -> None:
    trainer = _TinyTrainer()
    calls: list[str] = []

    def arm(name: str, delta: int) -> SwapArm:
        def apply(host: _TinyTrainer, batch: dict[str, int]) -> int:
            calls.append(name)
            return host.update(batch, delta)

        return SwapArm(name, apply)

    result = run_swap(
        {"token": 42},
        [arm("reward_a", 2), arm("placebo", 0)],
        _profile(),
        trainer=trainer,
        null=arm("null", 1),
    )
    assert calls == ["null", "reward_a", "placebo"]
    assert result.arm_order == ("null", "reward_a", "placebo")
    assert result.start_digest == result.final_digest
    assert "mechanism conformance" in result.statement
    assert "not a behavioural result" in result.statement


def test_run_swap_refuses_to_downgrade_placebo_to_an_optional_control() -> None:
    with pytest.raises(ValueError, match="placebo"):
        run_swap(
            {"token": 1},
            [SwapArm("candidate", lambda host, batch: host.update(batch, 1))],
            _profile(),
            trainer=_TinyTrainer(),
            null=SwapArm("null", lambda host, batch: host.update(batch, 0)),
        )


def test_real_pinned_trl_110_cpu_trainer_restores_full_state_after_each_arm(tmp_path: Path) -> None:
    """A real local GRPOTrainer run with random weights and no remote model/data."""
    import importlib.metadata
    import os
    import random
    import sys
    import tempfile
    from pathlib import Path

    import numpy as np
    import torch

    # The repository's test helper package is named ``packaging``. Remove only
    # that test root before datasets imports the third-party distribution.
    tests_root = Path(__file__).parents[1].resolve()
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != tests_root]
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from trl import GRPOConfig, GRPOTrainer

    assert importlib.metadata.version("trl") == "1.10.0"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["ACCELERATE_USE_CPU"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)
    random.seed(31)
    np.random.seed(31)
    torch.manual_seed(31)

    raw = Tokenizer(
        WordLevel(
            vocab={"<pad>": 0, "<unk>": 1, "<eos>": 2, "say": 3, "one": 4, "two": 5},
            unk_token="<unk>",
        )
    )
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<eos>",
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=6,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=1,
            bos_token_id=2,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    assert next(model.parameters()).device.type == "cpu"

    source_scorer_calls = 0

    def reward(prompts, completions, completion_ids, **kwargs):
        # The source scorer is deliberately stateful: any escaped per-arm
        # rescore would return different advantages before an intervention.
        nonlocal source_scorer_calls
        source_scorer_calls += 1
        # Two fixed values per GRPO generation group make the null replay a
        # real nonzero update while retaining no intervention offset.
        return [float(index % 2 + source_scorer_calls - 1) for index, _ in enumerate(completions)]

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward,
        args=GRPOConfig(
            output_dir=tempfile.mkdtemp(prefix="p-learn-swap-cpu-"),
            per_device_train_batch_size=2,
            num_generations=2,
            max_completion_length=3,
            max_steps=1,
            logging_strategy="no",
            report_to=[],
            use_cpu=True,
            optim="adamw_torch",
            save_strategy="no",
            disable_tqdm=True,
            gradient_checkpointing=False,
            learning_rate=1e-4,
        ),
        train_dataset=Dataset.from_dict({"prompt": ["say one", "say two"]}),
        processing_class=tokenizer,
    )
    assert trainer.accelerator.device.type == "cpu"
    trainer.create_optimizer_and_scheduler(num_training_steps=1)
    # These are real pinned-GRPO private buffers. Freeze clears them on the
    # normal path, so the profile must restore their populated initial values.
    trainer._pending_extra_logs["packet"] = ["preserve"]
    trainer._pending_metrics["packet"] = [1.0]
    pending_extra_before = copy.deepcopy(trainer._pending_extra_logs)
    pending_metrics_before = copy.deepcopy(trainer._pending_metrics)
    adapter = TRL110StateAdapter(trainer)
    baseline = capture_state(adapter, _profile())
    private_before = adapter.private_profile_digest()
    calls: list[str] = []

    batch = MaterializedBatch(({"prompt": "say one"}, {"prompt": "say two"}))
    interventions = {
        "null": Intervention("null", (0.0, 0.0)),
        "candidate": Intervention("candidate", (0.25, -0.25)),
        "placebo": Intervention("placebo", (-0.25, 0.25)),
    }
    replay_seen: list[ScoredReplayBundle] = []

    def real_arm(name: str) -> SwapArm:
        def apply(host: TRL110StateAdapter, supplied: ScoredReplayBundle) -> object:
            calls.append(name)
            replay_seen.append(supplied)
            return host.run_one_update(replay=supplied, intervention=interventions[name])

        return SwapArm(name, apply)

    result = run_swap(
        batch,
        [real_arm("candidate"), real_arm("placebo")],
        _profile(),
        trainer=adapter,
        null=real_arm("null"),
    )
    # Consume this actual pinned-TRL run through the product producer.  The
    # producer receives the retained runtime witnesses, never a hand-built arm.
    from reward_lens import contracts
    from reward_lens.product.compare.rung4 import (
        ControlledUpdateEvidence,
        controlled_update_compare,
    )
    from reward_lens.product.trace.record import TraceRecord

    trace = TraceRecord(
        "run_record",
        "local-pinned-trainer",
        "p-learn-swap-cpu",
        (),
        {},
        {},
        "local",
        "local pinned CPU fixture",
        "strict",
        True,
        "complete local fixture",
        "sha256:" + "2" * 64,
        ("digest:training_semantics", "digest:samples", "digest:policy"),
    )
    compared = controlled_update_compare(
        result,
        ControlledUpdateEvidence(
            trace,
            contracts.EntryProvenance(
                started="2026-09-26T00:00:00Z", duration_s=0.0, sandbox_tier="T0", offline=True
            ),
        ),
    )
    assert compared.entries[0].kind == "check"
    assert compared.entries[0].check.passed is True
    assert compared.entries[0].result["null_replay"]["passed"] is True
    assert compared.entries[0].result["placebo"]["passed"] is True
    import json

    from reward_lens.api._record import absent_record
    from reward_lens.product.audit.run import persist_assay_bundle

    assay = absent_record(
        command="compare", reproduce=["local pinned CPU rung four fixture"], path=Path(__file__)
    )
    assay.measurement.reward_statistics = [compared.entries[0]]
    assay.holes_from_entries()
    assay.assay_id = contracts.digest(assay)
    contracts.validate_record(assay.to_dict())
    persisted, paths = persist_assay_bundle(
        assay, project=None, name="pinned-rung4", fallback_dir=tmp_path
    )
    reopened = contracts.Assay.model_validate(
        json.loads(paths["assay"].read_text(encoding="utf-8"))
    )
    contracts.validate_record(reopened.to_dict(), raw=paths["assay"].read_bytes())
    survived = reopened.measurement.reward_statistics[0]
    assert survived.result["trace_record"]["source"] == "local-pinned-trainer"
    assert survived.result["null_replay"]["passed"] is True
    assert survived.result["placebo"]["passed"] is True
    assert persisted.assay_id == reopened.assay_id
    assert calls == ["null", "candidate", "placebo"]
    assert adapter.generation_count == 1
    assert source_scorer_calls == adapter.source_scorer_count == 1
    assert {replay.digest for replay in replay_seen} == {replay_seen[0].digest}
    assert replay_seen[0].base_scores == ((0.0, 1.0),)
    assert replay_seen[0].base_reward_witness
    assert replay_seen[0].base_reward_witness[0]["source_values"] == (0.0, 1.0)
    assert replay_seen[0].base_reward_witness[0]["returned_values"] == (0.0, 1.0)
    assert adapter.private_profile_digest() == private_before
    assert trainer._pending_extra_logs == pending_extra_before
    assert trainer._pending_metrics == pending_metrics_before
    assert result.start_digest == baseline.digest() == result.final_digest
    assert result.profile.version == "1.10.0"
    scored_replay_digests: set[str] = set()
    loss_input_digests: set[str] = set()
    for name, witness in result.arms:
        arm = witness["value"]
        assert arm["batch_bytes"] == batch.bytes
        assert arm["batch_digest"] == batch.digest
        scored_replay_digests.add(arm["scored_replay_digest"])
        loss_input_digests.add(arm["loss_input_digest"])
        assert arm["intervention"] == interventions[name]
        assert arm["optimizer_update_norm"] > 0.0
        assert witness["during_digest"] != witness["restored_digest"]
        assert {
            "adam_moments",
            "scheduler_counter",
            "rng_states",
            "dataloader_position",
            "trl_step",
            "state_global_step",
        }.issubset(witness["changed_objects"])
        reward_witness = arm["reward_witness"]
        assert reward_witness
        for reward_call in reward_witness:
            assert reward_call["intervention"] == interventions[name]
            assert reward_call["source_values"] == replay_seen[0].base_scores[0]
            assert reward_call["returned_values"] == tuple(
                value
                + interventions[name].reward_offsets[
                    index % len(interventions[name].reward_offsets)
                ]
                for index, value in enumerate(reward_call["source_values"])
            )
    assert len(scored_replay_digests) == 1
    assert len(loss_input_digests) == 1
    inventory = adapter._private_inventory()
    trainer._unexpected_pinned_mutation = {"forbidden": True}
    with pytest.raises(ValueError, match="unexpected pinned-path mutation"):
        adapter._changed_private_fields(inventory)
    del trainer._unexpected_pinned_mutation
