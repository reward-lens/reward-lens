"""Pinned-TRL CPU conformance for the rung-five optimizer-step clamp."""

from __future__ import annotations

import importlib.metadata
import json
import random
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from reward_lens.learning.contracts import EIGHT_OBJECTS, TrainerProfile
from reward_lens.learning.intervention.clamp import (
    ClampSpecification,
    PinnedShortForkExecutor,
    build_clamp_callback,
    run_short_fork,
)
from reward_lens.learning.intervention.swap import (
    Intervention,
    MaterializedBatch,
    TRL110StateAdapter,
    capture_state,
    same_start,
)
from reward_lens.product.compare.rung5 import (
    CPUConformanceRecordInvalid,
    Rung5Input,
    compare_verified_short_fork,
    load_cpu_conformance_record,
    write_cpu_conformance_record,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "learning" / "cpu_toy" / "fork"


def _fixture() -> dict[str, object]:
    return json.loads((FIXTURE / "fork.json").read_text(encoding="utf-8"))


def _profile() -> TrainerProfile:
    return TrainerProfile("1.10.0", EIGHT_OBJECTS, "cpu-toy-fork-v2")


def _pinned_cpu_route():
    """Construct the fixture's actual TRL 1.10.0 trainer and scored replay."""
    tests_root = Path(__file__).parents[1].resolve()
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != tests_root]
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from trl import GRPOConfig, GRPOTrainer

    fixture = _fixture()
    assert importlib.metadata.version("trl") == "1.10.0"
    torch.set_num_threads(int(fixture["limits"]["cpu_threads"]))
    random.seed(int(fixture["seed"]))
    np.random.seed(int(fixture["seed"]))
    torch.manual_seed(int(fixture["seed"]))
    vocab = {token: index for index, token in enumerate(fixture["vocab"])}
    raw = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="<pad>", unk_token="<unk>", eos_token="<eos>"
    )
    model_config = fixture["model"]
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocab),
            n_positions=model_config["n_positions"],
            n_ctx=model_config["n_positions"],
            n_embd=model_config["n_embd"],
            n_layer=model_config["n_layer"],
            n_head=model_config["n_head"],
            bos_token_id=vocab["<eos>"],
            eos_token_id=vocab["<eos>"],
            pad_token_id=vocab["<pad>"],
        )
    )
    target = fixture["target"]
    row, column = target["embedding_index"]
    with torch.no_grad():
        model.get_input_embeddings().weight[row, column] = float(target["value"])
    scorer_calls = 0

    def reward(prompts, completions, completion_ids, **kwargs):
        nonlocal scorer_calls
        scorer_calls += 1
        return [float(index % 2 + scorer_calls - 1) for index, _ in enumerate(completions)]

    training = fixture["training"]
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward,
        args=GRPOConfig(
            output_dir=tempfile.mkdtemp(prefix="p-learn-fork-cpu-"),
            per_device_train_batch_size=training["batch_size"],
            num_generations=training["generations"],
            max_completion_length=training["completion_length"],
            max_steps=1,
            logging_strategy="no",
            report_to=[],
            use_cpu=True,
            optim="adamw_torch",
            save_strategy="no",
            disable_tqdm=True,
            gradient_checkpointing=False,
            learning_rate=training["learning_rate"],
        ),
        train_dataset=Dataset.from_dict({"prompt": fixture["prompts"]}),
        processing_class=tokenizer,
    )
    trainer.create_optimizer_and_scheduler(num_training_steps=1)
    adapter = TRL110StateAdapter(trainer)
    callback = build_clamp_callback(
        _profile(),
        ClampSpecification(
            target_name=target["name"],
            target=lambda current: current.get_input_embeddings().weight[row, column],
            target_value=float(target["value"]),
            tolerance=float(target["tolerance"]),
            max_correction=float(fixture["limits"]["max_correction"]),
            max_iterations=int(fixture["limits"]["max_iterations"]),
        ),
    )
    callback.attach(trainer.optimizer, trainer.model)
    batch = MaterializedBatch(tuple({"prompt": prompt} for prompt in fixture["prompts"]))
    executor = PinnedShortForkExecutor(adapter, batch, _profile(), callback)
    useful = fixture["useful_task"]
    useful_inputs = tokenizer(useful["prompt"], return_tensors="pt")["input_ids"]
    useful_token = vocab[useful["token"]]

    def useful_task(current) -> float:
        with torch.no_grad():
            return float(current(useful_inputs).logits[0, -1, useful_token].cpu())

    return fixture, adapter, callback, executor, useful_task, scorer_calls


def test_fixture_constructs_a_real_pinned_trl_short_fork_with_replay_controls(
    tmp_path: Path,
) -> None:
    fixture, adapter, callback, executor, useful_task, scorer_calls = _pinned_cpu_route()
    offsets = fixture["interventions"]
    useful_name = fixture["useful_task"]["name"]
    result = run_short_fork(
        (
            executor.arm(
                "null", Intervention("null", tuple(offsets["null"])), {useful_name: useful_task}
            ),
            executor.arm(
                "null", Intervention("null", tuple(offsets["null"])), {useful_name: useful_task}
            ),
            executor.arm(
                "candidate",
                Intervention("candidate", tuple(offsets["candidate"])),
                {useful_name: useful_task},
            ),
            executor.arm(
                "placebo",
                Intervention("placebo", tuple(offsets["placebo"])),
                {useful_name: useful_task},
            ),
        ),
        profile=_profile(),
        clamp=callback,
        fixture_root=FIXTURE,
        limits=fixture["limits"],
    )
    record_path = tmp_path / "cpu-conformance.json"
    write_cpu_conformance_record(result, fixture_root=FIXTURE, path=record_path)
    record = load_cpu_conformance_record(record_path, fixture_root=FIXTURE)

    assert result.scope == "short_fork_behavior"
    assert result.null_spread_value == 0.0  # computed from real deterministic null executions
    assert adapter.generation_count == adapter.source_scorer_count == 1
    assert scorer_calls == 1  # source scorer ran once during freeze and never on an arm replay
    assert len(callback.readings) == len(result.arm_names) == 4
    assert all(reading.held for reading in callback.readings)
    assert result.target_values[2] == pytest.approx(float(fixture["target"]["value"]), abs=1e-6)
    assert len(result.useful_task_controls[useful_name]) == 4
    assert [witness["arm"] for witness in executor.witnesses] == [
        "null",
        "null",
        "candidate",
        "placebo",
    ]
    # The pinned trainer uses token-weighted DAPO loss. The frozen completions
    # have unequal non-padding lengths, so zero-sum group advantages do not
    # imply zero loss. Check the objective against the measured advantages and
    # frozen completion lengths, including distinct intervention outcomes.
    assert adapter.trainer.loss_type == "dapo"
    assert adapter.trainer.beta == 0.0
    assert adapter.trainer.current_gradient_accumulation_steps == 1
    completion_lengths = tuple(len(ids) for ids in executor.replay.generation_payload[1])
    assert completion_lengths == (1, 2)
    losses = [witness["value"]["loss"] for witness in executor.witnesses]
    for witness, loss in zip(executor.witnesses, losses):
        advantages = witness["value"]["advantages"]
        expected = -sum(
            advantage * length for advantage, length in zip(advantages, completion_lengths)
        ) / sum(completion_lengths)
        assert loss == pytest.approx(expected, abs=1e-6)
    assert losses[0] == losses[1]
    assert len(set(losses)) == 3
    assert len({witness["value"]["loss_input_digest"] for witness in executor.witnesses}) == 1
    assert all(
        witness["value"]["gradient_norm"] > 0.0
        and witness["value"]["optimizer_update_norm"] > 0.0
        and len(witness["value"]["reward_witness"]) == 1
        for witness in executor.witnesses
    )
    assert capture_state(adapter, _profile()).digest() == executor.baseline.digest()
    assert record["profile"]["mutable_objects"] == list(EIGHT_OBJECTS)
    assert record["profile"]["fingerprint"] == _profile().fingerprint
    from reward_lens.contracts import EntryProvenance

    # This is the original unbound test fixture. It remains valid for the real
    # CPU execution and record checks above, but must never acquire a claimed
    # external subject through the canonical entry path.
    with pytest.raises(CPUConformanceRecordInvalid, match="legacy unbound"):
        compare_verified_short_fork(
            Rung5Input(
                result=result,
                record_path=record_path,
                fixture_root=FIXTURE,
                subject_ref="sha256:" + "2" * 64,
                provenance=EntryProvenance(
                    started="2026-09-26T00:00:00Z", duration_s=0.0, sandbox_tier="T0", offline=True
                ),
            )
        )


def test_pinned_short_fork_names_the_first_same_start_divergence() -> None:
    _fixture_value, _adapter, _callback, executor, _useful_task, _scorer_calls = _pinned_cpu_route()
    changed = replace(executor.baseline, learning_rate=executor.baseline.learning_rate * 2)
    with pytest.raises(ValueError, match="SAME_START_REFUSAL: divergent learning rate"):
        same_start(changed, executor.baseline).require()
