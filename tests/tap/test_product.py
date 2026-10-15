"""Contract tests for the product-facing trainer reward tap.

They are intentionally red until P-TAP supplies ``reward_lens.tap.product``.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import pytest

from reward_lens.tap.product import (
    TAP_VALUE_CHANGED,
    TRAINER_UNSUPPORTED,
    TappedReward,
    TappedRubric,
    assert_value_unchanged,
    tap_reward,
    tap_rubric,
    verify_cpu_conformance_receipt,
    write_cpu_conformance_receipt,
)


class RecordingRunContext:
    def __init__(self, *, trainer_version: str = "1.13.0", step: int = 7) -> None:
        self.trainer_version = trainer_version
        self.step = step
        self.events: list[dict[str, object]] = []

    def append_tap_event(self, event: dict[str, object]) -> None:
        self.events.append(event)


def test_trl_reward_returns_the_exact_trainer_object_and_records_run_fields() -> None:
    result = [0.0, None, {"component": 1.25}]
    context = RecordingRunContext()

    def reward(*, prompts, completions, completion_ids, **kwargs):
        return result

    wrapped = tap_reward(reward, run_context=context)
    observed = wrapped(prompts=["p"], completions=["c"], completion_ids=[[1]])

    assert observed is result
    assert len(context.events) == 1
    event = context.events[0]
    assert event["adapter"] == "trl_reward"
    assert event["step"] == 7
    assert event["outcome"] == "returned"
    assert "value" not in event
    assert event["value_shape"]["kind"] == "sequence"
    assert event["effect"]["invocations"] == 1


def test_verifiers_rubric_returns_the_exact_value_and_signature_shape() -> None:
    value = {"rubric": [0.0, 1.0]}
    context = RecordingRunContext()

    def rubric(state, *, rubric_weight=1.0):
        return value

    wrapped = tap_rubric(rubric, run_context=context)
    assert wrapped({"completion": "x"}, rubric_weight=0.25) is value
    assert context.events[0]["adapter"] == "verifiers_rubric"
    assert "value" not in context.events[0]
    assert context.events[0]["value_shape"]["kind"] == "mapping"


def test_async_trl_reward_stays_async_and_preserves_the_exact_object() -> None:
    value = [1.0]
    context = RecordingRunContext()

    async def reward(*, prompts, completions, completion_ids, **kwargs):
        return value

    wrapped = tap_reward(reward, run_context=context)
    assert asyncio.iscoroutinefunction(wrapped)
    assert asyncio.run(wrapped(prompts=["p"], completions=["c"], completion_ids=[[1]])) is value
    assert "value" not in context.events[0]


def test_trainer_outside_supported_profile_refuses_with_its_version() -> None:
    context = RecordingRunContext(trainer_version="2.0.0")

    with pytest.raises(TRAINER_UNSUPPORTED, match=r"2\.0\.0"):
        tap_reward(lambda **_: [1.0], run_context=context)


def test_changed_value_refusal_names_wrapper_and_fixture() -> None:
    with pytest.raises(TAP_VALUE_CHANGED, match=r"wrapper=trl_reward fixture=identity-fixture"):
        assert_value_unchanged(
            [1.0],
            [1.0],
            wrapper="trl_reward",
            fixture="identity-fixture",
        )


def test_tap_recorder_failure_cannot_change_a_trainer_value() -> None:
    class BrokenContext(RecordingRunContext):
        def append_tap_event(self, event: dict[str, object]) -> None:
            raise RuntimeError("full sink")

    value = [None]
    assert tap_reward(lambda **_: value, run_context=BrokenContext())() is value


def test_mutating_event_sink_cannot_change_the_trainer_return_value() -> None:
    class MutatingContext(RecordingRunContext):
        def append_tap_event(self, event: dict[str, object]) -> None:
            assert "value" not in event
            event["value_shape"]["kind"] = "sink-mutated"
            super().append_tap_event(event)

    value = [1.0]
    observed = tap_reward(lambda **_: value, run_context=MutatingContext())()

    assert observed is value
    assert observed == [1.0]


def test_full_product_recording_effect_includes_a_slow_event_sink() -> None:
    class SlowContext(RecordingRunContext):
        def append_tap_event(self, event: dict[str, object]) -> None:
            time.sleep(0.02)
            super().append_tap_event(event)

    wrapped = tap_reward(lambda **_: [1.0], run_context=SlowContext())
    wrapped()
    assert wrapped.product_effect().added_ns_total >= 20_000_000


def test_public_tapped_result_types_are_the_returned_callable_interface() -> None:
    context = RecordingRunContext()
    assert isinstance(tap_reward(lambda **_: [1.0], run_context=context), TappedReward)
    assert isinstance(tap_rubric(lambda **_: 1.0, run_context=context), TappedRubric)


def test_tap_preserves_a_grader_exception_and_records_its_outcome() -> None:
    context = RecordingRunContext()
    original = ValueError("original grader failure")

    def reward(**kwargs):
        raise original

    wrapped = tap_reward(reward, run_context=context)
    with pytest.raises(ValueError, match="original grader failure") as raised:
        wrapped()

    assert raised.value is original
    assert context.events[0]["outcome"] == "raised"
    assert context.events[0]["exception_type"] == "ValueError"


def test_real_short_cpu_trl_training_preserves_rewards_and_measures_overhead(
    tmp_path: Path,
) -> None:
    """One real, local GRPO update with random weights; no model or tokenizer is downloaded."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["ACCELERATE_USE_CPU"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Pytest inserts `tests/` ahead of installed distributions. This repository has a test helper
    # package named `packaging`, so remove only that test root before datasets imports its real
    # dependency. It is fixture isolation, never a patch to datasets, TRL, or transformers.
    tests_root = Path(__file__).parents[1].resolve()
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != tests_root]

    import importlib.metadata

    import torch
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from trl import GRPOConfig, GRPOTrainer

    assert importlib.metadata.version("trl") == "1.13.0"
    torch.set_num_threads(1)
    vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2, "say": 3, "one": 4, "two": 5}

    def local_tokenizer() -> PreTrainedTokenizerFast:
        raw = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        raw.pre_tokenizer = Whitespace()
        return PreTrainedTokenizerFast(
            tokenizer_object=raw,
            pad_token="<pad>",
            unk_token="<unk>",
            eos_token="<eos>",
        )

    def run(*, tapped: bool) -> tuple[int, RecordingRunContext | None, list[list[float]], object]:
        torch.manual_seed(1234)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(vocab),
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
        returned: list[list[float]] = []

        def reward(prompts, completions, completion_ids, **kwargs):
            value = [float(bool(text)) for text in completions]
            returned.append(value)
            return value

        context = RecordingRunContext(trainer_version="1.13.0", step=0) if tapped else None
        reward_func = tap_reward(reward, run_context=context) if context else reward
        args = GRPOConfig(
            output_dir=tempfile.mkdtemp(prefix="p-tap-cpu-"),
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
        )
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_func,
            args=args,
            train_dataset=Dataset.from_dict({"prompt": ["say one", "say two"]}),
            processing_class=local_tokenizer(),
        )
        assert trainer.accelerator.device.type == "cpu"
        started = time.perf_counter_ns()
        trainer.train()
        return time.perf_counter_ns() - started, context, returned, reward_func

    # The first call bears one-time framework initialization. The remaining matched bare/tapped
    # arms are a wall-clock characterization, while `effect` is the exact tap-side measurement.
    run(tapped=False)
    bare = [run(tapped=False)[0] for _ in range(2)]
    tapped_arms = [run(tapped=True) for _ in range(2)]
    tapped = [arm[0] for arm in tapped_arms]
    context = tapped_arms[-1][1]
    returned = tapped_arms[-1][2]
    tapped_reward = tapped_arms[-1][3]
    assert context is not None
    assert len(returned) == len(context.events) == 1
    assert "value" not in context.events[0]
    assert context.events[0]["effect"]["invocations"] == 1
    assert context.events[0]["effect"]["added_ns_total"] > 0
    product_effect = tapped_reward.product_effect().as_record()
    assert product_effect["calls"] == 1
    assert product_effect["added_ns_total"] > 0

    bare_median = statistics.median(bare)
    tapped_median = statistics.median(tapped)
    measurement = {
        "product_calls": product_effect["calls"],
        "product_added_ns_total": product_effect["added_ns_total"],
        "product_added_ns_p50": product_effect["added_ns_p50"],
        "product_added_ns_p99": product_effect["added_ns_p99"],
        "product_added_ns_max": product_effect["added_ns_max"],
        "inner_added_ns_total": context.events[0]["effect"]["added_ns_total"],
        "wall_bare_median_ns": bare_median,
        "wall_tapped_median_ns": tapped_median,
    }
    fixture_id = "tap-product-local-random-gpt2-cpu-v1"
    receipt_path = Path(
        os.environ.get(
            "REWARD_LENS_TAP_RECEIPT_PATH", str(tmp_path / "cpu-conformance-receipt.json")
        )
    )
    receipt = write_cpu_conformance_receipt(
        receipt_path,
        fixture_id=fixture_id,
        runtime={
            "trl": __import__("trl").__version__,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
        measurement=measurement,
        constraints={"device": "cpu", "threads": 1, "model": "random GPT2Config"},
    )
    assert verify_cpu_conformance_receipt(receipt_path, fixture_id=fixture_id) == receipt
    tampered = json.loads(receipt_path.read_text())
    tampered["measurement"]["product_added_ns_total"] = 0
    receipt_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="TAP_RECEIPT_TAMPERED"):
        verify_cpu_conformance_receipt(receipt_path, fixture_id=fixture_id)
    tampered = json.loads(json.dumps(receipt))
    del tampered["measurement"]["product_added_ns_p99"]
    receipt_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="measurement.product_added_ns_p99"):
        verify_cpu_conformance_receipt(receipt_path, fixture_id=fixture_id)
    tampered = json.loads(json.dumps(receipt))
    tampered["fixture"]["id"] = "other-fixture"
    receipt_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="fixture.id"):
        verify_cpu_conformance_receipt(receipt_path, fixture_id=fixture_id)
    # Leave a valid generated receipt for an explicitly requested evidence path after exercising
    # both tamper controls. Ordinary pytest uses tmp_path and leaves no checked-in result.
    write_cpu_conformance_receipt(
        receipt_path,
        fixture_id=fixture_id,
        runtime=receipt["runtime"],
        measurement=measurement,
        constraints=receipt["constraints"],
    )

    print(
        "P-TAP local CPU GRPO fixture: "
        f"bare_median_ns={bare_median} tapped_median_ns={tapped_median} "
        f"wall_delta_ns={tapped_median - bare_median} "
        f"inner_tap_added_ns={context.events[0]['effect']['added_ns_total']} "
        f"product_tap_added_ns={product_effect['added_ns_total']} "
        f"receipt={receipt_path.name} fixture_id={fixture_id}"
    )
