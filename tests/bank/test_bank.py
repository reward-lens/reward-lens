from __future__ import annotations

import hashlib
import json
import os
import random
from decimal import Decimal
from pathlib import Path

import pytest

from reward_lens.bank import (
    BUDGET_LADDER,
    BankProvenance,
    Response,
    ResponseBank,
    TieBreak,
    best_of_n,
    load_bank,
    require_bank,
)
from reward_lens.contracts import canonical_bytes
from reward_lens.execution.limits import Limits
from reward_lens.execution.linux import LinuxSandbox
from reward_lens.graders.python_grader import PythonGrader


def _campaign_cache() -> Path:
    configured = os.environ.get("REWARD_LENS_CAMPAIGN_CACHE")
    if not configured:
        pytest.fail(
            "set REWARD_LENS_CAMPAIGN_CACHE to run the required campaign bank integration checks"
        )
    root = Path(configured)
    if not root.is_dir():
        pytest.fail(f"REWARD_LENS_CAMPAIGN_CACHE is not a directory: {root}")
    return root


class ConstantScorer:
    def score(self, task, response, **_kwargs):
        return Decimal("1")


class TextScorer:
    def score(self, task, response, **_kwargs):
        return Decimal(len(response))


@pytest.fixture
def execution_context() -> tuple[LinuxSandbox, Limits]:
    """A concrete local CPU sandbox context, including no dummy execution values."""
    return LinuxSandbox(tier="T0"), Limits(wall_s=5, cpu_s=5, network=True)


@pytest.fixture
def bank(tmp_path: Path) -> ResponseBank:
    source = tmp_path / "small.jsonl"
    source.write_text(
        '{"name":"small","source":"fixture","revision":"r1",'
        '"generating_command":"build small"}\n'
        '{"task_id":"task-1","responses":[{"text":"a"},{"text":"bb"},'
        '{"text":"ccc"},{"text":"dddd"}]}\n'
        '{"task_id":"task-2","responses":[{"text":"e"},{"text":"ff"},'
        '{"text":"ggg"},{"text":"hhhh"},{"text":"iiiii"},{"text":"jjjjjj"}]}\n',
        encoding="utf-8",
    )
    return load_bank(source, name="small")


def test_bank_carries_canonical_digest_and_source_provenance(bank: ResponseBank) -> None:
    assert bank.id == "small"
    assert len(bank.digest) == 64
    assert isinstance(bank.provenance, BankProvenance)
    assert bank.provenance.source_path.endswith("small.jsonl")
    assert bank.provenance.generating_command == "build small"
    assert bank.provenance.revision == "r1"
    assert bank.N == 4
    assert bank.n_per_task == {"task-1": 4, "task-2": 6}
    rows = [json.loads(line) for line in Path(bank.provenance.source_path).read_text().splitlines()]
    assert bank.digest == hashlib.sha256(canonical_bytes({"loaded_rows": rows})).hexdigest()
    assert all(
        isinstance(response, Response) and len(response.digest) == 64
        for task in bank.tasks
        for response in bank.responses_for(task)
    )


@pytest.mark.parametrize("name", ("hack-probes", "diagnostic-v3", "rb2-patch-40"))
def test_real_campaign_banks_load_and_select(
    name: str, execution_context: tuple[LinuxSandbox, Limits]
) -> None:
    bank = load_bank(_campaign_cache() / f"{name}.jsonl", name=name)
    sandbox, limits = execution_context
    result = best_of_n(
        bank,
        TextScorer(),
        n=1,
        seed=7,
        tie_break=TieBreak.FIRST_BY_DIGEST,
        sandbox=sandbox,
        limits=limits,
    )
    assert bank.N > 0
    assert bank.N == min(bank.n_per_task.values())
    assert bank.provenance.generating_command is None
    assert result.chosen
    assert result.tie_break is TieBreak.FIRST_BY_DIGEST


def test_best_of_n_is_seed_repeatable_and_records_the_tie_break(
    bank: ResponseBank, execution_context: tuple[LinuxSandbox, Limits]
) -> None:
    sandbox, limits = execution_context
    first = best_of_n(
        bank,
        ConstantScorer(),
        n=2,
        seed=11,
        tie_break=TieBreak.FIRST_BY_DIGEST,
        sandbox=sandbox,
        limits=limits,
    )
    second = best_of_n(
        bank,
        ConstantScorer(),
        n=2,
        seed=11,
        tie_break=TieBreak.FIRST_BY_DIGEST,
        sandbox=sandbox,
        limits=limits,
    )
    assert first == second
    assert first.tie_break is TieBreak.FIRST_BY_DIGEST
    assert first.ties_observed == 2
    replay = random.Random(11)
    for task in sorted(bank.tasks):
        sampled = replay.sample(list(bank.responses_for(task)), 2)
        assert first.chosen[task].digest == min(response.digest for response in sampled)


def test_budget_ladder_is_declared_and_budget_past_half_the_bank_refuses(
    bank: ResponseBank, execution_context: tuple[LinuxSandbox, Limits]
) -> None:
    assert BUDGET_LADDER == (1, 2, 4, 8, 16, 32)
    sandbox, limits = execution_context
    with pytest.raises(Exception) as caught:
        best_of_n(
            bank,
            ConstantScorer(),
            n=bank.N // 2 + 1,
            seed=0,
            tie_break=TieBreak.FIRST_BY_DIGEST,
            sandbox=sandbox,
            limits=limits,
        )
    error = caught.value
    assert error.code == "RL0132"
    assert error.exit_code == 4
    assert error.context == {"N": bank.N, "requested_budget": 3}
    assert "N=4" in error.message and "3" in error.message


def test_missing_bank_refuses_with_rl0134() -> None:
    with pytest.raises(Exception) as caught:
        require_bank(None)
    error = caught.value
    assert error.code == "RL0134"
    assert error.exit_code == 4
    assert error.context == {"bank": "required"}


def test_real_python_grader_receives_the_bank_task_and_response(
    tmp_path: Path, execution_context: tuple[LinuxSandbox, Limits]
) -> None:
    source = tmp_path / "real_grader.py"
    source.write_text(
        "def score(task, response):\n"
        "    if task.get('id') not in {'task-1', 'task-2'}:\n"
        "        raise AssertionError('wrong task id')\n"
        "    if task.get('prompt') not in {'first', 'second'}:\n"
        "        raise AssertionError('wrong prompt')\n"
        "    if response != 'safe':\n"
        "        raise AssertionError('wrong response')\n"
        "    return 1.0\n",
        encoding="utf-8",
    )
    bank_path = tmp_path / "real.jsonl"
    bank_path.write_text(
        '{"name":"real"}\n'
        '{"task_id":"task-1","prompt":"first","responses":[{"text":"safe"},{"text":"safe"}]}\n'
        '{"task_id":"task-2","prompt":"second","responses":[{"text":"safe"},{"text":"safe"}]}\n',
        encoding="utf-8",
    )
    sandbox, limits = execution_context
    selected = best_of_n(
        load_bank(bank_path, name="real"),
        PythonGrader.from_path(source),
        n=1,
        seed=0,
        tie_break=TieBreak.FIRST_BY_DIGEST,
        sandbox=sandbox,
        limits=limits,
    )
    assert {response.text for response in selected.chosen.values()} == {"safe"}


@pytest.mark.parametrize("body", ("return None", "raise RuntimeError('boom')"))
def test_unscored_and_grader_error_envelopes_are_not_selected(
    tmp_path: Path, execution_context: tuple[LinuxSandbox, Limits], body: str
) -> None:
    source = tmp_path / "negative_grader.py"
    source.write_text(f"def score(task, response):\n    {body}\n", encoding="utf-8")
    bank_path = tmp_path / "negative.jsonl"
    bank_path.write_text(
        '{"name":"negative"}\n'
        '{"task_id":"task","prompt":"one","responses":[{"text":"safe"},{"text":"safe"}]}\n',
        encoding="utf-8",
    )
    sandbox, limits = execution_context
    with pytest.raises(ValueError, match="unscored"):
        best_of_n(
            load_bank(bank_path, name="negative"),
            PythonGrader.from_path(source),
            n=1,
            seed=0,
            tie_break=TieBreak.FIRST_BY_DIGEST,
            sandbox=sandbox,
            limits=limits,
        )


def test_loaded_and_selected_state_rejects_mutation_without_digest_drift(
    bank: ResponseBank, execution_context: tuple[LinuxSandbox, Limits]
) -> None:
    digest = bank.digest
    task = bank.tasks[0]
    response = bank.responses_for(task)[0]
    sandbox, limits = execution_context
    selected = best_of_n(
        bank,
        ConstantScorer(),
        n=1,
        seed=0,
        tie_break=TieBreak.FIRST_BY_DIGEST,
        sandbox=sandbox,
        limits=limits,
    )
    with pytest.raises(TypeError):
        bank.responses[task] = ()
    with pytest.raises(TypeError):
        bank.task_payloads[task] = {"id": "mutated"}
    with pytest.raises(TypeError):
        selected.chosen[task] = response
    assert bank.digest == digest
    assert bank.responses_for(task) == tuple(bank.responses_for(task))


def test_input_mappings_are_defensively_copied() -> None:
    scores = {"judge": Decimal("1")}
    response = Response(digest="a" * 64, task_id="task", text="text", recorded_scores=scores)
    responses = {"task": (response,)}
    payloads = {"task": {"id": "task", "prompt": "p", "nested": ["old"]}}
    bank = ResponseBank(
        id="manual",
        digest="b" * 64,
        provenance=BankProvenance("source", None, None, "c" * 64),
        tasks=("task",),
        responses=responses,
        task_payloads=payloads,
    )
    scores["judge"] = Decimal("0")
    responses["task"] = ()
    payloads["task"]["nested"].append("new")
    assert bank.responses_for("task")[0].recorded_scores == {"judge": Decimal("1")}
    assert bank.task_for("task")["nested"] == ("old",)
