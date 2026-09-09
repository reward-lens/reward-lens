"""`fixtures/my_grader.py`, the bare grader of section 5.3: deterministic, `None` on malformed input."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "reward_lens"
    / "examples"
    / "code_reward"
    / "fixtures"
    / "my_grader.py"
)

MALFORMED = [
    None,
    {},
    {"prompt": "add two numbers"},
    {"prompt": "add two numbers", "expected": None},
    {"prompt": 7, "expected": "x"},
    [],
    "not a task at all",
]


@pytest.fixture(scope="session")
def my_grader():
    spec = importlib.util.spec_from_file_location("demo_my_grader", FIXTURE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["demo_my_grader"] = module
    spec.loader.exec_module(module)
    return module


def test_returns_none_on_malformed_input(my_grader) -> None:
    for task in MALFORMED:
        assert my_grader.score(task, "anything") is None, task
    assert my_grader.score({"prompt": "p", "expected": "yes"}, None) is None
    assert my_grader.score({"prompt": "p", "expected": "yes"}, 12) is None


def test_returns_a_float_on_a_valid_task(my_grader) -> None:
    value = my_grader.score({"prompt": "say yes", "expected": "yes"}, "yes")
    assert isinstance(value, float) and not isinstance(value, bool)
    assert 0.0 <= value <= 1.0


def test_identical_scores_over_one_hundred_repeats(my_grader) -> None:
    task = {"prompt": "name the capital of France", "expected": "Paris is the capital"}
    response = "I think Paris is the capital of France."
    values = {my_grader.score(task, response) for _ in range(100)}
    assert len(values) == 1
