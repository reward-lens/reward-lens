"""Shared fixtures for the code-reward example.

The demo tree is written once per session into a temporary directory, and the grader and the
independent outcome check are loaded from that written tree rather than from the package, because
what ships to a user is the written tree.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "reward_lens" / "examples" / "code_reward"


def load_module(path: Path, name: str):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass
class Demo:
    root: Path
    written: list[Path]
    tasks: list[dict]
    responses: list[dict]
    grader: object
    check: object

    def task(self, task_id: str) -> dict:
        for task in self.tasks:
            if task["task_id"] == task_id:
                return task
        raise KeyError(task_id)


@pytest.fixture(scope="session")
def demo(tmp_path_factory: pytest.TempPathFactory) -> Demo:
    from reward_lens.examples.code_reward import write

    root = tmp_path_factory.mktemp("code-reward")
    written = write(root)
    return Demo(
        root=root,
        written=written,
        tasks=read_jsonl(root / "tasks.jsonl"),
        responses=read_jsonl(root / "responses.jsonl"),
        grader=load_module(root / "grader.py", "demo_grader"),
        check=load_module(root / "outcome" / "check.py", "demo_outcome_check"),
    )
