"""Public behavior checks for six v3 fixes retained by the 3.1.0 source target."""

from __future__ import annotations

import ast
import itertools
from pathlib import Path
from typing import Any

import pytest

from reward_lens.core.types import Capability
from reward_lens.measure.base import capability_name
from reward_lens.tap.adapters.trl import TRLTap

ROOT = Path(__file__).resolve().parents[2]
PLANE_B = ("capture", "grad_h", "token_gradients", "hvp", "with_interventions")


class _TrainerShaped:
    def __init__(self) -> None:
        self.callbacks: list[Any] = []

    def add_callback(self, callback: Any) -> None:
        self.callbacks.append(callback)


def test_base_collection_has_explicit_white_box_selection() -> None:
    source = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "_WHITE_BOX_TOKENS" in source
    assert "collect_ignore.append" in source


def test_trl_tap_attaches_without_transformers() -> None:
    tap = TRLTap()
    trainer = _TrainerShaped()
    assert tap.attach(trainer) is trainer
    assert trainer.callbacks
    assert tap.adapter_exception_keys == []
    assert tap.adapter_exceptions == 0
    callback = trainer.callbacks[0]
    assert callable(getattr(callback, "on_step_end", None))
    assert callable(getattr(callback, "on_pre_optimizer_step", None))


def test_capability_names_are_stable_for_flags() -> None:
    members = [member for member in Capability if member.value]
    assert members
    for width in range(0, min(3, len(members)) + 1):
        for combo in itertools.combinations(members, width):
            value = Capability(0)
            for member in combo:
                value |= member
            assert capability_name(value) == (value.name or "NONE")
    with pytest.raises(TypeError):
        int(members[0])


def _function_source(path: Path, name: str) -> str:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node) or ""
    raise AssertionError(f"{name} is missing from {path}")


def test_debt_m_uses_backend_independent_count_assertions() -> None:
    source = _function_source(
        ROOT / "tests" / "acceptance" / "test_debt_m.py",
        "test_the_none_case_is_fiellers_unbounded_case_and_now_says_so",
    )
    assert "0.8 * examined" in source
    assert "0.8 * agreed" in source
    assert "{agreed} of {examined}" in source
    assert "{unbounded} of {agreed}" in source


def test_serving_boundary_obeys_attribute_lookup_contract() -> None:
    path = ROOT / "src" / "reward_lens" / "policy" / "vllm.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    boundary = classes["EngineBoundary"]
    bases = {base.id for base in boundary.bases if isinstance(base, ast.Name)}
    assert {"RuntimeError", "AttributeError"} <= bases
    serving = classes["ServingPolicy"]
    getattr_method = next(
        node for node in serving.body if isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
    )
    text = ast.get_source_segment(path.read_text(encoding="utf-8"), getattr_method) or ""
    assert "EngineBoundary" in text
    assert all(name in text for name in PLANE_B)
