"""Named results produced by actual execution-boundary events."""

from __future__ import annotations

from pathlib import Path

import pytest

from reward_lens.execution import Limits, run_python
from reward_lens.execution.errors import ExecutionBreachReason, ExecutionLimitBreached


def test_memory_backstop_emits_the_typed_resource_budget_reason(tmp_path: Path) -> None:
    """RLIMIT_AS refusal, not a fixture label, supplies the named resource-budget result."""
    # This test isolates the 32 MiB memory backstop. L3's real bubblewrap launch tree contains
    # trusted helper processes, so a one-process cap can name that independent boundary first.
    # Process-budget breaches are exercised separately in test_l0.py.
    limits = Limits(wall_s=3.0, cpu_s=3.0, memory_bytes=32 << 20, processes=8)
    result = run_python(
        "blob = bytearray(1024 * 1024 * 1024)\nprint(len(blob))",
        limits=limits,
        cwd=tmp_path,
        env={},
    )
    assert result.breach == "memory"
    assert result.breach_reason is ExecutionBreachReason.BUDGET_EXCEEDED
    with pytest.raises(ExecutionLimitBreached) as caught:
        result.raise_for_breach()
    error = caught.value
    assert error.code == "RL0410"
    assert error.default_exit_code == 7
    assert error.reason is ExecutionBreachReason.BUDGET_EXCEEDED
    assert error.dimension == "memory"
    assert error.context["reason"] == "BUDGET_EXCEEDED"
    assert error.context["limit"] == limits.memory_bytes
    assert "usd" not in error.context


def test_a_result_without_a_supervisor_breach_has_no_named_budget_reason(tmp_path: Path) -> None:
    result = run_python("print('bounded success')", limits=Limits(wall_s=3.0), cwd=tmp_path, env={})
    assert result.breach is None
    assert result.breach_reason is None
