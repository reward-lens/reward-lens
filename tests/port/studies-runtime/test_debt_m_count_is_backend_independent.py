"""Keep the debt-M AISI control aligned with the resolved-direction implementation.

The earlier floor-and-ratio guard predated the numerical-zero branch. The acceptance test now
partitions all 97 captured windows into numerical zero, resolved non-rising, and meaningful positive
directions; it also checks the Fieller boundary on every positive window. Its exact five-count
fixture is deliberate. Read the acceptance source without importing its optional artifacts package;
the acceptance suite runs the numerical check itself.
"""

from __future__ import annotations

import ast
import pathlib

ACCEPTANCE = pathlib.Path(__file__).resolve().parents[2] / "acceptance" / "test_debt_m.py"
TEST_NAME = "test_the_none_case_is_fiellers_unbounded_case_and_now_says_so"


def _node() -> ast.FunctionDef:
    tree = ast.parse(ACCEPTANCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == TEST_NAME:
            return node
    raise AssertionError(f"{TEST_NAME} is not in {ACCEPTANCE}")


def _source() -> str:
    return ast.get_source_segment(ACCEPTANCE.read_text(encoding="utf-8"), _node()) or ""


def test_the_real_series_has_an_exhaustive_numerical_direction_partition() -> None:
    """The captured fixture pins refusals separately from positive projections."""
    source = _source()
    assert "examined += 1" in source
    assert 'if "numerically resolved direction" in note:' in source
    assert "numeric_zero += 1" in source
    assert "non_rising += 1" in source
    assert "meaningful += 1" in source
    assert "unbounded += 1" in source
    assert (
        "(examined, numeric_zero, non_rising, meaningful, unbounded) == (97, 2, 40, 55, 50)"
        in source
    )


def test_numerical_zero_and_a_resolved_non_rising_slope_keep_distinct_refusals() -> None:
    source = _source()
    assert "assert ci is None" in source
    assert 'assert "not rising" in note' in source


def test_the_per_window_boundary_assertion_is_untouched() -> None:
    """Loosening the totals must not loosen the thing the test exists for."""
    source = _source()
    assert "(ci is None) == (old is None)" in source
    assert "the None boundary moved at step" in source
