"""No expected failure path produces a traceback. Twenty malformed invocations prove it.

Membership in the catalogue is not the claim. Each of the twenty is owed one reserved code and one
exit status, written down in OWED below, and each case is checked against its own row. A case that
escapes untyped fails by name rather than being rewritten into RL0900.
"""

from __future__ import annotations

import functools
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from reward_lens.errors import CATALOGUE
from reward_lens.errors.catalogue import EXIT_MEANINGS

DRIVER = Path(__file__).resolve().parent / "_malformed_driver.py"
REPO = Path(__file__).resolve().parents[2]
BLOCK = re.compile(
    r"^error: (?P<message>.+)\nhelp:  (?P<help>.+)\n {7}reward-lens explain (?P<code>RL[0-9]{4})\nexit: (?P<exit>[0-9]+)$",
    re.MULTILINE,
)

# What each of the twenty is owed: the reserved code, and the exit status that code carries.
# The five that reach `reward_lens.contracts` through an untyped path are owed RL0003 (a project
# file field that is not valid, with the field named) or RL0604 (a record that does not match the
# schema, with the instance path). They are P-CONTRACT's typed refusals; until those land, those
# rows fail here, which is the point of writing them down.
OWED: dict[str, tuple[str, int]] = {
    "explain an unknown code": ("RL0001", 4),
    "explain an empty code": ("RL0001", 4),
    "explain a lowercase code": ("RL0001", 4),
    "explain a short code": ("RL0001", 4),
    "explain a number": ("RL0001", 4),
    "make an unknown code": ("RL0001", 4),
    "make an empty code": ("RL0001", 4),
    "make from nothing": ("RL0001", 4),
    "raise an outcome error": ("RL0341", 4),
    "raise a capability error": ("RL0701", 5),
    "raise a budget error": ("RL0501", 6),
    "raise a pending decision": ("RL0801", 3),
    "raise an interruption": ("RL0130", 130),
    "validate an empty record": ("RL0604", 4),
    "validate something that is not a record": ("RL0604", 4),
    "validate a record from a later major": ("RL0601", 4),
    "build a record from nothing": ("RL0604", 4),
    "build a record from a list": ("RL0604", 4),
    "canonicalise a value with no JSON form": ("RL0604", 4),
    "read a project file with a bad reward kind": ("RL0003", 4),
}


@functools.cache
def _run() -> tuple[subprocess.CompletedProcess[str], list[dict[str, object]]]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    with tempfile.TemporaryDirectory() as work:
        report_path = Path(work) / "report.json"
        result = subprocess.run(
            [sys.executable, str(DRIVER), str(report_path)],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(REPO),
            timeout=180,
        )
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else []
    return result, report


def _outcome(case: str) -> dict[str, object]:
    _, report = _run()
    rows = [row for row in report if row["case"] == case]
    assert len(rows) == 1, f"{case}: the driver reported it {len(rows)} times"
    return rows[0]


def test_the_table_covers_the_twenty_cases_the_driver_runs() -> None:
    result, report = _run()
    assert result.returncode == 0, result.stderr[-2000:]
    assert [row["case"] for row in report] == list(OWED), "the table and the driver have drifted"
    assert len(OWED) == 20


@pytest.mark.parametrize("case", list(OWED))
def test_each_malformed_invocation_carries_the_code_and_exit_it_is_owed(case: str) -> None:
    code, exit_code = OWED[case]
    outcome = _outcome(case)
    assert "no_failure" not in outcome, f"{case}: meant to fail and did not"
    assert "untyped" not in outcome, (
        f"{case}: escaped as {outcome.get('untyped')} instead of {code}; "
        "a bare exception is a failed case, not an internal failure"
    )
    assert outcome["code"] == code, case
    assert outcome["exit"] == exit_code, case
    assert CATALOGUE[code].exit_code == exit_code, code
    assert exit_code in EXIT_MEANINGS, code

    # and what the caller saw is the two-line shape, naming that same code back.
    lines = str(outcome["rendered"]).splitlines()
    assert len(lines) == 3, case
    assert lines[0].startswith("error: ") and lines[0][7:].strip(), case
    assert lines[1].startswith("help:  ") and lines[1][7:].strip(), case
    assert lines[2] == f"       reward-lens explain {code}", case


def test_every_typed_case_reached_stderr_as_a_block_and_nothing_else_did() -> None:
    result, report = _run()
    typed = [row for row in report if "code" in row]
    blocks = list(BLOCK.finditer(result.stderr))
    assert len(blocks) == len(typed), result.stderr[-2000:]
    for block, row in zip(blocks, typed, strict=True):
        assert block.group("code") == row["code"], row["case"]
        assert int(block.group("exit")) == row["exit"], row["case"]


def test_no_traceback_reaches_stderr() -> None:
    result, _ = _run()
    assert "Traceback" not in result.stderr
    assert '  File "' not in result.stderr
    assert "Error:" not in result.stderr


def test_nothing_reaches_stdout_and_no_internal_module_is_named() -> None:
    result, _ = _run()
    assert result.stdout == ""
    assert not re.search(r"reward_lens[./][A-Za-z_]", result.stderr)
    assert not re.search(r"\b[A-Z]-[0-9]{1,3}\b|\bsection [0-9]|§", result.stderr)
