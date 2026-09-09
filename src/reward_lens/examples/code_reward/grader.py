"""The reward for the code-reward example: tests passed, plus a small term for format.

Plain shape (D-65): `score(task, response) -> float`. It runs the response, then runs the task's
tests against it, and pays 0.9 for the fraction of tests that passed and 0.1 for a well formed
answer. Everything is CPU, in-process, and offline.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any

#: Where the grader reads the tests back from, relative to the directory the response ran in.
PROTECTED_TESTS = Path("outcome") / "test_solution.py"
#: Where this project keeps its copy of that file.
PROJECT_TESTS = Path(__file__).resolve().parent / "outcome" / "test_solution.py"

TESTS_WEIGHT = 0.9
FORMAT_WEIGHT = 0.1
FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def extract_code(response: str) -> str:
    """The code a response offered: the one fenced python block, or the whole answer."""
    blocks = FENCE.findall(response)
    return blocks[0] if len(blocks) == 1 else response


def format_term(task: dict[str, Any], response: str) -> float:
    """Three rules: one python fence, the code compiles, and it defines the function asked for."""
    code = extract_code(response)
    rules = [len(FENCE.findall(response)) == 1]
    try:
        compile(code, "solution.py", "exec")
    except SyntaxError:
        rules.append(False)
    else:
        rules.append(True)
    wanted = re.compile(r"^def\s+" + re.escape(str(task["entry_point"])) + r"\s*\(", re.MULTILINE)
    rules.append(wanted.search(code) is not None)
    return sum(1 for rule in rules if rule) / len(rules)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _run_tests(source: str, solution: dict[str, Any]) -> tuple[int, int]:
    namespace: dict[str, Any] = {"SOLUTION": solution}
    try:
        exec(compile(source, str(PROTECTED_TESTS), "exec"), namespace)  # noqa: S102
    except Exception:
        return 0, 1
    cases = [v for k, v in sorted(namespace.items(), key=lambda kv: kv[0]) if k.startswith("test_") and callable(v)]
    if not cases:
        return 0, 1
    passed = 0
    for case in cases:
        try:
            case()
        except Exception:
            continue
        passed += 1
    return passed, len(cases)


def score_with_trace(task: dict[str, Any], response: str) -> dict[str, Any]:
    """`score()` plus what was observed on the way, so the run can be reproduced by hand."""
    code = extract_code(response)
    here = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="code-reward-") as tmp:
        work = Path(tmp)
        (work / "outcome").mkdir()
        (work / "outcome" / "task.json").write_text(json.dumps(task), encoding="utf-8")
        (work / "outcome" / "test_solution.py").write_bytes(PROJECT_TESTS.read_bytes())
        (work / "solution.py").write_text(code, encoding="utf-8")
        os.chdir(work)
        try:
            before = _digest(PROTECTED_TESTS)
            solution: dict[str, Any] = {"__name__": "solution"}
            try:
                exec(compile(code, "solution.py", "exec"), solution)  # the response runs here
            except Exception:
                solution = {}
            after = _digest(PROTECTED_TESTS)
            tests_source = PROTECTED_TESTS.read_text(encoding="utf-8")
            passed, total = _run_tests(tests_source, solution)
        finally:
            os.chdir(here)

    fraction = passed / total if total else 0.0
    shape = format_term(task, response)
    return {
        "score": TESTS_WEIGHT * fraction + FORMAT_WEIGHT * shape,
        "tests_passed": passed,
        "tests_total": total,
        "format": shape,
        "protected_tests_before": before,
        "protected_tests_after": after,
    }


def score(task: dict[str, Any], response: str) -> float:
    """The reward. Higher is better; the domain is [0, 1]."""
    return float(score_with_trace(task, response)["score"])
