"""The independent outcome check (D-39), and what qualifies it.

This file and `../grader.py` run the same protected suite over the same solution. One line separates
them: this one reads the suite from its own directory, into memory, before any candidate has run, so
nothing a candidate writes can reach it. The grader reads it back off disk from inside the directory
the candidate just executed in.

It deliberately does not import the grader. An outcome check that is a copy of the audited grader is
not independent of it, so the few lines they have in common are written out again here.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROTECTED_SOURCE = (HERE / "test_solution.py").read_text(encoding="utf-8")
FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)

#: Every way a candidate can fail, in the order they are looked for.
REASONS = ("empty_suite", "syntax_error", "exec_error", "missing_entry", "raises", "wrong_answer")


def _code(text: str) -> str:
    blocks = FENCE.findall(text)
    return blocks[0] if len(blocks) == 1 else text


def _result(passed: bool, reason: str, detail: str, cases: int = 0, total: int = 0) -> dict[str, Any]:
    return {"passed": passed, "reason": reason, "detail": detail, "passed_cases": cases, "total_cases": total}


def run_outcome(task: dict[str, Any], source: str) -> dict[str, Any]:
    """Run the protected suite against one candidate solution. The verdict is all or nothing."""
    cases = task.get("tests") or []
    if not cases:
        return _result(False, "empty_suite", "the task carries no test cases, so nothing is checked")

    code = _code(source)
    try:
        compiled = compile(code, "candidate.py", "exec")
    except SyntaxError as error:
        return _result(False, "syntax_error", str(error), 0, len(cases))

    here = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="outcome-check-") as tmp:
        work = Path(tmp)
        (work / "outcome").mkdir()
        (work / "outcome" / "task.json").write_text(json.dumps(task), encoding="utf-8")
        os.chdir(work)
        try:
            namespace: dict[str, Any] = {"__name__": "candidate"}
            try:
                exec(compiled, namespace)  # noqa: S102 - the candidate under test
            except Exception as error:
                return _result(False, "exec_error", f"{type(error).__name__}: {error}", 0, len(cases))
            if not callable(namespace.get(task["entry_point"])):
                return _result(False, "missing_entry", f"no callable named {task['entry_point']}", 0, len(cases))

            suite: dict[str, Any] = {"SOLUTION": namespace}
            exec(compile(PROTECTED_SOURCE, "outcome/test_solution.py", "exec"), suite)  # noqa: S102
            tests = [v for k, v in sorted(suite.items(), key=lambda kv: kv[0]) if k.startswith("test_") and callable(v)]
            passed = 0
            for test in tests:
                try:
                    test()
                except AssertionError as error:
                    return _result(False, "wrong_answer", str(error), passed, len(tests))
                except Exception as error:
                    return _result(False, "raises", f"{type(error).__name__}: {error}", passed, len(tests))
                passed += 1
            return _result(True, "ok", f"{passed} of {len(tests)} cases", passed, len(tests))
        finally:
            os.chdir(here)


def _tasks() -> dict[str, dict[str, Any]]:
    lines = (HERE.parent / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
    return {row["task_id"]: row for row in (json.loads(line) for line in lines if line.strip())}


def qualify() -> dict[str, Any]:
    """D-39's qualification: the known-good pass, the known-wrong fail for the reason they state."""
    tasks = _tasks()
    stated = json.loads((HERE / "known_wrong" / "reasons.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    classes: set[str] = set()

    good = sorted((HERE / "known_good").glob("*.py"))
    for path in good:
        result = run_outcome(tasks[path.stem], path.read_text(encoding="utf-8"))
        if not result["passed"]:
            failures.append(f"known good {path.name}: {result['reason']}, {result['detail']}")

    wrong = sorted((HERE / "known_wrong").glob("*.py"))
    for path in wrong:
        result = run_outcome(tasks[path.stem], path.read_text(encoding="utf-8"))
        expected = stated.get(path.name, {}).get("reason")
        classes.add(result["reason"])
        if result["passed"]:
            failures.append(f"known wrong {path.name} passed the suite")
        elif result["reason"] != expected:
            failures.append(f"known wrong {path.name}: {result['reason']}, expected {expected}")

    return {
        "qualified": not failures,
        "known_good": len(good),
        "known_wrong": len(wrong),
        "reason_classes": sorted(classes),
        "failures": failures,
    }
