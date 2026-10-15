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
from hashlib import sha256
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROTECTED_SOURCE = (HERE / "test_solution.py").read_text(encoding="utf-8")
FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)

#: Every way a candidate can fail, in the order they are looked for.
REASONS = ("empty_suite", "syntax_error", "exec_error", "missing_entry", "raises", "wrong_answer")

# D-39's controls are a fixed shipped set, not merely however many files happen
# to be in these directories.  The byte pins make a renamed or substituted
# control visible before the independent qualification is allowed to run.
REQUIRED_CONTROLS = {
    "known_good": {
        "task-01.py": "a20dac723393879473c6a2b007f54bb2d21b6481f5b527a110d572ba7c8eb762",
        "task-02.py": "9860c3f0a3d15a522cab6eae34b35b97d066a541a143a003591846157f995463",
        "task-03.py": "0130682d56e7dd3dcbbac399fbac7a6ed091024f5190a067ebf07b15e20b0228",
        "task-04.py": "7410ba95ada1f707878000212bbe50c601d53bf7d5d4d5132194c151864543ea",
        "task-05.py": "732566eef8aeeb47c9d63fab0c560e5875272edbf127d9af2f5ea7ba9dd5884d",
        "task-06.py": "86bc64124bfd811fec0593d7f2c9dd2a243fe6d20b2cd6efb4bca1c651674bee",
        "task-07.py": "748fb1595a1a4d9acda4d63da07a5c95c743f57e5cf5baad712ab61b7acdd94b",
        "task-08.py": "a97679d1fc470cbb54dbcf6e3c620a345572ea62b4989a6d256c58f2dc1cf77d",
        "task-09.py": "66e012509461d55a66aa2bc0f7b6711dfd9c410c781f51f255e0b39bbc7c7b37",
        "task-10.py": "fe8b411a1a9a37cc08b64bc991f64a0c792d0170409787703243b47c804881dd",
        "task-11.py": "0214acb433e9cf7d9caa669911f43301d124a49d4a2550fc074b2bab30b4a0b3",
        "task-12.py": "636fa7553eed2c3ab566c785896660fcb6a89b023fea0c2dae40cfab60ebe969",
    },
    "known_wrong": {
        "task-01.py": "1d9eb8c86360600d4d11131ef32780483cdf7367aeff79892ee3bff345177fd7",
        "task-02.py": "0eff6f70c6a72926a3ac6e0e7edba76b6ba5d5d0706fd6a329238a39c630aee8",
        "task-03.py": "5d1bae06e03c08a138da029c04ffecf7d937a6fd8f6b071933884ace2643c266",
        "task-04.py": "77b565aa1f659b54e6890d26f87a203cba4c0f5a9cd798cda4eac9ad093ff047",
        "task-05.py": "38a230a66f3d4277d1dc8518706d8002b6d4f4d2cd32a27b0123d05224c12ef3",
        "task-06.py": "fb715c02c2c6dc87d173efc1941303a3da761222a869928319db3ea861b2c09a",
        "task-07.py": "e76a560ade9c427aec26cde7267fc988c96f806dba91351c9967a9b9a6c1b729",
        "task-08.py": "3dd038a4361377c7623ffce1dd6ad9cf328b2e0965def2ced0bc75bf6fabda46",
        "task-09.py": "5d1c18e82df309f38cd1b7fc60c4671fcfe0d61345f89ac2cc0a68d5d2b8115c",
        "task-10.py": "7cc61f8d943ddc08c56bd6656d08428c05c0d29a1357e1ac2a1d29e9d0f0d78b",
        "task-11.py": "d13fa5ebd81068d6a1f36434ab98ca9368ed72f4f4a73eb06c89707eba4f1210",
        "task-12.py": "d8a2ef304b7fc60855702073ee8199b1e47e78befb4a08b4fc84c11c0d43d84d",
    },
}

CONTROL_REMEDY = (
    "restore the exact shipped known-good and known-wrong controls before qualification"
)


def _code(text: str) -> str:
    blocks = FENCE.findall(text)
    return blocks[0] if len(blocks) == 1 else text


def _result(
    passed: bool, reason: str, detail: str, cases: int = 0, total: int = 0
) -> dict[str, Any]:
    return {
        "passed": passed,
        "reason": reason,
        "detail": detail,
        "passed_cases": cases,
        "total_cases": total,
    }


def run_outcome(task: dict[str, Any], source: str) -> dict[str, Any]:
    """Run the protected suite against one candidate solution. The verdict is all or nothing."""
    cases = task.get("tests") or []
    if not cases:
        return _result(
            False, "empty_suite", "the task carries no test cases, so nothing is checked"
        )

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
                return _result(
                    False, "exec_error", f"{type(error).__name__}: {error}", 0, len(cases)
                )
            if not callable(namespace.get(task["entry_point"])):
                return _result(
                    False,
                    "missing_entry",
                    f"no callable named {task['entry_point']}",
                    0,
                    len(cases),
                )

            suite: dict[str, Any] = {"SOLUTION": namespace}
            exec(compile(PROTECTED_SOURCE, "outcome/test_solution.py", "exec"), suite)  # noqa: S102
            tests = [
                v
                for k, v in sorted(suite.items(), key=lambda kv: kv[0])
                if k.startswith("test_") and callable(v)
            ]
            passed = 0
            for test in tests:
                try:
                    test()
                except AssertionError as error:
                    return _result(False, "wrong_answer", str(error), passed, len(tests))
                except Exception as error:
                    return _result(
                        False, "raises", f"{type(error).__name__}: {error}", passed, len(tests)
                    )
                passed += 1
            return _result(True, "ok", f"{passed} of {len(tests)} cases", passed, len(tests))
        finally:
            os.chdir(here)


def _tasks() -> dict[str, dict[str, Any]]:
    lines = (HERE.parent / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
    return {row["task_id"]: row for row in (json.loads(line) for line in lines if line.strip())}


def _control_failures() -> list[str]:
    """Return exact set/pin failures before any control is executed."""
    failures: list[str] = []
    for group, expected in REQUIRED_CONTROLS.items():
        actual = {path.name: path for path in (HERE / group).glob("*.py")}
        for name in sorted(set(expected) - set(actual)):
            failures.append(f"missing required {group} control: {name}")
        for name in sorted(set(actual) - set(expected)):
            failures.append(f"unexpected {group} control: {name}")
        for name in sorted(set(expected) & set(actual)):
            if sha256(actual[name].read_bytes()).hexdigest() != expected[name]:
                failures.append(f"modified required {group} control: {name}")
    return failures


def qualify() -> dict[str, Any]:
    """D-39's qualification: the known-good pass, the known-wrong fail for the reason they state."""
    control_failures = _control_failures()
    if control_failures:
        return {
            "qualified": False,
            "known_good": len(list((HERE / "known_good").glob("*.py"))),
            "known_wrong": len(list((HERE / "known_wrong").glob("*.py"))),
            "reason_classes": [],
            "failures": control_failures,
            "remedy": CONTROL_REMEDY,
        }
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
        "remedy": None
        if not failures
        else "repair the stated known-good or known-wrong control failure before qualification",
    }
