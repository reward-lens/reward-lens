"""D-39: the independent outcome check, and what qualifies it."""

from __future__ import annotations

import json

import pytest


def test_twelve_known_good_solutions_pass(demo) -> None:
    good = sorted((demo.root / "outcome" / "known_good").glob("*.py"))
    assert len(good) == 12
    for path in good:
        task = demo.task(path.stem)
        result = demo.check.run_outcome(task, path.read_text(encoding="utf-8"))
        assert result["passed"] is True, (path.name, result)
        assert result["reason"] == "ok"


def test_twelve_known_wrong_solutions_fail_for_their_stated_reason(demo) -> None:
    stated = json.loads((demo.root / "outcome" / "known_wrong" / "reasons.json").read_text(encoding="utf-8"))
    wrong = sorted((demo.root / "outcome" / "known_wrong").glob("*.py"))
    assert len(wrong) == 12
    assert set(stated) == {p.name for p in wrong}
    for path in wrong:
        task = demo.task(path.stem)
        result = demo.check.run_outcome(task, path.read_text(encoding="utf-8"))
        assert result["passed"] is False, (path.name, result)
        assert result["reason"] == stated[path.name]["reason"], (path.name, result)
        assert stated[path.name]["detail"].strip()


def test_an_empty_suite_fails_qualification(demo) -> None:
    task = dict(demo.task(sorted((demo.root / "outcome" / "known_good").glob("*.py"))[0].stem))
    task["tests"] = []
    good = (demo.root / "outcome" / "known_good" / f"{task['task_id']}.py").read_text(encoding="utf-8")
    result = demo.check.run_outcome(task, good)
    assert result["passed"] is False
    assert result["reason"] == "empty_suite"


def test_qualification_reports_every_reference_solution(demo) -> None:
    report = demo.check.qualify()
    assert report["qualified"] is True
    assert report["known_good"] == 12
    assert report["known_wrong"] == 12
    assert report["reason_classes"] == sorted(set(report["reason_classes"]))


def test_the_check_is_not_a_duplicate_of_the_grader(demo) -> None:
    """The check reads its test source from its own directory, before any candidate runs."""
    source = (demo.root / "outcome" / "check.py").read_text(encoding="utf-8")
    assert "__file__" in source
    task = demo.task(sorted((demo.root / "outcome" / "known_good").glob("*.py"))[0].stem)
    tampering = (
        "from pathlib import Path\n"
        "Path('outcome').mkdir(exist_ok=True)\n"
        "Path('outcome/test_solution.py').write_text('def test_ok():\\n    pass\\n')\n"
        "def " + task["entry_point"] + "(*a, **k):\n    return None\n"
    )
    result = demo.check.run_outcome(task, tampering)
    assert result["passed"] is False
