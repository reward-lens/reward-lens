"""The protected test suite for the code-reward example: one test per case.

The task under test is read from `outcome/task.json` next to this file, and the namespace of the
candidate solution is handed in as `SOLUTION` by whoever runs this module. That is all the contract
there is, which is what lets the independent check in `check.py` run this exact source from a place
no candidate can reach.
"""

import json
from pathlib import Path

TASK = json.loads(Path("outcome/task.json").read_text(encoding="utf-8"))
ENTRY = TASK["entry_point"]
CASES = TASK["tests"]


def _make_case(index, case):
    def run():
        function = SOLUTION.get(ENTRY)  # noqa: F821 - injected by the runner
        if not callable(function):
            raise AssertionError("no callable named %s" % ENTRY)
        got = function(*case["args"])
        assert got == case["expect"], "case %d: %r is not %r" % (index, got, case["expect"])

    run.__name__ = "test_case_%02d" % index
    return run


for _index, _case in enumerate(CASES):
    globals()["test_case_%02d" % _index] = _make_case(_index, _case)
