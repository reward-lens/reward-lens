"""Twenty malformed invocations, each driven through the handler a caller would meet.

This stands in for the CLI until P-CLI lands: the same two-line rendering, the same exit codes,
the same promise. Each case runs inside the handler, and nothing is allowed to reach stderr as a
traceback. The exit status printed after each block is the one the process would have taken.

A case that raises something other than `RewardLensError` is a failed case, not a case to be
dressed up. Nothing here rewrites a bare exception into a code: the report names the exception
that escaped, and the test that reads the report fails on that case by name. Rewriting one into
RL0900 would turn every untyped escape into "reward-lens itself failed", which is a claim about
this project that a caller's own bad input has no business making.

Usage: `python _malformed_driver.py <report path>`. Stderr carries exactly what a caller would
see and nothing else; the JSON report carries the per-case outcome the test checks against its
table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

from reward_lens import contracts
from reward_lens.contracts import RewardLensError
from reward_lens.errors import explain, make, render_two_line


def _raise(error: BaseException) -> Any:
    raise error


# The later-major case names `$schema`, because that is the field the major is read from: a record
# that carries only `schema_version` never reaches the major check, and the case would be reporting
# a missing property under the name of a test for an unreadable major.
_LATER_MAJOR = "https://reward-lens.github.io/schema/assay/9.0/assay.schema.json"

CASES: list[tuple[str, Callable[[], Any]]] = [
    ("explain an unknown code", lambda: explain("RL9999")),
    ("explain an empty code", lambda: explain("")),
    ("explain a lowercase code", lambda: explain("rl0341")),
    ("explain a short code", lambda: explain("RL341")),
    ("explain a number", lambda: explain(341)),
    ("make an unknown code", lambda: make("RL9999")),
    ("make an empty code", lambda: make("")),
    ("make from nothing", lambda: make(None)),
    ("raise an outcome error", lambda: _raise(make("RL0341", path="./tests"))),
    ("raise a capability error", lambda: _raise(make("RL0701", extra="trace"))),
    ("raise a budget error", lambda: _raise(make("RL0501", cap="5.00"))),
    ("raise a pending decision", lambda: _raise(make("RL0801", decision="the policy to apply"))),
    ("raise an interruption", lambda: _raise(make("RL0130"))),
    ("validate an empty record", lambda: contracts.validate_record({})),
    ("validate something that is not a record", lambda: contracts.validate_record("not a record")),
    ("validate a record from a later major", lambda: contracts.validate_record({"$schema": _LATER_MAJOR})),
    ("build a record from nothing", lambda: contracts.Assay.model_validate({})),
    ("build a record from a list", lambda: contracts.Assay.model_validate([])),
    ("canonicalise a value with no JSON form", lambda: contracts.canonical_bytes(object())),
    ("read a project file with a bad reward kind", lambda: contracts.ProjectConfig.model_validate({"reward": {"kind": "nope"}})),
]


def main(report_path: str) -> int:
    report: list[dict[str, Any]] = []
    for name, case in CASES:
        try:
            case()
        except RewardLensError as caught:
            rendered = render_two_line(caught)
            sys.stderr.write(rendered + "\n")
            sys.stderr.write(f"exit: {caught.exit_code}\n")
            report.append(
                {
                    "case": name,
                    "code": caught.code,
                    "exit": caught.exit_code,
                    "rendered": rendered,
                }
            )
        except Exception as caught:  # noqa: BLE001 - a bare exception is a failed case
            report.append({"case": name, "untyped": type(caught).__name__})
        else:
            report.append({"case": name, "no_failure": True})
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
