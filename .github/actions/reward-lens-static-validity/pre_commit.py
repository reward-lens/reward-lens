#!/usr/bin/env python3
"""The installed pre-commit entry point for reward-lens' four fast validity checks.

This deliberately does not call ``reward-lens audit``.  The full audit measures and
persists many other panels; a pre-commit hook must instead exercise only the three
source checks and the input-handling check that the product calls its fourth static
validity check.  It creates no record and makes no recommendation from the other
Audit panels.
"""

from __future__ import annotations

import sys
from pathlib import Path

from reward_lens import contracts
from reward_lens.api.requests import AuditRequest
from reward_lens.execution import default_sandbox
from reward_lens.instruments.base import RunContext
from reward_lens.product.audit import instruments, static_checks
from reward_lens.product.audit.run import resolve
from reward_lens.store import Project


def _corpus(subject) -> instruments.Corpus:
    tasks = ()
    if subject.tasks_path is not None and subject.tasks_path.is_file():
        tasks = tuple(instruments.read_jsonl(subject.tasks_path, cap=1))
    return instruments.probe_corpus(tasks)


def _report(check_id: str, passed: bool | None, detail: str) -> None:
    state = "passed" if passed is True else "could not check" if passed is None else "failed"
    print(f"reward-lens {check_id}: {state}: {detail}")


def main() -> int:
    root = Path.cwd().resolve()
    project = Project.open(root)
    subject = resolve(AuditRequest(path=root), project)
    corpus = _corpus(subject)
    context = RunContext(sandbox=default_sandbox(), project=project)

    source_results = static_checks.run_static_checks(
        subject.source,
        source_name=subject.grader_path.name,
        entrypoint=subject.entrypoint,
        tasks=list(corpus.tasks),
    )
    failed = False
    for result in source_results:
        _report(result.entry_id, result.passed, result.observed)
        failed |= not result.passed

    handling = instruments.input_handling_entry(
        subject.grader_path,
        subject.entrypoint,
        corpus,
        context,
        subject_ref=contracts.digest({"source": subject.source}),
        trainer=subject.trainer,
        project_dir=subject.project_dir,
    )
    passed = handling.check.passed if handling.check is not None else None
    _report("validity.input_handling", passed, handling.measurand)
    failed |= passed is not True
    return int(failed)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"reward-lens static validity hook could not run: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(2)
