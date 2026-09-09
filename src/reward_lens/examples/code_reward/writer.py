"""`write()`: put the example on disk, and refuse to do it over the top of someone's work."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from reward_lens.contracts.errors import UsageError

from . import _bank

__all__ = ["MANIFEST", "build", "check_tasks", "write"]

#: What the written tree holds, in the order `init` prints it, with the line each entry is given.
#: The two counts are the measured ones from the bank and are written here rather than counted at
#: print time, so a bank that drifts fails a test instead of quietly renarrating itself. `outcome/`
#: carries its slash because it is the one entry that is a directory. Frozen by A-015: this is the
#: contract behind the wave-1 transcript, and `init` renders it and nothing else.
MANIFEST: tuple[tuple[str, str], ...] = (
    ("grader.py", "the reward: tests passed, plus a format term"),
    ("tasks.jsonl", "40 code tasks"),
    ("responses.jsonl", "120 sampled responses from a small model"),
    ("outcome/", "a protected test suite, the independent check"),
    ("rewardlens.yaml", "what the reward is, and what counts as success"),
)


def _refuse(task_id: Any, what: str) -> UsageError:
    return UsageError(
        code="RL0001",
        message=f"task {task_id!r} {what}, so the example would ship a task it cannot score",
        remediation="give every task an id, a prompt, an entry point and at least one test case",
        context={"task_id": task_id, "problem": what},
    )


def check_tasks(tasks: Iterable[dict[str, Any]]) -> None:
    """The writer's own consistency check. A task the example cannot score is never written."""
    for task in tasks:
        task_id = task.get("task_id") if isinstance(task, dict) else None
        if not isinstance(task, dict) or not isinstance(task_id, str) or not task_id:
            raise _refuse(task_id, "has no identifier")
        if not isinstance(task.get("prompt"), str) or not task["prompt"].strip():
            raise _refuse(task_id, "has no prompt")
        entry = task.get("entry_point")
        if not isinstance(entry, str) or not entry.isidentifier():
            raise _refuse(task_id, "has no entry point")
        cases = task.get("tests")
        if not isinstance(cases, list) or not cases:
            raise _refuse(task_id, "has no tests")
        for case in cases:
            if not isinstance(case, dict) or not isinstance(case.get("args"), list) or "expect" not in case:
                raise _refuse(task_id, "has a test case that is not an args and expect pair")


def build(dest: Path) -> list[Path]:
    """Regenerate the example from the bank. Used to ship it, and to prove it was not hand-tuned."""
    tasks = _bank.tasks()
    check_tasks(tasks)
    return _bank.build(Path(dest))


def write(dest: Path, *, force: bool = False) -> list[Path]:
    """Write the example into `dest` and return every file written.

    Refuses a destination that already holds something, because the first thing a demo must not do is
    overwrite the project someone pointed it at.
    """
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()) and not force:
        raise UsageError(
            code="RL0001",
            message=f"{dest} is not empty",
            remediation="choose an empty directory, or pass force=True to write into this one",
            context={"dest": str(dest)},
        )
    return build(dest)
