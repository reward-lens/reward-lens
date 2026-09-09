"""A small plain-shape grader, kept as the subject of a bare-grader audit.

It scores how much of the expected answer a response covers. It is deterministic, and it returns
`None` rather than guessing when the task or the response is not the shape it needs, which is the
first of D-65's three silent failure modes: TRL turns that `None` into a NaN and drops the function
for that row.
"""

from __future__ import annotations

import re
from typing import Any

WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))


def score(task: Any, response: Any) -> float | None:
    """Return the fraction of the expected answer's words the response used, or None if malformed."""
    if not isinstance(task, dict) or not isinstance(response, str):
        return None
    prompt = task.get("prompt")
    expected = task.get("expected")
    if not isinstance(prompt, str) or not isinstance(expected, str):
        return None
    wanted = _words(expected)
    if not wanted:
        return None
    return float(len(wanted & _words(response))) / float(len(wanted))
