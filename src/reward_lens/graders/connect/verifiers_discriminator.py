"""The two rules `verifiers` uses, written out so that neither needs `verifiers` installed.

Both are transcriptions of the vendored checkout at commit edafab8, and the test that guards them
does not trust the transcription: it lifts the framework's own `Rubric._is_group_func`
(`rubric.py:106-123`) and the binding expressions of `Rubric._call_individual_reward_func`
(`rubric.py:200-211`) out of that file with `ast`, runs them, and compares on the same callable.

Owned by P-CONNECT.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Iterable, Mapping, get_origin

__all__ = ["GROUP_INDICATORS", "bound_names", "is_group_rubric"]

#: `rubric.py:111-118`. A plural parameter name is what makes a rubric function a group function.
GROUP_INDICATORS: frozenset[str] = frozenset(
    {"states", "prompts", "completions", "answers", "tasks", "infos"}
)


def is_group_rubric(fn: Callable[..., Any]) -> bool:
    """True where `Rubric._is_group_func` is true: a plural parameter name, or a list return."""
    signature = inspect.signature(fn)
    parameter_names = set(signature.parameters.keys())
    return_annotation = signature.return_annotation
    returns_list = return_annotation is list or get_origin(return_annotation) is list
    return bool(parameter_names & GROUP_INDICATORS) or returns_list


def bound_names(fn: Callable[..., Any], available: Mapping[str, Any] | Iterable[str]) -> set[str]:
    """The names `Rubric` would pass to `fn` out of `available`, by the rule at `rubric.py:200-211`.

    A callable with `**kwargs` is handed everything; anything else is handed the subset of
    `available` whose names appear in its signature. That subset, not a positional order, is the
    whole binding contract, which is why an adapter that reorders parameters still works and one
    that renames them silently stops receiving a field.
    """
    signature = inspect.signature(fn)
    names = set(available)
    if any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
        return names
    return {name for name in names if name in signature.parameters}
