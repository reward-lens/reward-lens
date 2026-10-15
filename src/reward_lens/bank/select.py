"""Finite without-replacement response-bank selection."""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Mapping

from reward_lens.contracts import UsageError
from reward_lens.execution import Limits, Sandbox

from .types import Response, ResponseBank, TaskId

BUDGET_LADDER: tuple[int, ...] = (1, 2, 4, 8, 16, 32)


class TieBreak(enum.Enum):
    FIRST_BY_DIGEST = "first_by_digest"


@dataclass(frozen=True)
class Selection:
    chosen: Mapping[TaskId, Response]
    budget: int
    seed: int
    tie_break: TieBreak
    ties_observed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "chosen", MappingProxyType(dict(self.chosen)))


def _input_error(code: str, message: str, context: dict[str, object]) -> UsageError:
    """Keep reserved wave-3 input refusals typed until the catalogue row lands."""
    return UsageError(
        code=code, message=message, remediation="provide a finite response bank", context=context
    )


def require_bank(bank: ResponseBank | None) -> ResponseBank:
    """Refuse a compare invocation lacking the finite bank it needs."""
    if bank is None:
        raise _input_error("RL0134", "comparison requires a response bank", {"bank": "required"})
    return bank


def _score(
    scorer: Any,
    task: Mapping[str, object],
    response: Response,
    *,
    sandbox: Sandbox,
    limits: Limits,
) -> Decimal:
    """Read a scored envelope or a narrow test scorer without inventing an unscored value."""
    result = scorer.score(dict(task), response.text, sandbox=sandbox, limits=limits)
    value = getattr(result, "score", result)
    if value is None:
        raise ValueError("the grader returned an unscored response")
    return Decimal(str(value))


def best_of_n(
    bank: ResponseBank,
    scorer: Any,
    *,
    n: int,
    seed: int,
    tie_break: TieBreak,
    sandbox: Sandbox,
    limits: Limits,
) -> Selection:
    """Sample `n` candidates per task without replacement and select its scored winner."""
    bank = require_bank(bank)
    if n > bank.N // 2:
        raise _input_error(
            "RL0132",
            f"requested selection budget {n} exceeds safe bank budget {bank.N // 2} for N={bank.N}",
            {"N": bank.N, "requested_budget": n},
        )
    if n < 1:
        raise ValueError("selection budget must be at least one")
    if tie_break is not TieBreak.FIRST_BY_DIGEST:
        raise ValueError(f"unsupported declared tie break: {tie_break!r}")

    rng = random.Random(seed)
    chosen: dict[TaskId, Response] = {}
    ties_observed = 0
    for task in sorted(bank.tasks):
        sampled = rng.sample(list(bank.responses_for(task)), n)
        scored = [
            (
                response,
                _score(scorer, bank.task_for(task), response, sandbox=sandbox, limits=limits),
            )
            for response in sampled
        ]
        highest = max(score for _, score in scored)
        winners = [response for response, score in scored if score == highest]
        if len(winners) > 1:
            ties_observed += 1
        chosen[task] = min(winners, key=lambda response: response.digest)
    return Selection(
        chosen=chosen,
        budget=n,
        seed=seed,
        tie_break=tie_break,
        ties_observed=ties_observed,
    )
