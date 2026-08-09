"""The search loop, and the replay property it enforces before a finding is allowed to exist.

`seek` is small on purpose: propose, score, check that the score reproduces from the proposal's own
recorded inputs, stop. The checking is the part that matters. A finding enters the record only
after it has been reproduced from the bytes the record will carry, so "reproducible from its
recorded inputs" is a thing the code did rather than a thing the record says.
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence

from .budget import BudgetExhausted
from .protocol import (
    Actor,
    ActorRecord,
    FindingNotReplayable,
    Proposal,
    Scored,
    Scorer,
    SeekerState,
    is_exploit,
)

__all__ = ["replay", "seek"]


def _rotation(tasks: Sequence[dict], limit: int) -> Iterator[dict]:
    """The tasks, in order, over and over, up to `limit` proposals."""
    if not tasks:
        return
    for index in range(limit):
        yield dict(tasks[index % len(tasks)])


def seek(
    actor: Actor,
    *,
    tasks: Sequence[dict],
    intent: str,
    max_proposals: int = 8,
    stop_on_first_exploit: bool = True,
) -> ActorRecord:
    """Run one actor against one task set under its own budget, and return its record."""
    actor.begin(intent)  # type: ignore[attr-defined]
    attempts: list[Scored] = []
    stopped = "proposals_exhausted"
    for task in _rotation(tuple(tasks), max_proposals):
        state = SeekerState(
            task=task,
            intent=intent,
            attempts=tuple(attempts),
            budget_remaining_usd=actor.budget.remaining,
            calls_remaining=actor.budget.calls_remaining,
        )
        try:
            proposal = actor.propose(state)
        except BudgetExhausted:
            stopped = "budget_exhausted"
            break
        scored = actor.score(proposal)
        attempts.append(scored)
        if scored.exploit:
            assert_replays(scored, actor.scorer)  # type: ignore[attr-defined]
            stopped = "exploit_found"
            if stop_on_first_exploit:
                break
    return actor.record(stopped=stopped)  # type: ignore[call-arg]


def assert_replays(scored: Scored, scorer: Scorer) -> None:
    """Reproduce one finding from its recorded inputs, or refuse it at record time."""
    again = rescore(scored.proposal, scorer, threshold=scored.threshold)
    if (again.reward, again.outcome_passed, again.exploit) != (
        scored.reward,
        scored.outcome_passed,
        scored.exploit,
    ):
        raise FindingNotReplayable(
            f"the finding on proposal {scored.proposal.proposal_id} does not replay from its own "
            f"recorded inputs: it scored {scored.reward} with the check at "
            f"{scored.outcome_passed}, and replaying the same inputs scored {again.reward} with "
            f"the check at {again.outcome_passed}"
        )


def rescore(proposal: Proposal, scorer: Scorer, *, threshold: float) -> Scored:
    """Score a proposal from `proposal.inputs` alone, touching nothing else."""
    result = scorer.score(dict(proposal.inputs.get("task") or {}), proposal.response)
    return Scored(
        proposal=proposal,
        reward=result.reward,
        outcome_passed=result.outcome_passed,
        verdict=result.verdict,
        threshold=threshold,
        exploit=is_exploit(result, threshold),
        envelope_digest=result.envelope_digest,
    )


def replay(record: ActorRecord, scorer: Scorer) -> tuple[Scored, ...]:
    """Every score in a record, recomputed from the recorded inputs and nothing else.

    A record and this function are all a reader needs: no actor is built, no provider is called, no
    endpoint is reached. That is what makes the record the evidence rather than the run.
    """
    return tuple(
        rescore(scored.proposal, scorer, threshold=record.threshold) for scored in record.scores
    )


def _unused() -> None:  # pragma: no cover - keeps the import list honest for linters
    _ = Any
