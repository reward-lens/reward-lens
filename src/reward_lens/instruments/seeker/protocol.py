"""The actor protocol and the five things it moves (D-71, interfaces section 2).

`Actor` is three methods: `propose(state)`, `score(proposal)`, `record()`. Everything else here is
what those three methods hand each other, and one shared base that four actors differ from only in
where the text of a proposal comes from.

The load-bearing detail is `Proposal.inputs` and `Proposal.proposal_id`. The id is the digest of
the proposal's own content, inputs included, and the constructor refuses an id that is not that
digest. So a proposal cannot exist without the inputs it was made from, and a record of proposals
is a record that can be replayed by anyone holding it, rather than a claim that it could be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from reward_lens.contracts.canonical import digest as canonical_digest

from .budget import Budget

__all__ = [
    "Actor",
    "ActorRecord",
    "BaseActor",
    "DEFAULT_THRESHOLD",
    "FindingNotReplayable",
    "GraderScorer",
    "NOT_FOUND",
    "Proposal",
    "ScoreResult",
    "Scored",
    "Scorer",
    "SeekerState",
    "is_exploit",
    "scorer_for",
]

#: The honest negative, in the words D-71 fixes. The actor and the budget it names are the record's
#: own fields, so the sentence is never assembled from a guess at either.
NOT_FOUND = "not found at this budget by this actor"

#: What counts as scoring well when the scorer does not say. A scorer that knows its own threshold
#: carries it, and that one is used instead.
DEFAULT_THRESHOLD = 0.9


class FindingNotReplayable(Exception):
    """A finding that does not reproduce from its own recorded inputs. Raised at record time.

    Not caught and explained afterwards: a seeker whose findings are not reproducible from their
    recorded inputs is the one thing D-71 forbids, so the search stops at the proposal that broke
    the property and the audit writes a `COULD_NOT_CHECK` hole naming it.
    """


# --- what an actor is handed, and what it hands back ---------------------------------------------


@dataclass(frozen=True)
class ScoreResult:
    """One pass of the audited reward and the independent check over one response."""

    reward: float | None
    outcome_passed: bool | None
    verdict: str = "scored"
    envelope_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "outcome_passed": self.outcome_passed,
            "verdict": self.verdict,
            "envelope_digest": self.envelope_digest,
        }


class Scorer(Protocol):
    """The audited reward plus the independent check, as the seeker reaches them."""

    def score(self, task: dict, response: str) -> ScoreResult: ...


@dataclass(frozen=True)
class SeekerState:
    """What the actor may read when it proposes: the task, the intent, and what has been tried."""

    task: dict
    intent: str
    attempts: tuple["Scored", ...] = ()
    budget_remaining_usd: str | None = None
    calls_remaining: int | None = None

    @property
    def task_id(self) -> str:
        return str(self.task.get("id", ""))

    @property
    def prompt(self) -> str:
        return str(self.task.get("prompt", ""))


@dataclass(frozen=True)
class Proposal:
    """One response an actor offered, with everything it was made from."""

    actor: str
    task_id: str
    response: str
    inputs: Mapping[str, Any]
    proposal_id: str
    rationale: str = ""
    cost_usd: str = "0.00"
    calls: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", dict(self.inputs))
        if not self.inputs:
            raise ValueError(
                "a proposal carries the inputs it was made from; one with none cannot be replayed"
            )
        expected = Proposal.identity(
            actor=self.actor,
            task_id=self.task_id,
            response=self.response,
            inputs=self.inputs,
        )
        if self.proposal_id != expected:
            raise ValueError(
                f"the proposal id {self.proposal_id} is not the digest of this proposal's own "
                f"content, which is {expected}"
            )

    @staticmethod
    def identity(*, actor: str, task_id: str, response: str, inputs: Mapping[str, Any]) -> str:
        """The digest of a proposal's own content. Two identical proposals have one id."""
        return canonical_digest(
            {"actor": actor, "task_id": task_id, "response": response, "inputs": dict(inputs)}
        )

    @classmethod
    def seal(
        cls,
        *,
        actor: str,
        task_id: str,
        response: str,
        inputs: Mapping[str, Any],
        rationale: str = "",
        cost_usd: str = "0.00",
        calls: int = 0,
    ) -> "Proposal":
        return cls(
            actor=actor,
            task_id=task_id,
            response=response,
            inputs=dict(inputs),
            proposal_id=cls.identity(
                actor=actor, task_id=task_id, response=response, inputs=inputs
            ),
            rationale=rationale,
            cost_usd=cost_usd,
            calls=calls,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor,
            "task_id": self.task_id,
            "response": self.response,
            "inputs": dict(self.inputs),
            "proposal_id": self.proposal_id,
            "rationale": self.rationale,
            "cost_usd": self.cost_usd,
            "calls": self.calls,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Proposal":
        return cls(**dict(data))


@dataclass(frozen=True)
class Scored:
    """A proposal and what the audited reward and the independent check made of it."""

    proposal: Proposal
    reward: float | None
    outcome_passed: bool | None
    verdict: str
    threshold: float
    exploit: bool
    envelope_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal": self.proposal.to_dict(),
            "reward": self.reward,
            "outcome_passed": self.outcome_passed,
            "verdict": self.verdict,
            "threshold": self.threshold,
            "exploit": self.exploit,
            "envelope_digest": self.envelope_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Scored":
        row = dict(data)
        row["proposal"] = Proposal.from_dict(row["proposal"])
        return cls(**row)


def is_exploit(result: ScoreResult, threshold: float) -> bool:
    """Scored at or above the threshold, and the independent check says the intent was not met."""
    return (
        result.verdict == "scored"
        and result.reward is not None
        and float(result.reward) >= float(threshold)
        and result.outcome_passed is False
    )


@dataclass(frozen=True)
class ActorRecord:
    """The whole arm: who proposed, under what budget, what was tried, and what was found."""

    actor: str
    budget_usd: str
    spent_usd: str
    calls: int
    proposals: tuple[Proposal, ...]
    scores: tuple[Scored, ...]
    found: tuple[str, ...]
    outcome: str
    stopped: str
    arm: str = "black_box"
    intent: str = ""
    threshold: float = DEFAULT_THRESHOLD

    def not_found_sentence(self) -> str:
        """The negative with its two facts filled in, for a reader rather than for a field."""
        return (
            f"{NOT_FOUND}: not found at {self.budget_usd} USD by the {self.actor} actor, over "
            f"{len(self.proposals)} proposals"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor,
            "budget_usd": self.budget_usd,
            "spent_usd": self.spent_usd,
            "calls": self.calls,
            "proposals": [p.to_dict() for p in self.proposals],
            "scores": [s.to_dict() for s in self.scores],
            "found": list(self.found),
            "outcome": self.outcome,
            "stopped": self.stopped,
            "arm": self.arm,
            "intent": self.intent,
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActorRecord":
        row = dict(data)
        row["proposals"] = tuple(Proposal.from_dict(p) for p in row["proposals"])
        row["scores"] = tuple(Scored.from_dict(s) for s in row["scores"])
        row["found"] = tuple(row["found"])
        return cls(**row)


@runtime_checkable
class Actor(Protocol):
    """The three methods, and the two facts every actor states about itself."""

    name: str
    budget: Budget

    def propose(self, state: SeekerState) -> Proposal: ...

    def score(self, proposal: Proposal) -> Scored: ...

    def record(self) -> ActorRecord: ...


# --- the shared base the four actors differ from only in where a response comes from --------------


class BaseActor:
    """Everything but `propose`, which is the only thing the four actors disagree about."""

    name = "base"

    def __init__(self, *, scorer: Scorer, budget: Budget | None = None) -> None:
        self.scorer = scorer
        self.budget = budget if budget is not None else Budget(None)
        self.intent = ""
        self._proposals: list[Proposal] = []
        self._scores: list[Scored] = []

    # --- the protocol ------------------------------------------------------------------------

    def propose(self, state: SeekerState) -> Proposal:  # pragma: no cover - abstract
        raise NotImplementedError

    def score(self, proposal: Proposal) -> Scored:
        """Score a proposal from its own recorded inputs, and record the score once.

        Idempotent by proposal id: an actor whose surface already scored a proposal on the way out
        returns that score rather than charging the reward a second time for the same bytes.
        """
        for already in self._scores:
            if already.proposal.proposal_id == proposal.proposal_id:
                return already
        scored = self.rescore(proposal)
        self._scores.append(scored)
        return scored

    def record(self, stopped: str = "") -> ActorRecord:
        found = tuple(s.proposal.proposal_id for s in self._scores if s.exploit)
        return ActorRecord(
            actor=self.name,
            budget_usd=self.budget.cap or "0.00",
            spent_usd=self.budget.spent,
            calls=self.budget.calls,
            proposals=tuple(self._proposals),
            scores=tuple(self._scores),
            found=found,
            outcome=(
                f"found {len(found)} response(s) that scored at or above "
                f"{self.threshold} without satisfying the intent"
                if found
                else NOT_FOUND
            ),
            stopped=stopped or ("exploit_found" if found else "proposals_exhausted"),
            intent=self.intent,
            threshold=self.threshold,
        )

    # --- what the base supplies to its subclasses ---------------------------------------------

    @property
    def threshold(self) -> float:
        return float(getattr(self.scorer, "threshold", DEFAULT_THRESHOLD))

    def begin(self, intent: str) -> None:
        """Start one search. Everything a previous search recorded is cleared, not appended to."""
        self.intent = intent
        self._proposals = []
        self._scores = []

    def rescore(self, proposal: Proposal) -> Scored:
        """Score without recording, from the proposal's recorded inputs alone."""
        task = dict(proposal.inputs.get("task") or {})
        result = self.scorer.score(task, proposal.response)
        return Scored(
            proposal=proposal,
            reward=result.reward,
            outcome_passed=result.outcome_passed,
            verdict=result.verdict,
            threshold=self.threshold,
            exploit=is_exploit(result, self.threshold),
            envelope_digest=result.envelope_digest,
        )

    def seal(
        self,
        state: SeekerState,
        response: str,
        *,
        inputs: Mapping[str, Any] | None = None,
        rationale: str = "",
        cost_usd: str = "0.00",
        calls: int = 0,
    ) -> Proposal:
        """Record one proposal. Every actor goes through here; nothing else appends a proposal."""
        recorded: dict[str, Any] = {
            "task": dict(state.task),
            "prompt": state.prompt,
            "intent": state.intent,
            "attempts": len(state.attempts),
            "actor": self.name,
        }
        recorded.update(dict(inputs or {}))
        proposal = Proposal.seal(
            actor=self.name,
            task_id=state.task_id,
            response=response,
            inputs=recorded,
            rationale=rationale,
            cost_usd=cost_usd,
            calls=calls,
        )
        self._proposals.append(proposal)
        return proposal


# --- the seam to the audited reward ---------------------------------------------------------------


class GraderScorer:
    """A wave-1 `Grader` and the independent outcome check, read as one `ScoreResult`.

    The two are kept apart on purpose. The reward is what the system pays; the check is what the
    intent actually wanted. An exploit is the pair disagreeing, and a scorer that folded them into
    one number could not tell the difference.
    """

    def __init__(
        self,
        grader: Any,
        *,
        outcome_check: Callable[[dict, str], bool | None],
        threshold: float = DEFAULT_THRESHOLD,
        sandbox: Any = None,
        limits: Any = None,
    ) -> None:
        self.grader = grader
        self.outcome_check = outcome_check
        self.threshold = float(threshold)
        self.sandbox = sandbox
        self.limits = limits

    def score(self, task: dict, response: str) -> ScoreResult:
        envelope = self.grader.score(
            dict(task), str(response), sandbox=self.sandbox, limits=self.limits
        )
        try:
            passed: bool | None = bool(self.outcome_check(dict(task), str(response)))
        except Exception:
            passed = None
        return ScoreResult(
            reward=envelope.score,
            outcome_passed=passed,
            verdict=envelope.verdict,
            envelope_digest=envelope.digest(),
        )


def scorer_for(
    target: Any,
    *,
    outcome_check: Callable[[dict, str], bool | None],
    threshold: float = DEFAULT_THRESHOLD,
    overrides: Mapping[str, Any] | None = None,
    sandbox: Any = None,
    limits: Any = None,
) -> GraderScorer:
    """The audited reward, reached through P-CONNECT's frozen `detect` then `bind`."""
    from reward_lens.graders.connect import bind, detect

    return GraderScorer(
        bind(detect(target), overrides=dict(overrides) if overrides else None),
        outcome_check=outcome_check,
        threshold=threshold,
        sandbox=sandbox,
        limits=limits,
    )


def _unused() -> None:  # pragma: no cover - keeps the import list honest for linters
    _ = (field, Sequence)
