"""Acceptance-policy values over the canonical assay contract (D-13, D-72)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from typing import Literal

from reward_lens.contracts import UsageError

State = Literal["qualified", "rejected", "unresolved"]
CheckState = Literal["qualified", "rejected", "missing", "stale", "invalid", "inconclusive"]
EvidenceKey = str


class PolicyInvalid(UsageError):
    """RL0310: a policy file cannot safely decide a record."""

    def __init__(self, message: str) -> None:
        super().__init__(code="RL0310", message=message, remediation="correct the policy file")


class PolicyDigestMismatch(UsageError):
    """RL0311: a record did not name the exact policy it claims decided it."""

    def __init__(self, *, expected: str, recorded: str) -> None:
        super().__init__(
            code="RL0311",
            message="the policy digest named by the record does not match the policy file",
            remediation="reload the policy version that produced the record, or make a new decision",
            context={"expected": expected, "recorded": recorded},
        )


@dataclass(frozen=True)
class Multiplicity:
    """The registered multiplicity rule carried by an immutable policy file."""

    rule: str
    family: str


@dataclass(frozen=True)
class PolicyCheck:
    """One required evidence clause, in the one policy-file authority."""

    key: EvidenceKey
    source: str
    section: str | None
    entry_id: str | None
    alternate_entry_ids: tuple[str, ...]
    result_field: str | None
    fresh_for: timedelta | None
    threshold: str | None


@dataclass(frozen=True)
class Policy:
    id: str
    version: str
    digest: str
    checks: tuple[PolicyCheck, ...]
    default_interval: Decimal = Decimal("0.95")
    multiplicity: Multiplicity = Multiplicity("benjamini_hochberg_fdr", "registered family")
    actions: tuple[str, ...] = ()

    @property
    def required_evidence(self) -> tuple[EvidenceKey, ...]:
        """The policy's ordered required-evidence set, derived from its sole authority."""
        return tuple(check.key for check in self.checks)

    @property
    def freshness(self) -> dict[EvidenceKey, timedelta]:
        return {check.key: check.fresh_for for check in self.checks if check.fresh_for is not None}


@dataclass(frozen=True)
class CheckResult:
    state: CheckState
    reason: str


@dataclass(frozen=True)
class Consequence:
    option: str
    consequence: str


@dataclass(frozen=True)
class Decision:
    state: State
    action: str
    policy_id: str
    policy_digest: str
    evaluated: dict[EvidenceKey, CheckResult]
    reasons: list[str]

    def as_contract(self):
        """Return the existing contracts.Decision used by persisted Assay records."""
        from reward_lens.contracts.models import Decision as ContractDecision
        from reward_lens.contracts.models import PolicyRef

        return ContractDecision(
            state=self.state,
            action=self.action,
            policy=PolicyRef(id=self.policy_id, digest=self.policy_digest),
            reasons=self.reasons,
            tradeoff=None,
            signature=None,
            regression_cases=[],
        )


@dataclass(frozen=True)
class UnresolvedTradeoff(Decision):
    competing_consequences: tuple[Consequence, Consequence]

    def as_contract(self):
        from reward_lens.contracts.models import Tradeoff, TradeoffOption

        base = super().as_contract()
        base.tradeoff = Tradeoff(
            competing=[
                TradeoffOption(option=item.option, consequences=[item.consequence])
                for item in self.competing_consequences
            ]
        )
        return base


@dataclass(frozen=True)
class RegressionCase:
    id: str
    from_version: str
    evidence_ref: str
    depends_on: tuple[str, ...]
    dependency_digests: tuple[tuple[str, str], ...] = ()

    def needs_rerun(self, changed_dependencies: set[str]) -> bool:
        return bool(set(self.depends_on) & changed_dependencies)
