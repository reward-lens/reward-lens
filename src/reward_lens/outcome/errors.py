"""The refusals of the outcome path.

RL0302 is this packet's one allocation, made by amendment A-006. RL0001 is wave 1's usage refusal,
imported with the catalogue's own wording; RL0401 and RL0402 are P-EXEC's and are raised by the
separation certifier when the machine did not establish what a containment claim would assert.
RL0301 is P-AUDIT-1's, is a state rather than an exception, and is never built here.
"""

from __future__ import annotations

from reward_lens.contracts.errors import UsageError
from reward_lens.errors import make

__all__ = ["AcceptanceRoundExhausted", "PanelNotRegistered", "usage_refusal"]


def usage_refusal(detail: str, **context: object) -> UsageError:
    """RL0001 with the catalogue's text, for an invocation this packet will not carry out."""
    error = make("RL0001", detail=detail, **context)
    assert isinstance(error, UsageError)
    return error


class AcceptanceRoundExhausted(UsageError):
    """RL0302: a candidate set asked for a round beyond the one that was sealed for it.

    The default is one sealed comparison round per candidate set (D-40). A second seal, a replayed
    nonce and a request whose candidate or protocol digest matches no sealed round are all the same
    refusal, because they are all the same ask: score this candidate against the protected
    partition one more time. The way past it is a fresh partition or a new protocol, not a retry.
    """

    name = "ACCEPTANCE_ROUND_EXHAUSTED"

    def __init__(self, *, candidate_set: str, partition_id: str, reason: str) -> None:
        super().__init__(
            code="RL0302",
            message=(
                f"the acceptance round for candidate set {candidate_set} on partition "
                f"{partition_id} is exhausted: {reason}"
            ),
            remediation=(
                "seal a round against a fresh partition, or state a new protocol and seal a round "
                "under it; a second round on the same material would be scored against a partition "
                "this candidate set has already been measured on"
            ),
            context={
                "candidate_set": candidate_set,
                "partition_id": partition_id,
                "reason": reason,
            },
        )


class PanelNotRegistered(UsageError):
    """RL0001: a panel evaluator scored before it registered its membership and its estimand.

    Registering afterwards is backfilling: the membership and the estimand would then be chosen
    with the scores in hand, which is the post hoc rule D-39 forbids.
    """

    def __init__(self, *, panel_id: str, detail: str) -> None:
        base = usage_refusal(detail, panel_id=panel_id)
        super().__init__(
            code=base.code,
            message=base.message,
            remediation=base.remediation,
            context=dict(base.context),
        )
