"""The two refusals the trace spine owns: RL0610 and RL0611 (wave-2 interfaces, section 6).

Both sit in the record-integrity band and both exit 4, because both mean the same thing to the
caller: the record in front of you cannot support the reading you asked for. What separates them is
what the message has to name. `RECORD_INCOMPLETE` names the series it could not build, because a
reader who is told only that the record is incomplete has to go and find out which half is missing.
`RECORD_INTEGRITY` names the row and the rule, because a rule that fires without a row is a claim
about the whole file that nobody can check.
"""

from __future__ import annotations

from typing import Any

from reward_lens.contracts.errors import UsageError

__all__ = ["RecordIncomplete", "RecordIntegrity"]


class RecordIncomplete(UsageError):
    """RL0610. A named series the reading needs is not in the record.

    `context["series"]` is the list of missing series names, in the canonical order, and the
    message names them too: a machine reader takes the list, a person reads the sentence.
    """

    def __init__(
        self, *, message: str, remediation: str = "", context: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            code="RL0610", message=message, remediation=remediation, context=dict(context or {})
        )


class RecordIntegrity(UsageError):
    """RL0611. A named integrity rule failed, on a row this names.

    `context["rule"]` is the rule id and `context["row"]` is the row identity it failed on.
    """

    def __init__(
        self, *, message: str, remediation: str = "", context: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            code="RL0611", message=message, remediation=remediation, context=dict(context or {})
        )
