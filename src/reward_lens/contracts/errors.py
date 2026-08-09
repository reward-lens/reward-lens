"""`RewardLensError` and the exit-code family (D-22), plus the four codes P-CONTRACT reserves.

Every packet defines its own exceptions as subclasses of `RewardLensError` carrying a code the
wave-1 interface file reserved for it. The codes here are RL0601 to RL0604. `to_json()` is the
stderr twin: the three fields a machine reader needs, and nothing else.

Python's own JSON serialiser is not imported anywhere in this package: `to_json()` goes through
`rfc8785`, the only serialiser on any path that might be hashed or compared (D-10).
"""

from __future__ import annotations

from typing import Any

import rfc8785

__all__ = [
    "RewardLensError",
    "UsageError",
    "CapabilityUnavailable",
    "BudgetExceeded",
    "InternalFailure",
    "PendingDecision",
    "SchemaUnknownMajor",
    "EntryKindEvidenceMissing",
    "NumberNotQuantised",
    "RecordInvalid",
]


class RewardLensError(Exception):
    """The base of every refusal this project makes.

    `code` is `RL` plus four digits, allocated once in `reward_lens.errors`. `exit_code` is the
    process exit status the CLI uses; the subclasses below fix it per family.
    """

    default_exit_code: int = 7

    def __init__(
        self,
        *,
        code: str,
        message: str,
        remediation: str = "",
        context: dict[str, Any] | None = None,
        exit_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.remediation = remediation
        self.context: dict[str, Any] = dict(context or {})
        self.exit_code = self.default_exit_code if exit_code is None else exit_code

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.code}: {self.message}"

    def to_json(self) -> str:
        """The stderr twin: `{code, message, remediation}`, canonical bytes decoded."""
        return rfc8785.dumps(
            {
                "code": self.code,
                "message": self.message,
                "remediation": self.remediation,
            }
        ).decode("utf-8")


class UsageError(RewardLensError):
    """Exit 4: the caller asked for something the inputs cannot support."""

    default_exit_code = 4


class CapabilityUnavailable(RewardLensError):
    """Exit 5: the instrument is not in this build, or the machine cannot hold the tier."""

    default_exit_code = 5


class BudgetExceeded(RewardLensError):
    """Exit 6: a money cap was reached."""

    default_exit_code = 6


class InternalFailure(RewardLensError):
    """Exit 7: a defect in reward-lens itself."""

    default_exit_code = 7


class PendingDecision(RewardLensError):
    """Exit 3: the run stopped on a decision a person has to make."""

    default_exit_code = 3

    def __init__(
        self,
        *,
        code: str,
        message: str,
        remediation: str = "",
        context: dict[str, Any] | None = None,
        exit_code: int | None = None,
        argv: list[str] | None = None,
        command: str = "",
        decision: str = "",
    ) -> None:
        super().__init__(
            code=code,
            message=message,
            remediation=remediation,
            context=context,
            exit_code=exit_code,
        )
        self.argv: list[str] = list(argv or [])
        self.command = command
        self.decision = decision


# --- the four codes this packet reserves (wave-1 interfaces, section 8) -------------------------


class SchemaUnknownMajor(UsageError):
    """RL0601. The record declares a schema major this build cannot read (D-64).

    The original bytes are preserved on the error so a migration can still read them; nothing is
    rewritten, coerced or dropped.
    """

    def __init__(self, *, message: str, remediation: str, context: dict[str, Any]) -> None:
        super().__init__(code="RL0601", message=message, remediation=remediation, context=context)

    @property
    def original_bytes(self) -> bytes | None:
        return self.context.get("original_bytes")


class EntryKindEvidenceMissing(UsageError):
    """RL0602. An entry does not carry the validity evidence its kind calls for (D-11)."""

    def __init__(self, *, message: str, remediation: str, context: dict[str, Any]) -> None:
        super().__init__(code="RL0602", message=message, remediation=remediation, context=context)


class NumberNotQuantised(UsageError):
    """RL0603. A number carries more decimal places than its declared `x-scale` (D-10)."""

    def __init__(self, *, message: str, remediation: str, context: dict[str, Any]) -> None:
        super().__init__(code="RL0603", message=message, remediation=remediation, context=context)


class RecordInvalid(UsageError):
    """RL0604. The record does not validate against the frozen schema (D-66)."""

    def __init__(self, *, message: str, remediation: str, context: dict[str, Any]) -> None:
        super().__init__(code="RL0604", message=message, remediation=remediation, context=context)
