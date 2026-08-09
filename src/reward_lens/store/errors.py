"""What the store refuses, and the codes wave 1 reserved for it.

RL0620 and RL0621 are P-STORE's own. RL0003 (bad config field) is allocated to the shared usage
family in `reward_lens.errors`; the store raises it because `rewardlens.yaml` is owner-authored
configuration and a field-level error is what its reader needs (section 6.1). No code is minted
here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reward_lens.contracts import UsageError

__all__ = [
    "BadConfigField",
    "DependencyDigestMismatch",
    "RunNotFound",
    "SubjectVersionRewritten",
]


class BadConfigField(UsageError):
    """RL0003: `rewardlens.yaml` is missing, unreadable, or one of its fields is wrong.

    `extra` is the context of a refusal this one restates, so that nothing the first refusal
    carried is lost when the store re-raises it as its own public error. The store's own `field`
    and `path` are written last, because the store is what knows which file was being read.
    """

    def __init__(
        self,
        *,
        field: str,
        message: str,
        remediation: str,
        path: Path | None = None,
        extra: Mapping[str, Any] | None = None,
    ):
        context: dict[str, Any] = dict(extra or {})
        context["field"] = field
        context["path"] = str(path) if path is not None else None
        super().__init__(
            code="RL0003",
            message=message,
            remediation=remediation,
            context=context,
        )


class DependencyDigestMismatch(UsageError):
    """RL0620: a record's declared dependency does not match the project it is recorded into."""

    def __init__(self, *, digest: str, expected: str | None, found: str | None):
        super().__init__(
            code="RL0620",
            message=(
                f"the {digest} digest of this record is not the {digest} digest of this project: "
                f"the record declares {found}, the project computes {expected}"
            ),
            remediation=(
                f"re-measure against the current {digest}, or record this into the project it was "
                "measured against; reuse is valid only when every declared dependency matches"
            ),
            context={"digest": digest, "expected": expected, "found": found},
        )


class SubjectVersionRewritten(UsageError):
    """RL0620: a second, different measurement of a subject version that already has one.

    The same refusal as a disagreeing digest, and for the same reason: a recorded measurement of a
    version is that version's measurement. A later one describes a later version, so it carries a
    subject version of its own.
    """

    def __init__(self, *, version: str, existing: str, name: str):
        super().__init__(
            code="RL0620",
            message=(
                f"subject version {version} is already measured by {existing}; a recorded "
                f"measurement is never rewritten, and {name} holds a different record of it"
            ),
            remediation=(
                "change the reward system, so the new measurement carries its own subject "
                f"version, or read {existing}, which is the measurement of this one"
            ),
            context={
                "digest": "subject.version",
                "version": version,
                "existing": existing,
                "name": name,
            },
        )


class RunNotFound(UsageError):
    """RL0621: no run directory answers to that id or name."""

    def __init__(self, *, wanted: str, root: Path):
        super().__init__(
            code="RL0621",
            message=f"no run called {wanted} under {root}",
            remediation="reward-lens runs list shows every run this machine kept",
            context={"wanted": wanted, "root": str(root)},
        )
