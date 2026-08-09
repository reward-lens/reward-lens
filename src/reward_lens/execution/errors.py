"""The three refusals this module makes, on the codes wave 1 reserved for P-EXEC.

RL0401 and RL0402 are capability refusals (exit 5): the machine cannot hold the tier, or it holds
one below what the caller demanded. RL0410 is a breach of a limit the supervisor set; it is exit 7
when an internal caller lets it escape, and an adapter is expected to catch it and turn it into a
verdict instead.
"""

from __future__ import annotations

from reward_lens.contracts.errors import CapabilityUnavailable, RewardLensError

__all__ = [
    "SandboxTierUnavailable",
    "SandboxTierBelowRequired",
    "ExecutionLimitBreached",
]


class SandboxTierUnavailable(CapabilityUnavailable):
    """RL0401: a named tier's probe did not pass on this machine.

    The probe that failed is named, because "sandbox unavailable" is not actionable and
    "bwrap --unshare-all --ro-bind / / /bin/true exited 1" is.
    """

    def __init__(self, *, tier: str, probe: str, reason: str) -> None:
        super().__init__(
            code="RL0401",
            message=f"sandbox tier {tier} is not available on this machine: {reason}",
            remediation=(
                f"run `{probe}` to see the failure directly, or ask for a lower tier with "
                "--require-tier"
            ),
            context={"tier": tier, "probe": probe, "reason": reason},
        )


class SandboxTierBelowRequired(CapabilityUnavailable):
    """RL0402: the ladder stopped below the tier the caller required."""

    def __init__(self, *, required: str, held: str, reason: str) -> None:
        super().__init__(
            code="RL0402",
            message=(
                f"sandbox tier {held} held on this machine, below the required {required}: {reason}"
            ),
            remediation=(
                f"install what {required} needs, or lower --require-tier to {held}; a record "
                "written at a tier below the one required would claim containment that did not hold"
            ),
            context={"required": required, "held": held, "reason": reason},
        )


class ExecutionLimitBreached(RewardLensError):
    """RL0410: a run exceeded one of its limits, and the dimension is named."""

    default_exit_code = 7

    def __init__(self, *, dimension: str, limit: object, observed: object = None) -> None:
        super().__init__(
            code="RL0410",
            message=f"execution limit breached on {dimension} (limit {limit})",
            remediation=f"raise the {dimension} limit, or fix the grader that consumes it",
            context={"dimension": dimension, "limit": limit, "observed": observed},
        )
        self.dimension = dimension
