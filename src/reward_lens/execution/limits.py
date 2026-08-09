"""`Limits`, the breach dimensions, and the environment a grader is allowed to see.

The environment rule is the one worth stating plainly. T0 promises that no API key reaches grader
code, and a denylist of four variable names would not keep that promise: the base is an allowlist
built from nothing, and whatever the caller adds passes a credential filter on the way in. A caller
that hands over the whole of `os.environ` still gets a child with no key in it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .errors import SandboxTierBelowRequired

__all__ = [
    "Limits",
    "DEFAULT_LIMITS",
    "Breach",
    "BREACH_DIMENSIONS",
    "scrub_env",
    "SECRET_NAMES",
    "AS_MARGIN_MIN",
    "address_space_ceiling",
    "EGRESS_MECHANISM",
    "LANDLOCK_NET_ABI",
    "egress_mechanism",
    "require_egress_mechanism",
]

#: The six dimensions a breach can name. RL0410 carries one of these and nothing else.
Breach = Literal["wall", "cpu", "memory", "processes", "stdout", "artifact"]

BREACH_DIMENSIONS: tuple[str, ...] = (
    "wall",
    "cpu",
    "memory",
    "processes",
    "stdout",
    "artifact",
)


@dataclass(frozen=True, slots=True)
class Limits:
    """The per-call budget of D-37. Wall and CPU are separate because throttling makes them differ.

    `network` is false by default, and what enforces it is an OS filter named in
    `EGRESS_MECHANISM` for the tier the run is on; an environment variable is never the boundary.
    A tier with no entry there cannot honour the flag, and the sandbox refuses the run (RL0402)
    rather than accept a budget it will not keep.
    """

    wall_s: float = 30.0
    cpu_s: float = 30.0
    memory_bytes: int = 2 * 2**30
    processes: int = 64
    network: bool = False
    stdout_bytes: int = 1 << 20
    artifact_bytes: int = 64 << 20


DEFAULT_LIMITS = Limits()


#: How far above `memory_bytes` the `RLIMIT_AS` backstop sits.
#:
#: Setting `RLIMIT_AS` at the budget makes the allocator, not the supervisor, the thing that ends
#: a run: the kernel refuses the mapping that would cross the budget, the grader dies of
#: `MemoryError`, and the supervisor never observes a resident set over budget at all. Raising the
#: ceiling puts the rule back where the counters are. The margin has to clear a fresh
#: interpreter's own address space (tens of megabytes before the grader allocates anything) plus
#: whatever one allocation step adds, so it is the larger of 256 MiB and the budget itself; the
#: backstop remains, as a hard stop on a runaway reservation, at twice the budget or more.
AS_MARGIN_MIN = 256 << 20


def address_space_ceiling(memory_bytes: int) -> int:
    """The `RLIMIT_AS` value for a budget: above it, so the supervisor is what stops the run."""
    return int(memory_bytes) + max(AS_MARGIN_MIN, int(memory_bytes))


#: The Landlock ABI that first carries network access rights (`LANDLOCK_ACCESS_NET_*`).
LANDLOCK_NET_ABI = 4

#: What `network=False` is enforced by, tier by tier, in the words of the mechanism itself.
#:
#: A tier whose entry is `None` has no egress filter, and `require_egress_mechanism` refuses the
#: flag there instead of recording a containment that did not hold. L1 and L2 carry a Landlock
#: ruleset, which reaches TCP bind and connect and nothing else, so their entry says exactly that:
#: the complete claim belongs to L3, which has no interface to send on.
EGRESS_MECHANISM: dict[str, str | None] = {
    "T0": None,
    "L0": None,
    "L1": (
        "a Landlock ruleset with LANDLOCK_ACCESS_NET_BIND_TCP and LANDLOCK_ACCESS_NET_CONNECT_TCP "
        "in the handled set and no rule allowing either, so TCP bind and connect are denied; UDP "
        "and raw sockets are outside what a Landlock ruleset can reach"
    ),
    "L2": (
        "the L1 Landlock ruleset, so TCP bind and connect are denied, plus the syscall denylist; "
        "UDP and raw sockets are outside what a Landlock ruleset can reach"
    ),
    "L3": (
        "a network namespace of the run's own (bwrap --unshare-all), which has no interface at all "
        "beyond a loopback that is never brought up"
    ),
    "M0": None,
    "M1": "the sandbox-exec profile's (deny network*)",
    "W0": None,
    "W1": "an AppContainer token without the internet client capability",
}


def egress_mechanism(tier: str, *, landlock_abi: int | None = None) -> str | None:
    """What would filter egress at `tier` on this machine, or `None` if nothing would.

    The Landlock tiers are conditional on the ABI: before ABI 4 a ruleset has no network access
    rights at all, so L1 and L2 on such a kernel filter nothing and are reported as filtering
    nothing.
    """
    mechanism = EGRESS_MECHANISM.get(tier)
    if mechanism is None:
        return None
    if tier in ("L1", "L2") and (landlock_abi is None or landlock_abi < LANDLOCK_NET_ABI):
        return None
    return mechanism


def require_egress_mechanism(
    tier: str,
    *,
    network: bool,
    ladder: tuple[str, ...],
    landlock_abi: int | None = None,
) -> str | None:
    """Return the filter that will honour `network=False` at `tier`, or refuse with RL0402.

    `network=True` is a request for egress, not a containment claim, so there is nothing to
    enforce and nothing to refuse.
    """
    if network:
        return None
    mechanism = egress_mechanism(tier, landlock_abi=landlock_abi)
    if mechanism is not None:
        return mechanism
    lowest = next(
        (t for t in ladder if egress_mechanism(t, landlock_abi=landlock_abi) is not None), None
    )
    raise SandboxTierBelowRequired(
        required=lowest or "a tier with an egress filter",
        held=tier,
        reason=(
            f"network=False asks for egress to be filtered and {tier} has no filter for it; pass "
            f"network=True to run at {tier} with egress uncontained and say so in the record, or "
            f"require {lowest or 'a tier that has one'}"
        ),
    )


#: Variables a grader never sees, whoever passed them. The four D-37 names plus the shape of a
#: credential: anything whose name carries KEY, TOKEN, SECRET, PASSWORD, PASSWD or CREDENTIAL.
SECRET_NAMES: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "HF_TOKEN",
    "AWS_SECRET_ACCESS_KEY",
)

_SECRET_SHAPE = re.compile(
    r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|SESSION|COOKIE|AUTH)", re.IGNORECASE
)

#: Variables that change how the loader or the linker behaves, which a grader must not set for
#: the interpreter that runs it.
_LOADER_NAMES = frozenset(
    {
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "LD_AUDIT",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "PYTHONSTARTUP",
        "PYTHONEXECUTABLE",
        "PYTHONWARNINGS",
        "BASH_ENV",
        "ENV",
    }
)


def is_secret_name(name: str) -> bool:
    """True for anything that looks like a credential, by exact name or by shape."""
    return name in SECRET_NAMES or bool(_SECRET_SHAPE.search(name))


def scrub_env(requested: dict[str, str], *, cwd: Path) -> dict[str, str]:
    """Build the child's environment: an allowlisted base, then filtered caller entries.

    `HOME` and `TMPDIR` point at the working directory, so a grader that writes to either lands
    inside the one place L1 lets it write.
    """
    base = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(cwd),
        "TMPDIR": str(cwd),
        "TEMP": str(cwd),
        "TMP": str(cwd),
        "PWD": str(cwd),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "REWARD_LENS_SANDBOXED": "1",
    }
    for name in sorted(requested):
        if is_secret_name(name) or name in _LOADER_NAMES:
            continue
        base[name] = requested[name]
    # The base wins over a caller that tried to redirect the loader or unset determinism.
    base["PYTHONHASHSEED"] = "0"
    return base
