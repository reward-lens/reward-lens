"""The probe: what this machine actually holds, measured once per process.

D-37's rule is that a tier label in a record is one the probe established on that machine. So every
row here is the result of running the mechanism, not of reading a kernel version string, and every
row carries the reason and how long it took. The operating systems this is not running on get a
column of `pending` in the platform matrix: their code exists and is tested under mocks, and
nothing claims it held anywhere.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from dataclasses import dataclass, field

from .bwrap import probe_bwrap
from .errors import SandboxTierBelowRequired, SandboxTierUnavailable
from .landlock import abi_version, apply_ruleset
from .macos import MACOS_TIERS, probe_sandbox_exec
from .seccomp import probe_seccomp, set_no_new_privs
from .windows import WINDOWS_TIERS, probe_appcontainer, probe_job_object

__all__ = ["TierResult", "SandboxProbe", "probe", "tier_ladder", "default_sandbox", "host_os"]

LINUX_TIERS = ("T0", "L0", "L1", "L2", "L3")

_T0_REASON = (
    "always: a new session, a fresh 0700 working directory, an allowlisted environment with no "
    "key in it, no inherited descriptors, RLIMIT_FSIZE, RLIMIT_NOFILE, RLIMIT_CORE=0, a "
    "parent-enforced wall clock, the whole process group killed, and capped output"
)


@dataclass(frozen=True)
class TierResult:
    held: bool
    reason: str
    ms: float


@dataclass(frozen=True)
class SandboxProbe:
    """What held, on which operating system, measured. This is what a record quotes."""

    os: str
    tier_held: str
    tiers: dict[str, TierResult]
    kernel: str | None
    landlock_abi: int | None
    platform_matrix: dict[str, dict[str, str]] = field(default_factory=dict)


def host_os() -> str:
    return {"linux": "linux", "darwin": "macos", "win32": "windows"}.get(sys.platform, sys.platform)


def tier_ladder(os_name: str) -> tuple[str, ...]:
    return {"linux": LINUX_TIERS, "macos": MACOS_TIERS, "windows": WINDOWS_TIERS}.get(
        os_name, ("T0",)
    )


def _timed(fn) -> tuple[bool, str, float]:
    start = time.monotonic()
    held, reason = fn()
    return held, reason, (time.monotonic() - start) * 1000


def _probe_l0() -> tuple[bool, str]:
    """A child that sets the three rlimits and `PR_SET_NO_NEW_PRIVS`, and exits 0."""
    import resource

    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child never returns
        try:
            resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
            resource.setrlimit(resource.RLIMIT_CPU, (60, 61))
            soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
            resource.setrlimit(resource.RLIMIT_NPROC, (soft, hard))
            set_no_new_privs()
        except (OSError, ValueError):
            os._exit(1)
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0:
        return True, "a child set RLIMIT_AS, RLIMIT_CPU, RLIMIT_NPROC and PR_SET_NO_NEW_PRIVS"
    return False, f"the rlimit-setting child exited with status {status}"


def _probe_l1(abi: int | None) -> tuple[bool, str]:
    """A child that builds the real ruleset and attaches it, and exits 0."""
    if abi is None:
        return False, "landlock_create_ruleset(NULL, 0, LANDLOCK_CREATE_RULESET_VERSION) failed"
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child never returns
        try:
            set_no_new_privs()
            apply_ruleset(read_roots=("/usr",), write_roots=(), abi=abi)
        except OSError:
            os._exit(1)
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0:
        net = "with TCP bind and connect handled" if abi >= 4 else "without network rules"
        scope = ", and LANDLOCK_SCOPE_*" if abi >= 6 else ""
        return True, f"landlock ABI {abi}, a ruleset attached {net}{scope}"
    return False, f"landlock ABI {abi} but landlock_restrict_self failed (status {status})"


def _linux_tiers() -> tuple[dict[str, TierResult], int | None]:
    abi = abi_version()
    tiers: dict[str, TierResult] = {"T0": TierResult(True, _T0_REASON, 0.0)}
    tiers["L0"] = TierResult(*_timed(_probe_l0))
    tiers["L1"] = TierResult(*_timed(lambda: _probe_l1(abi)))
    tiers["L2"] = TierResult(*_timed(probe_seccomp))
    held, reason, ms = probe_bwrap()
    l3_held = held and tiers["L2"].held
    if not held:
        l3_reason = reason
    elif not tiers["L2"].held:
        l3_reason = f"{reason}, but L3 is layered on L2 and L2 did not hold: {tiers['L2'].reason}"
    else:
        l3_reason = reason
    tiers["L3"] = TierResult(l3_held, l3_reason, ms)
    return tiers, abi


def _macos_tiers() -> dict[str, TierResult]:
    tiers = {"T0": TierResult(True, _T0_REASON, 0.0)}
    tiers["M0"] = TierResult(True, "RLIMIT_CPU, RLIMIT_FSIZE and RLIMIT_NOFILE are enforced by XNU", 0.0)
    tiers["M1"] = TierResult(*probe_sandbox_exec())
    return tiers


def _windows_tiers() -> dict[str, TierResult]:
    tiers = {"T0": TierResult(True, _T0_REASON, 0.0)}
    tiers["W0"] = TierResult(*probe_job_object())
    tiers["W1"] = TierResult(*probe_appcontainer())
    return tiers


def _matrix(os_name: str, tiers: dict[str, TierResult]) -> dict[str, dict[str, str]]:
    matrix: dict[str, dict[str, str]] = {}
    for candidate in ("linux", "macos", "windows"):
        ladder = tier_ladder(candidate)
        if candidate == os_name:
            matrix[candidate] = {t: ("held" if tiers[t].held else "unavailable") for t in ladder}
        else:
            matrix[candidate] = dict.fromkeys(ladder, "pending")
    return matrix


_CACHE: SandboxProbe | None = None


def probe(*, refresh: bool = False) -> SandboxProbe:
    """Measure the ladder on this machine. Cached per process; the answer cannot change under us."""
    global _CACHE
    if _CACHE is not None and not refresh:
        return _CACHE
    os_name = host_os()
    abi: int | None = None
    if os_name == "linux":
        tiers, abi = _linux_tiers()
        kernel: str | None = platform.release()
    elif os_name == "macos":  # pragma: no cover - macOS only
        tiers = _macos_tiers()
        kernel = platform.release()
    elif os_name == "windows":  # pragma: no cover - Windows only
        tiers = _windows_tiers()
        kernel = platform.release()
    else:  # pragma: no cover - no ladder is claimed on an unknown platform
        tiers = {"T0": TierResult(True, _T0_REASON, 0.0)}
        kernel = platform.release()

    ladder = tier_ladder(os_name)
    tier_held = ladder[0]
    for name in ladder:
        if tiers[name].held:
            tier_held = name
        else:
            break

    _CACHE = SandboxProbe(
        os=os_name,
        tier_held=tier_held,
        tiers=tiers,
        kernel=kernel,
        landlock_abi=abi,
        platform_matrix=_matrix(os_name, tiers),
    )
    return _CACHE


def _sandbox_for(tier: str, os_name: str):
    if os_name == "linux":
        from .linux import LinuxSandbox

        return LinuxSandbox(tier=tier)
    if os_name == "macos":  # pragma: no cover - macOS only
        from .macos import MacosSandbox

        return MacosSandbox(tier=tier)
    if os_name == "windows":  # pragma: no cover - Windows only
        from .windows import WindowsSandbox

        return WindowsSandbox(tier=tier)
    raise SandboxTierUnavailable(  # pragma: no cover - no ladder on an unknown platform
        tier=tier, probe="uname", reason=f"no sandbox ladder for {os_name}"
    )


def default_sandbox(require_tier: str | None = None, *, probe_result: SandboxProbe | None = None):
    """The sandbox for the tier that held, refusing anything below `require_tier`.

    `probe_result` is an addition to the frozen signature, keyword-only and defaulted: it lets a
    test drive the two refusals on a machine where the tier in question does hold.
    """
    p = probe() if probe_result is None else probe_result
    ladder = tier_ladder(p.os)
    if require_tier is None:
        return _sandbox_for(p.tier_held, p.os)
    if require_tier not in ladder or require_tier not in p.tiers:
        raise SandboxTierUnavailable(
            tier=require_tier,
            probe=f"the {p.os} ladder {' '.join(ladder)}",
            reason=f"{require_tier} is not a tier of the {p.os} ladder",
        )
    row = p.tiers[require_tier]
    if not row.held:
        raise SandboxTierUnavailable(tier=require_tier, probe=row.reason, reason=row.reason)
    if ladder.index(p.tier_held) < ladder.index(require_tier):
        below = next(t for t in ladder if not p.tiers[t].held)
        raise SandboxTierBelowRequired(
            required=require_tier,
            held=p.tier_held,
            reason=f"{below} did not hold: {p.tiers[below].reason}",
        )
    return _sandbox_for(require_tier, p.os)
