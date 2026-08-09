"""The macOS ladder: M0 rlimits, M1 Seatbelt, and an RSS-sampling memory ceiling.

Nothing here is claimed on a machine that is not macOS. `probe()` marks the whole column pending
off-platform, and `MacosSandbox.run` refuses with RL0401 rather than quietly falling back to T0.

Two macOS facts drive the shape. `sandbox-exec` is deprecated and has no replacement for
command-line process sandboxing, so the deprecation is accepted and recorded in the probe's reason
rather than suppressed. And the memory ceiling cannot be an rlimit: XNU does not enforce
`RLIMIT_AS`, and `RLIMIT_NPROC` is per uid, so setting it low would kill the owner's other
processes. The supervisor's resident-set sampler is the ceiling, and the probe allocates past it.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

from .errors import SandboxTierUnavailable
from .limits import Limits

__all__ = [
    "SANDBOX_EXEC_PROBE_ARGV",
    "MacosSandbox",
    "probe_sandbox_exec",
    "seatbelt_profile",
    "MACOS_TIERS",
]

SANDBOX_EXEC_PROBE_ARGV = [
    "sandbox-exec",
    "-p",
    "(version 1)(allow default)",
    "/usr/bin/true",
]

MACOS_TIERS = ("T0", "M0", "M1")

_SYSTEM_READ_ROOTS = ("/usr", "/System", "/Library", "/bin", "/sbin", "/private/var/db/dyld")


def seatbelt_profile(work: Path, write_roots: Sequence[Path | str] = ()) -> str:
    """Deny by default, read the system paths, write the write roots, no network at all.

    Every write root is granted `file-read*` beside `file-write*`, which is the same rule the
    Linux ladder holds: a write root is a read root too.
    """
    lines = ["(version 1)", "(deny default)", "(allow process-exec*)", "(allow process-fork)"]
    lines += [f'(allow file-read* (subpath "{root}"))' for root in _SYSTEM_READ_ROOTS]
    for root in sorted({str(Path(work))} | {str(Path(one)) for one in write_roots}):
        lines.append(f'(allow file-read* (subpath "{root}"))')
        lines.append(f'(allow file-write* (subpath "{root}"))')
    lines.append('(allow file-write-data (literal "/dev/null"))')
    lines.append("(allow sysctl-read)")
    lines.append("(deny network*)")
    return "\n".join(lines) + "\n"


def probe_sandbox_exec() -> tuple[bool, str, float]:
    """Returns (held, reason, milliseconds). The reason records the deprecation."""
    if platform.system() != "Darwin":
        return False, "sandbox-exec is a macOS mechanism; pending on this platform", 0.0
    start = time.monotonic()
    try:
        done = subprocess.run(  # noqa: S603 - a fixed argv
            SANDBOX_EXEC_PROBE_ARGV, capture_output=True, timeout=2.0, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover - off-platform
        return False, f"{' '.join(SANDBOX_EXEC_PROBE_ARGV)} failed: {exc}", (
            time.monotonic() - start
        ) * 1000
    ms = (time.monotonic() - start) * 1000
    held = done.returncode == 0
    reason = (
        f"{' '.join(SANDBOX_EXEC_PROBE_ARGV)} exited {done.returncode}; the command is deprecated "
        "and still the only command-line Seatbelt front end, and that deprecation is recorded here "
        "rather than suppressed"
    )
    return held, reason, ms


class MacosSandbox:
    """M0 and M1 behind the same `Sandbox` protocol as the Linux ladder."""

    memory_enforcement = "supervisor_rss_sampling"

    def __init__(self, *, tier: str = "M1") -> None:
        if tier not in MACOS_TIERS:
            raise SandboxTierUnavailable(
                tier=tier, probe="macOS ladder", reason=f"unknown macOS tier {tier}"
            )
        self.tier = tier

    def rlimit_names(self, limits: Limits) -> tuple[str, ...]:
        """The rlimits macOS actually enforces. `RLIMIT_AS` and `RLIMIT_NPROC` are not among them."""
        return ("RLIMIT_CPU", "RLIMIT_FSIZE", "RLIMIT_NOFILE", "RLIMIT_CORE")

    def wrap(
        self, argv: list[str], *, cwd: Path, write_roots: Sequence[Path | str] = ()
    ) -> tuple[list[str], Path]:
        """M1 wraps the argv in `sandbox-exec -f <profile>`; M0 leaves it alone."""
        profile_path = Path(tempfile.mkdtemp(prefix="rl-seatbelt-")) / "profile.sb"
        profile_path.write_text(seatbelt_profile(cwd, write_roots), encoding="utf-8")
        if self.tier != "M1":
            return list(argv), profile_path
        return ["sandbox-exec", "-f", str(profile_path), *argv], profile_path

    def run(
        self,
        argv: list[str],
        *,
        limits: Limits,
        cwd: Path,
        env: dict[str, str],
        stdin: bytes | None = None,
        write_roots: Sequence[Path | str] = (),
    ):
        if sys.platform != "darwin":
            raise SandboxTierUnavailable(
                tier=self.tier,
                probe=" ".join(SANDBOX_EXEC_PROBE_ARGV),
                reason=f"the macOS ladder is pending on {sys.platform}; it is not claimed here",
            )
        from .limits import require_egress_mechanism  # pragma: no cover - macOS only
        from .supervisor import supervise  # pragma: no cover - macOS only

        require_egress_mechanism(  # pragma: no cover - macOS only
            self.tier, network=limits.network, ladder=MACOS_TIERS
        )
        wrapped, _profile = self.wrap(  # pragma: no cover - macOS only
            argv, cwd=cwd, write_roots=write_roots
        )
        return supervise(  # pragma: no cover - macOS only
            wrapped,
            limits=limits,
            cwd=cwd,
            env=env,
            stdin=stdin,
            tier=self.tier,
            sandbox_os="macos",
            child_setup=None,
        )
