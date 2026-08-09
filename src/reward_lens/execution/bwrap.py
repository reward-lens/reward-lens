"""The bubblewrap tier, probed and never assumed.

Ubuntu 23.10 and later ship `kernel.apparmor_restrict_unprivileged_userns=1`, and the
`bwrap-userns-restrict` AppArmor profile that lifts it needs root to install. So bubblewrap is not
the floor: it is a tier that is layered on top of L2 when, and only when, this command exits 0 on
this machine.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from .limits import Limits

__all__ = ["BWRAP_PROBE_ARGV", "probe_bwrap", "bwrap_argv"]

#: The probe D-37 names. It runs in well under 100 ms when bubblewrap is usable at all.
BWRAP_PROBE_ARGV = ["bwrap", "--unshare-all", "--ro-bind", "/", "/", "/bin/true"]

_PROBE_TIMEOUT_S = 0.5


def probe_bwrap() -> tuple[bool, str, float]:
    """Returns (held, reason, milliseconds). The duration is recorded, not discarded."""
    if shutil.which("bwrap") is None:
        return False, "bwrap is not on PATH", 0.0
    start = time.monotonic()
    try:
        done = subprocess.run(
            BWRAP_PROBE_ARGV,
            capture_output=True,
            timeout=_PROBE_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"{' '.join(BWRAP_PROBE_ARGV)} did not finish in {_PROBE_TIMEOUT_S}s", (
            time.monotonic() - start
        ) * 1000
    ms = (time.monotonic() - start) * 1000
    if done.returncode == 0:
        return True, f"{' '.join(BWRAP_PROBE_ARGV)} exited 0", ms
    detail = done.stderr.decode("utf-8", "replace").strip().splitlines()
    tail = detail[-1] if detail else ""
    return (
        False,
        f"{' '.join(BWRAP_PROBE_ARGV)} exited {done.returncode}: {tail}",
        ms,
    )


def bwrap_argv(
    argv: list[str],
    *,
    cwd: Path,
    limits: Limits,
    read_roots: tuple[str, ...],
    write_roots: tuple[str, ...] = (),
    seccomp_fd: int | None = None,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Wrap an argv in bubblewrap: everything read-only, the write roots writable, no network.

    `--clearenv` is on because the environment the child gets is the one the supervisor built, not
    a copy of the parent's; `--die-with-parent` is what stops an orphaned grader outliving the run.

    The working directory is always a write root, and `write_roots` names any others the caller
    needs, in the same order Landlock grants them at L1 and L2. `--bind` carries read as well as
    write, so a bound root is readable: that is what lets a process open a script that was staged
    for it. The binds come after `--tmpfs /tmp`, so a root under `/tmp` survives the tmpfs that
    replaces the rest of it, and the working directory is bound last so its own bind wins over a
    root that happens to contain it.
    """
    roots = [str(r) for r in write_roots if str(r) != str(cwd)]
    wrapped = [
        "bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--tmpfs",
        "/tmp",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
    ]
    for root in sorted(set(roots)):
        if Path(root).exists():
            wrapped += ["--bind", root, root]
    wrapped += [
        "--bind",
        str(cwd),
        str(cwd),
        "--chdir",
        str(cwd),
    ]
    if limits.network:  # pragma: no cover - egress is off by default and off in wave 1
        wrapped.remove("--unshare-all")
        wrapped[1:1] = ["--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts"]
    # Only the stated read roots are bound. Binding `/` read-only, which is what this did before,
    # confined writes but left every path on the machine readable, so L3 read less than L1 did and
    # the ladder was not monotone. `/proc`, `/dev` and `/tmp` are left to the mounts above: a bind
    # of the host's `/proc` over the namespace's own would undo the PID namespace.
    for root in sorted(set(read_roots)):
        if root not in {"/", "/proc", "/dev", "/tmp"} and Path(root).exists():
            wrapped += ["--ro-bind-try", root, root]
    for name in sorted(env or {}):
        wrapped += ["--setenv", name, (env or {})[name]]
    if seccomp_fd is not None:
        # L3 is L2 inside bubblewrap, so the denylist has to be installed by bwrap after it has
        # made its own namespaces: a filter applied outside would deny bwrap the `unshare` and
        # `mount` calls that are the tier.
        wrapped += ["--seccomp", str(seccomp_fd)]
    return wrapped + ["--"] + list(argv)
