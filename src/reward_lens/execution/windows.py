"""The Windows ladder: W0 job objects, W1 an AppContainer with zero capabilities.

Neither needs an administrator, which is why they are the ladder and Codex's dedicated-sandbox-user
design is not. Job objects are a better rlimit than rlimits: the memory limit is real, and
`KILL_ON_JOB_CLOSE` means a crashed parent cannot orphan a grader. AppContainer with no capabilities
is the cleanest no-network guarantee of any operating system here.

Nothing below is claimed on a machine that is not Windows: `probe()` marks the column pending and
the tests drive these paths with mocked `kernel32` and `userenv`.
"""

from __future__ import annotations

import ctypes
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from .errors import SandboxTierUnavailable
from .limits import Limits

__all__ = [
    "WindowsSandbox",
    "job_limit_flags",
    "probe_job_object",
    "probe_appcontainer",
    "WINDOWS_TIERS",
    "JOB_OBJECT_LIMIT_JOB_MEMORY",
    "JOB_OBJECT_LIMIT_JOB_TIME",
    "JOB_OBJECT_LIMIT_ACTIVE_PROCESS",
    "JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE",
]

WINDOWS_TIERS = ("T0", "W0", "W1")

JOB_OBJECT_LIMIT_JOB_TIME = 0x00000004
JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

JOBOBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9


def job_limit_flags() -> int:
    """The four limits D-37 names, and no fifth one smuggled in."""
    return (
        JOB_OBJECT_LIMIT_JOB_MEMORY
        | JOB_OBJECT_LIMIT_JOB_TIME
        | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
        | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    )


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in
                ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                 "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _kernel32():  # pragma: no cover - replaced by the tests off-platform
    return ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]


def _userenv():  # pragma: no cover - replaced by the tests off-platform
    return ctypes.WinDLL("userenv", use_last_error=True)  # type: ignore[attr-defined]


def probe_job_object() -> tuple[bool, str, float]:
    """Create a job, set the four limits on it, close it. No admin, no side effect."""
    start = time.monotonic()
    try:
        k32 = _kernel32()
    except (AttributeError, OSError) as exc:
        return False, f"job objects are a Windows mechanism; pending here ({exc})", 0.0
    handle = k32.CreateJobObjectW(None, None)
    if not handle:
        return False, "CreateJobObjectW returned NULL", (time.monotonic() - start) * 1000
    info = _ExtendedLimitInformation()
    info.BasicLimitInformation.LimitFlags = job_limit_flags()
    ok = k32.SetInformationJobObject(
        handle,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    k32.CloseHandle(handle)
    ms = (time.monotonic() - start) * 1000
    if not ok:
        return False, "SetInformationJobObject refused the four limits", ms
    return True, "CreateJobObjectW and SetInformationJobObject succeeded without admin", ms


def probe_appcontainer() -> tuple[bool, str, float]:
    """Create an AppContainer profile with zero capabilities, then delete it again."""
    start = time.monotonic()
    try:
        userenv = _userenv()
    except (AttributeError, OSError) as exc:
        return False, f"AppContainer is a Windows mechanism; pending here ({exc})", 0.0
    name = "reward-lens-probe"
    sid = ctypes.c_void_p()
    hr = userenv.CreateAppContainerProfile(name, name, name, None, 0, ctypes.byref(sid))
    ms = (time.monotonic() - start) * 1000
    if hr != 0:
        return False, f"CreateAppContainerProfile returned 0x{hr:08x}", ms
    userenv.DeleteAppContainerProfile(name)
    return True, "CreateAppContainerProfile succeeded with zero capabilities and no admin", ms


class WindowsSandbox:
    """W0 and W1 behind the same `Sandbox` protocol as the Linux ladder."""

    EXTENDED_LIMIT_INFORMATION = _ExtendedLimitInformation
    memory_enforcement = "job_object_JOB_MEMORY"

    def __init__(self, *, tier: str = "W1") -> None:
        if tier not in WINDOWS_TIERS:
            raise SandboxTierUnavailable(
                tier=tier, probe="Windows ladder", reason=f"unknown Windows tier {tier}"
            )
        self.tier = tier

    def create_job(self, limits: Limits) -> int:
        """One job carrying exactly the four limits, in the units Windows uses."""
        k32 = _kernel32()
        handle = k32.CreateJobObjectW(None, None)
        info = _ExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = job_limit_flags()
        info.BasicLimitInformation.PerJobUserTimeLimit = int(limits.wall_s * 10_000_000)
        info.BasicLimitInformation.ActiveProcessLimit = limits.processes
        info.JobMemoryLimit = limits.memory_bytes
        k32.SetInformationJobObject(
            handle,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        return handle

    def run(
        self,
        argv: list[str],
        *,
        limits: Limits,
        cwd,
        env: dict[str, str],
        stdin: bytes | None = None,
        write_roots: Sequence[Path | str] = (),
    ):
        # The Windows ladder is a job object, which bounds resources and not paths, so the write
        # roots are accepted and unused: there is no filesystem ruleset here to name them in.
        if sys.platform != "win32":
            raise SandboxTierUnavailable(
                tier=self.tier,
                probe="CreateJobObjectW / CreateAppContainerProfile",
                reason=f"the Windows ladder is pending on {sys.platform}; it is not claimed here",
            )
        from .limits import require_egress_mechanism  # pragma: no cover - Windows only
        from .supervisor import supervise  # pragma: no cover - Windows only

        require_egress_mechanism(  # pragma: no cover - Windows only
            self.tier, network=limits.network, ladder=WINDOWS_TIERS
        )
        job = self.create_job(limits)  # pragma: no cover - Windows only
        try:  # pragma: no cover - Windows only
            return supervise(
                list(argv),
                limits=limits,
                cwd=cwd,
                env=env,
                stdin=stdin,
                tier=self.tier,
                sandbox_os="windows",
                child_setup=None,
                on_spawn=lambda pid: _kernel32().AssignProcessToJobObject(job, pid),
            )
        finally:
            _kernel32().CloseHandle(job)
