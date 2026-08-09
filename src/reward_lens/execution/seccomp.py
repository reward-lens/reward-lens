"""A hand-built seccomp-BPF denylist, installed through `prctl` with ctypes.

The Python bindings are not options: `pyseccomp` is from 2021 and `python-prctl` from 2020. The
program is small enough to write out: check the architecture, then compare the syscall number
against the nine numbers D-37 names, returning EPERM for those and allowing everything else. A
foreign architecture is killed rather than allowed, because a filter written for the wrong syscall
table is worse than no filter at all.
"""

from __future__ import annotations

import ctypes
import os
import platform

__all__ = [
    "DENIED_SYSCALLS",
    "build_filter",
    "install_filter",
    "probe_seccomp",
    "SyscallTableUnknown",
]

PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2

SECCOMP_RET_KILL_PROCESS = 0x80000000
SECCOMP_RET_ERRNO = 0x00050000
SECCOMP_RET_ALLOW = 0x7FFF0000
EPERM = 1

BPF_LD = 0x00
BPF_W = 0x00
BPF_ABS = 0x20
BPF_JMP = 0x05
BPF_JEQ = 0x10
BPF_K = 0x00
BPF_RET = 0x06

AUDIT_ARCH_X86_64 = 0xC000003E
AUDIT_ARCH_AARCH64 = 0xC00000B7

#: `seccomp_data` is {nr: s32, arch: u32, instruction_pointer: u64, args[6]: u64}.
_OFF_NR = 0
_OFF_ARCH = 4

#: The nine syscalls of D-37's denylist, per architecture. `kexec*` and `process_vm_*` are two
#: numbers each, which is why the list is nine rather than seven.
DENIED_SYSCALLS: dict[str, dict[str, int]] = {
    "ptrace": {"x86_64": 101, "aarch64": 117},
    "mount": {"x86_64": 165, "aarch64": 40},
    "unshare": {"x86_64": 272, "aarch64": 97},
    "bpf": {"x86_64": 321, "aarch64": 280},
    "perf_event_open": {"x86_64": 298, "aarch64": 241},
    "kexec_load": {"x86_64": 246, "aarch64": 104},
    "kexec_file_load": {"x86_64": 320, "aarch64": 294},
    "process_vm_readv": {"x86_64": 310, "aarch64": 270},
    "process_vm_writev": {"x86_64": 311, "aarch64": 271},
}

_AUDIT_ARCH = {"x86_64": AUDIT_ARCH_X86_64, "aarch64": AUDIT_ARCH_AARCH64}


class SyscallTableUnknown(OSError):
    """This architecture's syscall numbers are not written down here, so no filter is claimed."""


class SockFilter(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("code", ctypes.c_uint16),
        ("jt", ctypes.c_uint8),
        ("jf", ctypes.c_uint8),
        ("k", ctypes.c_uint32),
    ]


class SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(SockFilter))]


def _stmt(code: int, k: int) -> SockFilter:
    return SockFilter(code=code, jt=0, jf=0, k=k & 0xFFFFFFFF)


def _jump(code: int, k: int, jt: int, jf: int) -> SockFilter:
    return SockFilter(code=code, jt=jt, jf=jf, k=k & 0xFFFFFFFF)


def machine() -> str:
    return platform.machine()


def build_filter(arch: str | None = None) -> list[SockFilter]:
    """The BPF program: architecture guard, then one comparison per denied syscall."""
    arch = machine() if arch is None else arch
    if arch not in _AUDIT_ARCH:
        raise SyscallTableUnknown(f"no seccomp syscall table written down for {arch}")
    numbers = [DENIED_SYSCALLS[name][arch] for name in sorted(DENIED_SYSCALLS)]
    n = len(numbers)
    prog: list[SockFilter] = [
        _stmt(BPF_LD | BPF_W | BPF_ABS, _OFF_ARCH),
        _jump(BPF_JMP | BPF_JEQ | BPF_K, _AUDIT_ARCH[arch], 1, 0),
        _stmt(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
        _stmt(BPF_LD | BPF_W | BPF_ABS, _OFF_NR),
    ]
    # Each match jumps forward to the EPERM return, which sits one past the allow.
    for i, nr in enumerate(numbers):
        prog.append(_jump(BPF_JMP | BPF_JEQ | BPF_K, nr, n - i, 0))
    prog.append(_stmt(BPF_RET | BPF_K, SECCOMP_RET_ALLOW))
    prog.append(_stmt(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM))
    return prog


def install_filter(arch: str | None = None) -> None:
    """Attach the filter to the calling process. `PR_SET_NO_NEW_PRIVS` must already be set."""
    prog = build_filter(arch)
    array = (SockFilter * len(prog))(*prog)
    fprog = SockFprog(len=len(prog), filter=array)
    libc = ctypes.CDLL(None, use_errno=True)
    ctypes.set_errno(0)
    rc = libc.prctl(
        ctypes.c_int(PR_SET_SECCOMP),
        ctypes.c_ulong(SECCOMP_MODE_FILTER),
        ctypes.byref(fprog),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    if rc != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER) failed")


def set_no_new_privs() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    ctypes.set_errno(0)
    rc = libc.prctl(
        ctypes.c_int(PR_SET_NO_NEW_PRIVS),
        ctypes.c_ulong(1),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    if rc != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")


def probe_seccomp() -> tuple[bool, str]:
    """Fork a child that installs the filter and exits 0. Nothing is assumed from the kernel name."""
    if platform.system() != "Linux":
        return False, "seccomp is a Linux mechanism"
    try:
        build_filter()
    except SyscallTableUnknown as exc:
        return False, str(exc)
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child never returns to the test runner
        try:
            set_no_new_privs()
            install_filter()
        except OSError:
            os._exit(1)
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0:
        return True, f"a child installed the {len(DENIED_SYSCALLS)}-syscall filter and exited 0"
    return False, f"the filter-installing child exited with status {status}"
