"""A ctypes shim over the three Landlock syscalls, with the kernel ABI read rather than assumed.

There is no maintained Python binding: PyPI's `landlock` is a 2025 development release and there is
no `pylandlock`, so a dependency here would be worse than a hundred lines of `ctypes`. The ABI
matters because the kernel refuses a ruleset carrying an access bit it does not know (EINVAL), so
the handled bits are downgraded to what `landlock_create_ruleset(NULL, 0,
LANDLOCK_CREATE_RULESET_VERSION)` reports on this machine.

ABI ladder: 1 (5.13) filesystem; 2 (5.19) refer; 3 (6.2) truncate; 4 (6.7) TCP bind and connect;
5 (6.10) ioctl on devices; 6 (6.12) scoped abstract sockets and signals; 7 (6.15) audit logging.
A kernel reporting more than we know is handled as the highest ABI we do know.
"""

from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path

__all__ = [
    "abi_version",
    "handled_fs_for_abi",
    "handled_net_for_abi",
    "scoped_for_abi",
    "apply_ruleset",
    "ACCESS_FS_BY_ABI",
    "ACCESS_FS_READ",
    "ACCESS_FS_WRITE",
    "ACCESS_FS_WRITE_ONLY",
    "ACCESS_NET_BIND_TCP",
    "ACCESS_NET_CONNECT_TCP",
    "LANDLOCK_SYSCALLS",
]

# --- syscall numbers ---------------------------------------------------------------------------
# 444/445/446 in the generic table; every architecture that has Landlock at all uses these.
NR_CREATE_RULESET = 444
NR_ADD_RULE = 445
NR_RESTRICT_SELF = 446
LANDLOCK_SYSCALLS = (NR_CREATE_RULESET, NR_ADD_RULE, NR_RESTRICT_SELF)

LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
LANDLOCK_RULE_PATH_BENEATH = 1
LANDLOCK_RULE_NET_PORT = 2

# --- filesystem access bits --------------------------------------------------------------------
FS_EXECUTE = 1 << 0
FS_WRITE_FILE = 1 << 1
FS_READ_FILE = 1 << 2
FS_READ_DIR = 1 << 3
FS_REMOVE_DIR = 1 << 4
FS_REMOVE_FILE = 1 << 5
FS_MAKE_CHAR = 1 << 6
FS_MAKE_DIR = 1 << 7
FS_MAKE_REG = 1 << 8
FS_MAKE_SOCK = 1 << 9
FS_MAKE_FIFO = 1 << 10
FS_MAKE_BLOCK = 1 << 11
FS_MAKE_SYM = 1 << 12
FS_REFER = 1 << 13  # ABI 2
FS_TRUNCATE = 1 << 14  # ABI 3
FS_IOCTL_DEV = 1 << 15  # ABI 5

#: Everything the kernel handles, per ABI. A ruleset asks for the highest row it can.
ACCESS_FS_BY_ABI: dict[int, int] = {
    1: (1 << 13) - 1,
    2: (1 << 14) - 1,
    3: (1 << 15) - 1,
    5: (1 << 16) - 1,
}

#: What a read-only root is allowed. No write bit of any kind appears here.
ACCESS_FS_READ = FS_EXECUTE | FS_READ_FILE | FS_READ_DIR
#: What a write root is allowed *beyond* reading: content, creation, removal, truncation.
#: `FS_TRUNCATE` is here because `open(path, "w")` on an existing file truncates it, and without
#: this bit that call is EACCES on a directory the ruleset otherwise grants in full.
ACCESS_FS_WRITE_ONLY = (
    FS_WRITE_FILE
    | FS_REMOVE_DIR
    | FS_REMOVE_FILE
    | FS_MAKE_CHAR
    | FS_MAKE_DIR
    | FS_MAKE_REG
    | FS_MAKE_SOCK
    | FS_MAKE_FIFO
    | FS_MAKE_BLOCK
    | FS_MAKE_SYM
    | FS_REFER
    | FS_TRUNCATE
    | FS_IOCTL_DEV
)
#: What a write root is allowed. **A write root is a read root too**: a process that may create a
#: file in a directory but not open one there cannot run a script it wrote, and that asymmetry is
#: what kept a staged grader from reading its own supervisor. The union is every bit this module
#: knows, masked to the ABI in `apply_ruleset`.
ACCESS_FS_WRITE = ACCESS_FS_READ | ACCESS_FS_WRITE_ONLY

ACCESS_NET_BIND_TCP = 1 << 0  # ABI 4
ACCESS_NET_CONNECT_TCP = 1 << 1  # ABI 4

SCOPE_ABSTRACT_UNIX_SOCKET = 1 << 0  # ABI 6
SCOPE_SIGNAL = 1 << 1  # ABI 6


class RulesetAttr(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),  # read only when size >= 16 (ABI 4)
        ("scoped", ctypes.c_uint64),  # read only when size >= 24 (ABI 6)
    ]


class PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class NetPortAttr(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("port", ctypes.c_uint64)]


def _libc() -> ctypes.CDLL:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    return libc


def abi_version() -> int | None:
    """The kernel's Landlock ABI, or None where Landlock is absent or the arch is not Linux."""
    if platform.system() != "Linux":
        return None
    try:
        libc = _libc()
    except OSError:  # pragma: no cover - libc is always loadable on Linux
        return None
    ctypes.set_errno(0)
    rc = libc.syscall(
        ctypes.c_long(NR_CREATE_RULESET),
        ctypes.c_void_p(0),
        ctypes.c_size_t(0),
        ctypes.c_uint32(LANDLOCK_CREATE_RULESET_VERSION),
    )
    if rc < 1:
        return None
    return int(rc)


def handled_fs_for_abi(abi: int) -> int:
    """The filesystem bits this ABI knows. Never more: an unknown bit is EINVAL, not a warning."""
    known = sorted(ACCESS_FS_BY_ABI)
    chosen = known[0]
    for level in known:
        if abi >= level:
            chosen = level
    return ACCESS_FS_BY_ABI[chosen]


def handled_net_for_abi(abi: int) -> int:
    """TCP bind and connect arrive at ABI 4; below that Landlock says nothing about the network."""
    if abi < 4:
        return 0
    return ACCESS_NET_BIND_TCP | ACCESS_NET_CONNECT_TCP


def scoped_for_abi(abi: int) -> int:
    """Abstract-socket and signal scoping arrive at ABI 6."""
    if abi < 6:
        return 0
    return SCOPE_ABSTRACT_UNIX_SOCKET | SCOPE_SIGNAL


def _attr_size(abi: int) -> int:
    if abi >= 6:
        return 24
    if abi >= 4:
        return 16
    return 8


def apply_ruleset(
    *,
    read_roots: tuple[str, ...],
    write_roots: tuple[str, ...],
    read_write_files: tuple[str, ...] = (),
    allow_net: bool = False,
    abi: int | None = None,
) -> int:
    """Build the ruleset and attach it to the calling process. Returns the ABI it was built for.

    Called in the forked child after `PR_SET_NO_NEW_PRIVS`. Every path is opened here, after the
    working directory exists, because Landlock rules are path based and a rule cannot name a
    directory that is not there yet.

    `write_roots` implies read on the same paths: each one is granted `ACCESS_FS_WRITE`, which is
    `ACCESS_FS_READ` plus the write bits. Naming a path in both sets is harmless and changes
    nothing. A path that does not exist is skipped rather than raising, because a caller may name
    a scratch root that this run did not need.
    """
    abi = abi_version() if abi is None else abi
    if abi is None:
        raise OSError("landlock unavailable")

    libc = _libc()
    handled_fs = handled_fs_for_abi(abi)
    handled_net = 0 if allow_net else handled_net_for_abi(abi)
    scoped = scoped_for_abi(abi)

    attr = RulesetAttr(
        handled_access_fs=handled_fs,
        handled_access_net=handled_net,
        scoped=scoped,
    )
    ctypes.set_errno(0)
    ruleset_fd = libc.syscall(
        ctypes.c_long(NR_CREATE_RULESET),
        ctypes.byref(attr),
        ctypes.c_size_t(_attr_size(abi)),
        ctypes.c_uint32(0),
    )
    if ruleset_fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")

    try:
        def add_path(path: str, access: int) -> None:
            if not os.path.exists(path):
                return
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathBeneathAttr(
                    allowed_access=access & handled_fs, parent_fd=ctypes.c_int32(fd).value
                )
                ctypes.set_errno(0)
                rc = libc.syscall(
                    ctypes.c_long(NR_ADD_RULE),
                    ctypes.c_int(ruleset_fd),
                    ctypes.c_int(LANDLOCK_RULE_PATH_BENEATH),
                    ctypes.byref(rule),
                    ctypes.c_uint32(0),
                )
                if rc < 0:
                    raise OSError(ctypes.get_errno(), f"landlock_add_rule failed for {path}")
            finally:
                os.close(fd)

        for path in sorted(set(read_roots)):
            add_path(path, ACCESS_FS_READ)
        for path in sorted(set(write_roots)):
            add_path(path, ACCESS_FS_WRITE)
        for path in sorted(set(read_write_files)):
            add_path(path, FS_READ_FILE | FS_WRITE_FILE | FS_IOCTL_DEV)

        ctypes.set_errno(0)
        rc = libc.syscall(ctypes.c_long(NR_RESTRICT_SELF), ctypes.c_int(ruleset_fd), ctypes.c_uint32(0))
        if rc < 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    finally:
        os.close(ruleset_fd)
    return abi


def interpreter_read_roots(executable: str) -> tuple[str, ...]:
    """The directories an interpreter needs to read to run at all.

    A ruleset that granted only `/usr` and `/lib` could not exec a virtualenv's Python, whose
    standard library and site-packages live under its own prefix. Stating that set is the honest
    alternative to either failing to restrict or failing to run.
    """
    import site
    import sys

    roots: set[str] = set()
    for candidate in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix):
        if candidate:
            roots.add(candidate)
    real = os.path.realpath(executable)
    roots.add(str(Path(real).parent))
    for entry in sys.path:
        if entry and os.path.isdir(entry):
            roots.add(os.path.realpath(entry))
    try:
        for entry in site.getsitepackages():
            if os.path.isdir(entry):
                roots.add(os.path.realpath(entry))
    except AttributeError:  # pragma: no cover - a virtualenv without the helper
        pass
    return tuple(sorted(roots))
