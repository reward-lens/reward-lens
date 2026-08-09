"""L2: the seccomp-BPF denylist, and the child that proves the filter installs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from reward_lens.execution import Limits, probe
from reward_lens.execution.linux import LinuxSandbox
from reward_lens.execution.seccomp import DENIED_SYSCALLS, build_filter, probe_seccomp

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="seccomp is Linux only")

requires_l2 = pytest.mark.skipif(
    not probe().tiers["L2"].held, reason="seccomp filter would not install"
)


def test_the_denylist_is_the_seven_families_d37_names() -> None:
    assert set(DENIED_SYSCALLS) == {
        "ptrace",
        "mount",
        "unshare",
        "bpf",
        "perf_event_open",
        "kexec_load",
        "kexec_file_load",
        "process_vm_readv",
        "process_vm_writev",
    }


def test_the_filter_is_a_well_formed_bpf_program() -> None:
    prog = build_filter()
    assert len(prog) >= len(DENIED_SYSCALLS) + 4
    assert all(len(bytes(ins)) == 8 for ins in prog)


def test_the_probe_is_a_child_that_installs_the_filter_and_exits_zero() -> None:
    held, reason = probe_seccomp()
    assert held is True, reason
    assert probe().tiers["L2"].held is held


@requires_l2
def test_a_denied_syscall_is_refused_inside_the_sandbox(work: Path) -> None:
    sandbox = LinuxSandbox(tier="L2")
    src = (
        "import ctypes, os\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "ctypes.set_errno(0)\n"
        "rc = libc.syscall(ctypes.c_long(101), ctypes.c_long(0), ctypes.c_long(0),\n"
        "                  ctypes.c_void_p(0), ctypes.c_void_p(0))\n"
        "print('ptrace', rc, ctypes.get_errno())\n"
        "ctypes.set_errno(0)\n"
        "rc = libc.syscall(ctypes.c_long(272), ctypes.c_long(0))\n"
        "print('unshare', rc, ctypes.get_errno())\n"
    )
    r = sandbox.run_source(src, limits=Limits(wall_s=15.0, cpu_s=15.0), cwd=work, env={})
    assert r.exit_code == 0, r.stderr
    lines = r.stdout.decode().splitlines()
    for line in lines:
        name, rc, errno = line.split()
        assert rc == "-1", line
        assert errno == "1", line  # EPERM, from the filter, not from the kernel's own check


@requires_l2
def test_the_child_reports_seccomp_mode_filter(work: Path) -> None:
    sandbox = LinuxSandbox(tier="L2")
    src = (
        "line = [l for l in open('/proc/self/status') if l.startswith('Seccomp:')]\n"
        "print(line[0].split()[1])\n"
    )
    r = sandbox.run_source(src, limits=Limits(wall_s=15.0, cpu_s=15.0), cwd=work, env={})
    assert r.exit_code == 0, r.stderr
    assert r.stdout.decode().strip() == "2"
