"""L1: the Landlock ruleset, with the ABI read from the kernel rather than assumed."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

from reward_lens.execution import Limits, probe
from reward_lens.execution.landlock import (
    ACCESS_FS_BY_ABI,
    ACCESS_NET_CONNECT_TCP,
    abi_version,
    handled_fs_for_abi,
    handled_net_for_abi,
    scoped_for_abi,
)
from reward_lens.execution.linux import LinuxSandbox

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Landlock is Linux only")


def _l1() -> bool:
    return probe().tiers["L1"].held


requires_l1 = pytest.mark.skipif(not _l1(), reason="Landlock unavailable on this kernel")


def test_abi_is_probed_not_assumed() -> None:
    """`landlock_create_ruleset(NULL, 0, LANDLOCK_CREATE_RULESET_VERSION)` is the whole probe."""
    abi = abi_version()
    p = probe()
    assert abi == p.landlock_abi
    assert abi is None or abi >= 1


def test_handled_bits_are_downgraded_to_the_abi() -> None:
    """A bit the running kernel does not know is never handed to it: EINVAL is the failure mode."""
    assert handled_fs_for_abi(1) == ACCESS_FS_BY_ABI[1]
    assert handled_fs_for_abi(3) == handled_fs_for_abi(4)  # ABI 4 adds network, not filesystem
    assert handled_fs_for_abi(99) == handled_fs_for_abi(5)
    assert handled_net_for_abi(3) == 0
    assert handled_net_for_abi(4) & ACCESS_NET_CONNECT_TCP
    assert scoped_for_abi(5) == 0
    assert scoped_for_abi(6) != 0


@requires_l1
def test_write_outside_the_working_directory_is_denied(work: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("before\n")
    sandbox = LinuxSandbox(tier="L1")
    src = (
        "import pathlib\n"
        f"target = pathlib.Path({str(outside)!r})\n"
        "try:\n"
        "    target.write_text('after\\n')\n"
        "except PermissionError as e:\n"
        "    print('DENIED', e.errno)\n"
        "else:\n"
        "    print('WROTE')\n"
        "pathlib.Path('inside.txt').write_text('ok\\n')\n"
        "print('INSIDE OK')\n"
    )
    r = sandbox.run_source(src, limits=Limits(wall_s=15.0, cpu_s=15.0), cwd=work, env={})
    assert r.exit_code == 0, r.stderr
    out = r.stdout.decode().splitlines()
    assert out[0].startswith("DENIED")
    assert out[1] == "INSIDE OK"
    assert outside.read_text() == "before\n"


@requires_l1
def test_a_new_file_outside_the_working_directory_cannot_be_created(
    work: Path, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    sandbox = LinuxSandbox(tier="L1")
    src = (
        "import pathlib\n"
        f"p = pathlib.Path({str(elsewhere / 'new.txt')!r})\n"
        "try:\n"
        "    p.write_text('x')\n"
        "except OSError as e:\n"
        "    print('DENIED', type(e).__name__)\n"
        "else:\n"
        "    print('CREATED')\n"
    )
    r = sandbox.run_source(src, limits=Limits(wall_s=15.0, cpu_s=15.0), cwd=work, env={})
    assert r.exit_code == 0, r.stderr
    assert r.stdout.decode().strip().startswith("DENIED")
    assert sorted(p.name for p in elsewhere.iterdir()) == []


@requires_l1
@pytest.mark.skipif(
    (abi_version() or 0) < 4, reason="TCP rules need Landlock ABI 4 (kernel 6.7)"
)
def test_tcp_connect_is_denied_where_the_abi_allows(work: Path) -> None:
    """The listener belongs to this process: the point is that the connect never reaches it."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        sandbox = LinuxSandbox(tier="L1")
        src = (
            "import socket\n"
            "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            "s.settimeout(3)\n"
            "try:\n"
            f"    s.connect(('127.0.0.1', {port}))\n"
            "except OSError as e:\n"
            "    print('DENIED', e.errno)\n"
            "else:\n"
            "    print('CONNECTED')\n"
        )
        r = sandbox.run_source(src, limits=Limits(wall_s=15.0, cpu_s=15.0), cwd=work, env={})
    finally:
        listener.close()
    assert r.exit_code == 0, r.stderr
    assert r.stdout.decode().strip().startswith("DENIED")


@requires_l1
def test_the_ruleset_records_its_read_set(work: Path) -> None:
    """The read set is stated, not implied: an interpreter needs its own prefix to exec."""
    sandbox = LinuxSandbox(tier="L1")
    roots = sandbox.read_roots_for(sys.executable)
    assert "/usr" in roots
    assert any(r.startswith(sys.prefix) for r in roots) or sys.prefix in roots
    r = sandbox.run_source("print(1 + 1)", limits=Limits(wall_s=15.0), cwd=work, env={})
    assert r.exit_code == 0, r.stderr
    assert r.stdout.decode().strip() == "2"
