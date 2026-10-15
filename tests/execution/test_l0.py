"""L0: the rlimits and `PR_SET_NO_NEW_PRIVS` that need no probe, each proved by exceeding it."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import pytest

import reward_lens.execution.linux as linux
from reward_lens.execution import Limits, probe, run_python
from reward_lens.execution.errors import SandboxTierUnavailable
from reward_lens.execution.limits import AS_MARGIN_MIN, address_space_ceiling
from reward_lens.execution.linux import LinuxSandbox, user_process_baseline

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the Linux ladder")


def _at_least_l0() -> bool:
    return probe().tiers["L0"].held


def test_l0_holds_on_this_machine() -> None:
    assert _at_least_l0(), probe().tiers["L0"].reason


def test_nproc_baseline_counts_real_background_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real `/proc/self/status` thread count, not only its process leader, is charged."""
    original_listdir = os.listdir
    monkeypatch.setattr(
        os,
        "listdir",
        lambda path: [str(os.getpid())] if path == "/proc" else original_listdir(path),
    )
    before = user_process_baseline()
    ready = threading.Barrier(5)
    release = threading.Event()

    def wait() -> None:
        ready.wait(timeout=5.0)
        release.wait(timeout=5.0)

    workers = [threading.Thread(target=wait) for _ in range(4)]
    try:
        for worker in workers:
            worker.start()
        ready.wait(timeout=5.0)
        assert user_process_baseline() == before + len(workers)
    finally:
        release.set()
        for worker in workers:
            if worker.ident is not None:
                worker.join(timeout=5.0)


def test_nproc_baseline_refuses_unreadable_proc(monkeypatch: pytest.MonkeyPatch) -> None:
    def unreadable(path: str) -> list[str]:
        assert path == "/proc"
        raise PermissionError("procfs access denied")

    monkeypatch.setattr(os, "listdir", unreadable)
    with pytest.raises(PermissionError, match="procfs access denied"):
        user_process_baseline()


def test_unreadable_proc_refuses_l0_but_does_not_block_t0(
    monkeypatch: pytest.MonkeyPatch, work: Path
) -> None:
    def unavailable() -> int:
        raise PermissionError("procfs access denied")

    monkeypatch.setattr(linux, "user_process_baseline", unavailable)
    limits = Limits(wall_s=5.0, cpu_s=5.0, network=True)
    with pytest.raises(SandboxTierUnavailable) as raised:
        LinuxSandbox(tier="L0").run_source("print('ok')", limits=limits, cwd=work, env={})
    assert raised.value.code == "RL0401"
    assert raised.value.context["tier"] == "L0"

    result = LinuxSandbox(tier="T0").run_source("print('ok')", limits=limits, cwd=work, env={})
    assert result.exit_code == 0, result.stderr
    assert result.tier == "T0"
    assert "rlimit_nproc" not in result.detail


def test_rlimit_as_is_the_backstop_and_sits_above_the_budget(work: Path) -> None:
    """The ceiling is above the budget, by the margin, and a 3 GB request is still refused.

    Setting it *at* the budget is what made the memory breach a race: the allocator ended the run
    before a resident set over budget could exist, so the supervisor's own rule was unreachable.
    """
    budget = 384 * 2**20
    assert address_space_ceiling(budget) == budget + max(AS_MARGIN_MIN, budget)
    assert address_space_ceiling(budget) > budget
    limits = Limits(wall_s=20.0, cpu_s=20.0, memory_bytes=budget)
    src = (
        "import resource\n"
        "print(resource.getrlimit(resource.RLIMIT_AS)[0])\n"
        "try:\n"
        "    b = bytearray(3 * 1024 * 1024 * 1024)\n"
        "except MemoryError:\n"
        "    print('MemoryError')\n"
        "else:\n"
        "    print('allocated', len(b))\n"
    )
    r = run_python(src, limits=limits, cwd=work, env={})
    out = r.stdout.decode().splitlines()
    assert int(out[0]) == address_space_ceiling(budget)
    assert out[1] == "MemoryError"
    assert r.detail["rlimit_as"] == address_space_ceiling(budget)


def test_memory_budget_kills_a_grader_that_grows_past_it(work: Path) -> None:
    """The outcome, not the sampler: a grader that grows past the budget is named `memory`.

    The grader touches every page it allocates, so its resident set passes the budget for certain
    and the kernel's own high-water mark for the child says so after the wait. Whether the 30 ms
    sampler caught it first or the backstop refused a mapping, the dimension is the same.
    """
    limits = Limits(wall_s=25.0, cpu_s=25.0, memory_bytes=256 * 2**20)
    src = (
        "held = []\n"
        "while True:\n"
        "    held.append(bytearray(8 * 1024 * 1024))\n"
        "    for i in range(0, len(held[-1]), 4096):\n"
        "        held[-1][i] = 1\n"
    )
    r = run_python(src, limits=limits, cwd=work, env={})
    assert r.breach == "memory"
    assert r.counters.peak_rss_bytes > limits.memory_bytes


def test_a_grader_that_reserves_without_touching_still_names_memory(work: Path) -> None:
    """The backstop's own outcome: `RLIMIT_AS` refuses the mapping, and the refusal is a breach.

    Address space a grader never touches never becomes a resident set, so the supervisor's rule
    cannot see it. What it can see is how the run ended.
    """
    limits = Limits(wall_s=20.0, cpu_s=20.0, memory_bytes=256 * 2**20)
    src = "b = bytearray(8 * 1024 * 1024 * 1024)\nprint(len(b))\n"
    r = run_python(src, limits=limits, cwd=work, env={})
    assert r.exit_code != 0
    assert b"MemoryError" in r.stderr
    assert r.breach == "memory"


def test_rlimit_cpu_is_set_and_a_spin_is_stopped(work: Path) -> None:
    limits = Limits(wall_s=30.0, cpu_s=2.0, memory_bytes=512 * 2**20)
    src = "import resource\nwhile True:\n    pass\n"
    r = run_python(src, limits=limits, cwd=work, env={})
    assert r.breach == "cpu"
    assert r.counters.cpu_s >= 1.0


def test_rlimit_nproc_is_relative_to_the_users_load_and_forks_hit_eagain(work: Path) -> None:
    """At explicit L0, NPROC counts real-UID tasks and leaves a supervisor observation slot."""
    limits = Limits(wall_s=25.0, cpu_s=25.0, processes=4, memory_bytes=512 * 2**20, network=True)
    src = (
        "import os, resource, sys\n"
        "soft, _ = resource.getrlimit(resource.RLIMIT_NPROC)\n"
        "print('soft', soft, flush=True)\n"
        "made = 0\n"
        "try:\n"
        "    for _ in range(2000):\n"
        "        pid = os.fork()\n"
        "        if pid == 0:\n"
        "            import time; time.sleep(60); os._exit(0)\n"
        "        made += 1\n"
        "except BlockingIOError:\n"
        "    print('EAGAIN after', made, flush=True)\n"
        "else:\n"
        "    print('NO LIMIT', made, flush=True)\n"
        "import time; time.sleep(60)\n"
    )
    r = LinuxSandbox(tier="L0").run_source(src, limits=limits, cwd=work, env={})
    out = r.stdout.decode().splitlines()
    assert r.tier == "L0"
    assert out[0].startswith("soft ")
    assert int(out[0].split()[1]) > 0
    assert r.detail["rlimit_nproc"] == int(out[0].split()[1])
    assert r.breach == "processes"
    assert r.counters.processes > limits.processes


def test_process_budget_names_the_processes_dimension(work: Path) -> None:
    limits = Limits(wall_s=20.0, cpu_s=20.0, processes=3, memory_bytes=512 * 2**20, network=True)
    src = (
        "import subprocess, sys, time\n"
        "kids = [subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "        for _ in range(10)]\n"
        "time.sleep(60)\n"
    )
    r = LinuxSandbox(tier="L0").run_source(src, limits=limits, cwd=work, env={})
    assert r.tier == "L0"
    assert r.breach == "processes"
    assert r.counters.processes > limits.processes


def test_no_new_privs_is_set(work: Path) -> None:
    """`NoNewPrivs: 1` in the child's status, which is what Landlock needs to attach."""
    limits = Limits(wall_s=10.0, cpu_s=10.0)
    src = (
        "line = [l for l in open('/proc/self/status') if l.startswith('NoNewPrivs')]\n"
        "print(line[0].split()[1])\n"
    )
    r = run_python(src, limits=limits, cwd=work, env={})
    assert r.exit_code == 0, r.stderr
    assert r.stdout.decode().strip() == "1"
