"""L0: the rlimits and `PR_SET_NO_NEW_PRIVS` that need no probe, each proved by exceeding it."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from reward_lens.execution import Limits, probe, run_python
from reward_lens.execution.limits import AS_MARGIN_MIN, address_space_ceiling

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the Linux ladder")


def _at_least_l0() -> bool:
    return probe().tiers["L0"].held


def test_l0_holds_on_this_machine() -> None:
    assert _at_least_l0(), probe().tiers["L0"].reason


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
    """`RLIMIT_NPROC` is per uid, so the ceiling is the user's baseline plus the budget."""
    limits = Limits(wall_s=25.0, cpu_s=25.0, processes=4, memory_bytes=512 * 2**20)
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
    r = run_python(src, limits=limits, cwd=work, env={})
    out = r.stdout.decode().splitlines()
    assert out[0].startswith("soft ")
    assert int(out[0].split()[1]) > 0
    assert r.breach in {"processes", "wall"}
    if r.breach == "processes":
        assert r.counters.processes > limits.processes


def test_process_budget_names_the_processes_dimension(work: Path) -> None:
    limits = Limits(wall_s=20.0, cpu_s=20.0, processes=3, memory_bytes=512 * 2**20)
    src = (
        "import subprocess, sys, time\n"
        "kids = [subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "        for _ in range(10)]\n"
        "time.sleep(60)\n"
    )
    r = run_python(src, limits=limits, cwd=work, env={})
    assert r.breach == "processes"


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
