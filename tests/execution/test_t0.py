"""T0: what holds on every machine, whatever the probe says.

Each limit here is proved by a run that tries to exceed it. A test that only asserted the rlimit
call was made would be the thing D-37 forbids.
"""

from __future__ import annotations

import os
import resource
import time
from pathlib import Path

import pytest

from reward_lens.contracts import Counters
from reward_lens.execution import Limits, default_sandbox, run_python
from reward_lens.execution.errors import ExecutionLimitBreached

from .conftest import secret_parent_env


def test_no_api_key_reaches_grader_code(work: Path, small: Limits) -> None:
    """The grader reads its own environment and finds none of the four keys the parent set."""
    src = (
        "import os\n"
        "names = ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'HF_TOKEN', 'AWS_SECRET_ACCESS_KEY')\n"
        "print(sorted(n for n in names if n in os.environ))\n"
        "print(sorted(k for k in os.environ if 'KEY' in k or 'TOKEN' in k or 'SECRET' in k))\n"
    )
    r = run_python(src, limits=small, cwd=work, env=secret_parent_env())
    assert r.exit_code == 0, r.stderr
    lines = r.stdout.decode().splitlines()
    assert lines[0] == "[]"
    assert lines[1] == "[]"


def test_environment_is_an_allowlist_not_the_parents(work: Path, small: Limits) -> None:
    """Nothing is inherited: a marker the parent sets is absent unless the caller passes it."""
    env = secret_parent_env()
    env["RL_MARKER_NOT_PASSED"] = "1"
    r = run_python(
        "import os; print('RL_MARKER_NOT_PASSED' in os.environ, os.environ.get('PYTHONHASHSEED'))",
        limits=small,
        cwd=work,
        env={},
    )
    assert r.exit_code == 0, r.stderr
    assert r.stdout.decode().strip() == "False 0"


def test_no_inherited_descriptors(work: Path, small: Limits) -> None:
    """Only the three standard streams are open in the child."""
    extra = open(work / "held-open.txt", "w")  # noqa: SIM115 - deliberately left open
    try:
        src = (
            "import os\n"
            "fds = sorted(int(f) for f in os.listdir('/proc/self/fd'))\n"
            "print([f for f in fds if f < 3])\n"
            "print(max(fds) <= 3)\n"
        )
        r = run_python(src, limits=small, cwd=work, env={})
    finally:
        extra.close()
    assert r.exit_code == 0, r.stderr
    assert r.stdout.decode().splitlines() == ["[0, 1, 2]", "True"]


def test_child_is_a_new_session(work: Path, small: Limits) -> None:
    """A session of its own, so the parent can kill the group without touching its own.

    The assertion is that the session and the process group are the same, rather than that both
    equal the pid: at L3 bubblewrap calls `setsid` and then forks inside a PID namespace, so the
    grader is not itself the session leader, and a pid compared across namespaces means nothing.
    """
    r = run_python(
        "import os; print(os.getsid(0) == os.getpgid(0))",
        limits=small,
        cwd=work,
        env={},
    )
    assert r.exit_code == 0, r.stderr
    assert r.stdout.decode().strip() == "True"


def test_working_directory_is_fresh_and_0700(work: Path, small: Limits) -> None:
    r = run_python(
        "import os, stat; print(oct(stat.S_IMODE(os.stat('.').st_mode)), sorted(os.listdir('.')))",
        limits=small,
        cwd=work,
        env={},
    )
    assert r.exit_code == 0, r.stderr
    mode, listing = r.stdout.decode().strip().split(" ", 1)
    assert mode == "0o700"
    assert listing == "['main.py']"


def test_rlimit_core_is_zero(work: Path, small: Limits) -> None:
    r = run_python(
        "import resource; print(resource.getrlimit(resource.RLIMIT_CORE))",
        limits=small,
        cwd=work,
        env={},
    )
    assert r.exit_code == 0, r.stderr
    assert r.stdout.decode().strip() == "(0, 0)"


def test_rlimit_nofile_is_lowered_and_bites(work: Path, small: Limits) -> None:
    """The descriptor ceiling is not the parent's, and opening past it fails."""
    parent_soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    src = (
        "import resource\n"
        "soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)\n"
        "print(soft)\n"
        "held = []\n"
        "try:\n"
        "    for i in range(soft + 64):\n"
        "        held.append(open('/dev/null'))\n"
        "except OSError as e:\n"
        "    print('EMFILE', e.errno)\n"
        "else:\n"
        "    print('NO LIMIT')\n"
    )
    r = run_python(src, limits=small, cwd=work, env={})
    assert r.exit_code == 0, r.stderr
    out = r.stdout.decode().splitlines()
    assert int(out[0]) < parent_soft
    assert out[1].startswith("EMFILE")


def test_wall_timeout_kills_and_records_counters(work: Path) -> None:
    """The parent's clock ends the run, and the counters survive the kill (section 7.2)."""
    limits = Limits(wall_s=1.0, cpu_s=30.0)
    r = run_python("import time; time.sleep(60)", limits=limits, cwd=work, env={})
    assert r.breach == "wall"
    assert r.exit_code != 0
    assert isinstance(r.counters, Counters)
    assert 0.5 <= r.counters.wall_s <= 8.0
    assert r.counters.processes >= 1
    assert r.counters.peak_rss_bytes > 0


def test_timeout_kills_the_whole_process_group(work: Path) -> None:
    """A grader that spawns children leaves none of them running.

    The evidence is a heartbeat rather than `kill(pid, 0)`, because at L3 the children live in a
    PID namespace and the numbers they report are not the host's. A file that stops growing is
    true in every namespace.
    """
    src = (
        "import subprocess, sys, time, pathlib\n"
        "beat = (\n"
        "    \"import time, pathlib\\n\"\n"
        "    \"p = pathlib.Path('beat.txt')\\n\"\n"
        "    \"while True:\\n\"\n"
        "    \"    p.open('a').write('x')\\n\"\n"
        "    \"    time.sleep(0.02)\\n\"\n"
        ")\n"
        "kids = [subprocess.Popen([sys.executable, '-c', beat]) for _ in range(3)]\n"
        "time.sleep(120)\n"
    )
    limits = Limits(wall_s=3.0, cpu_s=30.0, processes=64)
    beat = work / "beat.txt"
    r = run_python(src, limits=limits, cwd=work, env={})
    assert r.breach == "wall"
    found = sorted(work.glob("*/beat.txt"))
    assert len(found) == 1, "the children never started"
    settled = found[0].stat().st_size
    time.sleep(0.6)
    assert found[0].stat().st_size == settled
    assert not beat.exists()


def test_stdout_cap(work: Path, small: Limits) -> None:
    r = run_python(
        "import sys\n"
        "for _ in range(4096):\n"
        "    sys.stdout.write('x' * 1024)\n"
        "    sys.stdout.flush()\n",
        limits=small,
        cwd=work,
        env={},
    )
    assert r.breach == "stdout"
    assert len(r.stdout) <= small.stdout_bytes


def test_stderr_is_capped_too(work: Path, small: Limits) -> None:
    r = run_python(
        "import sys\n"
        "for _ in range(4096):\n"
        "    sys.stderr.write('e' * 1024)\n"
        "    sys.stderr.flush()\n",
        limits=small,
        cwd=work,
        env={},
    )
    assert r.breach == "stdout"
    assert len(r.stderr) <= small.stdout_bytes


def test_rlimit_fsize_stops_one_oversized_file(work: Path, small: Limits) -> None:
    """The kernel backstop for artifact bytes: SIGXFSZ, not a quiet 64 MB write."""
    r = run_python(
        "open('big.bin', 'wb').write(b'0' * (8 << 20))",
        limits=small,
        cwd=work,
        env={},
    )
    assert r.breach == "artifact"
    assert r.exit_code != 0


def test_artifact_budget_over_many_files(work: Path) -> None:
    limits = Limits(wall_s=20.0, cpu_s=20.0, artifact_bytes=1 << 18)
    src = (
        "for i in range(40):\n"
        "    open('f%02d.bin' % i, 'wb').write(b'0' * (32 << 10))\n"
        "print('wrote')\n"
    )
    r = run_python(src, limits=limits, cwd=work, env={})
    assert r.breach == "artifact"


def test_counters_are_filled_on_a_clean_run(work: Path, small: Limits) -> None:
    r = run_python("print('hello')", limits=small, cwd=work, env={})
    assert r.exit_code == 0
    c = r.counters
    assert c.wall_s > 0 and c.cpu_s >= 0
    assert c.peak_rss_bytes > 0
    assert c.processes >= 1
    assert c.output_bytes == len(r.stdout) + len(r.stderr)
    assert c.disk_bytes >= 0
    assert (c.calls, c.api_calls, c.input_tokens, c.output_tokens) == (1, 0, 0, 0)
    assert c.usd == "0.00"


def test_wall_and_cpu_are_counted_separately(work: Path) -> None:
    """A sleeping grader burns wall and almost no CPU (D-37's reason for two fields)."""
    limits = Limits(wall_s=10.0, cpu_s=10.0)
    r = run_python("import time; time.sleep(1.5)", limits=limits, cwd=work, env={})
    assert r.exit_code == 0, r.stderr
    assert r.counters.wall_s >= 1.4
    assert r.counters.cpu_s < 1.0


def test_result_raises_the_named_dimension(work: Path) -> None:
    limits = Limits(wall_s=1.0, cpu_s=30.0)
    r = run_python("import time; time.sleep(60)", limits=limits, cwd=work, env={})
    with pytest.raises(ExecutionLimitBreached) as exc:
        r.raise_for_breach()
    assert exc.value.code == "RL0410"
    assert exc.value.context["dimension"] == "wall"
    assert "wall" in exc.value.message


def test_sandbox_run_takes_argv_and_stdin(work: Path, small: Limits) -> None:
    sandbox = default_sandbox()
    r = sandbox.run(
        ["/bin/cat"],
        limits=small,
        cwd=work,
        env={},
        stdin=b"through the pipe\n",
    )
    assert r.exit_code == 0, r.stderr
    assert r.stdout == b"through the pipe\n"
    assert r.tier == sandbox.tier
