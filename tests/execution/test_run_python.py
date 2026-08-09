"""`run_python`: the convenience every adapter calls, and the determinism D-38 asks of it."""

from __future__ import annotations

from pathlib import Path

import pytest

from reward_lens.execution import Limits, run_python
from reward_lens.execution.limits import DEFAULT_LIMITS, scrub_env


def test_defaults_are_the_d37_budget() -> None:
    d = DEFAULT_LIMITS
    assert (d.wall_s, d.cpu_s) == (30.0, 30.0)
    assert d.memory_bytes == 2 * 2**30
    assert d.processes == 64
    assert d.network is False
    assert d.stdout_bytes == 1 << 20
    assert d.artifact_bytes == 64 << 20
    assert Limits() == d


def test_pythonhashseed_is_zero_in_a_fresh_interpreter(work: Path) -> None:
    """Set by re-exec, not in process, so `hash(str)` is stable across runs (D-38)."""
    src = "import os, sys; print(os.environ['PYTHONHASHSEED'], sys.flags.hash_randomization)"
    first = run_python(src, limits=Limits(wall_s=15.0), cwd=work, env={})
    second = run_python(src, limits=Limits(wall_s=15.0), cwd=work, env={})
    assert first.exit_code == 0, first.stderr
    assert first.stdout.decode().strip() == "0 0"
    assert first.stdout == second.stdout

    h = run_python("print(hash('reward-lens'))", limits=Limits(wall_s=15.0), cwd=work, env={})
    h2 = run_python("print(hash('reward-lens'))", limits=Limits(wall_s=15.0), cwd=work, env={})
    assert h.stdout == h2.stdout


def test_args_reach_the_source(work: Path) -> None:
    r = run_python(
        "import sys; print(sys.argv[1:])",
        args=["alpha", "beta"],
        limits=Limits(wall_s=15.0),
        cwd=work,
        env={},
    )
    assert r.exit_code == 0, r.stderr
    assert r.stdout.decode().strip() == "['alpha', 'beta']"


def test_each_call_gets_its_own_scratch_directory(work: Path) -> None:
    a = run_python("import os; print(os.getcwd())", limits=Limits(wall_s=15.0), cwd=work, env={})
    b = run_python("import os; print(os.getcwd())", limits=Limits(wall_s=15.0), cwd=work, env={})
    assert a.stdout != b.stdout
    for r in (a, b):
        assert r.stdout.decode().strip().startswith(str(work))


def test_a_raising_grader_comes_back_as_a_nonzero_exit_not_an_exception(work: Path) -> None:
    r = run_python("raise ValueError('boom')", limits=Limits(wall_s=15.0), cwd=work, env={})
    assert r.exit_code != 0
    assert b"ValueError: boom" in r.stderr
    assert r.breach is None


@pytest.mark.parametrize(
    "name",
    ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN", "AWS_SECRET_ACCESS_KEY", "MY_SECRET_TOKEN"],
)
def test_scrub_env_drops_anything_that_looks_like_a_credential(name: str) -> None:
    out = scrub_env({name: "value", "SAFE_SETTING": "1"}, cwd=Path("/tmp/x"))
    assert name not in out
    assert out["SAFE_SETTING"] == "1"


def test_scrub_env_builds_the_base_from_an_allowlist() -> None:
    out = scrub_env({}, cwd=Path("/tmp/x"))
    assert out["PYTHONHASHSEED"] == "0"
    assert out["HOME"] == "/tmp/x"
    assert out["TMPDIR"] == "/tmp/x"
    assert out["LC_ALL"] == "C.UTF-8"
    assert out["TZ"] == "UTC"
    assert out["PATH"] == "/usr/local/bin:/usr/bin:/bin"
    assert "LD_PRELOAD" not in out
