"""The frozen root help, the fourteen commands, and the cost of printing it."""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from conftest import SCRIPT
from reward_lens.cli import registry

CONTRACTED = list(registry.ORDER)

#: A-026 and A-027: the verbs this build contracts for but cannot run, and four it can.
ABSENT = ("trace", "compare", "forecast", "improve")
PRESENT = ("audit", "init", "doctor", "explain")


def test_root_help_matches_the_public_registry_contract(cli):
    run = cli()
    assert run.exit_code == 0
    assert run.stdout == registry.root_help()


def test_help_flag_prints_the_public_registry_contract(cli):
    assert cli("--help").stdout == registry.root_help()


def test_the_root_help_marks_every_verb_whose_engine_is_not_in_this_build(cli):
    """A-027: the mark is what RL0703 sends a reader to `--help` to find."""
    from reward_lens.cli import availability, registry

    lines = cli("--help").stdout.splitlines()
    marked = {line.split()[0] for line in lines if line.endswith(registry.ABSENT_MARK)}

    assert marked == set(ABSENT), lines
    assert marked == set(availability.marked_absent())
    for name in PRESENT:
        printed = [line for line in lines if line.startswith(f"  {name} ")]
        assert printed, name
        assert not printed[0].endswith(registry.ABSENT_MARK), printed[0]


def test_an_absent_verb_refuses_before_it_parses_and_keeps_its_help(cli):
    """A-026 and A-027: one answer, RL0703, exit 5, whether or not an argument came with it.

    The refusal sits in front of the parser, so a verb that could not have run either way never
    argues about its arguments first. `--help` is the one thing it can still honestly do.
    """
    bare = cli("trace")
    with_argument = cli("trace", "run-0123456789ab")

    assert bare.exit_code == with_argument.exit_code == 5, (bare.stderr, with_argument.stderr)
    assert bare.stderr == with_argument.stderr
    assert "RL0703" in bare.stderr
    assert "`trace` is not in this build" in bare.stderr
    assert "Traceback" not in bare.stderr
    assert bare.stdout == with_argument.stdout == ""

    helped = cli("trace", "--help")
    assert helped.exit_code == 0, helped.stderr
    assert helped.stdout.startswith("Usage: reward-lens trace")


def test_root_help_goes_to_stdout_and_stderr_stays_empty(cli):
    run = cli("--help")
    assert run.stderr == ""


def test_the_contracted_fourteen_are_the_commands_the_registry_carries():
    from reward_lens.cli import registry

    assert list(registry.COMMANDS) == CONTRACTED
    assert registry.ORDER == tuple(CONTRACTED)


@pytest.mark.parametrize("name", CONTRACTED)
def test_every_command_responds_to_help(cli, name):
    run = cli(name, "--help")
    assert run.exit_code == 0, run.stderr
    assert run.stdout.startswith("Usage: reward-lens " + name)


def test_version_flag_exists_and_names_the_version(cli):
    """The rule is that `--version` prints the installed distribution's version.

    A literal here would assert the state of the tree on the day it was written, and would go
    red on the first version bump for no reason a reader could act on.
    """
    from importlib.metadata import version

    run = cli("--version")
    assert run.exit_code == 0
    assert run.stdout.strip() == f"reward-lens {version('reward-lens')}"


def test_help_imports_no_verb_module(child_env):
    """D-30: a verb module is not imported before its verb runs."""
    probe = (
        "import sys;"
        "sys.argv=['reward-lens','--help'];"
        "import reward_lens.cli.main as m;"
        "\ntry:\n m.main()\nexcept SystemExit:\n pass\n"
        "print('VERBS=' + ','.join(sorted(k for k in sys.modules if k.startswith('reward_lens.cli.verbs.'))))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=child_env(), check=True
    ).stdout
    assert "VERBS=\n" in out + "\n", out.splitlines()[-1]


HEAVY = ("rich", "pydantic", "jsonschema_rs", "numpy", "pandas", "torch", "transformers", "httpx", "requests")


def test_import_set_after_help_holds_none_of_the_heavy_names(child_env):
    probe = (
        "import sys;"
        "sys.argv=['reward-lens','--help'];"
        "import reward_lens.cli.main as m;"
        "\ntry:\n m.main()\nexcept SystemExit:\n pass\n"
        "print('HEAVY=' + ','.join(sorted(n for n in %r if n in sys.modules)))" % (HEAVY,)
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=child_env(), check=True
    ).stdout
    assert "HEAVY=" in out
    assert out.rsplit("HEAVY=", 1)[1].strip() == ""


def test_help_completes_under_100_ms_as_the_minimum_of_twenty_runs(child_env):
    env = child_env()
    best = min(_one_help(env) for _ in range(20))
    assert best < 0.100, f"fastest of twenty --help runs was {best:.3f}s"


def _one_help(env) -> float:
    start = time.perf_counter()
    subprocess.run([str(SCRIPT), "--help"], stdout=subprocess.DEVNULL, env=env, check=True)
    return time.perf_counter() - start
