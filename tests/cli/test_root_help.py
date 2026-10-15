"""The current root help, the preserved base commands plus Join, and its printing cost."""

from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest
from conftest import REPO, SCRIPT, golden_body

CONTRACTED = json.loads((REPO / "fleet" / "CLI_COMMANDS.json").read_text(encoding="utf-8"))[
    "commands"
]
CURRENT_COMMANDS = CONTRACTED[:3] + ["join"] + CONTRACTED[3:] + ["version", "completion"]
# This is a static current-source checkpoint. Wave 1, Wave 2 and the target transcript remain
# frozen historical/end-state references; this candidate never selects an active demo wave.
ROOT_HELP = "candidates/current/root-help.txt"

#: Current engines are landed. The synthetic absence control below preserves A-026/A-027.
ABSENT = ()
PRESENT = ("audit", "trace", "compare", "join", "forecast", "improve", "init", "doctor", "explain")


def test_root_help_is_byte_identical_to_the_golden(cli):
    """The current checkpoint transcript, not an active or frozen wave transcript.

    The static current checkpoint includes the landed Join command. A separate control preserves
    every frozen target line and allows only that commissioned command addition.
    """
    run = cli()
    assert run.exit_code == 0
    assert run.stdout == golden_body(ROOT_HELP)


def test_help_flag_prints_the_same_root_help(cli):
    assert cli("--help").stdout == golden_body(ROOT_HELP)


def test_the_current_checkpoint_preserves_the_target_and_adds_the_landed_join():
    """Preserve the frozen base transcript while naming the commissioned Join surface."""
    from reward_lens.cli import registry

    target = golden_body("target/root-help.txt").split("\n")
    checkpoint = golden_body(ROOT_HELP).split("\n")

    additions = [line for line in checkpoint if line.startswith("  join ")]
    assert additions == [
        "  join        connect audit families to their saved trace and compare evidence"
    ]
    preserved = [line for line in checkpoint if not line.startswith("  join ")]
    assert len(preserved) == len(target)
    for want, have in zip(target, preserved):
        assert have.removesuffix(registry.ABSENT_MARK) == want, (want, have)
    assert not [line for line in checkpoint if line.endswith(registry.ABSENT_MARK)]


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


def test_an_absent_engine_refuses_before_it_parses_and_keeps_its_help(invoke, monkeypatch):
    """A-026 and A-027: one answer, RL0703, exit 5, whether or not an argument came with it.

    The refusal sits in front of the parser, so a verb that could not have run either way never
    argues about its arguments first. `--help` is the one thing it can still honestly do.
    """
    from reward_lens.api import _dispatch

    monkeypatch.setitem(_dispatch.ENGINES, "improve", "reward_lens.product.nothing_here:run")
    bare = invoke("improve")
    with_argument = invoke("improve", ".")

    assert bare.exit_code == with_argument.exit_code == 5, (bare.stderr, with_argument.stderr)
    assert bare.stderr == with_argument.stderr
    assert "RL0703" in bare.stderr
    assert "`improve` is not in this build" in bare.stderr
    assert "Traceback" not in bare.stderr
    assert bare.stdout == with_argument.stdout == ""

    helped = invoke("improve", "--help")
    assert helped.exit_code == 0, helped.stderr
    # The in-process fixture uses Click's inferred `cli` program name; the subprocess
    # command-help controls above separately require the public `reward-lens` name.
    assert helped.stdout.startswith("Usage: cli improve")


def test_root_help_goes_to_stdout_and_stderr_stays_empty(cli):
    run = cli("--help")
    assert run.stderr == ""


def test_the_current_registry_preserves_the_base_commands_and_adds_join():
    from reward_lens.cli import registry

    assert list(registry.COMMANDS) == CURRENT_COMMANDS
    assert registry.ORDER == tuple(CURRENT_COMMANDS)


@pytest.mark.parametrize("name", CURRENT_COMMANDS)
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


def test_version_leaf_matches_the_flag(cli):
    leaf = cli("version")
    assert leaf.exit_code == 0, leaf.stderr
    assert leaf.stdout == cli("--version").stdout


@pytest.mark.parametrize("shell", ("bash", "zsh", "fish"))
def test_completion_leaf_emits_click_source(cli, shell):
    from click.shell_completion import get_completion_class

    from reward_lens.cli.main import cli as root

    run = cli("completion", shell)
    expected = get_completion_class(shell)(root, {}, "reward-lens", "_REWARD_LENS_COMPLETE")
    assert run.exit_code == 0, run.stderr
    assert run.stdout == expected.source()
    assert "_REWARD_LENS_COMPLETE" in run.stdout


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


HEAVY = (
    "rich",
    "pydantic",
    "jsonschema_rs",
    "numpy",
    "pandas",
    "torch",
    "transformers",
    "httpx",
    "requests",
)


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
