"""Correction C-004: the inherited import layer, made true to the base closure.

Every assertion here runs in the clean venv, in a fresh `-I` interpreter, with the working
directory outside the checkout. The claim is not "these modules import" (they import fine in a
developer environment that has numpy sitting there from something else); it is "these modules
import when numpy, scipy, pandas, sklearn, pydantic-settings, rich, typer and torch are not
installed at all, and none of them is in `sys.modules` when the import returns".
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from .conftest import import_probe

#: The six modules the wave-1 audit is built on. Interfaces section 7.
BASE_IMPORTS = [
    "reward_lens",
    "reward_lens.core.reading",
    "reward_lens.access.report",
    "reward_lens.verifier.coverage",
    "reward_lens.verifier.replay",
    "reward_lens.verifier.attack",
]

TINY_GRADER = textwrap.dedent(
    '''
    """A grader small enough to read, with two branches and one normalisation step."""


    def grade(answer: str, target: str) -> float:
        if answer == target:
            return 1.0
        if answer.strip().lower() == target.strip().lower():
            return 0.5
        return 0.0
    '''
).strip()


@pytest.mark.parametrize("module", BASE_IMPORTS)
def test_module_imports_on_the_base_closure(clean_python: Path, module: str) -> None:
    result = import_probe(clean_python, module)
    assert result["returncode"] == 0, (
        f"`import {module}` failed in a venv holding only the wheel:\n{result['stderr']}"
    )


@pytest.mark.parametrize("module", BASE_IMPORTS)
def test_module_pulls_nothing_numeric(clean_python: Path, module: str) -> None:
    result = import_probe(clean_python, module)
    if result["returncode"] != 0:
        pytest.fail(f"`import {module}` failed before the sys.modules check:\n{result['stderr']}")
    assert result["present"] == [], (
        f"`import {module}` left {result['present']} in sys.modules. D-30 and D-58: nothing "
        "numeric, nothing that renders, on the base path."
    )


# The three instrument runs of interfaces section 7. They were `xfail(strict=True)` through attempt
# 1, each blocked by a run-time dependency: `coverage` was in the `[verifier]` extra (A-002 moved
# it to base), `attack.py` gated libcst on an extra A-001 had already dissolved, and the evidence
# codec touched the numpy proxy to answer type checks that could only ever answer False. All three
# are fixed and the marks are gone.
#
# The reference numbers below were measured once in a venv holding the wheel *with the verifier
# extra*, which is the environment that has every optional dependency available, and are asserted
# against the base-wheel run so a degraded answer on the base closure cannot pass as a Reading.
# The command, run from the worktree after `uv build --no-sources --wheel -o dist`:
#
#     uv venv <tmp>/venv-verifier --python 3.11
#     uv pip install --python <tmp>/venv-verifier/bin/python 'dist/reward_lens-3.1.0-py3-none-any.whl[verifier]'
#     <tmp>/venv-verifier/bin/python -I -B <script building the subject and corpus below>
#
# It printed, and the same script against `.venv-clean/bin/python` printed identically:
#
#     COVERAGE type CoverageReading refusal False
#     COVERAGE stmts 6 / 6 arcs 4 / 4 rung 1 n 3
#     REPLAY type Evidence refusal False
#     REPLAY quantity env.replay_fidelity n_tasks 3 n_reproduced 3 n_mismatched 0 fidelity 1.0
#     ATTACK type Evidence refusal False
#     ATTACK observable AttackSurfaceInventory rung 1 accesses 0 taints 2 validated [False, False]
#             sources ['parameter answer', 'parameter target']
#
# `Reading` is `Any` at runtime (core/reading.py:246; it is a real union only under a type
# checker), so "is a Reading" cannot be an isinstance check. What the assertions below say instead
# is the enforceable half: the value is the instrument's own declared success type, and
# `is_refusal` says False. `measure_coverage` is annotated `CoverageReading | Refusal` rather than
# `Reading`, so its success type is `CoverageReading`, not `Evidence`.


def _instrument_probe(clean_python: Path, tmp_path: Path, body: str) -> dict[str, object]:
    grader = tmp_path / "tiny_grader.py"
    grader.write_text(TINY_GRADER, encoding="utf-8")
    # Built by concatenation rather than a dedented template: `body` arrives already flush left, so
    # a `textwrap.dedent` over the whole thing finds no common prefix and leaves the template's own
    # indentation in place, which the interpreter then rejects.
    preamble = [
        "from pathlib import Path",
        "from reward_lens.verifier import ListCorpus, Rollout, VerifierUnderTest",
        f"subject = VerifierUnderTest(source_path=Path({str(grader)!r}), entrypoint='grade')",
        "corpus = ListCorpus(rollouts=(",
        "    Rollout('r1', {'answer': 'cat', 'target': 'cat'}, 1.0),",
        "    Rollout('r2', {'answer': ' Cat ', 'target': 'cat'}, 0.5),",
        "    Rollout('r3', {'answer': 'dog', 'target': 'cat'}, 0.0),",
        "))",
    ]
    return import_probe(clean_python, "sys", extra_code="\n".join([*preamble, body]))


def test_measure_coverage_runs_to_a_reading(clean_python: Path, tmp_path: Path) -> None:
    result = _instrument_probe(
        clean_python,
        tmp_path,
        "from reward_lens.core.reading import is_refusal\n"
        "from reward_lens.verifier.coverage import CoverageReading, measure_coverage\n"
        "reading = measure_coverage(subject, corpus)\n"
        "assert isinstance(reading, CoverageReading), type(reading).__name__\n"
        "assert not is_refusal(reading), reading\n"
        "assert (reading.statements_covered, reading.statements_total) == (6, 6), reading\n"
        "assert (reading.branch_arcs_covered, reading.branch_arcs_total) == (4, 4), reading\n"
        "assert reading.uncovered_branch_arcs == (), reading.uncovered_branch_arcs\n"
        "assert (reading.rung, reading.n_rollouts) == (1, 3), reading\n"
        "assert reading.errors == {}, reading.errors",
    )
    assert result["returncode"] == 0, result["stderr"]
    assert result["present"] == [], result["present"]


def test_replay_fidelity_runs_to_a_reading(clean_python: Path, tmp_path: Path) -> None:
    result = _instrument_probe(
        clean_python,
        tmp_path,
        "from reward_lens.core.evidence import Evidence\n"
        "from reward_lens.core.reading import is_refusal\n"
        "from reward_lens.verifier.replay import replay_fidelity\n"
        "reading = replay_fidelity(subject, corpus)\n"
        "assert isinstance(reading, Evidence), type(reading).__name__\n"
        "assert not is_refusal(reading), reading\n"
        "assert reading.quantity == 'env.replay_fidelity', reading.quantity\n"
        "report = reading.value\n"
        "assert (report.n_tasks, report.n_reproduced) == (3, 3), report\n"
        "assert (report.n_mismatched, report.n_unreplayable) == (0, 0), report\n"
        "assert report.replay_fidelity == 1.0, report.replay_fidelity",
    )
    assert result["returncode"] == 0, result["stderr"]
    assert result["present"] == [], result["present"]


def test_attack_surface_runs_to_a_reading(clean_python: Path, tmp_path: Path) -> None:
    result = _instrument_probe(
        clean_python,
        tmp_path,
        "from reward_lens.core.evidence import Evidence\n"
        "from reward_lens.core.reading import is_refusal\n"
        "from reward_lens.verifier.attack import attack_surface\n"
        "reading = attack_surface(subject)\n"
        "assert isinstance(reading, Evidence), type(reading).__name__\n"
        "assert not is_refusal(reading), reading\n"
        "assert reading.observable == 'AttackSurfaceInventory', reading.observable\n"
        "surface = reading.value\n"
        "assert (surface.rung, len(surface.accesses)) == (1, 0), surface\n"
        # Both parameters of `grade` reach the returned score and neither passes a validator, so
        # the taint analysis libcst does finds exactly two unvalidated paths. This is the assertion
        # that would go quiet if the libcst path silently stopped running.
        "assert len(surface.taints) == 2, surface.taints\n"
        "assert sorted(t.source for t in surface.taints) == "
        "['parameter answer', 'parameter target'], surface.taints\n"
        "assert [t.validated for t in surface.taints] == [False, False], surface.taints",
    )
    assert result["returncode"] == 0, result["stderr"]
    assert result["present"] == [], result["present"]


# --------------------------------------------------------------------------------------------
# Refusal cases
# --------------------------------------------------------------------------------------------


def test_a_source_path_that_does_not_exist_raises_before_any_instrument(
    clean_python: Path, tmp_path: Path
) -> None:
    """The documented error, and it is the subject's, not the instrument's.

    The brief asked whether each instrument on a missing source path returns a refusal Reading or
    raises. Measured: none of the three is reached. `VerifierUnderTest.__post_init__` raises
    `FileNotFoundError` at construction, with a message that names the path and says what the
    series needs, so all three instruments share one contract for this input and the refusal
    vocabulary of `core/reading.py` never comes into it. Asserting on the message as well as the
    type, because a bare `FileNotFoundError` from a later `open()` would satisfy the type alone.
    """
    result = _instrument_probe(
        clean_python,
        tmp_path,
        "from pathlib import Path\n"
        "missing = Path(str(subject.source_path) + '.no_such_file.py')\n"
        "assert not missing.exists()\n"
        "try:\n"
        "    VerifierUnderTest(source_path=missing, entrypoint='grade')\n"
        "except FileNotFoundError as exc:\n"
        "    assert 'no verifier source at' in str(exc), str(exc)\n"
        "    assert str(missing) in str(exc), str(exc)\n"
        "else:\n"
        "    raise AssertionError('a missing source path constructed a subject')",
    )
    assert result["returncode"] == 0, result["stderr"]


def test_sensitivity_imports_on_the_base_closure_and_names_the_verifier_extra(
    clean_python: Path,
) -> None:
    """`require_extra` is per module, so importing D9 costs nothing until it reaches numpy.

    Two claims, both on the base wheel. The first is that importing
    `reward_lens.verifier.sensitivity` succeeds and pulls nothing numeric with it: the numpy proxy
    is built at import time and resolves nothing until an attribute is read off it. The second is
    section 7's rule about the seam, that the proxy carries `extra="verifier"`, so the first numpy
    use raises `ExtraRequiredError` naming the extra and the command that installs it rather than a
    bare `ModuleNotFoundError` from the middle of a calculation.

    The second claim could not hold until attempt 4, and the reason is worth keeping here.
    `require_extra` decides whether an extra is installed by looking for one probe module, and
    `EXTRA_PROBE["verifier"]` was `coverage`, which A-002 had moved into the base closure so
    `measure_coverage` reaches a Reading with no extras. The probe was satisfied on every base
    install, `require_extra` returned, and the numpy import behind it was what the caller saw. The
    row now names `SALib`. The `ModuleNotFoundError` arm below is what that regression looks like
    if it ever comes back, and it fails with the reason rather than with a bare type mismatch.
    """
    result = import_probe(
        clean_python,
        "reward_lens.verifier.sensitivity",
        extra_code=(
            "import reward_lens.verifier.sensitivity as s\n"
            "from reward_lens.core.extras import ExtraRequiredError\n"
            "assert 'numpy' not in sys.modules, 'importing D9 imported numpy'\n"
            "try:\n"
            "    s.np.array([1.0])\n"
            "except ExtraRequiredError as exc:\n"
            "    assert 'verifier' in str(exc), str(exc)\n"
            '    assert "pip install \'reward-lens[verifier]\'" in str(exc), str(exc)\n'
            "    assert isinstance(exc, ImportError), type(exc)\n"
            "except ModuleNotFoundError as exc:\n"
            "    raise AssertionError('the first numpy use raised a bare ModuleNotFoundError for '\n"
            "        + str(exc.name) + ': the verifier probe is satisfied on the base closure '\n"
            "        'again, so require_extra cannot fire')\n"
            "else:\n"
            "    raise AssertionError('numpy resolved on the base closure')\n"
        ),
    )
    assert result["returncode"] == 0, result["stderr"]
    assert result["present"] == [], result["present"]


def test_require_extra_refuses_the_verifier_extra_on_the_base_closure(clean_python: Path) -> None:
    """The same gate called directly, with no proxy in front of it.

    The test above goes through the lazy numpy proxy, which is one seam; this is the other, the
    one `verifier/fuzz.py` and `verifier/metamorphic.py` call before importing hypothesis or
    crosshair. Neither of those is in the base closure either, so the refusal here is the message
    those subsystems now give instead of a `ModuleNotFoundError` two frames further in.
    """
    result = import_probe(
        clean_python,
        "reward_lens.core.extras",
        extra_code=(
            "from reward_lens.core.extras import ExtraRequiredError, require_extra\n"
            "try:\n"
            "    require_extra('verifier', subsystem='x')\n"
            "except ExtraRequiredError as exc:\n"
            "    assert str(exc).startswith('x needs'), str(exc)\n"
            "    assert \"'verifier' extra\" in str(exc), str(exc)\n"
            '    assert "pip install \'reward-lens[verifier]\'" in str(exc), str(exc)\n'
            "    assert isinstance(exc, ImportError), type(exc)\n"
            "else:\n"
            "    raise AssertionError('require_extra(verifier) returned on a base install')\n"
        ),
    )
    assert result["returncode"] == 0, result["stderr"]
    assert result["present"] == [], result["present"]


def test_verifier_package_does_not_gate_its_whole_namespace(clean_python: Path) -> None:
    """C-004 and interfaces section 7: `require_extra` applies per module, never to the whole
    verifier package. `import reward_lens.verifier` on a base install used to raise
    `ExtraRequiredError` from `verifier/__init__.py:42`."""
    result = import_probe(clean_python, "reward_lens.verifier")
    assert result["returncode"] == 0, result["stderr"]


def test_core_resolves_its_numpy_backed_names_lazily(clean_python: Path) -> None:
    """PEP 562. `import reward_lens.core` must not pull `core.evidence` (numpy) or `core.config`
    (pydantic-settings), but `reward_lens.core.Evidence` must still resolve for anyone who has the
    numeric stack, and `dir()` must still list the public surface so tab completion works."""
    result = import_probe(
        clean_python,
        "reward_lens.core",
        extra_code=(
            "import reward_lens.core as c\n"
            'assert "Evidence" in dir(c), "PEP 562 __dir__ dropped the lazy names"\n'
            'assert "Reading" in dir(c)\n'
            "assert c.Reading is not None\n"
            "try:\n"
            "    c.NoSuchName\n"
            "except AttributeError:\n"
            "    pass\n"
            "else:\n"
            '    raise AssertionError("a lazy __getattr__ must still raise AttributeError")\n'
        ),
    )
    assert result["returncode"] == 0, result["stderr"]
    assert result["present"] == [], result["present"]


#: Every row of `EXTRA_PROBE`, frozen the way `conftest.EXPECTED_CLOSURE` freezes the base closure.
#: The case below reads the table back out of the installed wheel and compares it with this list,
#: so a row added to the mapping and not added here fails every case instead of going unmeasured.
PROBED_EXTRAS = [
    "trace",
    "judge",
    "train",
    "sigstore",
    "verifier",
    "white-box",
    "organisms",
    "viz",
    "record",
]


@pytest.mark.parametrize("extra", PROBED_EXTRAS)
def test_no_probe_module_is_present_on_the_base_closure(clean_python: Path, extra: str) -> None:
    """The mapping's own rule, measured: a probe is present if and only if its extra is.

    `EXTRA_PROBE["verifier"]` named `coverage` for as long as it took A-002 to move `coverage`
    into the base closure and nobody to reread the row. A probe the base install satisfies is a
    guard that cannot fire: `require_extra` returns, and the subsystem dies further in on whatever
    it actually needed, with the error the guard exists to replace. This case is what would have
    caught that, and what catches the next base dependency to collide with a probe.

    The empty probe is asserted rather than skipped: `judge` declares no runtime dependency yet,
    so its row is the one that is meant to return, and a row that quietly became empty would
    otherwise stop being measured here.
    """
    result = import_probe(
        clean_python,
        "reward_lens.core.extras",
        extra_code=(
            "import importlib, importlib.util\n"
            "from reward_lens.core.extras import EXTRA_PROBE, ExtraRequiredError, require_extra\n"
            f"assert sorted(EXTRA_PROBE) == {sorted(PROBED_EXTRAS)!r}, sorted(EXTRA_PROBE)\n"
            f"extra = {extra!r}\n"
            "probe = EXTRA_PROBE[extra]\n"
            "if not probe:\n"
            "    require_extra(extra, subsystem='the probe audit')\n"
            "else:\n"
            "    assert importlib.util.find_spec(probe) is None, (\n"
            "        probe + ' is on the base closure, so require_extra(' + extra\n"
            "        + ') can never raise'\n"
            "    )\n"
            "    try:\n"
            "        importlib.import_module(probe)\n"
            "    except ModuleNotFoundError:\n"
            "        pass\n"
            "    else:\n"
            "        raise AssertionError(probe + ' imports on the base closure')\n"
            "    try:\n"
            "        require_extra(extra, subsystem='the probe audit')\n"
            "    except ExtraRequiredError as exc:\n"
            "        assert extra in str(exc), str(exc)\n"
            "    else:\n"
            "        raise AssertionError('require_extra(' + extra + ') returned on a base install')\n"
        ),
    )
    assert result["returncode"] == 0, result["stderr"]
    assert result["present"] == [], result["present"]
