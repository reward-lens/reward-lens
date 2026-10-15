"""The base closure: what a `pip install reward-lens` actually drags down.

D-58's rule is not "under N packages", it is that `uv pip list` in a clean venv prints a set with
nothing numeric, nothing that renders and nothing that talks to a network in it, and that the set
is written into the release notes beside the cold `uvx reward-lens --help` time. So the set is the
assertion and this file writes the artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .conftest import EXPECTED_CLOSURE, FORBIDDEN, normalise

CLOSURE_DOC = Path(__file__).resolve().parent / "CLOSURE.md"


def test_closure_is_exactly_the_expected_set(installed: dict[str, str]) -> None:
    got = set(installed)
    assert got == set(EXPECTED_CLOSURE), (
        f"the base closure moved.\n  added:   {sorted(got - set(EXPECTED_CLOSURE))}\n"
        f"  missing: {sorted(set(EXPECTED_CLOSURE) - got)}\n"
        "A new name here is a new base dependency, which D-58 forbids without a decision record."
    )


def test_nothing_numeric_or_rendering_is_in_the_closure(installed: dict[str, str]) -> None:
    intruders = sorted(n for n in installed if n in {normalise(f) for f in FORBIDDEN})
    assert intruders == [], f"forbidden distributions in the base closure: {intruders}"


def test_the_eight_declared_packages_are_present(installed: dict[str, str]) -> None:
    """D-58 plus A-001/A-002 and offline signatures: the eight declared names."""
    declared = [
        "click",
        "cryptography",
        "pyyaml",
        "pydantic",
        "jsonschema-rs",
        "rfc8785",
        "libcst",
        "coverage",
    ]
    missing = [n for n in declared if normalise(n) not in installed]
    assert missing == [], f"declared base dependencies missing from the clean venv: {missing}"


def test_the_closure_has_the_platform_count(installed: dict[str, str]) -> None:
    """The count the release notes quote, including libcst's Python 3.13-only dependency.

    ``cryptography`` deliberately adds ``cffi`` and ``pycparser`` for offline signatures; current
    libcst resolution adds ``pyyaml-ft`` only on Python 3.13. Asserted separately from the set
    so a count change is visible rather than buried in a set difference.
    """
    expected = 16 if sys.version_info[:2] == (3, 13) else 15
    assert len(installed) == expected, (
        f"the base closure is {len(installed)}, not {expected}: {sorted(installed)}"
    )


def test_closure_document_is_written(installed: dict[str, str]) -> None:
    """Write the measured closure where the release notes can quote it.

    The cold `uvx reward-lens --help` time belongs beside it. The console script is now present,
    but the clean-machine measurement belongs to final release qualification, so the line says
    that rather than carrying a placeholder number that would read as a measurement.
    """
    rows = "\n".join(f"| {name} | {version} |" for name, version in sorted(installed.items()))
    CLOSURE_DOC.write_text(
        "# The measured base closure\n\n"
        "Written by `tests/packaging/test_closure.py`. This is what `uv pip list` prints in a venv\n"
        "holding only the built wheel, which is the count D-58 says binds.\n\n"
        f"{len(installed)} distributions.\n\n"
        "| distribution | version |\n|---|---|\n"
        f"{rows}\n\n"
        "## Addendum A-002: what `coverage` costs the base closure\n\n"
        "`verifier/coverage.py` imports the `coverage` distribution inside `trace_corpus`, and D-63\n"
        "says the first hour runs with no extras, so `measure_coverage` has to reach a Reading on\n"
        "the base closure. Three measurements taken on this platform (Linux x86_64, CPython 3.11.13)\n"
        "before A-002 was applied:\n\n"
        "- **Wheel size.** `coverage-7.16.0-cp311-cp311-manylinux1_x86_64.manylinux_2_28_x86_64.\n"
        "  manylinux_2_5_x86_64.whl` is 255,846 bytes (249.8 KiB), read from the PyPI JSON API for\n"
        "  release 7.16.0 and selected by the cp311 manylinux x86_64 tag this interpreter resolves.\n"
        "- **Distributions added beyond itself: none.** `uv venv` then `uv pip install\n"
        "  'coverage>=7.15.0'` into an otherwise empty 3.11 venv, then `uv pip list`, prints one\n"
        "  row: `coverage 7.16.0`.\n"
        "- **Cold `import coverage`.** 152, 162 and 185 ms cumulative over three runs of\n"
        "  `python -X importtime -c 'import coverage'` with every `__pycache__` removed between\n"
        "  runs; 70 ms warm. The number is the whole subtree, stdlib imports included, which is\n"
        "  what a caller pays.\n\n"
        "The alternative A-002 rejected was a second tracer on `sys.settrace` (3.11) and\n"
        "`sys.monitoring` (3.12+), which would replace a tracer whose semantics the inherited tests\n"
        "pin, to save one pure-payload distribution. `requires-python >= 3.11` is unchanged.\n\n"
        "## Cold `uvx reward-lens --help`\n\n"
        "The console script is present and checked against the installed wheel. A cold `uvx`\n"
        "measurement on a clean machine is pending final release qualification; no timing claim\n"
        "is made here. D-30's ceiling is 100 ms and gate 5 checks the import set as well as\n"
        "the clock.\n",
        encoding="utf-8",
    )
    assert CLOSURE_DOC.exists()
