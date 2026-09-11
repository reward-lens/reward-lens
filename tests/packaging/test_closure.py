"""The base closure: what a `pip install reward-lens` actually drags down.

D-58's rule is not "under N packages", it is that `uv pip list` in a clean venv prints a set with
nothing numeric, nothing that renders and nothing that talks to a network in it, and that the set
is written into the release notes beside the cold `uvx reward-lens --help` time. So the set is the
assertion and this file writes the artifact.
"""

from __future__ import annotations

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


def test_the_seven_declared_packages_are_present(installed: dict[str, str]) -> None:
    """D-58 plus addenda A-001 and A-002: the seven declared names."""
    declared = [
        "click",
        "pyyaml",
        "pydantic",
        "jsonschema-rs",
        "rfc8785",
        "libcst",
        "coverage",
    ]
    missing = [n for n in declared if normalise(n) not in installed]
    assert missing == [], f"declared base dependencies missing from the clean venv: {missing}"


def test_the_closure_is_twelve_distributions(installed: dict[str, str]) -> None:
    """The count the release notes quote: seven declared, four from pydantic, and reward-lens.

    Asserted separately from the set so a diff in the release-notes number is its own failure
    rather than a line buried in a set difference.
    """
    assert len(installed) == 12, f"the base closure is {len(installed)}, not 12: {sorted(installed)}"


def test_closure_document_is_written(installed: dict[str, str]) -> None:
    """Write the measured closure where the release notes can quote it.

    The cold `uvx reward-lens --help` time belongs beside it and cannot be measured until P-CLI
    lands a runnable console script, so the line says so rather than carrying a placeholder number
    that would read as a measurement.
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
        "pending P-CLI. The console script's entry point is in the wheel's metadata and is asserted\n"
        "by `test_wheel_metadata.py`, but `reward_lens.cli.main:main` does not exist yet, so there\n"
        "is nothing to time. D-30's ceiling is 100 ms and gate 5 checks the import set as well as\n"
        "the clock.\n",
        encoding="utf-8",
    )
    assert CLOSURE_DOC.exists()
