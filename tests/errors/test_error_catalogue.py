"""The catalogue covers every code that exists, and every entry stands on its own.

The reserved list below is transcribed from section 8 of the wave-1 interface file, not imported
from the catalogue, so that this test measures the implementation against the allocation rather
than against itself.
"""

from __future__ import annotations

import re
from pathlib import Path

from reward_lens.errors import CATALOGUE, ErrorSpec, check_spec
from reward_lens.errors.catalogue import (
    CONTRACTED_COMMANDS,
    EXIT_MEANINGS,
    RANGE_EXCEPTIONS,
    RANGES,
)

RESERVED_BY_THE_INTERFACES = (
    "RL0001",
    "RL0002",
    "RL0003",
    "RL0004",
    "RL0120",
    "RL0130",
    "RL0201",
    "RL0202",
    "RL0203",
    "RL0210",
    "RL0211",
    "RL0212",
    "RL0213",
    "RL0214",
    "RL0215",
    "RL0231",
    "RL0301",
    "RL0302",
    "RL0341",
    "RL0401",
    "RL0402",
    "RL0410",
    "RL0501",
    "RL0601",
    "RL0602",
    "RL0603",
    "RL0604",
    "RL0620",
    "RL0621",
    "RL0701",
    "RL0702",
    "RL0703",
    "RL0710",
    "RL0801",
    "RL0900",
)

CODE_SHAPE = re.compile(r"^RL[0-9]{4}$")
ANY_CODE = re.compile(r"RL[0-9]{4}")
INTERNAL_MODULE = re.compile(r"reward_lens[./][A-Za-z_]")
DOCUMENT_SECTION = re.compile(r"\b[A-Z]-[0-9]{1,3}\b|\bsection [0-9]|§")
RUNNABLE_VERB = re.compile(r"run: reward-lens ([a-z][a-z-]*)")

SRC = Path(__file__).resolve().parents[2] / "src"


def test_the_catalogue_is_exactly_the_reserved_allocation() -> None:
    assert sorted(CATALOGUE) == sorted(RESERVED_BY_THE_INTERFACES)


def test_every_key_matches_the_code_it_holds() -> None:
    for key, spec in CATALOGUE.items():
        assert key == spec.code
        assert CODE_SHAPE.match(spec.code), key


def test_every_code_written_under_src_is_in_the_catalogue() -> None:
    found: dict[str, str] = {}
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in ANY_CODE.finditer(text):
            found.setdefault(match.group(0), str(path.relative_to(SRC.parent)))
    missing = {code: where for code, where in found.items() if code not in CATALOGUE}
    assert not missing, f"codes raised in src/ that the catalogue lacks: {missing}"


def test_every_entry_passes_the_consistency_check() -> None:
    problems = {code: check_spec(spec) for code, spec in CATALOGUE.items()}
    assert not {code: found for code, found in problems.items() if found}


def test_an_entry_with_no_remedy_fails_the_consistency_check() -> None:
    spec = ErrorSpec(
        code="RL0001",
        title="a title",
        cause="a cause",
        remedies=[],
        exit_code=4,
    )
    assert any("remedy" in problem for problem in check_spec(spec))


def test_an_entry_whose_only_remedy_is_blank_fails_the_consistency_check() -> None:
    spec = ErrorSpec(
        code="RL0001",
        title="a title",
        cause="a cause",
        remedies=["   "],
        exit_code=4,
    )
    assert any("remedy" in problem for problem in check_spec(spec))


def test_an_entry_with_no_runnable_command_fails_the_consistency_check() -> None:
    spec = ErrorSpec(
        code="RL0001",
        title="a title",
        cause="a cause",
        remedies=["think about it harder"],
        exit_code=4,
    )
    assert any("remedy" in problem for problem in check_spec(spec))


def test_every_exit_code_is_one_the_table_defines() -> None:
    for code, spec in CATALOGUE.items():
        assert spec.exit_code in EXIT_MEANINGS, (code, spec.exit_code)


def test_each_code_sits_in_its_range_or_is_a_recorded_exception() -> None:
    for code, spec in CATALOGUE.items():
        band = code[2:4]
        assert band in RANGES, code
        fixed = RANGES[band][1]
        if fixed is None or spec.exit_code == fixed:
            continue
        assert code in RANGE_EXCEPTIONS, (code, spec.exit_code, fixed)
        assert RANGE_EXCEPTIONS[code].strip(), code


def test_every_command_a_remedy_tells_you_to_run_is_one_the_cli_ships() -> None:
    named = 0
    for code, spec in CATALOGUE.items():
        for text in (spec.cause, spec.title, *spec.remedies):
            for verb in RUNNABLE_VERB.findall(text):
                named += 1
                assert verb in CONTRACTED_COMMANDS, (code, verb)
    assert named >= len(CATALOGUE) // 2


def test_no_entry_names_an_internal_module_or_a_document_section() -> None:
    for code, spec in CATALOGUE.items():
        for text in (spec.title, spec.cause, *spec.remedies):
            assert not INTERNAL_MODULE.search(text), (code, text)
            assert not DOCUMENT_SECTION.search(text), (code, text)


def test_every_surface_is_one_of_the_three() -> None:
    for code, spec in CATALOGUE.items():
        assert spec.surface in {"error", "finding", "state"}, (code, spec.surface)
