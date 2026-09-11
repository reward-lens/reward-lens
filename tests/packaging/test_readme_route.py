"""The README's route from a stranger's terminal to a 3.1.0 report (A-026).

The cold read of wave 1 opened `README.md` and found the 3.x library's front page: no `init`, no
`audit`, no `doctor` anywhere in it. A reader who installs the distribution this package builds has
no way to reach the product from the file that ships inside the wheel. This module pins the route
back in: a short section at the top, the three commands in the order a stranger runs them, and the
four verbs the help names that this build does not answer yet, so the README does not promise what
`audit` cannot deliver.

The section is delimited by HTML comments rather than by the next `## ` heading, because the 3.x
prose that follows it opens with paragraphs and no heading of its own. The file already uses a
comment marker for its generated refusal-reason block, so the idiom is not new here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"

#: The delimiters. Both are load-bearing: they are how a test, a generator or a reviewer can find
#: where the 3.1.0 route stops and the inherited 3.0 library reference starts.
OPEN_MARKER = "<!-- 3.1.0-route -->"
CLOSE_MARKER = "<!-- /3.1.0-route -->"

#: The section's own heading.
HEADING = "## 3.1.0: audit a reward system"

#: The brief's ceiling, markers included. A route a stranger reads before deciding whether to
#: install is worth about a screen; past that it is documentation, and documentation lives further
#: down the same file.
MAX_LINES = 20

#: The three commands, in the order the first-hour transcript runs them. Substring match, because
#: the section prints them as a shell block and the flags are load-bearing.
COMMANDS = (
    "reward-lens init --example code-reward ./demo",
    "reward-lens audit ./demo",
    "reward-lens doctor",
)

#: The verbs the CLI's help lists that this build does not answer. Naming them in the README is the
#: cheapest correction available for a help text that lists more than it does.
ABSENT_VERBS = ("trace", "compare", "forecast", "improve")

#: The install line the packaging already documents, further down the same file.
INSTALL = "pip install ."


def read_lines() -> list[str]:
    return README.read_text(encoding="utf-8").split("\n")


def section_bounds(lines: list[str]) -> tuple[int, int]:
    """Return the half-open line range of the 3.1.0 section, both markers included."""
    opens = [i for i, line in enumerate(lines) if line.strip() == OPEN_MARKER]
    closes = [i for i, line in enumerate(lines) if line.strip() == CLOSE_MARKER]
    assert opens, f"{OPEN_MARKER} is not in README.md; the route to 3.1.0 is gone"
    assert closes, f"{CLOSE_MARKER} is not in README.md; the 3.1.0 route has no end"
    assert len(opens) == len(closes) == 1, f"markers appear {len(opens)}/{len(closes)} times"
    assert opens[0] < closes[0], "the 3.1.0 route's markers are inverted"
    return opens[0], closes[0] + 1


@pytest.fixture(scope="module")
def section() -> list[str]:
    lines = read_lines()
    start, end = section_bounds(lines)
    return lines[start:end]


def test_readme_exists() -> None:
    assert README.is_file(), f"{README} is missing"


def test_section_carries_its_heading(section: list[str]) -> None:
    assert HEADING in section, f"the 3.1.0 route does not carry {HEADING!r}"


def test_section_is_the_first_section() -> None:
    """Nothing from the 3.x library stands between the title and the route."""
    lines = read_lines()
    start, _ = section_bounds(lines)
    earlier = [line for line in lines[:start] if line.startswith("## ")]
    assert not earlier, f"sections precede the 3.1.0 route: {earlier}"
    titles = [i for i, line in enumerate(lines) if line.startswith("# ")]
    assert titles, "README.md has no title"
    assert titles[0] < start, "the 3.1.0 route sits above the title"


def test_section_is_short(section: list[str]) -> None:
    assert len(section) <= MAX_LINES, (
        f"the 3.1.0 route is {len(section)} lines, over the {MAX_LINES} ceiling"
    )


def test_section_says_what_three_one_zero_is(section: list[str]) -> None:
    text = "\n".join(section)
    for phrase in ("assay record", "report"):
        assert phrase in text, f"the 3.1.0 route does not say {phrase!r}"


def test_section_names_the_install(section: list[str]) -> None:
    text = "\n".join(section)
    assert INSTALL in text, f"the 3.1.0 route does not name {INSTALL!r}"
    assert "unreleased" in text


@pytest.mark.parametrize("command", COMMANDS)
def test_section_names_each_command(section: list[str], command: str) -> None:
    text = "\n".join(section)
    assert command in text, f"the 3.1.0 route does not name {command!r}"


def test_commands_are_in_running_order(section: list[str]) -> None:
    text = "\n".join(section)
    positions = [text.index(command) for command in COMMANDS]
    assert positions == sorted(positions), f"the three commands are out of order: {positions}"


@pytest.mark.parametrize("verb", ABSENT_VERBS)
def test_section_names_each_absent_verb(section: list[str], verb: str) -> None:
    text = "\n".join(section)
    assert f"`{verb}`" in text, f"the 3.1.0 route does not name the absent verb {verb!r}"


def test_absent_verbs_are_not_offered_as_commands(section: list[str]) -> None:
    """Naming a verb as absent is the point; printing it in the shell block is the failure."""
    text = "\n".join(section)
    for verb in ABSENT_VERBS:
        assert f"reward-lens {verb}" not in text, (
            f"the route runs `reward-lens {verb}`, which this build does not answer"
        )


def test_section_points_json_at_the_envelope(section: list[str]) -> None:
    text = "\n".join(section)
    assert "--format json" in text, "the 3.1.0 route does not name `--format json`"
    assert "envelope" in text, "the 3.1.0 route does not say what `--format json` writes"


def test_section_distinguishes_the_inherited_3_0_reference(section: list[str]) -> None:
    text = "\n".join(section)
    assert "inherited 3.0 library surface" in text
    assert "active 3.1.0 product" in text


def test_section_carries_no_unearned_numbers(section: list[str]) -> None:
    """The example's shape is measured; anything else in this section would not be.

    Single digits are version numbers (`3.1.0`, `3.0`). Any other run of digits is a claim, and the
    only two the brief allows are the example's 40 tasks and 120 responses.
    """
    text = "\n".join(section)
    found = {int(match) for match in re.findall(r"\d{2,}", text)}
    assert found <= {40, 120}, (
        f"the 3.1.0 route carries numbers beyond the example's shape: {sorted(found - {40, 120})}"
    )
