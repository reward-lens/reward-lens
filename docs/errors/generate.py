"""Write the pages under docs/errors/ from the catalogue.

Run it from the repository root: `python docs/errors/generate.py`. The pages are committed, and a
test regenerates them into a temporary directory and compares byte for byte, so a catalogue change
that does not reach the pages fails rather than drifts.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from reward_lens.errors import CATALOGUE  # noqa: E402
from reward_lens.errors.catalogue import EXIT_MEANINGS, RANGES, ErrorSpec  # noqa: E402


def page(spec: ErrorSpec) -> str:
    lines = [
        f"# {spec.code}",
        "",
        spec.title,
        "",
        "## What it means",
        "",
        spec.long_form_cause,
        "",
        "## What this means for the run",
        "",
    ]
    if spec.surface == "finding":
        lines += [
            f"Nothing stops. {spec.code} is a finding the record carries, and a decision resting "
            "on the section it lands in stays unresolved until the finding is dealt with or "
            "accepted.",
            "",
        ]
    elif spec.surface == "state":
        lines += [
            f"Nothing stops. {spec.code} is a state the record carries: it says what the "
            "measurement could not qualify, rather than that something went wrong.",
            "",
        ]
    if spec.verdicts:
        lines += [spec.verdict_rule, ""]
    lines += [
        EXIT_MEANINGS[spec.exit_code],
        "",
        "## What usually causes it, and what resolves each",
        "",
    ]
    for remedy in spec.long_form_remedies:
        lines.append(f"- {remedy}")
    lines += [
        "",
        "## Offline",
        "",
        f"`reward-lens explain {spec.code}` prints this with the network off.",
        "",
    ]
    return "\n".join(lines)


def index() -> str:
    lines = [
        "# Error codes",
        "",
        "Every refusal reward-lens makes carries a code that does not change between releases.",
        "The code is on the second line of the error, and the long form of any of them prints",
        "with the network off:",
        "",
        "```text",
        "reward-lens explain RL0341",
        "```",
        "",
        "## Every code",
        "",
        "| Code | What it is | Exit |",
        "| --- | --- | --- |",
    ]
    for code in sorted(CATALOGUE):
        spec = CATALOGUE[code]
        lines.append(f"| [{code}]({code}.md) | {spec.title} | {spec.exit_code} |")
    lines += [
        "",
        "## What the exit statuses mean",
        "",
        "| Exit | Meaning |",
        "| --- | --- |",
    ]
    for status in sorted(EXIT_MEANINGS):
        meaning = EXIT_MEANINGS[status].split(": ", 1)[1]
        lines.append(f"| {status} | {meaning} |")
    lines += [
        "",
        "## How the numbers are grouped",
        "",
        "| Band | What it covers |",
        "| --- | --- |",
    ]
    for band in sorted(RANGES):
        lines.append(f"| RL{band}xx | {RANGES[band][0]} |")
    lines.append("")
    return "\n".join(lines)


def write_pages(dest: Path) -> list[Path]:
    """Write the index and one page per code into `dest`, and return what was written."""
    dest.mkdir(parents=True, exist_ok=True)
    written = [dest / "README.md"]
    written[0].write_text(index(), encoding="utf-8")
    for code in sorted(CATALOGUE):
        path = dest / f"{code}.md"
        path.write_text(page(CATALOGUE[code]), encoding="utf-8")
        written.append(path)
    return written


def main() -> int:
    for path in write_pages(HERE):
        print(path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
