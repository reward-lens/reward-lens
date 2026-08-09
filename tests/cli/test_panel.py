"""Panel ordering and glyph behavior for public CLI output."""

from __future__ import annotations

import pathlib

import pytest


GLYPHS = "○✘!✔"
DARK = "○"

#: Interfaces section 9's frozen order: the word each panel prints, and the section it reads.
PANELS = (
    ("validity", "validity"),
    ("soundness", "soundness"),
    ("reach", "reach"),
    ("exploits", "exploits"),
    ("framing", "framing"),
    ("signal", "signal"),
    ("cost", "reward_statistics"),
    ("trace", "trace"),
    ("forecast", "forecast"),
    ("calibration", "calibration"),
)
WORDS = tuple(word for word, _ in PANELS)
SECTION = dict(PANELS)


def is_dark(doc: dict, word: str) -> bool:
    """The rule, read off the record: a panel is dark when every entry of its section is absent.

    The tests below compute this rather than naming the glyph they expect, because which sections
    a build happens to fill changes as instruments land, and the panel rule does not.
    """
    entries = list((doc.get("measurement") or {}).get(SECTION[word]) or [])
    return not entries or all(entry.get("kind") == "absence" for entry in entries)


def printed_order(doc: dict) -> list[str]:
    """The frozen order with the lit panels stably ahead of the dark ones."""
    lit = [word for word in WORDS if not is_dark(doc, word)]
    return lit + [word for word in WORDS if is_dark(doc, word)]


@pytest.fixture
def honest(tmp_path):
    from reward_lens import api

    grader = tmp_path / "grader.py"
    grader.write_text("def score(task, response):\n    return 1.0\n", encoding="utf-8")
    return api.audit(api.AuditRequest(path=grader))


def panel_lines(text: str) -> list[tuple[str, str]]:
    """The (glyph, section) pairs of a rendered panel block, in the order they were printed."""
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped[:1] in GLYPHS and len(stripped.split()) >= 2:
            out.append((stripped[0], stripped.split()[1]))
    return out


def test_project_panel_order_is_stable() -> None:
    assert WORDS == (
        "validity",
        "soundness",
        "reach",
        "exploits",
        "framing",
        "signal",
        "cost",
        "trace",
        "forecast",
        "calibration",
    )

def test_every_panel_glyph_is_read_off_the_record_it_renders(honest):
    """Ten panels, and each one dark exactly when its section holds nothing but absences.

    A dark panel says what is missing, which it can only get from the hole; a lit one carries one
    of the three lit glyphs, and which of the three is the business of the tests below it.
    """
    from reward_lens.cli.format import text as formatter

    doc = honest.to_dict()
    rendered = formatter.render(honest, command="audit", shape="project")
    pairs = panel_lines(rendered)
    assert [name for _, name in pairs] == printed_order(doc)

    dark = [word for word in WORDS if is_dark(doc, word)]
    lit = [word for word in WORDS if word not in dark]
    assert lit and dark, f"this record exercises only one arm of the rule: lit={lit} dark={dark}"

    for glyph, name in pairs:
        if name in dark:
            assert glyph == DARK, f"{name} holds only absences and must be dark, not {glyph!r}"
        else:
            assert glyph in "✘!✔", f"{name} holds a measured entry and must not be dark"

    detail = {
        line.strip().split()[1]: line.strip()
        for line in rendered.splitlines()
        if line.strip()[:1] in GLYPHS and len(line.strip().split()) >= 2
    }
    for word in dark:
        assert "not measured:" in detail[word], detail[word]


def test_a_lit_panel_sorts_above_the_dark_ones(honest):
    from reward_lens.cli.format import text as formatter
    from reward_lens.contracts import Assay

    doc = honest.to_dict()
    entry = dict(doc["measurement"]["reach"][0])
    entry["kind"] = "detection"
    entry["state"] = "partial"
    entry.pop("absence", None)
    entry["entry_id"] = "reach.exposure_inventory"
    entry["detection"] = {
        "target": "exposure to the reward's stated scope",
        "material": None,
        "unqualified": True,
        "unqualified_reason": "a static inventory has no measured limits or error rates",
    }
    entry["result"] = {"covered": 0.62}
    entry["limitations"] = ["static: no executed witness"]
    doc["measurement"]["reach"] = [entry]
    doc["holes"] = [hole for hole in doc["holes"] if hole["section"] != "reach"]
    rendered = formatter.render(Assay.model_validate(doc), command="audit", shape="project")
    pairs = panel_lines(rendered)
    order = [name for _, name in pairs]
    glyph = dict((name, mark) for mark, name in pairs)

    dark = [word for word in WORDS if is_dark(doc, word)]
    assert "reach" not in dark and glyph["reach"] == "!"
    assert dark, "this record leaves no dark panel for a lit one to sort above"
    assert order.index("reach") < min(order.index(word) for word in dark)
    assert order == printed_order(doc)


def test_an_error_level_finding_darkens_its_section_to_a_cross(honest):
    from reward_lens.cli.format import text as formatter
    from reward_lens.contracts import Assay

    doc = honest.to_dict()
    doc["findings"] = [
        {
            "id": "RGX-local-0001",
            "rule": "validity.static.writable_then_read",
            "level": "error",
            "kind": "fail",
            "severity_rationale": "the reward reads a file the graded process can write",
            "scope": "evaluator_comparison",
            "entries": ["validity.instrument_absent"],
            "partial_fingerprint": "writable_then_read:grader.py",
            "message": "grader.py line 14 reads outcome/test_solution.py",
            "code": "RL0201",
        }
    ]
    rendered = formatter.render(Assay.model_validate(doc), command="audit", shape="project")
    glyph = dict((name, glyph) for glyph, name in panel_lines(rendered))
    assert glyph["validity"] == "✘"


def test_the_bare_grader_shape_folds_the_rest_away(honest):
    from reward_lens.cli.format import text as formatter

    rendered = formatter.render(honest, command="audit", shape="grader")
    names = [name for _, name in panel_lines(rendered)]
    assert names[:1] == ["validity"]
    assert set(names) <= {"validity", "replay", "soundness", "reach", "exploits", "signal"}
    assert not {"framing", "cost", "trace", "forecast", "calibration"} & set(names), rendered


def test_the_formatter_computes_no_statistic(honest):
    """It reads the record's own numbers; it never derives one the record does not carry."""
    source = pathlib.Path(
        __import__("reward_lens.cli.format.text", fromlist=["text"]).__file__
    ).read_text(encoding="utf-8")
    for banned in ("statistics.", "math.sqrt", "numpy", "mean(", "stdev("):
        assert banned not in source, banned


def _short_finding_record(honest):
    """The same record with its validity absences dropped, so the detail fits on one line."""
    from reward_lens.contracts import Assay

    doc = _long_finding_record(honest).to_dict()
    doc["measurement"]["validity"] = [
        entry for entry in doc["measurement"]["validity"] if entry.get("kind") != "absence"
    ]
    doc["findings"][0]["entries"] = [doc["measurement"]["validity"][0]["entry_id"]]
    return Assay.model_validate(doc)


def _long_finding_record(honest):
    from reward_lens.contracts import Assay

    doc = honest.to_dict()
    doc["findings"] = [
        {
            "id": "RGX-local-0001",
            "rule": "validity.static.writable_then_read",
            "level": "error",
            "kind": "fail",
            "severity_rationale": "the reward reads a file the graded process can write",
            "scope": "evaluator_comparison",
            "entries": ["validity.instrument_absent"],
            "partial_fingerprint": "writable_then_read:grader.py",
            "message": (
                "grader.py line 91 reads outcome/test_solution.py, and line 56 runs the graded "
                "response in the same working directory, so the response can write what the "
                "grader then reads"
            ),
            "code": "RL0201",
        }
    ]
    return Assay.model_validate(doc)


def test_a_finding_continues_a_section_whose_detail_fits_one_line(honest):
    """`first-hour.txt` lines 20 to 22: the detail fits, so the finding is a continuation of it.

    The panel order is frozen, so a finding that carried a glyph where the section had already
    said everything on one line read as an eleventh panel.
    """
    from reward_lens.cli.format import text as formatter

    rendered = formatter.render(_short_finding_record(honest), command="audit", shape="project")
    names = [name for _, name in panel_lines(rendered)]

    assert names.count("validity") == 1, rendered
    assert len(names) == len(set(names)) == len(WORDS), names

    body = [line for line in rendered.splitlines() if "grader.py line 91" in line]
    assert body, rendered
    assert body[0].strip()[:1] not in GLYPHS, body[0]


def test_a_finding_after_a_wrapped_detail_is_still_a_continuation(honest):
    """A-028: the detail took two lines, and the finding still continues them.

    The rule this replaces gave a finding its own glyph row once the detail had wrapped, which is
    what printed `validity` twice in `audit-unfamiliar.txt`. One glyph line per section, always.
    """
    from reward_lens.cli.format import text as formatter

    rendered = formatter.render(_long_finding_record(honest), command="audit", shape="project")
    lines = rendered.splitlines()
    opened = [line for line in lines if "grader.py line 91" in line]
    names = [name for _, name in panel_lines(rendered)]

    assert opened, rendered
    assert opened[0].startswith(" " * formatter._INDENT), opened[0]
    assert opened[0].strip()[:1] not in GLYPHS, opened[0]
    assert names.count("validity") == 1, rendered
    assert len(names) == len(set(names)) == len(WORDS), names


def test_two_findings_of_one_section_print_one_glyph_line_and_two_continuations(honest):
    """A-028: however many findings a section carries, it prints one head and they all continue.

    Two error-level findings under `validity` used to print two `✘ validity` rows, which read as
    two panels for one section. Both messages and both codes still reach the reader.
    """
    from reward_lens.cli.format import text as formatter
    from reward_lens.contracts import Assay

    doc = _long_finding_record(honest).to_dict()
    second = dict(doc["findings"][0])
    second.update(
        id="RGX-local-0002",
        rule="validity.static.grader_is_not_deterministic",
        severity_rationale="the reward reads the clock",
        partial_fingerprint="reads_clock:grader.py",
        message="grader.py line 12 reads the wall clock, so two runs of one response can disagree",
        code="RL0202",
    )
    doc["findings"].append(second)
    rendered = formatter.render(Assay.model_validate(doc), command="audit", shape="project")
    lines = rendered.splitlines()
    names = [name for _, name in panel_lines(rendered)]

    assert names.count("validity") == 1, rendered
    assert len(names) == len(set(names)) == len(WORDS), names
    for excerpt in ("grader.py line 91", "grader.py line 12"):
        opened = [line for line in lines if excerpt in line]
        assert opened, f"{excerpt} is not in the panel block\n{rendered}"
        assert opened[0].startswith(" " * formatter._INDENT), opened[0]
        assert opened[0].strip()[:1] not in GLYPHS, opened[0]
    for code in ("RL0201", "RL0202"):
        assert [line for line in lines if line.rstrip().endswith(code)], f"{code} is not printed"


def test_bare_grader_prints_one_validity_head_with_continuations(honest):
    from reward_lens.cli.format import text as formatter

    rendered = formatter.render(_long_finding_record(honest), command="audit", shape="grader")
    block = rendered.splitlines()

    heads = [line for line in block if line.strip().startswith(formatter.CROSS + " validity")]
    assert len(heads) == 1, block
    assert [line for line in block if line.rstrip().endswith("RL0201")], block
    for line in block:
        if "grader.py line 91" in line or line.rstrip().endswith("RL0201"):
            assert line.startswith(" " * formatter._INDENT), repr(line)
            assert line.strip()[:1] not in GLYPHS, repr(line)

def test_a_finding_wraps_under_the_detail_column_and_ends_with_its_code(honest):
    """The message is wrapped, every line of it sits in the detail column, and the code is last."""
    from reward_lens.cli.format import text as formatter

    rendered = formatter.render(_long_finding_record(honest), command="audit", shape="project")
    lines = rendered.splitlines()
    first = next(i for i, line in enumerate(lines) if "grader.py line 91" in line)
    coded = next(i for i, line in enumerate(lines) if line.rstrip().endswith("RL0201"))

    assert coded > first, "the code stands on the last line of the finding, not the first"
    for line in lines[first + 1 : coded + 1]:
        assert line.startswith(" " * formatter._INDENT), repr(line)
        assert len(line) <= formatter._INDENT + formatter._PANEL_WRAP + len("RL0201") + formatter._CODE_GAP
    column = lines[coded].index("RL0201")
    text_end = len(lines[coded][:column].rstrip())
    expected = formatter._CODE_COLUMN if text_end <= formatter._CODE_ELBOW else text_end + formatter._CODE_GAP
    assert column == expected, repr(lines[coded])
