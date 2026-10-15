"""A-015: the record renders the wave-1 transcripts, block for block, from fields alone.

The claim these tests make is narrow and it is the one that matters: hand the text surface a
record that carries the sentences, the counts and the clauses, and it prints the transcript. No
pipeline runs here, nothing is read off a path, and no example exists on disk, so a line that
matches matches because the record carried it.

The goldens are the controller's and frozen. They are read the way the harness reads them:
`conftest.golden_body` drops the invocation line, `fleet/golden/run_demo.py:parse_golden` splits
the rest into `(command, lines)`, and `run_demo.mask` applies the volatile rules of
`fleet/golden/VOLATILE.md` to both sides so that a duration, a file size or a date in a filename
cannot fail a comparison that is about wording, order and column. The numeral mask is off
(`numerals=False`): every number below is the record's, and a test that masked numbers would not
notice if the formatter invented one.

A failure names the first line that differs and prints both sides, so the report of a broken
layout stays one line long.
"""

from __future__ import annotations

import importlib.util
from itertools import zip_longest

import pytest
from conftest import GOLDEN, golden_body

import fixtures
from reward_lens.cli.format import text as formatter

WAVE = (GOLDEN / "CURRENT_WAVE").read_text(encoding="utf-8").strip()
PROJECT_GOLDEN = "first-hour.txt"
GRADER_GOLDEN = "audit-unfamiliar.txt"


def harness():
    """`fleet/golden/run_demo.py`, imported by location: it is a script, not an installed module."""
    spec = importlib.util.spec_from_file_location("run_demo", GOLDEN / "run_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def golden_block(name: str, command: str) -> list[str]:
    """The expected output of one command of a golden, masked the way the harness masks it."""
    run_demo = harness()
    body = golden_body(f"{WAVE}/{name}")
    blocks = run_demo.parse_golden(body)
    if blocks:
        matching = [lines for cmd, lines in blocks if cmd.startswith(command)]
        assert len(matching) == 1, f"{name} does not hold exactly one `{command}` block"
        lines = matching[0]
    else:  # a single-command transcript: `golden_body` has already removed its invocation line
        lines = body.split("\n")
    return run_demo.mask("\n".join(lines), name, run_demo.volatile(), numerals=False).split("\n")


def masked(rendered: str, name: str) -> list[str]:
    run_demo = harness()
    return run_demo.mask(rendered, name, run_demo.volatile(), numerals=False).split("\n")


def first_difference(expected: list[str], got: list[str]) -> str | None:
    for index, (want, have) in enumerate(zip_longest(expected, got)):
        if want != have:
            return f"line {index} differs\n  golden: {want!r}\n  record: {have!r}"
    return None


def paragraphs(lines: list[str]) -> list[list[str]]:
    """The lines grouped on the blank lines that separate the blocks of the layout."""
    groups: list[list[str]] = [[]]
    for line in lines:
        if line.strip():
            groups[-1].append(line)
        elif groups[-1]:
            groups.append([])
    return [group for group in groups if group]


# --- the two records, against the two transcripts -----------------------------------------------


@pytest.fixture
def project(tmp_path):
    """The project audit, with the report file the Report line stats on disk."""
    report = tmp_path / "demo" / "assays" / "code-reward-v1-2026-09-11.assay.html"
    report.parent.mkdir(parents=True)
    report.write_text("<html></html>", encoding="utf-8")

    def render(record=None, **over):
        return formatter.render(
            record if record is not None else fixtures.project_record(**over),
            plan=fixtures.Plan(estimate_s=2),
            artifacts={"report": "assays/code-reward-v1-2026-09-11.assay.html"},
            project=str(tmp_path / "demo"),
            cwd=str(tmp_path),
        )

    return render


@pytest.fixture
def grader(tmp_path):
    report = tmp_path / "assays" / "my_grader-2026-09-11.assay.html"
    report.parent.mkdir(parents=True)
    report.write_text("<html></html>", encoding="utf-8")

    def render(record=None, **over):
        return formatter.render(
            record if record is not None else fixtures.grader_record(**over),
            plan=fixtures.Plan(estimate_s=2),
            artifacts={"report": str(report)},
            cwd=str(tmp_path),
        )

    return render


def test_the_project_record_renders_the_project_transcript(project):
    expected = golden_block(PROJECT_GOLDEN, "reward-lens audit")
    difference = first_difference(expected, masked(project(), PROJECT_GOLDEN))

    assert difference is None, difference


def test_the_grader_record_renders_the_bare_grader_transcript(grader):
    expected = golden_block(GRADER_GOLDEN, "reward-lens audit")
    difference = first_difference(expected, masked(grader(), GRADER_GOLDEN))

    assert difference is None, difference


def test_the_bare_grader_validity_block_is_one_head_and_its_continuations(grader):
    """A-028, byte for byte: `audit-unfamiliar.txt` lines 6 to 10, as the bare grader renders them.

    The whole-transcript test above would also catch a regression here, but it would report it as
    "line 4 differs". This one names the rule: the detail wraps onto a second line, the RL0210
    finding follows it as three continuation lines, and no second `✘ validity` head appears.
    """
    expected = golden_block(GRADER_GOLDEN, "reward-lens audit")[4:9]
    got = masked(grader(), GRADER_GOLDEN)[4:9]
    difference = first_difference(expected, got)

    assert difference is None, difference
    assert got[0].lstrip().startswith(formatter.CROSS + " validity"), got
    assert [line for line in got if line.rstrip().endswith("RL0210")] == [got[-1]], got
    assert [line for line in got[1:] if line.lstrip()[:1] in "○✘!✔"] == [], got


@pytest.mark.parametrize(
    "name,which",
    [(PROJECT_GOLDEN, "project"), (GRADER_GOLDEN, "grader")],
)
def test_the_blocks_line_up_one_for_one(name, which, project, grader):
    """Block by block: the same count of blocks, and the same lines inside each one."""
    render = project if which == "project" else grader
    expected = paragraphs(golden_block(name, "reward-lens audit"))
    got = paragraphs(masked(render(), name))

    assert len(got) == len(expected), f"{len(expected)} blocks in the golden, {len(got)} rendered"
    for index, (want, have) in enumerate(zip(expected, got)):
        difference = first_difference(want, have)
        assert difference is None, f"block {index}: {difference}"


# --- the same layout over different numbers -----------------------------------------------------


def test_a_project_record_with_other_numbers_renders_those_numbers(project):
    """The layout reads the record: nothing the golden counts is held by the formatter."""
    rendered = project(
        line=27, checks=9, failed=2, not_in_build=3, exposures=4, tasks=64, responses=256
    )

    assert "  Subject   code-reward v1  ·  grader.py  ·  64 tasks  ·  256 responses" in rendered
    flat = " ".join(rendered.split())  # the sentences wrap, so the numbers are read off one line
    assert "7 of 9 checks passed; 3 checks are not in this build" in flat
    assert "4 surfaces exposed, by static analysis only" in flat
    assert "grader.py line 27 reads outcome/test_solution.py" in flat
    assert "the 256 responses used that path" in flat
    for golden_number in (
        "40 tasks",
        "120 responses",
        "4 of 5 checks",
        "1 surface exposed",
        "line 91",
    ):
        assert golden_number not in flat


def test_a_grader_record_with_other_numbers_renders_those_numbers(grader):
    rendered = grader(checks=6, failed=1, not_in_build=2, task_set_checks=5, repeats=7)

    flat = " ".join(rendered.split())
    assert (
        "5 of 6 checks passed; 2 are not in this build; the 5 task-set checks need a task set"
        in flat
    )
    assert "7 repeats, identical output, no hidden state" in flat
    for golden_number in (
        "4 of 5 checks",
        "100 repeats",
        "4 are not in this build",
        "the 3 task-set",
    ):
        assert golden_number not in flat


def test_the_verdict_counts_follow_the_decision_and_not_the_panels(project):
    """`1 blocking finding, and 8 required panels could not run.` is read off `decision.reasons`."""
    record = fixtures.project_record()
    record["decision"]["reasons"] = [
        "outcome_unqualified:validity.outcome_isolation",
        "blocking_finding:RL0201",
        "blocking_finding:RL0202",
        "required_missing:signal",
    ]

    assert "2 blocking findings, and 1 required panel could not run." in project(record)


def test_the_subject_prints_the_shape_clause_the_record_carries(grader):
    """The clause is the record's wording, framed and not added to.

    The record says `plain shape`, noun and all. A formatter that appends the noun to whatever it
    is handed prints `plain shape shape` on that record, and a record whose shape is worded any
    other way gets the same noun stapled to it, so what is checked here is that the clause is
    passed through: two records, two wordings, neither of them the formatter's.
    """
    assert "`score(task, response)`, plain shape" in grader()
    assert "plain shape shape" not in grader()
    assert "`score(task, response)`, batched by prompt" in grader(shape="batched by prompt")


def test_the_validity_counts_carry_the_build_and_the_task_set_and_no_other_absence(grader):
    """The absences this line counts are two: what is not in the build, and what tasks would give.

    A validity absence that needs a response bank is stated in the remedy table under `To measure
    the rest`, which is where the record puts the remedy. Counted here as well, it arrives as a
    fourth clause inside a tally, so the line grows a sentence whenever the record grows an
    absence. The record below carries three of them and the line is the one the transcript holds.
    """
    rendered = grader(checks=6, failed=1, not_in_build=2, task_set_checks=5, response_bank_checks=3)
    flat = " ".join(rendered.split())

    assert (
        "5 of 6 checks passed; 2 are not in this build; the 5 task-set checks need a task set"
        in flat
    )
    assert "response-bank" not in flat
    assert "the 3 " not in flat

    # The article is the sentence's, not the record's. `product/audit/absences.py` writes the bare
    # input into `needs`, so a record off a real run says `task set` where `grader_record` says
    # `a task set`, and the clause has to read the same either way: rendered off the store's words
    # without the article added, the transcript prints `need task set`.
    stored = fixtures.grader_record(
        checks=6, failed=1, not_in_build=2, task_set_checks=5, response_bank_checks=3
    )
    for entry in stored["measurement"]["validity"]:
        note = (entry.get("extensions") or {}).get(fixtures.EXT) or {}
        if note.get("needs") == "a task set":
            note["needs"] = "task set"

    assert (
        "5 of 6 checks passed; 2 are not in this build; the 5 task-set checks need a task set"
        in " ".join(grader(stored).split())
    )


# --- the version label (A-015) ------------------------------------------------------------------


def test_the_subject_prints_the_name_and_the_version_label(project):
    """`code-reward v1` is the name and `version.id` with the `<name>-` prefix taken off it."""
    assert "  Subject   code-reward v1  ·  " in project()
    assert formatter._version_label("code-reward", "code-reward-v1") == "v1"


def test_a_version_id_without_the_name_prefix_is_the_label_itself(project):
    rendered = project(version_id="2026.09.1")

    assert "  Subject   code-reward 2026.09.1  ·  " in rendered
    assert formatter._version_label("code-reward", "2026.09.1") == "2026.09.1"


def test_the_prefix_is_taken_off_once_and_only_at_the_front(project):
    """A prefix in the middle of the id is not a prefix, and the name alone is never printed."""
    assert formatter._version_label("code-reward", "v1-code-reward") == "v1-code-reward"
    assert "  Subject   code-reward" in project(version_id="code-reward-code-reward-v1")
    assert "code-reward code-reward-v1  ·  " in project(version_id="code-reward-code-reward-v1")


# --- the other two blocks of the transcripts ----------------------------------------------------


def test_init_writes_the_manifest_the_transcript_shows(cli, tmp_path):
    """`init`'s block is the example's own manifest, printed by the verb that wrote the files."""
    run = cli(
        "init",
        "--example",
        "code-reward",
        "./demo",
        env={"REWARD_LENS_FORMAT": "text"},
        cwd=tmp_path,
    )
    expected = golden_block(PROJECT_GOLDEN, "reward-lens init")
    difference = first_difference(expected, masked(run.stdout, PROJECT_GOLDEN))

    assert run.exit_code == 0, run.stderr
    assert difference is None, difference


#: The ladder the doctor golden describes: the id each rung is settled under, the status the
#: golden's lines imply, and the reason the panel groups the unavailable rungs on. The printed
#: fragments are the ladder's own, so what is pinned here is the report and not the wording.
GOLDEN_LADDER: tuple[tuple[str, str, str], ...] = (
    ("panel.validity", "available", "measured on this machine, in part"),
    ("panel.soundness", "unavailable", "not in this build"),
    ("panel.reach", "available", "measured on this machine, in part"),
    ("panel.exploits", "unavailable", "not in this build"),
    ("panel.framing", "unavailable", "not in this build"),
    ("panel.signal", "unavailable", "not in this build"),
    ("panel.cost", "unavailable", "not in this build"),
    ("verb.trace", "unavailable", "not in this build"),
    ("verb.compare", "unavailable", "not in this build"),
    ("verb.forecast", "unavailable", "not in this build"),
    ("verb.improve", "unavailable", "not in this build"),
    ("seeker.api", "unavailable", "not in this build"),
    ("seeker.local", "unavailable", "not in this build"),
    ("calibration", "unavailable", "no reference material for this substrate yet"),
)

#: The panel's sections, in the order it prints the ones it has. The two lists are printed only
#: when they hold something, so a report is checked against the order of this and not its length.
DOCTOR_SECTIONS: tuple[str, ...] = (
    "  Install",
    "  Sandbox",
    "  What this machine can measure now",
    "  What it cannot, and what each would take",
)


def golden_capabilities():
    """The `Capabilities` the doctor golden describes, from fields alone: nothing is measured.

    The installation is the golden's own (no extras), and the probe is the one whose tier line the
    harness masks and whose ABI is the one at which a Landlock ruleset filters egress, which is the
    Network line the golden carries.
    """
    from reward_lens.api import Capabilities, Capability, InstallInfo
    from reward_lens.execution import SandboxProbe, TierResult

    tiers = ("T0", "L0", "L1", "L2", "L3")
    probe = SandboxProbe(
        os="linux",
        tier_held="L3",
        tiers={name: TierResult(held=True, reason=f"{name} held", ms=1.0) for name in tiers},
        kernel="0.0.0-golden",
        landlock_abi=4,
        platform_matrix={},
    )
    return Capabilities(
        capabilities=[
            Capability(id=rung, status=status, reason=reason)
            for rung, status, reason in GOLDEN_LADDER
        ],
        sandbox=probe,
        install=InstallInfo(version="3.1.0", python="3.11.13", platform="linux x86_64", extras=()),
    )


def test_doctor_renders_its_golden_through_render_text(monkeypatch):
    """The verb's panel is the product's `render_text`, so that is what is compared here.

    The report handed over is built, never measured. `doctor()` reports what the machine running
    the tests holds, and that moves: a development checkout carries the `trace` extra where a clean
    install carries none, and the extras are a line of this panel. A golden comparison against a
    moving report is evidence of the machine rather than of the renderer, so the machine is kept
    out of it and the live report is checked for shape by the test below.
    """
    # The module, not `access.doctor` the verb the package exports under that name.
    doctor = importlib.import_module("reward_lens.product.access.doctor")

    # The one line that is neither the argument's nor a constant of the build: whether this
    # installation can audit local graders is measured when the panel renders.
    monkeypatch.setattr(doctor, "_audits_local_graders", lambda: True)

    expected = golden_block("doctor.txt", "reward-lens doctor")
    rendered = doctor.render_text(golden_capabilities())
    difference = first_difference(expected, masked(rendered, "doctor.txt"))

    assert difference is None, difference


def test_the_live_doctor_report_prints_in_the_shape_the_golden_has():
    """What the machine reports is checked for shape and for nothing else.

    Every value here belongs to the machine: its tier, its extras, which rungs it holds. So this
    asserts the sections in order and the labels that introduce the fields, and reads none of the
    values that follow them. It is the half of the old comparison that survives a clean install.
    """
    from reward_lens import api
    from reward_lens.product.access.doctor import render_text

    lines = render_text(api.doctor()).split("\n")

    headings = [line for line in lines if line in DOCTOR_SECTIONS]
    assert headings == [name for name in DOCTOR_SECTIONS if name in headings], headings
    assert {"  Install", "  Sandbox"} <= set(headings), headings

    install = lines.index("  Install")
    assert lines[install + 1].startswith("    reward-lens ")
    assert "  ·  python " in lines[install + 1]
    assert lines[install + 2].startswith("    extras installed: ")
    assert "audit on local graders: " in lines[install + 2]

    sandbox = lines.index("  Sandbox")
    assert "probed in " in lines[sandbox + 1] and lines[sandbox + 1].endswith(" ms, held")
    assert lines[sandbox + 2].startswith("    per call: ")

    tail = [line for line in lines if line.startswith(("  Network", "  Cost"))]
    assert len(tail) == 2, tail
    assert tail[0].startswith("  Network   ") and tail[1].startswith("  Cost      $")


# --- the remedy table (A-015) -------------------------------------------------------------------


def remedy_lines(lines: list[str]) -> list[str]:
    """The `To measure the rest` block: its sentence and the rows under it, up to the blank line."""
    start = next(index for index, line in enumerate(lines) if "To measure the rest," in line)
    block = []
    for line in lines[start:]:
        if not line.strip():
            break
        block.append(line)
    return block


def test_the_remedy_table_prints_a_row_for_every_input_the_record_asks_for(grader):
    """Four needs, four rows, against the transcript that froze them.

    The record names its inputs the way the store writes them ("run record"), and the table prints
    them the way a reader says them ("a training run record"). Matched on the label, the fourth
    row asked whether "a training run record" was inside "run record" and dropped the row a record
    off a real run always asks for.
    """
    expected = remedy_lines(golden_block(GRADER_GOLDEN, "reward-lens audit"))
    record = fixtures.needs_record("task set", "sampled responses", "protected check", "run record")
    got = remedy_lines(masked(grader(record=record), GRADER_GOLDEN))
    difference = first_difference(expected, got)

    assert len(expected) == 5, f"the transcript's table is a sentence and four rows, not {expected}"
    assert difference is None, difference


def test_the_remedy_table_prints_only_what_is_asked_for_and_in_the_table_s_order(grader):
    """Three needs, stated in another order: three rows, in the transcript's order, and no fourth."""
    record = fixtures.needs_record("run record", "protected check", "task set")
    got = remedy_lines(grader(record=record).split("\n"))

    assert [line.strip().split("  ")[0] for line in got[1:]] == [
        "a task set",
        "a protected check",
        "a training run record",
    ]
    assert "sampled responses" not in "\n".join(got)


def test_the_sampled_responses_row_states_the_bank_the_record_counted(grader):
    """The number in that row is the record's, when the record has one: 128 is the transcript's."""
    counted = fixtures.needs_record("sampled responses", responses=512)
    uncounted = fixtures.needs_record("sampled responses")

    assert (
        remedy_lines(grader(record=counted).split("\n"))[1]
        .strip()
        .startswith("512 sampled responses")
    )
    assert (
        remedy_lines(grader(record=uncounted).split("\n"))[1]
        .strip()
        .startswith("128 sampled responses")
    )
