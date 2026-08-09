"""A-019: what the text surface prints is read off the record, and never off the path.

Three things are settled here. The layout follows `subject.context.task_set`, so these tests hand
the renderer a record and never a path. The Subject clauses the schema's closed `subject` cannot
carry are read from `extensions[TRANSCRIPT_EXTENSION].subject` and from nowhere else. And the
`replay` panel, whose entry is the one the bare-grader transcript gives a line of its own, has a
detail in every state its entry can be in.
"""

from __future__ import annotations

import pytest

from reward_lens.cli.format import text as formatter

EXT = formatter.TRANSCRIPT_EXTENSION


def record(*, task_set=None, named=None, entries=None, holes=None, findings=None) -> dict:
    """A record with the fields the renderer reads and nothing else."""
    return {
        "subject": {
            "reward_system": {"id": "rs-1", "name": "demo"},
            "version": {"id": "demo-3", "digest": "sha256:" + "a" * 64},
            "context": {"configuration": "as committed", "policy": None, "task_set": task_set},
        },
        "extensions": {EXT: {"subject": named} if named is not None else {}},
        "measurement": {"validity": list(entries or [])},
        "holes": list(holes or []),
        "findings": list(findings or []),
        "decision": {"state": "unresolved", "reasons": ["required_missing:reach"]},
        "cost": {"wall_s": 1, "api_calls": 0, "usd": "0.00"},
    }


def replay_entry(**fields) -> dict:
    return dict({"entry_id": formatter.REPLAY_ENTRY, "section": "validity", "kind": "check"}, **fields)


def replay_line(doc: dict) -> str:
    for line in formatter.render(doc, shape="grader").splitlines():
        if line.strip().startswith(tuple(formatter.GLYPHS)) and " replay " in line:
            return line.strip()
    raise AssertionError("no replay panel line was printed")


# --- the layout is keyed on the record (A-019) --------------------------------------------------


def test_a_record_with_no_task_set_is_the_bare_grader_layout():
    assert formatter.shape_for(record(task_set=None)) == "grader"
    assert formatter.shape_for(record(task_set="tasks.jsonl")) == "project"


def test_the_four_wordings_follow_the_record_and_no_path_is_consulted():
    """The two transcripts differ in these four, and this is the field that separates them.

    Neither call names a file or a directory, and nothing here exists on disk. A renderer that
    keyed on the path could not tell these two records apart at all.
    """
    bare = formatter.render(record(task_set=None))
    project = formatter.render(record(task_set="tasks.jsonl"))

    assert formatter._NO_CHECK in bare
    assert formatter._UNRESOLVED not in bare
    assert formatter._REPORT_NOTE not in bare and formatter._NEXT not in bare

    assert formatter._UNRESOLVED in project
    assert formatter._NO_CHECK not in project
    assert "0 blocking findings, and 1 required panel could not run." in project


def test_a_directory_is_not_what_makes_a_project(tmp_path):
    """The path and the record disagree on purpose: a directory audited with no task set.

    This is the case attempt 4 got wrong. The old rule read the path, so it printed the project's
    counts sentence over a record that has no counts to state.
    """
    rendered = formatter.render(record(task_set=None), cwd=str(tmp_path))

    assert formatter._NO_CHECK in rendered
    assert formatter._UNRESOLVED not in rendered


# --- the Subject clauses come from the extension (A-019) ----------------------------------------


def test_the_project_subject_clauses_are_read_from_the_extension():
    subject = subject_line(record(task_set="tasks.jsonl", named={"grader": "grader.py", "tasks": 12, "responses": 128}))

    assert subject == "  Subject   demo 3  ·  grader.py  ·  12 tasks  ·  128 responses"


def test_the_bare_grader_subject_clauses_are_read_from_the_extension():
    named = {"grader": "grader", "signature": "score(task, response)", "shape": "plain shape"}
    subject = subject_line(record(task_set=None, named=named))

    assert subject == "  Subject   grader  ·  `score(task, response)`, plain shape  ·  no task set supplied"


def test_a_record_without_the_extension_subject_drops_the_clauses():
    """Not a substitute from somewhere else: the clause is the record's or it is not printed."""
    project = subject_line(record(task_set="tasks.jsonl"))
    bare = subject_line(record(task_set=None))

    assert project == "  Subject   demo 3"
    assert bare == "  Subject   demo  ·  no task set supplied"


def test_the_schema_subject_is_not_read_for_those_clauses():
    """A record that carries the old names on the closed `subject` prints none of them."""
    doc = record(task_set="tasks.jsonl")
    doc["subject"]["reward_system"]["entry"] = "grader.py"
    doc["subject"]["context"]["tasks"] = 12
    doc["subject"]["context"]["responses"] = 128

    assert subject_line(doc) == "  Subject   demo 3"


def subject_line(doc: dict) -> str:
    for line in formatter.render(doc).splitlines():
        if line.startswith("  Subject"):
            return line
    raise AssertionError("no Subject line was printed")


# --- the replay detail, in every state its entry can be in --------------------------------------


def test_an_absent_replay_entry_prints_the_reason_it_names():
    entry = replay_entry(kind="absence", absence={"missing_access": "no rollouts to replay"})

    assert replay_line(record(entries=[entry])) == "○ replay        not measured: no rollouts to replay"


def test_an_absent_replay_entry_with_no_reason_falls_back_to_its_section_hole():
    entry = replay_entry(kind="absence", absence={})
    holes = [{"section": "validity", "missing_access": "no response bank"}]

    assert replay_line(record(entries=[entry], holes=holes)) == "○ replay        not measured: no response bank"


def test_a_measured_replay_entry_prints_the_summary_the_record_carries():
    entry = replay_entry(
        check={"passed": True, "scope_tested": "8 inputs, each graded 2 times"},
        result={"summary": "100 repeats, 0 disagreed", "n_nondeterministic": 0},
    )

    assert replay_line(record(entries=[entry])) == "✔ replay        100 repeats, 0 disagreed"


def test_a_measured_replay_entry_with_no_summary_falls_back_to_its_own_fields():
    """The defect attempt 4 found: `validity.replay_determinism` is present, and the instrument
    writes `n_attempted`, `repeats` and the two counts but no `summary`, so the line was empty."""
    entry = replay_entry(
        check={"passed": True, "scope_tested": "8 inputs graded twice"},
        result={"n_attempted": 8, "repeats": 2, "n_nondeterministic": 0, "n_unreplayable": 0},
    )

    assert replay_line(record(entries=[entry])) == (
        "✔ replay        8 inputs graded twice, 0 disagreed, 0 could not be replayed"
    )


def test_a_replay_entry_with_neither_falls_back_to_the_predicate_it_asserts():
    entry = replay_entry(check={"passed": True, "predicate": "every input re-graded yields the same score"})

    assert replay_line(record(entries=[entry])) == (
        "✔ replay        every input re-graded yields the same score"
    )


@pytest.mark.parametrize("entries", [[], [replay_entry(kind="absence", absence={})]])
def test_the_replay_panel_never_prints_an_empty_detail(entries):
    assert replay_line(record(entries=entries)).split(maxsplit=2)[2:] != []
