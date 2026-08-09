"""A-025: the `--format json` envelope is section 5.7's projection of the record.

The record is the archive and the envelope is what a machine reads to decide, so the two are not
the same document. What is settled here is where they part: the envelope drops the record's own
bookkeeping from `decision`, names the reward system the way the Subject line names it, and
carries each dark panel's absence as a code and a short remedy rather than as the paragraph the
transcript prints. Each test hands `envelope_from_record` a fixture record, because the question
is what the projection reads off a record and not what any one run happens to produce.
"""

from __future__ import annotations

import copy

import fixtures
from reward_lens.cli.format import text as formatter
from reward_lens.cli.verbs import _shared

EXT = formatter.TRANSCRIPT_EXTENSION


def envelope(record: dict) -> dict:
    return _shared.envelope_from_record(record, command="audit")


# --- decision -----------------------------------------------------------------------------------


def test_the_decision_is_the_four_fields_section_5_7_shows():
    """`tradeoff`, `signature` and `regression_cases` are the record's, not the envelope's."""
    record = fixtures.project_record()
    record["decision"] = {
        "state": "unresolved",
        "action": "train_on_this_version",
        "policy": None,
        "reasons": ["blocking_finding:RL0201", "required_missing:soundness"],
        "tradeoff": None,
        "signature": None,
        "regression_cases": [],
    }

    decision = envelope(record)["decision"]

    assert list(decision) == ["state", "action", "policy", "reasons"]
    assert decision["state"] == "unresolved"
    assert decision["action"] == "train_on_this_version"


def test_the_reasons_keep_the_order_the_record_wrote_them_in():
    """The order is the record's: the blocking findings, then the unqualified outcome, then the
    required sections in printed order. Re-sorting here would put the envelope at odds with the
    transcript a reader has in front of them."""
    record = fixtures.project_record()
    ordered = [
        "blocking_finding:RL0201",
        "outcome_unqualified:protected_test_suite",
        "required_missing:soundness",
        "required_missing:calibration",
    ]
    record["decision"] = {"state": "unresolved", "action": "hold", "policy": None, "reasons": ordered}

    assert envelope(record)["decision"]["reasons"] == ordered


def test_a_record_with_no_decision_leaves_the_envelope_null():
    record = fixtures.project_record()
    record["decision"] = None

    assert envelope(record)["decision"] is None


# --- subject ------------------------------------------------------------------------------------


def test_the_subject_names_the_system_and_the_version_the_way_the_subject_line_does():
    """`code-reward v1` on screen is `code-reward` and `v1` here, not the version id twice."""
    record = fixtures.project_record(name="code-reward", version_id="code-reward-v1")

    subject = envelope(record)["subject"]

    assert subject["reward_system"] == "code-reward"
    assert subject["version"] == "v1"
    assert subject["version_digest"] == fixtures.DIGEST
    assert subject["policy"] is None


def test_the_version_is_the_id_when_it_does_not_start_with_the_name():
    """A-015's rule, read through the envelope: only the `<name>-` prefix is stripped."""
    record = fixtures.project_record(name="code-reward", version_id="2026.09.1")

    assert envelope(record)["subject"]["version"] == "2026.09.1"


def test_the_task_set_is_the_file_the_run_was_pointed_at_and_what_it_hashed_to():
    record = fixtures.project_record(task_set=fixtures.DIGEST, task_file="tasks.jsonl")

    assert envelope(record)["subject"]["task_set"] == f"tasks.jsonl@{fixtures.DIGEST}"


def test_a_task_set_with_no_file_name_in_the_record_keeps_the_digest_alone():
    """The file name is read from the extension and from nowhere else; a record that carries no
    name gets none invented for it."""
    record = fixtures.project_record(task_set=fixtures.DIGEST, task_file=None)

    assert envelope(record)["subject"]["task_set"] == fixtures.DIGEST


def test_a_bare_grader_has_no_task_set():
    record = fixtures.grader_record()

    assert record["subject"]["context"]["task_set"] is None
    assert envelope(record)["subject"]["task_set"] is None


# --- holes --------------------------------------------------------------------------------------


def test_each_dark_panel_carries_the_absence_code_and_the_short_remedy():
    """One row per dark panel, `missing` as a code and `remedy` as the short form, both read off
    the entry the holes index names."""
    record = fixtures.project_record(codes=True)

    holes = envelope(record)["holes"]

    assert [row["section"] for row in holes] == [
        "soundness",
        "exploits",
        "framing",
        "signal",
        "reward_statistics",
        "trace",
        "forecast",
        "calibration",
    ]
    assert all(row["state"] == "NOT_MEASURED" for row in holes)
    assert [(row["missing"], row["remedy"]) for row in holes] == [
        fixtures.DARK_CODES[row["section"]] for row in holes
    ]


def test_a_section_whose_absence_names_an_input_carries_that_inputs_flag():
    """The short remedy for an input gap is the command that supplies it, not a later build."""
    record = fixtures.project_record(codes=True)

    holes = {row["section"]: row for row in envelope(record)["holes"]}

    assert holes["trace"]["missing"] == "run_record"
    assert holes["trace"]["remedy"] == "reward-lens trace <run>"
    assert holes["calibration"]["missing"] == "reference_material"


def test_an_older_record_says_what_it_says_and_no_code_is_supplied_for_it():
    """A record written before the codes existed carries neither on its absences. Its row falls
    back to the sentence the record does carry: the envelope reports that record, and a code it
    never wrote is not one to invent on its behalf."""
    record = fixtures.project_record(codes=False)
    assert not any("entry_id" in row for row in record["holes"])

    holes = {row["section"]: row for row in envelope(record)["holes"]}

    assert holes["soundness"]["missing"] == "not in this build"
    assert holes["trace"]["missing"] == "no run record supplied"
    assert holes["calibration"]["missing"] == "no reference material for this substrate"


def test_a_stamped_record_missing_only_the_short_remedy_still_falls_back_for_it():
    """The two are read one by one, so a half-stamped absence is not silently emptied."""
    record = fixtures.project_record(codes=True)
    entry = record["measurement"]["soundness"][0]
    del entry["extensions"][EXT]["remedy_short"]
    record["holes"][0]["remedy"] = "run a build whose soundness battery has landed"

    holes = {row["section"]: row for row in envelope(record)["holes"]}

    assert holes["soundness"]["missing"] == "instrument_not_in_this_build"
    assert holes["soundness"]["remedy"] == "run a build whose soundness battery has landed"


# --- findings and execution -----------------------------------------------------------------------


def test_the_findings_carry_the_keys_the_golden_orders_them_in():
    record = fixtures.project_record()

    findings = envelope(record)["findings"]

    assert findings, "the fixture record carries a blocking finding"
    assert list(findings[0]) == [
        "id",
        "rule",
        "level",
        "kind",
        "scope",
        "entry_kind",
        "arm",
        "witness",
    ]
    assert findings[0]["entry_kind"] == "check", "the entry the finding stands against"


def test_a_measuring_run_reuses_nothing():
    """A-016: `execution.reused` is null unless a run was served from an earlier one."""
    record = fixtures.project_record()

    assert envelope(record)["execution"]["reused"] is None


def test_the_projection_does_not_edit_the_record_it_reads():
    record = fixtures.project_record(codes=True)
    before = copy.deepcopy(record)

    envelope(record)

    assert record == before
