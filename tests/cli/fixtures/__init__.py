"""Records that carry every field the text surface reads, and nothing the surface invents.

A-015 freezes the wave-1 transcripts as the contract for the text surface, and A-016 and A-019
settle where each sentence in them lives: the record carries the wording, the extension at
`extensions[TRANSCRIPT_EXTENSION]` carries the clauses the schema's closed `subject` cannot, and
the formatter only chooses order, glyph and column. These builders are that record. Every sentence
below is a field a real run writes, so a test can hand one of these to `format.text.render` and
compare the result against the golden line for line without a pipeline, an example on disk or a
network call.

Two shapes, because the two transcripts are two shapes: `project_record` is an audit with a task
set and `grader_record` is the bare grader, whose `subject.context.task_set` is null (A-019).
Every number a sentence states is a keyword argument, so a second record with different numbers
can be built from the same layout. That is what shows the layout reads the record: the formatter
holds no number of its own, and the counts it does compute (`4 of 5 checks passed`, the absence
tallies, the verdict's two counts) follow the entries and reasons the record carries.
"""

from __future__ import annotations

from reward_lens.cli.format.text import REPLAY_ENTRY, TRANSCRIPT_EXTENSION

EXT = TRANSCRIPT_EXTENSION

#: A record's digest is volatile (`fleet/golden/VOLATILE.md`): its shape is checked, its hex is
#: not, so a fixture may carry any well-formed one.
DIGEST = "sha256:" + "1f" * 32


class Plan:
    """What `_plan_block` reads off a plan: the panel list's length, and two numbers."""

    def __init__(self, panels=("validity", "reach"), paid_calls=0, estimate_s=2):
        self.panels = list(panels)
        self.paid_calls = paid_calls
        self.estimate_s = estimate_s


def check(entry_id, section, *, passed=True, **fields) -> dict:
    """A measured check entry. `check.passed` is what the panel's glyph reads."""
    entry = {"entry_id": entry_id, "section": section, "kind": "check", "check": {"passed": passed}}
    for key in ("predicate", "scope_tested"):
        if key in fields:
            entry["check"][key] = fields.pop(key)
    entry.update(fields)
    return entry


def absence(entry_id, section, *, missing_access=None, needs=None, **fields) -> dict:
    """An absence entry: what it lacked, and under `needs` the input that would supply it."""
    entry = {"entry_id": entry_id, "section": section, "kind": "absence", "absence": {}}
    if missing_access:
        entry["absence"]["missing_access"] = missing_access
    if needs:
        entry["extensions"] = {EXT: {"needs": needs}}
    entry.update(fields)
    return entry


def finding(finding_id, *, level, message, code=None, entries=(), rule=None) -> dict:
    out = {"id": finding_id, "level": level, "message": message, "entries": list(entries)}
    if code:
        out["code"] = code
    if rule:
        out["rule"] = rule
    return out


def _plural(count, noun) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


# --- the project audit --------------------------------------------------------------------------


#: A-025 point 3: what each dark section's absence is missing, as a code, and the short remedy.
#: A record written before that addendum carries neither, which is what `codes=False` models.
DARK_CODES = {
    "soundness": ("instrument_not_in_this_build", "a later build"),
    "exploits": ("instrument_not_in_this_build", "a later build"),
    "framing": ("instrument_not_in_this_build", "a later build"),
    "signal": ("instrument_not_in_this_build", "a later build"),
    "reward_statistics": ("instrument_not_in_this_build", "a later build"),
    "trace": ("run_record", "reward-lens trace <run>"),
    "forecast": ("run_record", "reward-lens trace <run>"),
    "calibration": ("reference_material", "none for this substrate yet"),
}


def project_record(
    *,
    line: int = 91,
    checks: int = 5,
    failed: int = 1,
    not_in_build: int = 7,
    exposures: int = 1,
    tasks: int = 40,
    responses: int = 120,
    wall_s: int = 31,
    name: str = "code-reward",
    version_id: str = "code-reward-v1",
    grader: str = "grader.py",
    task_set: str = "tasks.jsonl",
    task_file: str | None = None,
    codes: bool = False,
) -> dict:
    """An audit with a task set: the project layout, with a blocking finding and a dark tail.

    The numbers are arguments and not constants because the formatter must be shown reading them:
    `checks` and `failed` move the validity counts, `not_in_build` moves the absence tally,
    `exposures` and `line` are stated inside sentences the record carries, and `tasks` and
    `responses` are the Subject clauses the extension holds.
    """
    culprit = "validity.outcome_isolation"
    entries = [check(f"validity.check_{index}", "validity") for index in range(checks - failed)]
    entries.append(
        check(
            culprit,
            "validity",
            passed=False,
            result={
                "headline": "This reward reads a file the graded process can write.",
                "explanation": (
                    f"{grader} line {line} runs the tests in outcome/, and outcome/ sits inside "
                    f"the directory the graded response runs in. Whether any of the {responses} "
                    "responses used that path is not established here: the executed witness is "
                    "not in this build."
                ),
            },
        )
    )
    entries.extend(
        check(f"validity.failed_{index}", "validity", passed=False) for index in range(failed - 1)
    )
    entries.extend(
        absence(f"validity.absent_{index}", "validity", missing_access="not in this build")
        for index in range(not_in_build)
    )
    reach = [
        check(
            "reach.static_surfaces",
            "reach",
            limitations=["static analysis only"],
            result={
                "summary": f"{_plural(exposures, 'surface')} exposed, by static analysis only",
                "exposure_id": "REACH-EDIT-TESTS",
                "note": "no executed witness: not in this build",
            },
        )
    ]
    dark = {
        "soundness": "not in this build",
        "exploits": "not in this build",
        "framing": "not in this build",
        "signal": "not in this build",
        "reward_statistics": "not in this build",
        "trace": "no run record supplied",
        "forecast": "no run record to forecast from",
        "calibration": "no reference material for this substrate",
    }
    holes = [
        {"section": section, "state": "NOT_MEASURED", "missing_access": why}
        for section, why in dark.items()
    ]
    measurement = {"validity": entries, "reach": reach}
    named = {"grader": grader, "tasks": tasks, "responses": responses}
    if task_file:
        named["task_file"] = task_file
    if codes:
        # The index names an entry and the entry carries the code, so a record that has the codes
        # has the absences they sit on: a hole row pointing at nothing is not a shape that exists.
        for row in holes:
            code, remedy = DARK_CODES[row["section"]]
            row["entry_id"] = f"{row['section']}.instrument_absent"
            measurement[row["section"]] = [
                absence(
                    row["entry_id"],
                    row["section"],
                    missing_access=row["missing_access"],
                    extensions={EXT: {"missing_code": code, "remedy_short": remedy}},
                )
            ]
    return {
        "subject": {
            "reward_system": {"id": "rs-code-reward", "name": name},
            "version": {"id": version_id, "digest": DIGEST},
            "context": {"configuration": "as committed", "policy": None, "task_set": task_set},
        },
        "extensions": {EXT: {"subject": named}},
        "measurement": measurement,
        "holes": holes,
        "findings": [
            finding(
                "RL0201",
                level="error",
                code="RL0201",
                rule="validity.outcome_isolation",
                entries=[culprit],
                message=(
                    f"{grader} line {line} reads outcome/test_solution.py, which the graded "
                    "process can write"
                ),
            )
        ],
        "decision": {
            "state": "unresolved",
            "action": "hold",
            "reasons": [
                "outcome_unqualified:validity.outcome_isolation",
                "blocking_finding:RL0201",
                # Every dark section, calibration included: the run that froze the transcript held
                # no reference material for this substrate either, so the verdict counts eight.
                *[f"required_missing:{section}" for section in dark],
            ],
        },
        "cost": {"wall_s": wall_s, "api_calls": 0, "usd": "0.00"},
    }


# --- the bare grader audit ----------------------------------------------------------------------


def grader_record(
    *,
    checks: int = 5,
    failed: int = 1,
    not_in_build: int = 4,
    task_set_checks: int = 3,
    response_bank_checks: int = 0,
    repeats: int = 100,
    wall_s: int = 19,
    grader: str = "my_grader.py",
    signature: str = "score(task, response)",
    shape: str = "plain shape",
) -> dict:
    """An audit with no task set: the grader layout, with the two sentences and the remedy table.

    `subject.context.task_set` is null, which is the whole of what makes this the grader shape
    (A-019). `repeats` is stated inside the replay entry's own summary, and the absence groups are
    what the validity counts read.
    """
    culprit = "validity.malformed_input"
    entries = [check(REPLAY_ENTRY, "validity", scope_tested=f"{repeats} repeats")]
    entries[0]["result"] = {
        "summary": f"{repeats} repeats, identical output, no hidden state",
        "n_nondeterministic": 0,
    }
    entries.extend(check(f"validity.check_{index}", "validity") for index in range(checks - failed - 1))
    entries.append(check(culprit, "validity", passed=False))
    entries.extend(
        check(f"validity.failed_{index}", "validity", passed=False) for index in range(failed - 1)
    )
    entries.extend(
        absence(f"validity.absent_{index}", "validity", missing_access="not in this build")
        for index in range(not_in_build)
    )
    entries.extend(
        absence(f"validity.needs_tasks_{index}", "validity", needs="a task set")
        for index in range(task_set_checks)
    )
    # A validity absence the task set would not supply: the remedy table's to state, not the
    # counts line's, so a record can carry one and the line has to stay the same length.
    entries.extend(
        absence(f"validity.needs_bank_{index}", "validity", needs="a response bank")
        for index in range(response_bank_checks)
    )
    measurement = {
        "validity": entries,
        "soundness": [absence("soundness.protected", "soundness", needs="a protected check")],
        "signal": [absence("signal.spread", "signal", needs="128 sampled responses")],
        "trace": [absence("trace.join", "trace", needs="a training run record")],
    }
    dark = {
        "soundness": "no known-good or known-wrong solutions supplied",
        "reach": "no task set, so nothing to reach into",
        "exploits": "needs a task set, or a response bank",
        "signal": "needs sampled responses grouped by prompt",
    }
    return {
        "subject": {
            "reward_system": {"id": "rs-my-grader", "name": grader},
            "version": {"id": "my_grader-v1", "digest": DIGEST},
            "context": {"configuration": "as committed", "policy": None, "task_set": None},
        },
        "extensions": {
            EXT: {
                "subject": {"grader": grader, "signature": signature, "shape": shape},
                "could_establish": (
                    "the grader is deterministic and replayable, and it has one input-handling "
                    "defect."
                ),
                "could_not": "everything that needs tasks or responses.",
            }
        },
        "measurement": measurement,
        "holes": [{"section": section, "missing_access": why} for section, why in dark.items()],
        "findings": [
            finding(
                "RL0210",
                level="error",
                code="RL0210",
                rule="validity.malformed_input",
                entries=[culprit],
                message=(
                    "the grader returns None on malformed input, and no trainer is declared "
                    "under `reward.trainer`, so what becomes of that value in training is not "
                    "known here"
                ),
            )
        ],
        "decision": {
            "state": "unresolved",
            "action": "hold",
            "reasons": ["outcome_unqualified:validity.protected_check"],
        },
        "cost": {"wall_s": wall_s, "api_calls": 0, "usd": "0.00"},
    }


#: Which section of a record states each input the remedy table can be asked for, in the words the
#: store writes into `needs` (`product/audit/absences.py`: TASK_SET, SAMPLED_RESPONSES,
#: PROTECTED_CHECK, RUN_RECORD). `grader_record` states all four in the older, longer wording a
#: reader would use; a record off a real run states them this way.
NEED_SECTIONS = {
    "task set": "validity",
    "sampled responses": "signal",
    "protected check": "soundness",
    "run record": "trace",
}


def needs_record(*needs: str, responses: int | None = None, **over) -> dict:
    """The bare grader, restated so its absences ask for exactly `needs` and in the store's words.

    An absence whose input is not listed keeps its entry and drops its `needs` note, which is what
    a record does when that input arrived: the panel line stays and the remedy row goes.
    `responses` is the count of a response bank the record had, stated where the subject clauses
    are, for the row that asks for sampled responses.
    """
    record = grader_record(**over)
    wanted = {NEED_SECTIONS[need]: need for need in needs}
    for section, entries in record["measurement"].items():
        for entry in entries:
            if not (entry.get("extensions") or {}).get(EXT, {}).get("needs"):
                continue
            if section in wanted:
                entry["extensions"][EXT]["needs"] = wanted[section]
            else:
                entry.pop("extensions")
    if responses is not None:
        record["extensions"][EXT]["subject"]["responses"] = responses
    return record
