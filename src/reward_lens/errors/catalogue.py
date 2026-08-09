"""Every code this project allocates, with the cause and the remedies that go with it.

One table, read by the two-line renderer, by `explain`, and by the generator that writes the pages
under `docs/errors/`. Nothing else allocates a code: a packet that needs a new one asks for it here,
so that no two packets mint the same number.

A code's text is written for the person who hit it. It never names a file inside this project and
never points at a document: what it points at is a command.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

__all__ = [
    "CATALOGUE",
    "CONTRACTED_COMMANDS",
    "EXIT_MEANINGS",
    "ErrorSpec",
    "RANGES",
    "RANGE_EXCEPTIONS",
    "check_spec",
    "fill",
    "surface_for",
]


@dataclass(frozen=True)
class ErrorSpec:
    """One row of the catalogue.

    `cause` and each entry of `remedies` are templates: `{name}` is filled from the context a
    caller passes to `make`, and a name with nothing behind it renders as `<name>` rather than
    raising, because an error path is the worst place to meet a second error.

    That fallback is for the error path, not for reading. `explain` and the generated page are
    read with no context at all, so an entry whose template names a slot also carries the long
    form: the same sentence with the slot replaced by what the value would have said. Read those
    through `long_form_cause` and `long_form_remedies`, which fall back to the template for an
    entry that names no slot and so needs no second wording. `check_spec` refuses a long form
    that still carries a slot, which is what keeps `explain` free of them.

    `surface` is additive to the five frozen fields. Seven codes are findings or states rather
    than exit paths, and both `explain` and the generated pages have to say which they are.

    `verdicts` marks a code with two surfaces. Such a code is an error, carrying `exit_code`, only
    when what it reports was reward-lens's own; when the graded process was the one it happened to,
    nothing about reward-lens failed, and the outcome is one of `verdicts` recorded against that
    process with no process exit at all. `verdict_rule` is the sentence that picks one, and it
    reaches both `explain` and the generated page, because a reader who meets the code needs to
    know which of the two they are looking at.
    """

    code: str
    title: str
    cause: str
    remedies: list[str]
    exit_code: int
    surface: str = "error"
    verdicts: tuple[str, ...] = ()
    verdict_rule: str = ""
    long_cause: str = ""
    long_remedies: tuple[str, ...] = ()
    context_keys: tuple[str, ...] = field(default=(), compare=False)

    @property
    def long_form_cause(self) -> str:
        """The cause as a reader with no context meets it, on `explain` and on the page."""
        return self.long_cause or self.cause

    @property
    def long_form_remedies(self) -> tuple[str, ...]:
        """The remedies as a reader with no context meets them, in the order they are offered."""
        return self.long_remedies or tuple(self.remedies)


# The exit statuses, and what each one tells a caller. The hazard on 4 is worth stating every time:
# a shell usually reserves 2 for a usage mistake, and here 2 never means that.
EXIT_MEANINGS: dict[int, str] = {
    0: "reward-lens exits 0: the work completed, and any decision it was asked for qualified.",
    1: "reward-lens exits 1: the decision asked for was rejected by the policy it was put to.",
    2: (
        "reward-lens exits 2: the decision asked for is unresolved, because the evidence is "
        "missing or inconclusive. Rejected and unresolved never collapse into each other."
    ),
    3: (
        "reward-lens exits 3: a decision only you can make is pending, and there was no terminal "
        "to ask on. The payload names the flag and the command to run."
    ),
    4: (
        "reward-lens exits 4: usage, configuration or input contract. Nothing was measured and no "
        "record was written. Many shells use 2 for a usage mistake; here 2 never means that."
    ),
    5: (
        "reward-lens exits 5: a capability, service or extra the run needed is unavailable, so the "
        "measurement that rests on it cannot be made."
    ),
    6: "reward-lens exits 6: the money cap was reached. The record up to that point was written.",
    7: "reward-lens exits 7: reward-lens itself failed, or an integrity check did.",
    130: "reward-lens exits 130: the run was interrupted.",
}

# The bands, and the exit status each one fixes. `None` means the band mixes statuses on purpose.
RANGES: dict[str, tuple[str, int | None]] = {
    "00": ("usage", 4),
    "01": ("input contract", 4),
    "02": ("validity findings", None),
    "03": ("outcome", None),
    "04": ("execution", 5),
    "05": ("budget", 6),
    "06": ("record integrity", 4),
    "07": ("capability or extra unavailable", 5),
    "08": ("pending decision", 3),
    "09": ("internal", 7),
}

# Two codes sit outside the exit status their band fixes. Both were allocated by name, and the
# reason each one departs is written down here rather than left to be inferred.
RANGE_EXCEPTIONS: dict[str, str] = {
    "RL0130": (
        "allocated by name as the interrupt code, and 130 is the status a shell reports for one. "
        "The rest of the band carries input contract errors at exit 4."
    ),
    "RL0410": (
        "a breached execution limit is exit 7 when reward-lens broke the limit itself, and a "
        "verdict rather than an exit when the graded code did. The rest of the band carries "
        "missing capabilities at exit 5."
    ),
}

# Every command the root help lists. A remedy may name one of these and nothing else: a remedy
# pointing at a verb that does not ship is a lie the reader only discovers by typing it.
CONTRACTED_COMMANDS: frozenset[str] = frozenset(
    {
        "audit",
        "trace",
        "compare",
        "forecast",
        "improve",
        "open",
        "export",
        "import",
        "runs",
        "init",
        "doctor",
        "explain",
        "describe",
        "mcp",
    }
)

_RECORD = "to see what the record holds, run: reward-lens open <the record path>"


def _spec(
    code: str,
    title: str,
    cause: str,
    remedies: list[str],
    exit_code: int,
    surface: str = "error",
    verdicts: tuple[str, ...] = (),
    verdict_rule: str = "",
) -> ErrorSpec:
    return ErrorSpec(
        code=code,
        title=title,
        cause=cause,
        remedies=remedies,
        exit_code=exit_code,
        surface=surface,
        verdicts=verdicts,
        verdict_rule=verdict_rule,
    )


_SPECS: tuple[ErrorSpec, ...] = (
    # --- RL00xx, usage -------------------------------------------------------------------------
    _spec(
        "RL0001",
        "the invocation is not one this build can carry out",
        "reward-lens cannot carry out this invocation: {detail}",
        [
            "check the spelling of the command, its flags and its arguments, or run: "
            "reward-lens --help",
            "if the flag is real but the value is not, the command's own help lists what it "
            "takes: run: reward-lens <the command> --help",
            "if a run was meant, the ones on this machine are listed by: run: reward-lens runs",
        ],
        4,
    ),
    _spec(
        "RL0002",
        "an identifier you supplied is not usable",
        "the value given for {field} is not a usable identifier: {detail}",
        [
            "quote the value in the shell and drop any control character, null byte or `..` "
            "segment, then run: reward-lens <the same command> again",
            "if the id came from an earlier run, the real ones are listed by: run: reward-lens "
            "runs",
            _RECORD + ", which carries the ids it was written with",
        ],
        4,
    ),
    _spec(
        "RL0003",
        "a field in the project file is not valid",
        "the project file field {field} is not valid: {detail}",
        [
            "correct the field in rewardlens.yaml, then check the whole file at once: run: "
            "reward-lens doctor",
            "if you are unsure what the field takes, a fresh project ships every field with a "
            "working value: run: reward-lens init --example code_reward",
        ],
        4,
    ),
    _spec(
        "RL0004",
        "the output format asked for is not one this build writes",
        "{format} is not an output format reward-lens writes",
        [
            "the formats are text, json, jsonl, sarif and github: run: reward-lens audit . "
            "--format json",
            "to fix it for a whole session rather than per command, set REWARD_LENS_FORMAT in the "
            "environment",
        ],
        4,
    ),
    # --- RL01xx, input contract (and the interrupt, allocated by name) ---------------------------
    _spec(
        "RL0120",
        "the reach panel was given no task set",
        (
            "the reach panel runs the grader for {subject} against a task, and no task set was "
            "supplied"
        ),
        [
            "supply a task set: run: reward-lens audit . --tasks tasks.jsonl",
            "to read what the panel would have measured before you supply one: run: reward-lens "
            "explain RL0120",
            "if reach is not what this run should measure, leave the panel out rather than "
            "pointing it at an invented task, and the record carries its absence",
        ],
        4,
    ),
    _spec(
        "RL0130",
        "the run was interrupted",
        "the run stopped on an interrupt from the terminal, and the partial record was kept",
        [
            "pick it up where it stopped: run: reward-lens audit --resume {run_id}",
            "to see what was written before the interrupt: run: reward-lens runs",
        ],
        130,
    ),
    # --- RL02xx, validity findings ---------------------------------------------------------------
    _spec(
        "RL0201",
        "the grader reads a file the graded code can write",
        (
            "the grader reads {path} after the graded code has had its chance to write it, so a "
            "response can set its own score"
        ),
        [
            "read the file before the response runs, or keep it outside the graded working "
            "directory, then measure again: run: reward-lens audit .",
            "if the file has to be written by the response, hash it before and after and score "
            "the difference rather than the contents",
            _RECORD + ", which names the read and the write that pair up",
        ],
        2,
        surface="finding",
    ),
    _spec(
        "RL0202",
        "the answer is reachable from inside the graded process",
        (
            "the reference answer for {task} is readable from inside the graded process, so a "
            "response can copy it instead of earning the score"
        ),
        [
            "keep the reference outside the graded working directory and hand it to the grader in "
            "the task, then measure again: run: reward-lens audit .",
            "if the reference has to be on disk, put it behind the sandbox boundary rather than "
            "inside it",
            _RECORD + ", which names the path that carried it",
        ],
        2,
        surface="finding",
    ),
    _spec(
        "RL0203",
        "a branch of the grader never decides anything",
        (
            "the branch at {location} is either never reached or decides only on the task, so it "
            "cannot separate one response from another"
        ),
        [
            "delete the branch, or give it a condition that reads the response, then measure "
            "again: run: reward-lens audit .",
            "if the branch is meant for inputs you have not written tasks for yet, write one and "
            "the branch becomes reachable",
            _RECORD + ", which names the branch and what reached it",
        ],
        2,
        surface="finding",
    ),
    _spec(
        "RL0210",
        "the grader returns nothing on malformed input",
        (
            "the grader returned no score for {case}, so that response is unscored rather than "
            "scored badly"
        ),
        [
            "decide what a malformed response is worth and return that number, then measure "
            "again: run: reward-lens audit .",
            "if returning nothing is deliberate, find out what your trainer does with a missing "
            "score before you rely on it: some drop the sample and some read it as zero",
            _RECORD + ", which lists every case that came back unscored",
        ],
        2,
        surface="finding",
    ),
    _spec(
        "RL0211",
        "the grader raises on input it will meet",
        (
            "the grader raised on {case} instead of returning a score, so that response has no "
            "measurement at all"
        ),
        [
            "catch the malformed case inside the grader and return a score for it, then measure "
            "again: run: reward-lens audit .",
            "if the raise is guarding a real defect, it belongs before scoring starts, not in the "
            "middle of a batch",
            _RECORD + ", which carries the input that caused it",
        ],
        2,
        surface="finding",
    ),
    _spec(
        "RL0212",
        "the grader returns a boolean where a score belongs",
        (
            "the grader returned True or False for {case}, which collapses to 1.0 and 0.0 and "
            "hides every difference in between"
        ),
        [
            "return the number the boolean stands for, then measure again: run: reward-lens "
            "audit .",
            "if two levels are all you want, that is a decision worth stating rather than "
            "inferring: say so in rewardlens.yaml",
            _RECORD + ", which counts how many responses landed on each of the two values",
        ],
        2,
        surface="finding",
    ),
    _spec(
        "RL0213",
        "a reward function returned None and the trainer would make it NaN",
        "{entry} returned None for a row, and a trainer turns that into NaN for that row",
        [
            "return a number for every row, scoring 0.0 where the check does not apply, then "
            "measure again: run: reward-lens audit .",
            "None cannot mean no opinion here: the NaN spreads through the row's reward and that "
            "function stops counting, without anything being logged",
            _RECORD + ", which carries the row that came back None and the value it became",
        ],
        2,
        surface="finding",
    ),
    _spec(
        "RL0214",
        "a reward function raised and the framework would score it 0.0",
        (
            "{entry} raised {exception} while scoring a row, and the framework logs it and "
            "records 0.0 for that reward function"
        ),
        [
            "let the exception reach the caller, or catch it and return a score you chose on "
            "purpose, then measure again: run: reward-lens audit .",
            "a swallowed exception and an earned zero are the same number in the record, so a "
            "reward function that always raises reads as one that never rewards",
            _RECORD + ", which names the exception and the row it was raised on",
        ],
        2,
        surface="finding",
    ),
    _spec(
        "RL0215",
        "a reward function returned a boolean and float() accepted it",
        (
            "{entry} returned a bool for a row, and float() takes it, so 1.0 and 0.0 are recorded "
            "as though they were measured"
        ),
        [
            "return a float that says how much of the check was met, then measure again: run: "
            "reward-lens audit .",
            "if the check really is pass or fail, say so where the reward is declared, so a "
            "reader can tell two values apart from a scale that collapsed to two",
            _RECORD + ", which carries what was returned and the float it became",
        ],
        2,
        surface="finding",
    ),
    _spec(
        "RL0231",
        "the policy's context discloses the grader",
        "{origin} discloses the grader to the policy: {inference}",
        [
            "take the disclosure out of the context the policy sees, then measure again: run: "
            "reward-lens audit .",
            "if the policy has to be told how it is scored, say what the task rewards rather than "
            "how the grader checks it",
            _RECORD + ", which quotes the text and names where it was read",
        ],
        2,
        surface="finding",
    ),
    # --- RL03xx, outcome -------------------------------------------------------------------------
    _spec(
        "RL0301",
        "the correctness claim is not qualified by an outcome check",
        (
            "no outcome check stood behind the scores, so nothing here says a high score means "
            "correct work"
        ),
        [
            "point the run at a protected test suite: run: reward-lens audit . --outcome "
            "./outcome",
            "to see whether a suite is usable before you rely on it: run: reward-lens doctor "
            "--outcome ./outcome",
            "if there is no suite and there will not be one, the record says so as an absence, "
            "which is an honest answer rather than a missing one",
        ],
        2,
        surface="state",
    ),
    _spec(
        "RL0302",
        "this candidate set has already had its one acceptance round",
        (
            "the acceptance round for candidate set {candidate_set} on partition {partition_id} "
            "is exhausted: {reason}"
        ),
        [
            "seal a round against a fresh partition: run: reward-lens audit . --outcome "
            "<another protected directory>",
            "if the protocol itself changed, state the new one and seal a round under it, so the "
            "record says which protocol the second reading was taken under",
            "a second reading on the same material is a reading of a partition this candidate set "
            "has already been measured on, which is why it is refused rather than counted",
        ],
        4,
    ),
    _spec(
        "RL0341",
        "the outcome check collected no tests",
        (
            "the outcome check at {path} collected 0 tests, so it cannot qualify a correctness "
            "claim"
        ),
        [
            "check the path, or run: reward-lens doctor --outcome {path}",
            "if the suite collects from somewhere else, run it yourself first and pass the "
            "directory that worked: run: reward-lens audit . --outcome <that directory>",
            "an empty directory and a directory of helpers with no test in it look the same to a "
            "collector: check that the file names match the collector's rule",
        ],
        4,
    ),
    # --- RL04xx, execution -----------------------------------------------------------------------
    _spec(
        "RL0401",
        "no sandbox tier is available on this machine",
        "this machine offers no isolation tier reward-lens will run graded code in: {detail}",
        [
            "to see which tiers were probed and why each one failed: run: reward-lens doctor",
            "on a Debian or Ubuntu machine the strongest tier available without root comes from "
            "one package: run: sudo apt install bubblewrap",
            "in a container, the tier depends on what the container was started with, not on what "
            "is installed inside it",
        ],
        5,
    ),
    _spec(
        "RL0402",
        "the sandbox tier asked for is stronger than this machine holds",
        "the run required tier {required} and this machine holds {held}",
        [
            "to see what each tier needs and which one this machine holds: run: reward-lens "
            "doctor",
            "if the weaker tier is acceptable for this measurement, drop the requirement and let "
            "the record carry the tier that was used: run: reward-lens audit .",
        ],
        5,
    ),
    _spec(
        "RL0410",
        "an execution limit was breached",
        (
            "reward-lens reached its own {dimension} limit while making this measurement, so what "
            "stopped is reward-lens and not the work it was grading"
        ),
        [
            "to see the limits this build applies and the counters the run reached: run: "
            "reward-lens doctor",
            "raise the limit if the measurement genuinely needs the room, then measure again: "
            "run: reward-lens audit .",
            "when it was the graded work that hit a limit instead, nothing here failed and the "
            "record carries a verdict on that process: " + _RECORD,
        ],
        7,
        verdicts=("timeout", "resource_exhausted"),
        verdict_rule=(
            "RL0410 is an error, and exits 7, only when the limit breached was one of "
            "reward-lens's own dimensions. When the graded process was the one that hit the "
            "limit, nothing here failed: the record carries the verdict timeout for a time limit "
            "and resource_exhausted for any other dimension against that process, and no process "
            "exit belongs to it."
        ),
    ),
    # --- RL05xx, budget --------------------------------------------------------------------------
    _spec(
        "RL0501",
        "the money cap was reached",
        "the run reached its cap of {cap} and stopped, and the record up to that point was written",
        [
            "raise the cap and continue where it stopped: run: reward-lens audit --resume "
            "<the run id> --max-budget-usd <a higher cap>",
            "to see what was already paid for and kept: run: reward-lens runs",
            "to find out what a run will cost before paying for it, ask for the plan rather than "
            "the run: run: reward-lens audit . --dry-run",
        ],
        6,
    ),
    # --- RL06xx, record integrity ----------------------------------------------------------------
    _spec(
        "RL0601",
        "the record was written by a later version",
        (
            "the record declares schema major {major}, which this build cannot read, and the file "
            "was left exactly as it was"
        ),
        [
            "upgrade to a build that reads it: run: pip install --upgrade reward-lens",
            "to see which schema this build does read: run: reward-lens describe",
            "the file is not damaged and nothing was rewritten: an older build refusing a newer "
            "record is the contract working",
        ],
        4,
    ),
    _spec(
        "RL0602",
        "an entry does not carry the evidence its kind calls for",
        "entry {entry_id} is of kind {kind} but carries none of the evidence that kind requires",
        [
            _RECORD + ", which shows every entry that falls short",
            "add the evidence the kind calls for, or record the entry as an absence naming what "
            "is missing and why, which is what an honest gap looks like",
        ],
        4,
    ),
    _spec(
        "RL0603",
        "a number carries more precision than it was measured at",
        (
            "the value {value} for {field} carries more decimal places than the scale that field "
            "is declared at"
        ),
        [
            _RECORD + ", which names the field and the value that failed",
            "round once, where the number is recorded, not where it is read: a float computation "
            "carries digits the measurement never had",
        ],
        4,
    ),
    _spec(
        "RL0604",
        "the record does not match the schema",
        "the record is not valid at {location}: {detail}",
        [
            _RECORD + ", which is where the failing path lives",
            "a record this build wrote should never fail this check: keep the file if one did, "
            "because the file is the evidence",
        ],
        4,
    ),
    _spec(
        "RL0620",
        "something this record depends on has changed since it was written",
        "the record depends on {name} at digest {expected}, and what is on disk now is {actual}",
        [
            "measure again against what is on disk now: run: reward-lens audit .",
            "to see which entries still stand and which went stale: run: reward-lens compare "
            "<the earlier record> <the later record>",
        ],
        4,
    ),
    _spec(
        "RL0621",
        "no run by that name or id",
        "there is no run called {name} on this machine",
        [
            "list the runs this machine holds: run: reward-lens runs",
            "runs live under the state directory, so a run started with a different XDG_STATE_HOME "
            "is not visible from here",
        ],
        4,
    ),
    # --- RL07xx, capability or extra unavailable ---------------------------------------------------
    _spec(
        "RL0701",
        "the feature asked for is not in this build",
        (
            "{capability} is not installed in this environment, so the measurement that rests on "
            "it cannot be made"
        ),
        [
            "install the extra that carries it: run: pip install 'reward-lens[{extra}]'",
            "to see every capability, what it unlocks and what it costs: run: reward-lens doctor",
            "the rest of the run still has an answer: what could not be measured is recorded as "
            "an absence rather than left out",
        ],
        5,
    ),
    _spec(
        "RL0702",
        "the adapter cannot prove it holds a capability the run needs",
        (
            "the adapter for {family} does not prove {capability}, so a measurement resting on it "
            "would not be honest"
        ),
        [
            "to see what the adapter proved and what it only claims: run: reward-lens doctor",
            "run the measurements that do not need it and let the record carry the rest as "
            "absences: run: reward-lens audit .",
        ],
        5,
    ),
    _spec(
        "RL0703",
        "the verb is not in this build",
        "`{verb}` is not in this build, so this invocation cannot be carried out",
        [
            "the verbs this build carries are listed by: run: reward-lens --help (a verb not in "
            "this build is marked so on that list)",
            "a later release carries it; the report and the record you already have are unchanged "
            "by its absence",
            "what can be measured here, and what it would cost: run: reward-lens doctor",
        ],
        5,
    ),
    _spec(
        "RL0710",
        "the reward has a shape this build does not adapt",
        "{entry} has the {shape} shape, which reward-lens 3.1.0 does not adapt: {detail}",
        [
            "wrap it in a function that takes a prompt and a completion and returns a float, then "
            "point the project file's reward entry at the wrapper: run: reward-lens init . "
            "--detect",
            "the shapes this build does adapt, with a fixture for each: run: reward-lens init "
            "--example code-reward ./demo",
            "an adapter that half works is worse than a refusal: a guessed shape would score every "
            "row and say nothing about whether the score meant anything",
        ],
        5,
    ),
    # --- RL08xx, pending decision ------------------------------------------------------------------
    _spec(
        "RL0801",
        "a decision only you can make is waiting, and there is no terminal to ask on",
        "the run stopped on a decision only you can make ({decision}), with no terminal to ask on",
        [
            "make the decision on the command line and run it again: run: reward-lens audit . "
            "--policy <the policy to apply>",
            "in an automated job, set REWARD_LENS_NON_INTERACTIVE and pass every decision as a "
            "flag, so a run never waits on a terminal that is not there",
        ],
        3,
    ),
    # --- RL09xx, internal ---------------------------------------------------------------------------
    _spec(
        "RL0900",
        "reward-lens failed internally",
        (
            "reward-lens failed while {action}, which is a defect in reward-lens rather than in "
            "anything you gave it"
        ),
        [
            "check that the install is intact and the environment is what you think: run: "
            "reward-lens doctor",
            "the run directory keeps the log and whatever was written before the failure: run: "
            "reward-lens runs",
        ],
        7,
    ),
)

# The long form of every entry whose template names a `{name}` slot. `explain` and the generated
# page are read by someone who has not hit the error yet, so there is no context to fill a slot
# from and nothing sensible to show in its place; each entry here says the same thing with the
# slot replaced by what the value would have said. The first element rewords the cause, or is
# empty where the cause names no slot; the second rewords remedies by position, and carries only
# the ones that change. A code with no slot anywhere is absent, and `check_spec` fails any entry
# whose long form still carries a slot, so a new code cannot quietly reach a reader with one.
_LONG_FORMS: dict[str, tuple[str, dict[int, str]]] = {
    "RL0001": (
        "reward-lens cannot carry out this invocation, and the message names the part of it that "
        "could not be made sense of",
        {},
    ),
    "RL0002": (
        "the value given for the field the message names is not a usable identifier, and the "
        "message says what makes it one reward-lens will not take",
        {},
    ),
    "RL0003": (
        "the project file field the message names is not valid, and the message says what is "
        "wrong with it",
        {},
    ),
    "RL0004": ("the output format asked for is not one reward-lens writes", {}),
    "RL0130": (
        "",
        {0: "pick it up where it stopped: run: reward-lens audit --resume <the run id>"},
    ),
    "RL0120": (
        "the reach panel runs the grader for the subject the message names against a task, and no "
        "task set was supplied",
        {},
    ),
    "RL0201": (
        "the grader reads the file the finding names after the graded code has had its chance to "
        "write it, so a response can set its own score",
        {},
    ),
    "RL0202": (
        "the reference answer for the task the finding names is readable from inside the graded "
        "process, so a response can copy it instead of earning the score",
        {},
    ),
    "RL0203": (
        "the branch the finding names is either never reached or decides only on the task, so it "
        "cannot separate one response from another",
        {},
    ),
    "RL0210": (
        "the grader returned no score for the case the finding names, so that response is "
        "unscored rather than scored badly",
        {},
    ),
    "RL0211": (
        "the grader raised on the case the finding names instead of returning a score, so that "
        "response has no measurement at all",
        {},
    ),
    "RL0212": (
        "the grader returned True or False for the case the finding names, which collapses to "
        "1.0 and 0.0 and hides every difference in between",
        {},
    ),
    "RL0213": (
        "a reward function returned None for a row, and a trainer turns that into NaN for that row",
        {},
    ),
    "RL0214": (
        "a reward function raised the exception the message names while scoring a row, and the "
        "framework logs it and records 0.0 for that reward function",
        {},
    ),
    "RL0215": (
        "a reward function returned a bool for a row, and float() takes it, so 1.0 and 0.0 are "
        "recorded as though they were measured",
        {},
    ),
    "RL0231": (
        "the context the policy sees discloses the grader, and the finding quotes the disclosure "
        "and names the part of the context it was read from",
        {},
    ),
    "RL0302": (
        "the one acceptance round this candidate set was sealed for on this partition has already "
        "been used, or the request carried a nonce, candidate digest or protocol digest that "
        "matches no sealed round",
        {},
    ),
    "RL0341": (
        "the outcome check at the path the message names collected 0 tests, so it cannot qualify "
        "a correctness claim",
        {0: "check the path, or run: reward-lens doctor --outcome <the outcome directory>"},
    ),
    "RL0401": (
        "this machine offers no isolation tier reward-lens will run graded code in, and the "
        "message names what each tier it probed was missing",
        {},
    ),
    "RL0402": (
        "the run required a stronger isolation tier than this machine holds, and the message "
        "names both of them",
        {},
    ),
    "RL0410": (
        "reward-lens reached one of its own limits while making this measurement, so what stopped "
        "is reward-lens and not the work it was grading",
        {},
    ),
    "RL0501": (
        "the run reached the money cap it was given and stopped, and the record up to that point "
        "was written",
        {},
    ),
    "RL0601": (
        "the record declares a schema major this build cannot read, and the file was left exactly "
        "as it was",
        {},
    ),
    "RL0602": (
        "the entry the message names carries none of the evidence its kind requires",
        {},
    ),
    "RL0603": (
        "the value recorded for the field the message names carries more decimal places than the "
        "scale that field is declared at",
        {},
    ),
    "RL0604": (
        "the record is not valid at the location the message names, and the message says what "
        "failed there",
        {},
    ),
    "RL0620": (
        "the record depends on something at the digest it recorded, and what is on disk now "
        "hashes to a different one",
        {},
    ),
    "RL0621": ("there is no run by that name or id on this machine", {}),
    "RL0701": (
        "the capability the message names is not installed in this environment, so the "
        "measurement that rests on it cannot be made",
        {0: "install the extra that carries it: run: pip install 'reward-lens[<the extra>]'"},
    ),
    "RL0702": (
        "the adapter the run was given does not prove the capability the measurement needs, so a "
        "measurement resting on it would not be honest",
        {},
    ),
    "RL0703": (
        "the verb the message names is not in this build, so this invocation cannot be carried out",
        {},
    ),
    "RL0710": (
        "the reward entry the message names has a shape reward-lens 3.1.0 does not adapt, and the "
        "message says which shape was found and what about it could not be adapted",
        {},
    ),
    "RL0801": (
        "the run stopped on a decision only you can make, which the message names, with no "
        "terminal to ask on",
        {},
    ),
    "RL0900": (
        "reward-lens failed at the step the message names, which is a defect in reward-lens "
        "rather than in anything you gave it",
        {},
    ),
}


def _with_long_form(spec: ErrorSpec) -> ErrorSpec:
    """One row with its long form attached, or unchanged where its template names no slot."""
    entry = _LONG_FORMS.get(spec.code)
    if entry is None:
        return spec
    cause, remedies = entry
    return replace(
        spec,
        long_cause=cause or spec.cause,
        long_remedies=tuple(
            remedies.get(index, remedy) for index, remedy in enumerate(spec.remedies)
        ),
    )


CATALOGUE: dict[str, ErrorSpec] = {spec.code: _with_long_form(spec) for spec in _SPECS}

# A `{name}` slot, as a template writes it. The long form may carry none.
_SLOT = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")


def fill(template: str, context: dict[str, Any], *, placeholder: str = "<{key}>") -> str:
    """Fill a template's `{name}` slots from `context`, leaving the rest visible.

    A name with nothing behind it becomes `<name>` rather than raising. An error path is the worst
    place to meet a second error, and a reader who sees `<path>` knows exactly what is missing.
    """

    class _Missing(dict):  # type: ignore[type-arg]
        def __missing__(self, key: str) -> str:
            return placeholder.format(key=key)

    return template.format_map(_Missing(context))


def surface_for(code: str, *, internal: bool) -> tuple[str, int | None]:
    """Which surface a code takes on this occasion, and the exit status that goes with it.

    `internal` says the thing the code reports was reward-lens's own: its own limit, its own
    dimension. A code with no second surface is an error either way, because a caller cannot talk
    reward-lens out of its own exit status. A code that has one is an error only when `internal`,
    and otherwise a verdict the record carries against the graded process, which is not an exit:
    the second element is then `None`, and there is nothing for a caller to exit with.

    Raises `KeyError` for a code this build does not hold, because a caller asking for the surface
    of an unknown code has a bug rather than a bad input.
    """
    spec = CATALOGUE[code]
    if internal or not spec.verdicts:
        return ("error", spec.exit_code)
    return ("verdict", None)


def check_spec(spec: ErrorSpec) -> list[str]:
    """Everything wrong with one entry, as plain sentences. An empty list means it is usable."""
    problems: list[str] = []

    if not (len(spec.code) == 6 and spec.code.startswith("RL") and spec.code[2:].isdigit()):
        problems.append(f"{spec.code}: the code is not RL followed by four digits")
    for name, text in (("title", spec.title), ("cause", spec.cause)):
        if not text.strip():
            problems.append(f"{spec.code}: the {name} is empty")
        elif "\n" in text:
            problems.append(f"{spec.code}: the {name} runs to more than one line")

    if not spec.remedies:
        problems.append(f"{spec.code}: there is no remedy")
    for index, remedy in enumerate(spec.remedies):
        if not remedy.strip():
            problems.append(f"{spec.code}: remedy {index} is blank")
        if "\n" in remedy:
            problems.append(f"{spec.code}: remedy {index} runs to more than one line")
    if spec.remedies and not any("run: " in remedy for remedy in spec.remedies):
        problems.append(f"{spec.code}: no remedy carries a command to run")

    if spec.long_remedies and len(spec.long_remedies) != len(spec.remedies):
        problems.append(
            f"{spec.code}: the long form offers {len(spec.long_remedies)} remedies and the "
            f"template offers {len(spec.remedies)}"
        )
    long_form = [("cause", spec.long_form_cause)]
    long_form += [(f"remedy {i}", text) for i, text in enumerate(spec.long_form_remedies)]
    for name, text in long_form:
        slot = _SLOT.search(text)
        if slot is not None:
            problems.append(
                f"{spec.code}: the long form of the {name} still carries {slot.group(0)}, and "
                "explain is read with no context to fill it from"
            )

    if spec.exit_code not in EXIT_MEANINGS:
        problems.append(f"{spec.code}: exit {spec.exit_code} is not in the table")
    if spec.surface not in {"error", "finding", "state"}:
        problems.append(f"{spec.code}: {spec.surface} is not a surface")

    if bool(spec.verdicts) != bool(spec.verdict_rule.strip()):
        problems.append(
            f"{spec.code}: a second surface needs both the verdicts it lands as and the rule that "
            "picks between them"
        )
    if spec.verdicts and spec.surface != "error":
        problems.append(f"{spec.code}: only a code whose first surface is an error takes verdicts")
    for verdict in spec.verdicts:
        if verdict not in spec.verdict_rule:
            problems.append(f"{spec.code}: the rule does not say when it lands as {verdict}")

    return problems
