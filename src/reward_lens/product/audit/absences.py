"""What this build does not measure, said out loud, once per panel and once per missing check.

D-18: a hole is a first-class state with three values, each naming the missing access and the
remedy. Nothing here is a zero and nothing here is `inconclusive`. The catalogue is data so that a
wave-2 packet lands its panel and deletes exactly one row.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

from reward_lens import contracts
from reward_lens.instruments.base import RunContext

__all__ = [
    "DYNAMIC_CHECKS",
    "INPUTS",
    "PANEL_ABSENCES",
    "PANEL_NEEDS",
    "PROTECTED_CHECK",
    "RUN_RECORD",
    "SAMPLED_RESPONSES",
    "TASK_SET",
    "TASK_SET_CHECKS",
    "TRANSCRIPT_EXTENSION",
    "classify",
    "could_establish",
    "could_not",
    "could_not_check",
    "import_diagnosis",
    "needs_note",
    "panel_absences",
    "stamp_codes",
    "validity_absences",
]

#: The one string the fleet agreed on for "this build ships no instrument for this". A-015 freezes
#: it as the transcript's reason sentence, which is what the surface prints after "not measured: ".
NOT_IN_BUILD = "not in this build"

#: The extension key the transcript fields live under. A-015 calls it `extensions.transcript`, and
#: the frozen schema will not admit that name: `extensions` carries `propertyNames` on
#: `^https?://[^\s]+$` at both the pydantic and the JSON-schema layer, so a bare key fails
#: `validate_record`. This is the same namespace spelled the way the contract allows. The handoff
#: proposes the contract change that would let the shorter name be used.
TRANSCRIPT_EXTENSION = "https://reward-lens.github.io/x/transcript"

#: The four inputs an absence can name. A reader brings one of these and some absence lifts; the
#: surface groups the absences by the input they name to build the "To measure the rest" table.
TASK_SET = "task set"
SAMPLED_RESPONSES = "sampled responses"
PROTECTED_CHECK = "protected check"
RUN_RECORD = "run record"
INPUTS: tuple[str, ...] = (TASK_SET, SAMPLED_RESPONSES, PROTECTED_CHECK, RUN_RECORD)


#: The code an absence carries for what is missing, and the shortest true remedy for it (A-025
#: point 3). The long `missing_access` sentence and the long `remedy` are what the transcript
#: prints; these are what a machine reads and what the envelope projects, so a reader comparing two
#: records can group the holes without parsing English. The vocabulary is closed at six: a seventh
#: kind of missing thing is a decision, not a new string.
MISSING_INSTRUMENT = "instrument_not_in_this_build"
MISSING_RUN_RECORD = "run_record"
MISSING_REFERENCE_MATERIAL = "reference_material"
MISSING_TASK_SET = "task_set"
MISSING_RESPONSE_BANK = "response_bank"
MISSING_PROTECTED_CHECK = "protected_check"

#: Input this absence names -> (code, the flag or command that supplies it). An input gap's short
#: remedy is the flag, because the flag is the whole of what the reader has to do.
_BY_INPUT: dict[str, tuple[str, str]] = {
    TASK_SET: (MISSING_TASK_SET, "--tasks t.jsonl"),
    SAMPLED_RESPONSES: (MISSING_RESPONSE_BANK, "--responses r.jsonl"),
    PROTECTED_CHECK: (MISSING_PROTECTED_CHECK, "--outcome ./tests"),
    RUN_RECORD: (MISSING_RUN_RECORD, "reward-lens trace <run>"),
}

#: The codes and short remedies for an absence that names no input. A run record is here as well as
#: in `_BY_INPUT` because a panel can want one without the reader having been asked for one yet.
_BY_SENTENCE: tuple[tuple[str, str, str], ...] = (
    ("reference material", MISSING_REFERENCE_MATERIAL, "none for this substrate yet"),
    ("run record", MISSING_RUN_RECORD, "reward-lens trace <run>"),
)

#: What an absence gets when nothing more specific fits: this build did not ship the instrument, or
#: shipped one that could not run, and the thing a reader waits for either way is a later build.
#: A refusal and an import failure land here, because the six codes have no word for "refused" and
#: inventing a seventh is a decision above this module.
_FALLBACK = (MISSING_INSTRUMENT, "a later build")


def classify(entry: contracts.Entry) -> tuple[str, str]:
    """The code and the short remedy for one absence: what is missing, and the shortest fix."""
    block = (entry.extensions or {}).get(TRANSCRIPT_EXTENSION) or {}
    needs = block.get("needs")
    if needs in _BY_INPUT:
        return _BY_INPUT[needs]
    sentence = (getattr(entry.absence, "missing_access", "") or "").lower()
    for phrase, code, remedy in _BY_SENTENCE:
        if phrase in sentence:
            return code, remedy
    return _FALLBACK


def stamp_codes(entries: dict[str, list[contracts.Entry]]) -> None:
    """Put the code and the short remedy on every absence in the record, in place (A-025 point 3).

    One pass over the assembled entries rather than a stamp at each of the dozen places an absence
    is built: the `needs` note some of them carry is added after the entry exists, so a stamp at
    construction would have read the note before it was there, and half the absences would have
    fallen to the fallback for no reason a reader could see.
    """
    for group in entries.values():
        for entry in group:
            if entry.kind != "absence":
                continue
            code, remedy = classify(entry)
            extensions = dict(entry.extensions or {})
            block = dict(extensions.get(TRANSCRIPT_EXTENSION, {}))
            block["missing_code"] = code
            block["remedy_short"] = remedy
            extensions[TRANSCRIPT_EXTENSION] = block
            entry.extensions = extensions


def needs_note(entry: contracts.Entry, input_name: str) -> contracts.Entry:
    """Mark an absence with the input that would lift it, and hand the entry back.

    Only an absence a reader can act on carries this. "not in this build" with nothing to bring is
    not a request for an input, and an entry that named one would put a row in the surface's table
    that no file closes.
    """
    extensions = dict(entry.extensions or {})
    block = dict(extensions.get(TRANSCRIPT_EXTENSION, {}))
    block["needs"] = input_name
    extensions[TRANSCRIPT_EXTENSION] = block
    entry.extensions = extensions
    return entry

#: The four dynamic checks of section 7.3 that wave 1 does not run (replay is the fifth, and it
#: does run). Entry id suffix, measurand, what it would need, what it costs not to have it.
DYNAMIC_CHECKS: tuple[tuple[str, str, str, str], ...] = (
    (
        "fresh_state_sterility",
        "whether the grader starts each episode from the state it claims to start from",
        NOT_IN_BUILD,
        "no claim that one episode's state cannot change another's score",
    ),
    (
        "episode_isolation",
        "whether cross-episode and concurrent runs of the grader stay isolated",
        NOT_IN_BUILD,
        "no claim that a concurrent run scores the same as a serial one",
    ),
    (
        "timeout_behaviour",
        "what the grader returns when the graded process exceeds its wall clock",
        NOT_IN_BUILD,
        "no claim about what a timeout is worth to the trainer",
    ),
    (
        "resource_exhaustion",
        "what the grader returns when the graded process exhausts memory or processes",
        NOT_IN_BUILD,
        "no claim about what an exhausted run is worth to the trainer",
    ),
)

#: The three task-set checks, which run only when a task set is supplied (section 7.3).
TASK_SET_CHECKS: tuple[tuple[str, str, str], ...] = (
    (
        "task_validity",
        "whether every task has at least one known-good solution that passes the outcome check",
        "no claim that the tasks are solvable as written",
    ),
    (
        "pass_rate_floor",
        "whether any task is unsolved by every sampled response, and so uninformative here",
        "no claim that every task separates this policy's responses",
    ),
    (
        "flakiness",
        "whether one known-good solution scored k times yields one score",
        "no claim that a task's score is stable across repeats",
    ),
)

#: Section -> (entry id suffix, measurand, missing access, remedy, the claim its absence costs).
PANEL_ABSENCES: tuple[tuple[str, str, str, str, str, str], ...] = (
    (
        "soundness",
        "instrument_absent",
        "known-good solutions accepted and known-wrong solutions rejected, with their rates",
        NOT_IN_BUILD,
        "supply known-good and known-wrong solutions and run a build whose soundness battery has "
        "landed; this build ships none of it",
        "no soundness claim about this reward system",
    ),
    (
        "exploits",
        "instrument_absent",
        "responses that score well without satisfying the intent, by both arms",
        NOT_IN_BUILD,
        "run a build whose seeker and mutation-guided search have landed; the black-box arm is "
        "opt-in and budgeted, and neither arm is in this build",
        "no claim that this reward system resists exploitation",
    ),
    (
        "framing",
        "instrument_absent",
        "what the policy can infer about the grader from its context",
        NOT_IN_BUILD,
        "run a build whose framing panel has landed; this build ships none of it",
        "no claim about what the policy can infer about the grader",
    ),
    (
        "reward_statistics",
        "instrument_absent",
        "the components, weights, aggregation, clipping and the price of a call",
        NOT_IN_BUILD,
        "run a build whose reward-statistics panel has landed; this build ships none of it",
        "no claim about the composition or the price of this reward",
    ),
    (
        "signal",
        "instrument_absent",
        "the within-group contrast, the effective sample size, the style share and the "
        "disagreement between components (D-43)",
        NOT_IN_BUILD,
        "run a build whose signal panel has landed, with sampled responses grouped by prompt; "
        "this build ships none of it",
        "no claim that the score carries usable signal",
    ),
    (
        "trace",
        "instrument_absent",
        "what a training run under this reward system actually optimised",
        "no run record supplied",
        "record a training run and pass it to `reward-lens trace <run>`",
        "no claim about what training optimised",
    ),
    (
        "forecast",
        "instrument_absent",
        "what this reward system is expected to do to a policy",
        "no run record to forecast from",
        "record a training run, then issue a forecast against it before the run that tests it",
        "no forecast about this reward system",
    ),
    (
        "calibration",
        "instrument_absent",
        "how the measured quantities line up with the outcomes later observed",
        "no reference material for this substrate",
        "supply a certified reference material for this substrate, or wait for one to be "
        "published; nothing here can be calibrated without one",
        "no claim that these measurements are calibrated",
    ),
)


#: Section -> (the input that would lift this panel, the reason to give when it is not supplied,
#: the remedy that goes with that reason). A `None` keeps the row's own reason or remedy: the
#: `cost` panel wants sampled responses and would still have no instrument once they arrived, so
#: it names the input without claiming the input is what is blocking it. A panel no input lifts
#: (`framing`, `calibration`) is not here at all.
PANEL_NEEDS: dict[str, tuple[str, str | None, str | None]] = {
    "soundness": (
        PROTECTED_CHECK,
        "no known-good or known-wrong solutions supplied",
        "supply a protected check with `--outcome <dir>`, and the known-good and known-wrong "
        "solutions it accepts and rejects",
    ),
    "exploits": (
        TASK_SET,
        "needs a task set, or a response bank",
        "pass a task set with `--tasks <file>`, or a response bank with `--responses <file>`",
    ),
    "signal": (
        SAMPLED_RESPONSES,
        "needs sampled responses grouped by prompt",
        "pass sampled responses with `--responses <file>`, grouped by prompt",
    ),
    "reward_statistics": (SAMPLED_RESPONSES, None, None),
    "trace": (RUN_RECORD, None, None),
    "forecast": (RUN_RECORD, None, None),
}


#: The noun each input goes by in the sentence that says what the run could not establish. The
#: absence's own `needs` note is the store's word for the input ("task set"); this is the same
#: input as the sentence says it, so "task set" and "sampled responses" read as "tasks or
#: responses" rather than as a list of field names.
NEEDS_NOUN: dict[str, str] = {
    TASK_SET: "tasks",
    SAMPLED_RESPONSES: "responses",
    PROTECTED_CHECK: "a protected check",
    RUN_RECORD: "a run record",
}

#: What each input stands on. A protected check grades solutions to tasks and a run record is a run
#: over tasks with responses sampled from it, so a reader who brought either one alone, with no
#: tasks, would still measure nothing with it. The sentence that says what a run could not reach
#: names only the inputs that are not waiting on another missing one: the rest are in the remedy
#: table under it, in this order, each with what it unlocks (A-020). Two primitives and two that
#: rest on them is the whole of it, and the table is where a reader reads the order.
STANDS_ON: dict[str, tuple[str, ...]] = {
    TASK_SET: (),
    SAMPLED_RESPONSES: (),
    PROTECTED_CHECK: (TASK_SET,),
    RUN_RECORD: (TASK_SET, SAMPLED_RESPONSES),
}

#: The checks whose passing says something about the grader as a whole, and the words they earn.
#: A check is here only when its result is a property of the grader rather than the absence of one
#: defect class: the defect clause of the same sentence already carries the rest, so a row here for
#: every passing check would say the same thing twice and at four times the length.
ESTABLISHED_BY: tuple[tuple[str, str], ...] = (
    ("validity.replay_determinism", "deterministic and replayable"),
)

#: What a failing check is a defect *of*, for the sentence's defect clause. A check with no row
#: falls back to its own entry id with the underscores opened out, which reads as a defect name
#: because the entry ids are named for what they measure.
DEFECT_NOUN: dict[str, str] = {
    "validity.input_handling": "input-handling",
    "validity.writable_then_read": "isolation",
    "validity.input_leakage": "leakage",
    "validity.unreachable_score_branch": "unreachable-branch",
}

#: Counting words, up to the point where a digit reads better than a word.
_COUNT_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def _count(n: int) -> str:
    return _COUNT_WORDS[n] if n < len(_COUNT_WORDS) else str(n)


def _listed(items: Sequence[str], last: str = "and") -> str:
    """`a`, `a and b`, `a, b and c`: the joining the two sentences use.

    `last` is the word before the final item, because one sentence lists what was established
    (all of it, so "and") and the other lists what would lift an absence (any of it, so "or").
    """
    items = list(items)
    if len(items) < 3:
        return f" {last} ".join(items)
    return ", ".join(items[:-1]) + f" {last} " + items[-1]


def _passed(record: Any, entry_id: str) -> bool:
    for entry in record.entries():
        if entry.entry_id == entry_id:
            return entry.kind != "absence" and entry.check is not None and entry.check.passed
    return False


def could_establish(record: Any) -> str:
    """What this run settled, in one sentence, composed from the record it is about.

    Two clauses and neither is written down for a particular run: what the checks in
    `ESTABLISHED_BY` passed, and what the findings say failed. A record with neither says so; a
    sentence that read "the grader is fine" over a record with no passing check in it is the thing
    this must not do.
    """
    clauses: list[str] = []
    established = [phrase for entry_id, phrase in ESTABLISHED_BY if _passed(record, entry_id)]
    if established:
        clauses.append("the grader is " + _listed(established))
    nouns: dict[str, int] = {}
    for finding in record.findings:
        if finding.kind != "fail":
            continue
        for entry_id in finding.entries or ():
            noun = DEFECT_NOUN.get(entry_id, entry_id.split(".", 1)[-1].replace("_", "-"))
            nouns[noun] = nouns.get(noun, 0) + 1
    if nouns:
        clauses.append(
            "it has "
            + _listed(
                [
                    f"{_count(n)} {noun} defect" + ("s" if n != 1 else "")
                    for noun, n in nouns.items()
                ]
            )
        )
    if not clauses:
        return "nothing: no check this build ships returned a result on this grader."
    return ", and ".join(clauses) + "."


def could_not(record: Any) -> str:
    """What this run could not reach, in one sentence, composed from the absences' `needs` notes.

    The absences that name a missing input are the only ones here. An absence that names no input
    is one no reader can close by supplying something, so putting it in a sentence about what to
    bring next would be telling a reader to fetch what does not exist.
    """
    missing: set[str] = set()
    for entry in record.entries():
        block = (entry.extensions or {}).get(TRANSCRIPT_EXTENSION) or {}
        needs = block.get("needs")
        if needs:
            missing.add(needs)
    wanted = [
        NEEDS_NOUN.get(name, name)
        for name in INPUTS
        if name in missing and not (set(STANDS_ON.get(name, ())) & missing)
    ]
    if not wanted:
        return "nothing for want of an input: every absence left is one this build ships no instrument for."
    return "everything that needs " + _listed(wanted, "or") + "."


def validity_absences(
    ctx: RunContext, *, subject_ref: str, has_task_set: bool
) -> list[contracts.Entry]:
    """The seven validity checks wave 1 does not run: four dynamic and three task-set."""
    entries: list[contracts.Entry] = []
    for suffix, measurand, missing, claim in DYNAMIC_CHECKS:
        entries.append(
            ctx.absence(
                "validity",
                f"validity.{suffix}",
                measurand,
                missing,
                "run a build whose dynamic validity checks have landed; this one runs replay "
                "determinism and no other",
                (claim,),
                subject_ref=subject_ref,
            )
        )
    for suffix, measurand, claim in TASK_SET_CHECKS:
        missing = (
            NOT_IN_BUILD if has_task_set else "no task set was supplied, and this check needs one"
        )
        remedy = (
            "run a build whose task-set checks have landed; this build ships none of them"
            if has_task_set
            else "pass a task set with `--tasks <file>`, or point the audit at a project whose "
            "rewardlens.yaml names one"
        )
        entry = ctx.absence(
            "validity",
            f"validity.{suffix}",
            measurand,
            missing,
            remedy,
            (claim,),
            subject_ref=subject_ref,
        )
        if not has_task_set:
            needs_note(entry, TASK_SET)
        entries.append(entry)
    return entries


def panel_absences(
    ctx: RunContext,
    *,
    subject_ref: str,
    skip: Iterable[str] = (),
    supplied: Iterable[str] = (),
) -> dict[str, list[contracts.Entry]]:
    """One honest absence for every panel wave 1 does not fill.

    `supplied` is the inputs this run was actually given, drawn from `INPUTS`. A panel whose input
    is missing says so in the reason a reader acts on ("no known-good or known-wrong solutions
    supplied") rather than in the one only a new release closes; a panel whose input is there
    falls back to the build reason, because that is then what is missing.
    """
    skipped = set(skip)
    have = set(supplied)
    out: dict[str, list[contracts.Entry]] = {}
    for section, suffix, measurand, missing, remedy, claim in PANEL_ABSENCES:
        if section in skipped:
            continue
        wanted: str | None = None
        row = PANEL_NEEDS.get(section)
        if row is not None and row[0] not in have:
            wanted, reason, alternative = row
            missing = reason or missing
            remedy = alternative or remedy
        entry = ctx.absence(
            section,
            f"{section}.{suffix}",
            measurand,
            missing,
            remedy,
            (claim,),
            subject_ref=subject_ref,
        )
        if wanted is not None:
            needs_note(entry, wanted)
        out.setdefault(section, []).append(entry)
    return out


#: Requirement text up to the first version specifier, extra marker or bracket.
_REQUIREMENT_NAME = re.compile(r"[<>=!~;\[\s(]")
#: The extra a `Requires-Dist` line is conditional on.
_EXTRA_MARKER = re.compile(r"""extra\s*==\s*["']([^"']+)["']""")
#: The module name inside a `ModuleNotFoundError` whose `name` attribute was not set.
_NAMED_MODULE = re.compile(r"No module named ['\"]([^'\"]+)['\"]")


def _deepest_import_error(failure: BaseException) -> ImportError | None:
    """The `ImportError` furthest down `failure`'s cause chain, or None if there is none.

    The exception that reaches the audit is rarely the import that broke: an adapter catches the
    `ModuleNotFoundError`, fails on the `None` it then holds, and raises an `AttributeError` that
    names nothing a reader can install. The chain still carries the original, so the hole is
    written from the bottom of the chain rather than from the top.
    """
    seen: set[int] = set()
    current: BaseException | None = failure
    found: ImportError | None = None
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ImportError):
            found = current
        current = current.__cause__ or current.__context__
    return found


def _extras_carrying(module: str) -> tuple[str, ...]:
    """Every declared reward-lens extra whose requirements carry `module`, in name order.

    Read out of the installed distribution's metadata rather than out of `pyproject.toml`, so the
    instruction the hole gives is the one that holds for the wheel actually installed. The match is
    on the distribution name, so a project whose import name differs from its distribution name
    (`scikit-learn` against `sklearn`) falls through to the module name, which is still runnable.
    """
    try:
        from importlib.metadata import metadata

        declared = metadata("reward-lens").get_all("Requires-Dist") or []
    except Exception:  # pragma: no cover - a tree installed without metadata
        return ()
    wanted = module.split(".")[0].replace("_", "-").lower()
    found: set[str] = set()
    weight: dict[str, int] = {}
    for requirement in declared:
        text = str(requirement)
        name = _REQUIREMENT_NAME.split(text.strip(), maxsplit=1)[0]
        marker = _EXTRA_MARKER.search(text)
        # `all` carries every extra, so naming it would tell a reader to install the whole project
        # to get one module.
        if marker is None or marker.group(1) == "all":
            continue
        weight[marker.group(1)] = weight.get(marker.group(1), 0) + 1
        if name.replace("_", "-").lower() == wanted:
            found.add(marker.group(1))
    # Smallest first, so the command offered is the least that supplies the module: numpy is in
    # four extras, and `organisms` would pull a training stack in to get it.
    return tuple(sorted(found, key=lambda extra: (weight.get(extra, 0), extra)))


def import_diagnosis(failure: BaseException) -> tuple[str | None, str | None]:
    """`(module, the command that installs it)` when `failure` came from an import, else `(None, None)`."""
    error = _deepest_import_error(failure)
    if error is None:
        return None, None
    module = str(getattr(error, "name", "") or "")
    if not module:
        named = _NAMED_MODULE.search(str(error))
        module = named.group(1) if named else ""
    if not module:
        return None, None
    extras = _extras_carrying(module)
    if not extras:
        return module, f"pip install {module}"
    command = f"pip install 'reward-lens[{extras[0]}]'"
    if len(extras) > 1:
        command += f" (the extras carrying {module} are " + ", ".join(sorted(extras)) + ")"
    return module, command


def could_not_check(
    ctx: RunContext,
    *,
    section: str,
    entry_id: str,
    measurand: str,
    failure: BaseException,
    remedy: str,
    subject_ref: str,
    affected_claims: Sequence[str] = (),
) -> contracts.Entry:
    """An exception from loading or calling a user's grader, as a hole that names it (D-18).

    A hole that says only that something failed cannot be acted on. When the failure was an import,
    the hole names the module that would not import, the exception that raised it and the command
    that installs it, so the reader can close the hole without reproducing the run.
    """
    detail = f"{type(failure).__name__}: {failure}".strip()
    module, command = import_diagnosis(failure)
    if module is not None and command is not None:
        missing = (
            f"the grader needs {module}, which this environment does not have, so it could not be "
            f"imported here: {detail}; install it with {command}"
        )
        remedy = f"{command}; then {remedy}"
    else:
        missing = f"the grader could not be loaded or called here: {detail}"
    return ctx.absence(
        section,
        entry_id,
        measurand,
        missing,
        remedy,
        tuple(affected_claims),
        subject_ref=subject_ref,
        state="COULD_NOT_CHECK",
    )
