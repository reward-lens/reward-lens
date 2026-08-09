"""`--format text`: the record rendered for a person, and nothing computed.

Every number here is read off the record, the plan or the file the run wrote. The formatter decides
order, glyph, column and wording; it never derives a statistic, because a statistic that appears
only in the rendering is a claim with no entry behind it.

The layout is the wave-1 transcripts (`fleet/golden/wave-1/first-hour.txt` and
`audit-unfamiliar.txt`), which A-015 freezes as the contract: the record carries the sentences, the
example carries its manifest, and this module puts them in columns. Every constant below was
measured off those two files rather than chosen, and the docstring of each says which line it was
read from, so a later reader can check the column instead of trusting it.

The panel rule is interfaces section 9. Its canonical order is validity, soundness, reach,
exploits, framing, signal, cost (the reward_statistics section), trace, forecast, calibration; the
project golden shows reach second, ahead of a dark soundness, so the printed order is that
canonical order with the lit panels stably ahead of the dark ones. The two agree whenever every
panel is lit.
"""

from __future__ import annotations

import os
import textwrap

#: A-016: the record's transcript fields live under this key, and the CLI carries it as a literal.
TRANSCRIPT_EXTENSION = "https://reward-lens.github.io/x/transcript"

#: (section in the record, the word the panel prints)
CANONICAL = (
    ("validity", "validity"),
    ("soundness", "soundness"),
    ("reach", "reach"),
    ("exploits", "exploits"),
    ("framing", "framing"),
    ("signal", "signal"),
    ("reward_statistics", "cost"),
    ("trace", "trace"),
    ("forecast", "forecast"),
    ("calibration", "calibration"),
)

#: A bare-grader audit shows these, and folds the rest into "What it could not". `replay` is not a
#: section of the record: it is the `validity.replay_determinism` entry, which the unfamiliar
#: golden gives a panel line of its own because on a bare grader it is most of what ran.
GRADER_SHAPE = ("validity", "replay", "soundness", "reach", "exploits", "signal")

#: The entry the `replay` panel reads.
REPLAY_ENTRY = "validity.replay_determinism"

DARK = "○"
CROSS = "✘"
PARTIAL = "!"
TICK = "✔"
GLYPHS = (DARK, CROSS, PARTIAL, TICK)

#: `  <glyph> <label padded>` puts a panel's detail at column 18 (first-hour lines 20 to 32).
_LABEL_WIDTH = 14
_INDENT = 4 + _LABEL_WIDTH
#: The detail and a finding wrap to this width under the detail column. The transcripts pin it
#: between 63 and 65: first-hour line 21 breaks after "which the" and unfamiliar line 6 after
#: "task-set", and neither break survives a width outside that range.
_PANEL_WRAP = 64
#: A code stands at column 68 (first-hour lines 22 and 23) while the text before it ends at or
#: before column 60; past that it follows the text by six spaces (unfamiliar line 9, whose RL0210
#: stands at 67). The elbow is the golden's, not a preference: see the handoff's premise checks.
_CODE_COLUMN = 68
_CODE_ELBOW = 60
_CODE_GAP = 6

#: `  <label padded>` puts Subject, Plan, Reused, Verdict, Report and Next at column 12.
_FIELD_WIDTH = 10
#: The narrative under the panels (first-hour lines 36 to 38).
_NARRATIVE_INDENT = 2
_NARRATIVE_WRAP = 77
#: "What this run could establish" and its answer (unfamiliar lines 16 to 18).
_ESTABLISH_WIDTH = 32
_ESTABLISH_WRAP = 45
#: The three columns of "To measure the rest" (unfamiliar lines 21 to 24).
_REMEDY_COLUMNS = (23, 27)

#: The input the second row asks for, in the store's words, and the transcript's own count of it.
_SAMPLED = "sampled responses"

#: The remedy table of the unfamiliar golden, in its order: the input an absence names under
#: `extensions[TRANSCRIPT_EXTENSION].needs`, then the three columns printed for it. The store
#: writes the bare input into `needs` (`product/audit/absences.py`: "task set", "sampled
#: responses", "protected check", "run record") and the table prints the reader's wording of the
#: same input, so a row is matched on the need it names and never on the label it prints: "a
#: training run record" is not inside "run record", and matching on the label dropped the row.
REMEDIES = (
    ("task set", "a task set", "reach, exploits", "reward-lens audit x --tasks t.jsonl"),
    (_SAMPLED, f"128 {_SAMPLED}", "signal, cost, selection", "--responses r.jsonl"),
    ("protected check", "a protected check", "soundness, verdict", "--outcome ./tests"),
    ("run record", "a training run record", "the join, reward stats", "reward-lens trace <run>"),
)

_UNRESOLVED = "Unresolved is a verdict. It is not a pass and it is not a failure."
_NO_CHECK = "No independent check was supplied, so no correctness claim was made."
_OUTCOME_UNQUALIFIED = "the outcome check is unqualified (OUTCOME_UNQUALIFIED)"
_REPORT_NOTE = "Open it anywhere. It needs no install and no network."
_NEXT = "reward-lens trace <your run>   to see whether this was ever selected"


def shape_for(record) -> str:
    """A-019: a bare grader is a record whose `subject.context.task_set` is null, and nothing else.

    The two transcripts choose between four wordings, and what separates them is what the audit
    had to work with, not what the caller typed. A directory can be audited with no task set and a
    file can be audited with one; keying on the path would then print the project's counts sentence
    over a record that has no counts, and the remedy table over a record that needs no remedy. The
    record is the only thing that knows, so the record is what is asked.
    """
    doc = record.to_dict() if hasattr(record, "to_dict") else dict(record)
    context = (doc.get("subject") or {}).get("context") or {}
    return "grader" if context.get("task_set") is None else "project"


def render(
    record,
    *,
    command: str = "audit",
    shape: str | None = None,
    out=None,
    plan=None,
    artifacts=None,
    project: str | None = None,
    reused=None,
    cwd: str | None = None,
) -> str:
    doc = record.to_dict() if hasattr(record, "to_dict") else dict(record)
    shape = shape or shape_for(doc)
    lines = [""]
    lines.extend(_subject_block(doc, shape=shape))
    lines.extend(_plan_block(plan))
    lines.extend(_reused_block(reused))
    lines.append("")
    lines.extend(_panel_block(doc, shape=shape))
    lines.append("")
    if shape == "project":
        lines.extend(_narrative_block(doc))
    else:
        lines.extend(_establish_block(doc))
        lines.extend(_remedy_block(doc))
    lines.extend(_verdict_block(doc, shape=shape))
    lines.append("")
    lines.extend(_report_block(artifacts, shape=shape, cwd=cwd, project=project))
    lines.extend(_footer(doc))
    lines.append("")
    return "\n".join(_paint(lines, out))


# --- the record, read by the names A-015 and A-016 settle ------------------------------------


def _extension(holder) -> dict:
    return ((holder or {}).get("extensions") or {}).get(TRANSCRIPT_EXTENSION) or {}


def _entries(doc, section) -> list:
    return list((doc.get("measurement") or {}).get(section) or [])


def _entry(doc, entry_id):
    for entries in (doc.get("measurement") or {}).values():
        for entry in entries or ():
            if entry.get("entry_id") == entry_id:
                return entry
    return None


def _holes(doc, section) -> list:
    return [hole for hole in (doc.get("holes") or []) if hole.get("section") == section]


def _findings(doc, section) -> list:
    ids = {entry.get("entry_id") for entry in _entries(doc, section)}
    out = []
    for finding in doc.get("findings") or []:
        touched = set(finding.get("entries") or ())
        if touched & ids or (finding.get("rule") or "").startswith(section + "."):
            out.append(finding)
    return out


def _failed_entries(doc) -> set:
    """Entry ids an error-level finding stands against: A-015 counts those as failed checks."""
    failed = set()
    for finding in doc.get("findings") or []:
        if finding.get("level") == "error":
            failed.update(finding.get("entries") or ())
    return failed


def glyph_for(doc, section) -> str:
    if section == "replay":
        entry = _entry(doc, REPLAY_ENTRY)
        if entry is None or entry.get("kind") == "absence":
            return DARK
        return CROSS if (entry.get("check") or {}).get("passed") is False else TICK
    entries = _entries(doc, section)
    findings = _findings(doc, section)
    if not entries or all(entry.get("kind") == "absence" for entry in entries):
        return DARK
    if any(finding.get("level") == "error" for finding in findings):
        return CROSS
    if any(entry.get("state") == "partial" for entry in entries):
        return PARTIAL
    if any(str(note).startswith("static") for entry in entries for note in entry.get("limitations") or ()):
        return PARTIAL
    if any((entry.get("check") or {}).get("passed") is False for entry in entries):
        return CROSS
    if any(finding.get("level") == "warning" for finding in findings):
        return PARTIAL
    return TICK


# --- the blocks -------------------------------------------------------------------------------


def _field(label: str, value: str) -> str:
    return f"  {label:<{_FIELD_WIDTH}}{value}"


def _version_label(name, version_id) -> str:
    """A-015: the version id with `<name>-` stripped when it starts with it, else the id."""
    if not version_id:
        return ""
    if name and version_id.startswith(f"{name}-"):
        return version_id[len(name) + 1 :]
    return version_id


def _subject_block(doc, *, shape: str) -> list:
    """A-019: the clauses the schema's closed `subject` cannot carry come from the extension.

    `subject.reward_system` is `id` and `name`, and `subject.context` is `configuration`, `policy`
    and `task_set`. The entry file, the two measured counts, the signature and the shape are none
    of those, so they live at the record's top level under
    `extensions[TRANSCRIPT_EXTENSION].subject` and are read from there and nowhere else. A record
    that does not carry one drops its clause rather than substituting another field for it.
    """
    subject = doc.get("subject") or {}
    system = subject.get("reward_system") or {}
    context = subject.get("context") or {}
    named = _extension(doc).get("subject") or {}
    parts = []
    if shape == "project":
        name = system.get("name") or system.get("id") or "-"
        label = _version_label(system.get("name"), (subject.get("version") or {}).get("id"))
        parts.append(f"{name} {label}".strip())
        if named.get("grader"):
            parts.append(str(named["grader"]))
        if named.get("tasks") is not None:
            parts.append(f"{named['tasks']} tasks")
        if named.get("responses") is not None:
            parts.append(f"{named['responses']} responses")
    else:
        parts.append(str(named.get("grader") or system.get("name") or system.get("id") or "-"))
        if named.get("signature"):
            # The clause is the record's own wording, printed as it stands: a record that says
            # `plain shape` says the noun itself, and a formatter that appended one said it twice.
            shaped = f", {named['shape']}" if named.get("shape") else ""
            parts.append(f"`{named['signature']}`{shaped}")
        parts.append(str(context.get("task_set")) if context.get("task_set") else "no task set supplied")
    return [_field("Subject", "  ·  ".join(part for part in parts if part))]


def _plan_block(plan) -> list:
    """A-016: the panel count, the paid calls and `Plan.estimate_s`, and nothing computed here."""
    if plan is None:
        return []
    panels = len(getattr(plan, "panels", ()) or ())
    parts = [f"{panels} panels", f"{getattr(plan, 'paid_calls', 0)} paid calls"]
    estimate = getattr(plan, "estimate_s", 0) or 0
    if estimate:
        parts.append(f"about {estimate} s")
    return [_field("Plan", "  ·  ".join(parts))]


def _reused_block(reused) -> list:
    """A-016, D-28: a run that returned a stored record says so, above the record it returned."""
    if reused is None:
        return []
    says = reused.get("says") if isinstance(reused, dict) else getattr(reused, "says", None)
    return [_field("Reused", str(says or ""))]


def _panel_block(doc, *, shape: str) -> list:
    labels = dict(CANONICAL)
    wanted = CANONICAL if shape == "project" else tuple(
        (section, labels.get(section, section)) for section in GRADER_SHAPE
    )
    rows = [(section, label, glyph_for(doc, section)) for section, label in wanted]
    lit = [row for row in rows if row[2] != DARK]
    dark = [row for row in rows if row[2] == DARK]
    lines = []
    for section, label, glyph in lit + dark:
        lines.extend(_panel(doc, section, label, glyph))
    return lines


def _panel(doc, section, label, glyph) -> list:
    head = f"  {glyph} {label:<{_LABEL_WIDTH}}"
    detail, code, note = _detail(doc, section)
    lines = _wrapped(detail, _INDENT, _PANEL_WRAP, head=head)
    _place_code(lines, code)
    if note:
        lines.extend(_wrapped(note, _INDENT, _PANEL_WRAP))
    if glyph == DARK:
        # A dark section measured nothing, so it has nothing to qualify: the transcripts carry no
        # finding under one, and a finding printed there would read as a result of a panel that
        # did not run.
        return lines
    # A-028: one glyph line per section. Every finding of a lit section prints under the panel's
    # detail as continuation lines, whether the detail wrapped and whether it is the first finding
    # or the fifth, so two findings can never print two `validity` heads.
    for finding in _findings(doc, section):
        if finding.get("level") not in ("error", "warning"):
            continue
        message = str(finding.get("message") or finding.get("severity_rationale") or finding.get("rule") or "")
        if not message:
            continue
        block = _wrapped(message, _INDENT, _PANEL_WRAP, head=None)
        _place_code(block, str(finding.get("code") or finding.get("id") or ""))
        lines.extend(block)
    return lines


def _wrapped(text: str, indent: int, width: int, head: str | None = None) -> list:
    pad = " " * indent
    pieces = textwrap.wrap(str(text), width=width) if str(text).strip() else [""]
    return [(head if index == 0 and head is not None else pad) + piece for index, piece in enumerate(pieces)]


def _place_code(lines: list, code: str) -> None:
    if not code or not lines:
        return
    end = len(lines[-1])
    column = _CODE_COLUMN if end <= _CODE_ELBOW else end + _CODE_GAP
    lines[-1] = lines[-1].ljust(column) + code


def _detail(doc, section):
    """-> (the detail sentence, the code that stands to its right, the line under it)."""
    if section == "replay":
        entry = _entry(doc, REPLAY_ENTRY)
        if entry is None or entry.get("kind") == "absence":
            return _absence_detail(entry, _holes(doc, "validity")), "", ""
        result = entry.get("result") or {}
        summary = result.get("summary")
        if not summary:
            return _replay_fallback(entry), "", ""
        return str(summary), str(result.get("exposure_id") or ""), str(result.get("note") or "")
    entries = _entries(doc, section)
    if not entries or all(entry.get("kind") == "absence" for entry in entries):
        holes = _holes(doc, section)
        missing = holes[0].get("missing_access") if holes else None
        if not missing:
            absences = [(entry.get("absence") or {}) for entry in entries]
            missing = next((a.get("missing_access") for a in absences if a.get("missing_access")), None)
        return f"not measured: {missing or 'no evidence was supplied'}", "", ""
    if section == "validity":
        return _validity_counts(doc), "", ""
    result = _measured(entries)
    return str(result.get("summary") or _fallback_detail(entries)), str(result.get("exposure_id") or ""), str(result.get("note") or "")


def _absence_detail(entry, holes) -> str:
    """Why an absent entry was not measured: its own reason, then its section's hole, then neither.

    The entry is asked first here and the hole is asked first in the section case below. A hole
    stands for a whole dark section and is the only thing that can speak for one; an entry that is
    absent on its own carries the reason it is absent, and that reason is the specific one.
    """
    missing = ((entry or {}).get("absence") or {}).get("missing_access")
    if not missing and holes:
        missing = holes[0].get("missing_access")
    return f"not measured: {missing or 'no evidence was supplied'}"


def _replay_fallback(entry) -> str:
    """The replay line for a record whose replay entry carries no `result.summary`.

    Every clause names a field of the entry: `check.scope_tested` is the entry's own sentence about
    what it covered, and the two counts are its `result`'s. Nothing is computed from anything else,
    and a record that carries the summary never reaches here.
    """
    result = entry.get("result") or {}
    clauses = []
    scope = (entry.get("check") or {}).get("scope_tested")
    if scope:
        clauses.append(str(scope))
    for field, word in (("n_nondeterministic", "disagreed"), ("n_unreplayable", "could not be replayed")):
        if result.get(field) is not None:
            clauses.append(f"{result[field]} {word}")
    if clauses:
        return ", ".join(clauses)
    return str((entry.get("check") or {}).get("predicate") or _fallback_detail([entry]))


def _measured(entries) -> dict:
    for entry in entries:
        if entry.get("kind") != "absence" and entry.get("result"):
            return entry["result"]
    return {}


def _fallback_detail(entries) -> str:
    measured = [entry for entry in entries if entry.get("kind") != "absence"]
    count = len(measured)
    return f"{count} {'entry' if count == 1 else 'entries'}"


def _validity_counts(doc) -> str:
    entries = _entries(doc, "validity")
    failed = _failed_entries(doc)
    checks = [entry for entry in entries if entry.get("kind") in ("check", "witness")]
    passed = sum(
        1
        for entry in checks
        if (entry.get("check") or {}).get("passed") is not False and entry.get("entry_id") not in failed
    )
    clauses = []
    if checks:
        clauses.append(f"{passed} of {len(checks)} checks passed")
    # The line carries two absence counts and no others: what is not in this build, and what the
    # task set would supply. An absence that needs a response bank, a protected check or a run
    # record is stated once, in the remedy table under `To measure the rest`, and repeating it here
    # is what put a whole sentence inside a tally.
    not_built: list[str] = []
    task_set: list[str] = []
    for entry in entries:
        if entry.get("kind") != "absence":
            continue
        absence = entry.get("absence") or {}
        needs = _extension(entry).get("needs") or _extension(absence).get("needs")
        if needs:
            if _slug(str(needs)) == "task-set":
                task_set.append(str(needs))
            continue
        missing = str(absence.get("missing_access") or "")
        if "build" in missing:
            not_built.append(missing)
    if not_built:
        clauses.append(_absence_clause(len(not_built), not_built[0], needs=False, sole=not task_set))
    if task_set:
        clauses.append(_absence_clause(len(task_set), task_set[0], needs=True, sole=False))
    return "; ".join(clauses) or "no validity entry in this record"


def _absence_clause(count: int, missing: str, *, needs: bool, sole: bool) -> str:
    """The wording is the record's: only the count, the article and the frame are added."""
    if needs:
        return f"the {count} {_slug(missing)} checks need {_article(missing)}"
    if "build" in missing:
        return f"{count} checks are not in this build" if sole else f"{count} are not in this build"
    return f"{count} {missing}"


def _article(value: str) -> str:
    """The determiner the sentence needs in front of the bare input the store writes.

    `product/audit/absences.py` writes `task set`, not `a task set`, so the clause that ends the
    validity line has to supply the article or the transcript loses it. A wording that already
    carries a determiner, or that opens on a count (`128 sampled responses`), is passed through:
    the article is the sentence's and doubling it reads `a a task set`.
    """
    head = value.split(" ", 1)[0].lower()
    if not head or head in {"a", "an", "the"} or head[0].isdigit():
        return value
    return f"{'an' if head[0] in 'aeiou' else 'a'} {value}"


def _slug(value: str) -> str:
    return value.removeprefix("a ").removeprefix("an ").replace(" ", "-")


def _narrative_block(doc) -> list:
    """The headline and the explanation the check entry carries (A-015)."""
    for finding in doc.get("findings") or []:
        if finding.get("level") != "error":
            continue
        for entry_id in finding.get("entries") or ():
            result = (_entry(doc, entry_id) or {}).get("result") or {}
            headline, explanation = result.get("headline"), result.get("explanation")
            if headline or explanation:
                lines = []
                if headline:
                    lines.append(f"  {headline}")
                if explanation:
                    if lines:
                        lines.append("")
                    lines.extend(_wrapped(explanation, _NARRATIVE_INDENT, _NARRATIVE_WRAP))
                return lines + [""]
    return []


def _establish_block(doc) -> list:
    """A-016: the two sentences the record composes, at the record's top level."""
    extension = _extension(doc)
    lines = []
    for label, key in (("What this run could establish", "could_establish"), ("What it could not", "could_not")):
        sentence = extension.get(key)
        if not sentence:
            continue
        pad = " " * (2 + _ESTABLISH_WIDTH)
        head = f"  {label:<{_ESTABLISH_WIDTH}}"
        lines.extend(_wrapped(sentence, len(pad), _ESTABLISH_WRAP, head=head))
    return lines + [""] if lines else []


def _needs(doc) -> list:
    out = []
    for entries in (doc.get("measurement") or {}).values():
        for entry in entries or ():
            needs = _extension(entry).get("needs") or _extension(entry.get("absence") or {}).get("needs")
            if needs:
                out.append(str(needs).lower())
    return out


def _remedy_block(doc) -> list:
    named = _needs(doc)
    rows = [row for row in REMEDIES if any(row[0] in need for need in named)]
    if not rows:
        return []
    # A record that counted a response bank states the row in its own count; a record that counted
    # none leaves the transcript's number, which is an example and not a measurement.
    counted = (_extension(doc).get("subject") or {}).get("responses")
    lines = ["  To measure the rest, bring one of these, in order of what it unlocks:"]
    first, second = _REMEDY_COLUMNS
    for need, input_name, unlocks, command in rows:
        if need == _SAMPLED and counted is not None:
            input_name = f"{counted} {_SAMPLED}"
        lines.append(f"    {input_name:<{first}}{unlocks:<{second}}{command}")
    return lines + [""]


def _verdict_block(doc, *, shape: str) -> list:
    decision = doc.get("decision") or {}
    state = decision.get("state") or "no decision requested"
    reasons = [str(reason) for reason in decision.get("reasons") or ()]
    head = state
    if any(reason.startswith("outcome_unqualified:") for reason in reasons):
        head = f"{state}   ·   {_OUTCOME_UNQUALIFIED}"
    lines = [_field("Verdict", head)]
    pad = " " * (2 + _FIELD_WIDTH)
    if shape == "project":
        counted = _counts_sentence(doc, reasons)
        if counted:
            lines.append(pad + counted)
        if state == "unresolved":
            lines.append(pad + _UNRESOLVED)
    else:
        lines.append(pad + _NO_CHECK)
    return lines


def _counts_sentence(doc, reasons) -> str:
    blocking = set()
    for reason in reasons:
        if not reason.startswith("blocking_finding:"):
            continue
        named = reason.split(":", 1)[1]
        for finding in doc.get("findings") or ():
            if named in (finding.get("id"), finding.get("rule")):
                blocking.add(finding.get("id") or finding.get("rule"))
                break
        else:
            blocking.add(named)
    missing = {reason.split(":", 1)[1] for reason in reasons if reason.startswith("required_missing:")}
    if not blocking and not missing:
        return ""
    findings = f"{len(blocking)} blocking finding" + ("" if len(blocking) == 1 else "s")
    panels = f"{len(missing)} required panel" + ("" if len(missing) == 1 else "s")
    return f"{findings}, and {panels} could not run."


def _report_block(artifacts, *, shape: str, cwd: str | None, project: str | None = None) -> list:
    """A-019: the envelope's paths are relative to the project, and this line is relative to here.

    They are the same file named for two readers. The envelope is kept and moved, so it names the
    path from the project it belongs to; the transcript line is typed back into this shell, so it
    names the path from the working directory. `project` is what joins the one to the other, which
    is exactly what a consumer of the envelope has to do with it.
    """
    report = (artifacts or {}).get("report")
    if not report:
        return []
    report = os.path.join(project, report) if project and not os.path.isabs(report) else report
    lines = [_field("Report", f"{_relative(report, cwd)}   {_size(report)}")]
    if shape == "project":
        lines.append(" " * (2 + _FIELD_WIDTH) + _REPORT_NOTE)
        lines.append("")
        lines.append(_field("Next", _NEXT))
    lines.append("")
    return lines


def _relative(path: str, cwd: str | None) -> str:
    """The path as the caller would type it from here: the transcripts show it relative to cwd."""
    base = cwd or os.getcwd()
    try:
        relative = os.path.relpath(path, base)
    except ValueError:  # pragma: no cover - a different drive on Windows
        return str(path)
    if relative.startswith(".."):
        return str(path)
    return relative if relative.startswith(".") else f"./{relative}"


def _size(path: str) -> str:
    try:
        size = os.path.getsize(path)
    except OSError:  # pragma: no cover - the renderer never fails over a stat
        return ""
    return f"{size // 1024} KB" if size >= 1024 else f"{size} B"


def _footer(doc) -> list:
    cost = doc.get("cost") or {}
    return [
        f"  {_seconds(cost.get('wall_s', 0))} s  ·  "
        f"{cost.get('api_calls', 0)} paid calls  ·  ${cost.get('usd', '0.00')}"
    ]


def _seconds(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number.is_integer() else str(value)


def _paint(lines, out) -> list:
    if out is None:
        return lines
    painted = []
    for line in lines:
        if len(line) > 3 and line[:2] == "  " and line[2] in GLYPHS and line[3] == " ":
            painted.append("  " + out.paint(line[2]) + line[3:])
        else:
            painted.append(line)
    return painted
