"""The options every verb carries, and the two functions that open and close a verb.

The global options live on the commands rather than on the root group, which is what the frozen
root help promises when it says `Per-command options: reward-lens audit --help`, and what lets
`reward-lens audit ./demo --format json` parse with the flag after the argument.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import pathlib
import sys

import click

from .. import config, ids, output

#: The record suffix the store writes, and the page the renderer puts beside it.
RECORD_SUFFIX = ".assay.json"
REPORT_SUFFIX = ".assay.html"

#: The internal-failure hook of D-22's exit 7. Set alone it is refused; see `selftest_raise`.
SELFTEST_FLAG = "REWARD_LENS_SELFTEST"
SELFTEST_RAISE = "REWARD_LENS_SELFTEST_RAISE"

FORMAT_HELP = "text, json, jsonl, sarif or github; text on a TTY and json off it" + config.env_note("format")
EPILOG = config.PRECEDENCE_LINE


def common(function):
    """Format, colour, progress, --fields, --exit-zero, --no-config, --yes, --non-interactive."""
    for decorator in reversed(
        [
            click.option("-f", "--format", "format_", type=str, default=None, help=FORMAT_HELP),
            click.option("--json", "json_flag", is_flag=True, help="shorthand for --format json"),
            click.option("--color", "colour", type=click.Choice(["auto", "always", "never"]),
                         default=None, help="when to colour the output" + config.env_note("color")),
            click.option("--no-color", "no_colour", is_flag=True, help="alias for --color never"),
            click.option("--progress", type=click.Choice(["auto", "plain", "quiet"]), default=None,
                         help="progress on stderr" + config.env_note("no_progress")),
            click.option("--no-progress", "no_progress", is_flag=True, help="alias for --progress quiet"),
            click.option("--fields", default=None, help="bound the result to these dotted fields"),
            click.option("--exit-zero", is_flag=True, help="exit 0 even when the decision is rejected or unresolved"),
            click.option("--no-config", is_flag=True, help="ignore every configuration file"),
            click.option("-y", "--yes", is_flag=True, help="make the decision this command would ask for"),
            click.option("--non-interactive", is_flag=True,
                         help="never ask; print the pending action and exit 3" + config.env_note("non_interactive")),
        ]
    ):
        function = decorator(function)
    return function


def measuring(function):
    """The options a command that executes, spends or writes also carries."""
    for decorator in reversed(
        [
            click.option("--offline/--no-offline", default=None,
                         help="block every outbound request" + config.env_note("offline")),
            click.option("--max-budget-usd", default=None,
                         help="cap the whole tree of work in dollars" + config.env_note("max_budget_usd")),
            click.option("-n", "--dry-run", is_flag=True, help="plan, price and print; execute nothing"),
            click.option("--seeker", type=click.Choice(["off", "api", "local", "agent"]), default="off",
                         help="how hard to look for exploits"),
            click.option("--name", default=None, help="a human name for this run"),
            click.option("--resume", default=None, help="continue a run by id or name"),
            click.option("--seed", type=int, default=None, help="the seed this run records and reuses"),
            click.option("--only", multiple=True, help="restrict the run to these instruments"),
        ]
    ):
        function = decorator(function)
    return function


def make_output(opts: dict) -> output.Output:
    """Resolve the streams and the format by D-29's precedence, and refuse an unknown format."""
    no_config = bool(opts.get("no_config"))
    fmt = output.resolve_format(
        opts.get("format_"), json_flag=bool(opts.get("json_flag")), stdout_is_tty=sys.stdout.isatty()
    )
    colour_choice = "never" if opts.get("no_colour") else opts.get("colour")
    progress = "quiet" if opts.get("no_progress") else (opts.get("progress") or "auto")
    if progress == "auto" and (fmt != "text" or not sys.stderr.isatty()):
        progress = "plain" if sys.stderr.isatty() else "quiet"
    if config.resolve("no_progress", None, default=None, no_config=no_config):
        progress = "quiet"
    fields = tuple(part.strip() for part in (opts.get("fields") or "").split(",") if part.strip())
    return output.Output(
        format=fmt,
        colour=output.want_colour(colour_choice, stream=sys.stdout),
        progress=progress,
        fields=fields,
        exit_zero=bool(opts.get("exit_zero")),
        yes=bool(opts.get("yes")),
        non_interactive=output.is_non_interactive(bool(opts.get("non_interactive"))),
        no_config=no_config,
    )


def verb(name: str, *, normalise=None):
    """Wrap a verb callback so the Output exists before anything can fail.

    `normalise` runs over the parsed options before the format is resolved. One verb needs it:
    `export` names an artefact kind with a word the shared `--format` would otherwise refuse, and
    the refusal happens in `make_output`, before the callback is reached.
    """

    def decorate(function):
        @functools.wraps(function)
        @click.pass_context
        def wrapper(ctx, **opts):
            if normalise is not None:
                normalise(opts)
            out = make_output(opts)
            # Every context up the chain, not just this one: a failure inside a subcommand is
            # caught at the root group, and the handler there renders through `ctx.obj`. Without
            # the walk it finds nothing and falls back to a bare text Output, which silently
            # drops the structured envelope the caller asked for.
            cursor = ctx
            while cursor is not None:
                cursor.obj = out
                cursor = cursor.parent
            selftest_raise()
            ctx.exit(function(out=out, **opts))

        return wrapper

    return decorate


def artifacts_for_record(record, *roots) -> tuple[dict, str | None]:
    """Where the pipeline put this record, found by the record's own id.

    -> (the envelope's `artifacts`, the directory those paths are relative to).

    Section 5.7 says the envelope names the files the command left behind, and the api hands the
    verb a record and not the paths it wrote. So the verb asks the store which name holds the
    record it was just given. Matching on `assay_id` is what makes it *this* record rather than
    merely the newest one: a project holds many, and the newest is only a shortcut when its id
    agrees. A root that holds no such record contributes nothing, which is why this is safe to
    call from a verb that may or may not have written anything.
    """
    record_id = _assay_id(record)
    if not record_id:
        return {}, None
    for root in roots:
        if root in (None, ""):
            continue
        found = _written_at(pathlib.Path(str(root)), record_id)
        if found is not None:
            directory = _directory_of(pathlib.Path(str(root)))
            return _artifact_pair(found, directory), str(directory)
    return {}, None


def _directory_of(root: pathlib.Path) -> pathlib.Path:
    """The project directory a root names: itself, or the directory a bare grader file sits in."""
    return root if root.is_dir() else root.parent if str(root.parent) else pathlib.Path(".")


def _assay_id(record) -> str:
    if isinstance(record, dict):
        return str(record.get("assay_id") or "")
    return str(getattr(record, "assay_id", "") or "")


def _artifact_pair(assay_path: pathlib.Path, root: pathlib.Path) -> dict:
    """The record, and the page beside it when the renderer actually wrote one.

    Relative to the project directory (`assays/<name>.assay.json`), which A-019 settles and the
    golden envelope carries. The envelope does outlive the working directory it was written from,
    which was attempt 2's reason for making these absolute; an absolute path does not survive the
    move either, because it names this machine's copy of the project rather than the project. What
    survives is the path from the project root, because the envelope already names the project: its
    `subject` carries the reward system and the version the record is of, and a consumer that keeps
    an envelope joins that project's directory with this path. Gate 39 reads `artifacts.assay` that
    way. The working-directory form the transcript shows is the text renderer's, made by joining
    the same two parts for a reader who is standing in this shell.
    """
    assay_path = pathlib.Path(os.path.abspath(assay_path))
    base = pathlib.Path(os.path.abspath(root))
    artifacts = {"assay": _under(assay_path, base)}
    report = assay_path.with_name(assay_path.name[: -len(RECORD_SUFFIX)] + REPORT_SUFFIX)
    if report.is_file():
        artifacts["report"] = _under(report, base)
    return artifacts


def _under(path: pathlib.Path, base: pathlib.Path) -> str:
    """`path` named from `base`, with posix separators, or absolute when it is not under it."""
    try:
        relative = path.relative_to(base)
    except ValueError:
        return str(path)
    return relative.as_posix()


def _written_at(root: pathlib.Path, record_id: str) -> pathlib.Path | None:
    project = _project_at(root)
    if project is not None:
        name = _name_holding(project, record_id)
        return project.path_for(name) if name is not None else None
    return _loose_record(root, record_id)


def _project_at(root: pathlib.Path):
    """The project rooted here, or None. A path that is not a project is not an error."""
    try:
        from reward_lens.store import CONFIG_NAME, Project
    except ImportError:  # pragma: no cover - the store is part of the same distribution
        return None
    if not (root / CONFIG_NAME).is_file():
        return None
    try:
        return Project.open(root)
    except Exception:
        return None


def _name_holding(project, record_id: str) -> str | None:
    latest = project.latest()
    names = project.names()
    if latest is not None and names and _assay_id(latest) == record_id:
        return names[-1]
    for name in reversed(names):
        try:
            if _assay_id(project.open_record(name)) == record_id:
                return name
        except Exception:
            continue
    return None


def _loose_record(root: pathlib.Path, record_id: str) -> pathlib.Path | None:
    """A bare-grader audit writes `assays/` beside the grader, with no index to ask."""
    directory = (root if root.is_dir() else root.parent) / "assays"
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob("*" + RECORD_SUFFIX), reverse=True)
    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("assay_id") == record_id:
            return candidate
    return None


def selftest_raise() -> None:
    """D-22's exit 7, reachable from a command line without patching anything.

    An internal failure is by definition a defect, so no ordinary invocation reaches it and the
    gate that has to demonstrate exit 7 has nothing to run. This is that invocation, and it is
    deliberately awkward to reach by accident: `REWARD_LENS_SELFTEST_RAISE` says what to raise,
    and it does nothing at all unless `REWARD_LENS_SELFTEST=1` is set alongside it. Set on its
    own it is refused as a bad value rather than honoured, so a stray variable in an environment
    can never turn a working install into one that reports a defect in itself.
    """
    message = os.environ.get(SELFTEST_RAISE)
    if not message:
        return
    if os.environ.get(SELFTEST_FLAG) != "1":
        from reward_lens import errors

        raise errors.make(
            "RL0003",
            field=SELFTEST_RAISE,
            detail=(
                f"{SELFTEST_RAISE} is the internal-failure self-test hook and is refused unless "
                f"{SELFTEST_FLAG}=1 is set with it"
            ),
        )
    raise RuntimeError(message)


#: A-016: the `Reused` fields the envelope carries verbatim, in the order A-016 names them.
REUSED_FIELDS = ("name", "assay_id", "subject_version", "record_path", "rendered_report", "wrote_manifest", "says")


def deliver_record(
    out: output.Output, record, *, command: str, artifacts=None, project=None, plan=None, reused=None
) -> int:
    """Turn an Assay into the envelope, the text panel and the exit code, in one place.

    A-026, superseding A-016 on this point: the exit code is the stored verdict's, the same code
    the measuring run gave, so two readers of one assay never disagree about whether it passed.
    D-28's "a no-op that exits 0 and says so" is read as "not an error, and says so": the saying is
    the `Reused` line and the seven `execution.reused` fields, and what a reused run never returns
    is an error class (4, 6). The verdict block, `decision` and the exit code all carry the same
    answer whether the work was done now or read back.

    A-019: no verb tells the renderer which layout to print. The layout follows the record, and the
    record is here, so `render` reads it off the record itself and a path never reaches that
    decision. `project` is only the directory `artifacts`' paths are relative to.
    """
    from ..format import text as text_format

    reused_fields = reused_from(reused)
    document = envelope_from_record(record, command=command, artifacts=artifacts, reused=reused_fields)
    rendered = (
        text_format.render(
            record,
            command=command,
            out=out,
            plan=plan,
            artifacts=artifacts,
            project=project,
            reused=reused_fields,
        )
        if out.format == "text"
        else ""
    )
    code = output.exit_code_for(document["decision"])
    return out.deliver(document=document, text=rendered, exit_code=code)


def reused_from(reused) -> dict | None:
    """The `Reused` object as the envelope carries it: those seven fields, verbatim, or nothing."""
    if reused is None:
        return None
    if isinstance(reused, dict):
        source = reused
    elif hasattr(reused, "model_dump"):
        source = reused.model_dump()
    elif dataclasses.is_dataclass(reused) and not isinstance(reused, type):
        source = dataclasses.asdict(reused)
    else:
        source = {field: getattr(reused, field, None) for field in REUSED_FIELDS}
    return {field: _plain(source.get(field)) for field in REUSED_FIELDS}


def _plain(value):
    return str(value) if isinstance(value, pathlib.PurePath) else value


def envelope_from_record(record, *, command: str, artifacts=None, reused=None) -> dict:
    """Section 5.7's envelope, read off the record and nothing else."""
    doc = record.to_dict() if hasattr(record, "to_dict") else dict(record)
    provenance = doc.get("provenance") or {}
    cost = doc.get("cost") or {}
    return output.envelope(
        command=command,
        execution={
            "state": "complete",
            "started": (doc.get("created") or None),
            "duration_s": cost.get("wall_s"),
            "sandbox_tier": provenance.get("sandbox_tier"),
            "offline": provenance.get("offline"),
            "spend_usd": cost.get("usd"),
            "reused": reused,
        },
        subject=_subject(doc),
        decision=_decision(doc),
        findings=[
            {
                "id": finding.get("id"),
                "rule": finding.get("rule"),
                "level": finding.get("level"),
                "kind": finding.get("kind"),
                "scope": finding.get("scope"),
                "entry_kind": _entry_kind(doc, finding),
                "arm": finding.get("arm"),
                "witness": finding.get("witness_path"),
            }
            for finding in doc.get("findings") or []
        ],
        holes=_dark_sections(doc),
        artifacts=artifacts or {},
    )


#: A-025 point 1: the projection section 5.7 shows, in this order. The record's `decision` also
#: carries `tradeoff`, `signature` and `regression_cases`; a reader of the envelope is deciding
#: whether to train on this version, and those three are the record's own bookkeeping.
DECISION_FIELDS = ("state", "action", "policy", "reasons")


def _decision(doc: dict) -> dict | None:
    """The four fields, `reasons` in the order the record wrote them. A record with no decision
    (a run that failed before one) keeps the envelope's null rather than an empty projection."""
    decision = doc.get("decision")
    if decision is None:
        return None
    return {field: decision.get(field) for field in DECISION_FIELDS}


def _subject(doc: dict) -> dict:
    """A-025 point 4: what the Subject line names, in machine form.

    `reward_system` is the declared name and `version` is the label that line prints, so the two
    read the record the way the text layout reads it: the envelope and the transcript name the
    same system, and a reader who saw `code-reward v1` on screen finds `code-reward` and `v1` here
    rather than the version id twice.
    """
    from ..format import text as text_format

    subject = doc.get("subject") or {}
    system = subject.get("reward_system") or {}
    version = subject.get("version") or {}
    return {
        "reward_system": system.get("name") or system.get("id"),
        "version": text_format._version_label(system.get("name"), version.get("id")) or None,
        "version_digest": version.get("digest"),
        "policy": (subject.get("context") or {}).get("policy"),
        "task_set": _task_set(doc),
    }


def _task_set(doc: dict) -> str | None:
    """`<task file>@<digest>`: the file the run was pointed at, and what it hashed to.

    A bare grader has no task set and gets null. A record that carries the digest but no file name
    (nothing writes one today except the audit) keeps the digest alone: a file name the record does
    not hold is not one this function can invent.
    """
    from ..format import text as text_format

    subject = doc.get("subject") or {}
    digest = (subject.get("context") or {}).get("task_set")
    if not digest:
        return None
    named = text_format._extension(doc).get("subject") or {}
    task_file = named.get("task_file")
    return f"{task_file}@{digest}" if task_file else digest


def _entry_kind(doc: dict, finding: dict) -> str | None:
    """The kind of the entry the finding stands against, which is what makes it a finding."""
    named = set(finding.get("entries") or ())
    for entries in (doc.get("measurement") or {}).values():
        for entry in entries or ():
            if entry.get("entry_id") in named:
                return entry.get("kind")
    return None


def _entries_by_id(doc: dict) -> dict:
    """The record's measurement entries by id. The `holes` index names an entry and the entry is
    what carries the absence's own notes, so a row of the index is read through this."""
    return {
        entry.get("entry_id"): entry
        for entries in (doc.get("measurement") or {}).values()
        for entry in entries or ()
        if entry.get("entry_id")
    }


def _dark_sections(doc: dict) -> list:
    """One row per section that is wholly dark, not one per absent entry.

    A hole in the envelope is a panel a reader cannot read, and the reader counts panels. The
    record's own `holes` index is the per-entry list, which counts eight sections as fifteen
    absences and makes a build with one dark panel and seven absent checks look worse than a build
    with seven dark panels. The section's reason and remedy are its absences', which agree inside
    a section because the section went dark for one reason.

    A-025 point 3: `missing` is the absence's code and `remedy` its short form, both read off the
    entry the index names. A record written before the codes existed carries neither, and its row
    falls back to the sentence the record does carry: the envelope says what that record says, and
    a code it never wrote is not one to supply on its behalf.
    """
    from ..format import text as text_format

    entries = _entries_by_id(doc)
    rows = []
    for section, _label in text_format.CANONICAL:
        if text_format.glyph_for(doc, section) != text_format.DARK:
            continue
        holes = [hole for hole in doc.get("holes") or () if hole.get("section") == section]
        if not holes:
            continue
        hole = holes[0]
        block = text_format._extension(entries.get(hole.get("entry_id")))
        rows.append(
            {
                "section": section,
                "state": hole.get("state"),
                "missing": block.get("missing_code") or hole.get("missing_access"),
                "remedy": block.get("remedy_short") or hole.get("remedy"),
            }
        )
    return rows


def refuse_without_yes(out: output.Output, *, decision: str, argv: list[str], command: str) -> None:
    """D-31: a destructive or expensive step with no TTY and no --yes is a pending action, not a hang."""
    if out.yes:
        return
    error = output.pending_decision(decision=decision, argv=argv)
    error.context["command_name"] = command
    raise error


def budget_guard(out: output.Output, *, seeker: str, max_budget_usd, command: str) -> None:
    """D-33: no paid work starts without a cap that could cover it."""
    if seeker in (None, "off", "local"):
        return
    if max_budget_usd is None:
        raise _no_budget(command)
    try:
        cap = float(str(max_budget_usd))
    except ValueError as bad:
        from reward_lens import errors

        raise errors.make("RL0003", field="--max-budget-usd", detail=str(max_budget_usd)) from bad
    if cap <= 0:
        from reward_lens import errors

        raise errors.make(
            "RL0501",
            cap=str(max_budget_usd),
            spent="0.00",
            detail="the cap cannot cover a paid seeker, so no paid work was started",
            command=command,
        )


def _no_budget(command: str):
    from reward_lens import errors

    return errors.make(
        "RL0501",
        cap="none",
        spent="0.00",
        detail="a paid seeker needs --max-budget-usd, because no paid work starts without a named budget",
        command=command,
    )


def not_in_this_build(what: str, *, remedy: str = "a later build"):
    from reward_lens import errors

    return errors.make("RL0701", extra=what, capability=what, detail=remedy)


def identifier(value: str, *, field: str) -> str:
    return ids.validate_identifier(value, field=field)
