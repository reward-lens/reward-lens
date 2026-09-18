"""The audit spine: ten panels, and every one of them either runs or writes its absence.

Section 7.3 and D-41. The default path is a local grader, no key, no network, no paid call. Wave 1
fills two panels for real: `validity` with the four static checks, D10 replay determinism and D1
decision coverage, and `reach` with D8's static exposure inventory, which is labelled static and is
never an executed reach claim. Everything else is an `absence` entry with its missing access and
its remedy, and `holes` is rebuilt from those entries so the index cannot drift from them.

The one forbidden thing is a panel that prints a plausible number it did not compute. Every number
in the record comes back from an instrument; every instrument that refused is a hole carrying the
refusal's own words; and every exception from loading or calling a user's grader is a
`COULD_NOT_CHECK` hole naming it, never a zero and never `inconclusive`.
"""

from __future__ import annotations

import hashlib
import inspect
import platform
import re
import sys
import time
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from reward_lens import contracts
from reward_lens.instruments.base import RunContext, utc_now
from reward_lens.verifier import VerifierUnderTest

from . import absences, instruments, registry, static_checks
from .plan import SECTIONS as _SECTIONS
from .plan import estimate_s as _estimate_s
from .plan import panels as _panels

__all__ = ["Reused", "last_reuse", "run"]

#: The finding id each code is recorded under. `RGX-local-0001` is the writable-then-read finding.
FINDING_IDS: dict[str, str] = {
    "RL0201": "RGX-local-0001",
    "RL0202": "RGX-local-0002",
    "RL0203": "RGX-local-0003",
    "RL0210": "RGX-local-0010",
    "RL0211": "RGX-local-0011",
    "RL0212": "RGX-local-0012",
    "RL0301": "RGX-local-0301",
}

#: The finding codes wave-2 interfaces section 6 allocates to a panel. A discovered panel raises its
#: own findings and the runner carries them; what the runner will not do is carry a code the
#: catalogue does not own under a panel's name, because `explain <code>` would then have nothing to
#: say about a code a reader just read off the record. A finding that declares no code is not
#: checked here: an id in the local band (P-REACH's `RGX-local-12xx`) is a name, not a catalogue
#: claim, and section 6 allocates codes, not ids.
PANEL_FINDING_CODES: frozenset[str] = frozenset(
    ["RL0213", "RL0214", "RL0215"]
    + [f"RL02{number}" for number in range(21, 30)]
    + ["RL0231"]
    + [f"RL02{number}" for number in range(41, 46)]
    + ["RL0251", "RL0261"]
)



#: The reward system's name when nothing survives the cleaning. Stated once, not invented at the
#: call site, so every path that has to produce a name lands on the same string.
DEFAULT_SYSTEM_NAME = "reward-system"

#: The bundle manifest's file name. It is written beside the record it names, and every path in it
#: is relative to it, so the digest the page carries survives the bundle being copied or moved.
MANIFEST_NAME = "manifest.json"

#: How many times `_write` may seal the record while the manifest digest settles. Three is what
#: the settling takes: no digest, then the digest that produced, then the pass that repeats it.
#: The rest is slack, and running out of it is a failure rather than a fallback.
MANIFEST_PASSES = 6

#: How much of the subject version's digest goes into a record's name. Eight hex is what tells two
#: versions of one subject apart on a day; it is a label on the file, not the identity, which stays
#: the whole digest inside the record.
NAME_DIGEST_CHARS = 8


@dataclass(frozen=True)
class Reused:
    """The run that measured nothing: which stored record answered, and what made it the answer.

    This is the run's provenance rather than the record's. Nothing inside the sealed record says a
    word about reuse, because the record returned is the one already on disk, byte for byte, and a
    measurement that says which of the runs that handed it over is talking is not the same
    measurement. So the sentence lives beside the record, on the run, and `last_reuse()` is where a
    caller reads it. Section 5.7's `ResultEnvelope.execution` is where it belongs on the way out,
    and `contracts`, `api` and `cli` are frozen for this packet; the handoff proposes the field.
    """

    #: The stored record's name, which is also the stem of the report and the record file.
    name: str
    #: The record that answered. Unchanged: reuse never writes a new id.
    assay_id: str
    #: The subject version both the store and the project now agree on.
    subject_version: str
    #: The record file on disk, as a string so the report is JSON without a converter.
    record_path: str
    #: True when the bundle had lost its report and this run composed it again from the record.
    rendered_report: bool
    #: True when the bundle had lost its manifest and this run wrote it again from the record.
    wrote_manifest: bool
    #: The one sentence a surface can print without knowing any of the above.
    says: str


#: Set by every `run`: a `Reused` when the audit returned a stored record, `None` when it measured.
#: A context variable rather than a module global so a second audit on another thread or task
#: cannot answer for this one; a thread starts on an empty context, so its `set` stays its own.
_REUSE: ContextVar[Reused | None] = ContextVar("reward_lens_audit_reuse", default=None)


def last_reuse() -> Reused | None:
    """What the last `run` in this context did instead of measuring, or `None` if it measured."""
    return _REUSE.get()


def method_identity(method: Any) -> str:
    """What tells one measurement method from another: its rule id and its parameter digest.

    The pair, not either half. The id alone says which rule was applied and not how: two audits
    that ran `reach.oversight` with different probe budgets both write `id="reach.oversight"`. The
    parameter digest alone is not a name, and `contracts.digest` of an empty parameter set is the
    same eight-hundred-times-over digest for every method that takes no parameters. `version` is
    deliberately not in here: it is the method's own release string, and an instrument that fixes a
    typo in its procedure text has not measured anything differently.
    """
    return f"{getattr(method, 'id', '')}@{getattr(method, 'params_digest', '')}"


def _requested(discovered: Iterable[Any]) -> dict[str, str | None]:
    """Entry id -> the method identity the plan would write under it, over the discovered panels.

    This is the request's half of section 6.1's question, and it has to be answerable before any
    instrument runs, so it is read off what the instruments declare rather than off what they
    return. The entry id is the one the runner itself uses at the call site: `instrument.id`, which
    is what `guarded_many` labels the instrument's entries and its absence with. The identity is
    `method_identity` of the `method` an instrument declares up front, or `None` where it declares
    none, meaning the plan can say which instrument will run but not with what parameters, and the
    check falls back to coverage for that one.

    Panel names are deliberately not the key. A panel is a section of the record and a section can
    hold many methods, so keying on the section answers "has a panel landed" and cannot answer "has
    a method landed inside a panel that was already there" -- the case the principal's reviewer
    reproduced, where a second instrument registered in an already-populated `reach` was handed the
    old assay and nothing ran.
    """
    wanted: dict[str, str | None] = {}
    for panel in discovered:
        section = str(getattr(panel, "section", ""))
        for instrument in getattr(panel, "instruments", ()):
            entry_id = str(getattr(instrument, "id", f"{section}.instrument"))
            declared = getattr(instrument, "method", None)
            wanted[entry_id] = None if declared is None else method_identity(declared)
    return wanted


def _identities_under(record: contracts.Assay, entry_id: str) -> frozenset[str]:
    """Every method identity the record holds under `entry_id`; empty when it holds none.

    An instrument's entry ids are its own id or a name below it: `soundness.battery` returning
    three entries writes `soundness.battery.0` and its siblings, and returning nothing leaves the
    runner's absence under `soundness.battery` exactly. So the record answers for an instrument
    when it holds either, and the prefix carries the separating dot so that `reach.oversight` is
    not read as covering `reach.oversight_added`, which is a different instrument.
    """
    prefix = f"{entry_id}."
    return frozenset(
        method_identity(entry.method)
        for entry in record.entries()
        if entry.entry_id == entry_id or entry.entry_id.startswith(prefix)
    )


def _answers(previous: contracts.Assay, wanted: dict[str, str | None]) -> bool:
    """Whether `previous` measured what the request would measure (section 6.1).

    Containment one way only: every requested entry id has to be in the record, measured by the
    identity the request names where it names one. What the record holds and the request does not
    ask for does not disqualify it, because a record of a wider build answers a narrower request
    on the part that was asked -- and a method that changed inside a panel nobody asked about
    changed nothing this request can see.
    """
    for entry_id, identity in wanted.items():
        held = _identities_under(previous, entry_id)
        if not held:
            return False
        if identity is not None and identity not in held:
            return False
    return True


def _standing_record(
    project: Any, *, discovered: Iterable[Any] = ()
) -> tuple[str, contracts.Assay] | None:
    """The store's record of the subject as it stands now, if it already holds one.

    The index is the list of candidates and the newest is asked first, because a project that has
    moved and moved back wants the measurement of where it is, not the oldest one that matches.
    A candidate is the answer only when three things hold, and they are not one thing. The project,
    asked for its version through that record's own method set, has to produce exactly the version
    digest the record claims. And `changed_since` has to be empty: that is the store's own
    dependency check over the nine digests, and it is the question the brief asks, "has anything
    the record depends on changed". And the record has to have measured what this request would
    measure, which `_answers` asks as the pairs (entry id, method identity) the plan would produce
    -- the same key the store's `_check_not_a_rewrite` refuses a rewrite on, so a record that
    answers here is exactly a record a second write would have been refused as a rewrite of.

    A file the store cannot open is not an answer and not an error either. `_check_not_a_rewrite`
    already steps over unreadable and non-record files in the same directory, and a reuse check
    that raised where the write path shrugs would make an audit fail on a stray file. Nothing is
    suppressed by stepping over it: a project whose config or grader is the reason a candidate
    could not be versioned falls through to the measuring path, which asks the same questions and
    raises there.
    """
    if project is None:
        return None
    wanted = _requested(discovered)
    try:
        names = project.names()
    except OSError:
        return None
    for name in reversed(names):
        try:
            previous = project.open_record(name)
            current = project.version_of(previous).digest()
        except (OSError, ValueError, contracts.RewardLensError):
            # `ValueError` covers a file that is not JSON; `RecordInvalid` (RL0604) covers a file
            # that is JSON and is not a record, which is what a decoy in `assays/` looks like.
            continue
        if current != previous.subject.version.digest:
            continue
        if project.changed_since(previous):
            continue
        if not _answers(previous, wanted):
            # Section 6.1: an unchanged dependency set is not enough. A record that answers has to
            # have measured what this request would measure. Three ways it has not: a panel that
            # was dark when the record was written and is filled now, a method added beside one
            # that was already in a populated panel, and a method whose parameters have moved since.
            # In all three, handing the record back would answer a wider or a different question
            # with the old measurement while reporting that no instrument ran.
            continue
        return name, previous
    return None


def _reuse(name: str, previous: contracts.Assay, *, project: Any) -> contracts.Assay:
    """Hand back the stored record and complete its bundle, without rewriting anything on disk.

    The record is the file: it is not re-serialised, so the bytes a caller gets back are the bytes
    a reader has. The manifest and the report are functions of those bytes and of nothing else, so
    a bundle that lost either can have it back exactly as it was first written, which is why the
    two are composed here rather than copied. A bundle that still has them is left alone, including
    the case where `manifest.json` has since been replaced by a later audit of a changed version:
    rewriting it would only move the superseded report from this bundle to that one.
    """
    from reward_lens.render.report import render_to

    path = project.path_for(name)
    payload = path.read_bytes()
    manifest = _manifest(
        record_name=path.name, payload=payload, assay_id=previous.assay_id
    )
    digest = contracts.digest(manifest)

    manifest_path = path.with_name(MANIFEST_NAME)
    wrote_manifest = not manifest_path.is_file()
    if wrote_manifest:
        manifest_path.write_bytes(manifest_bytes(manifest))

    report_path = path.with_name(f"{name}.assay.html")
    rendered_report = not report_path.is_file()
    if rendered_report:
        render_to(previous, report_path, bundle_manifest_digest=digest)

    _REUSE.set(
        Reused(
            name=name,
            assay_id=previous.assay_id,
            subject_version=previous.subject.version.digest,
            record_path=str(path),
            rendered_report=rendered_report,
            wrote_manifest=wrote_manifest,
            says=(
                f"no instrument ran: {name} already measures subject version "
                f"{previous.subject.version.digest}, and nothing it depends on has changed"
            ),
        )
    )
    return previous


def _clean(text: str) -> str:
    """The cleaning `_ident` does, without the identifier fallback `_ident` ends on."""
    return re.sub(r"[^A-Za-z0-9_.:/-]", "-", text).strip("-")[:200]


def _ident(text: str) -> str:
    cleaned = _clean(text)
    return cleaned if cleaned and cleaned[0].isalnum() else "subject"


def _resolved_name(root: Any) -> str:
    """The project directory's own name, resolved first.

    `Project.open` keeps the root exactly as it was handed in, so a project opened as `.` has
    `Path(root).name == ""`. Resolving is what gives that directory a name to be known by.
    """
    try:
        return Path(root).resolve().name
    except OSError:
        return Path(root).name


def _system_name(declared: Any, root: Any) -> str:
    """The reward system's name, total over every root a project can be opened from.

    A config that declares a name uses it. One that does not is named for its project directory,
    resolved and then cleaned the way `_ident` cleans. When nothing survives that,
    `DEFAULT_SYSTEM_NAME` is the name. The empty string is never returned, which is what
    `RewardSystemRef.name` refused with `string_too_short`.
    """
    if declared:
        cleaned = _clean(str(declared))
        if cleaned:
            return cleaned
    return _clean(_resolved_name(root)) or DEFAULT_SYSTEM_NAME


def _os_name() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    return "macos" if sys.platform == "darwin" else "windows"


@dataclass
class AuditSubject:
    """What the audit is pointed at, once the request and the project have been reconciled."""

    grader_path: Path
    entrypoint: str
    project_dir: Path | None
    tasks_path: Path | None
    responses_path: Path | None
    trainer: str | None
    name: str
    source: str


def resolve(req: Any, project: Any) -> AuditSubject:
    """The grader, its entrypoint and its inputs. A grader that is not there is RL0001."""
    path = Path(req.path)
    entrypoint = "score"
    tasks_path = Path(req.tasks) if getattr(req, "tasks", None) else None
    responses_path = Path(req.responses) if getattr(req, "responses", None) else None
    trainer = None
    if project is not None:
        root = Path(project.root)
        config = project.config
        entry = str(config.reward.entry)
        if ":" in entry:
            module, _, entrypoint = entry.partition(":")
            grader_path = (root / module.replace(".", "/")).with_suffix(".py")
        else:
            grader_path = root / entry
        trainer = getattr(config.reward, "trainer", None)
        if tasks_path is None and config.tasks is not None:
            tasks_path = root / config.tasks.path
        if responses_path is None and getattr(config, "responses", None) is not None:
            responses_path = root / config.responses.path
        project_dir: Path | None = root
        name = _ident(_resolved_name(root))
    elif path.is_dir():
        raise contracts.UsageError(
            code="RL0001",
            message=f"{path} is not a reward-lens project: it holds no rewardlens.yaml",
            remediation=(
                "run `reward-lens init --example code-reward <dir>` to make one, or point the "
                "audit straight at a grader file"
            ),
            context={"path": str(path)},
        )
    else:
        grader_path = path
        project_dir = None
        name = _ident(path.name)
    if not grader_path.is_file():
        raise contracts.UsageError(
            code="RL0001",
            message=f"no grader at {grader_path}",
            remediation=(
                "point the audit at a Python file that defines the scoring function, or at a "
                "project whose rewardlens.yaml names one under `reward.entry`"
            ),
            context={"path": str(grader_path)},
        )
    if project is None:
        project_dir = grader_path.parent if grader_path.parent != Path("") else None
    return AuditSubject(
        grader_path=grader_path,
        entrypoint=entrypoint,
        project_dir=project_dir,
        tasks_path=tasks_path,
        responses_path=responses_path,
        trainer=trainer,
        name=name,
        source=grader_path.read_text(encoding="utf-8"),
    )


def _static_entry(
    result: static_checks.StaticResult,
    ctx: RunContext,
    *,
    subject_ref: str,
    source_name: str,
    source_digest: str,
) -> contracts.Entry:
    """A static check as a `check`, or, when it failed, as a `witness` by artifact identity."""
    method = contracts.Method(
        id=result.entry_id,
        version="1.0.0",
        params_digest=contracts.digest({"check": result.entry_id, "predicate": result.predicate}),
        procedure=(
            "the grader's own source is parsed with the standard library's ast and the check's "
            "predicate is evaluated over the call sites and branches it names; nothing is executed"
        ),
    )
    common: dict[str, Any] = {
        "entry_id": result.entry_id,
        "section": "validity",
        "kind": "check",
        "measurand": result.measurand,
        "method": method,
        "scope": "evaluator_comparison",
        "subject_ref": subject_ref,
        "depends_on": ["digest:source"],
        "state": "complete",
        "provenance": ctx.provenance(arm="white_box"),
        "limitations": list(result.limitations),
        "result": dict(result.result) | {"observed": result.observed},
    }
    if result.passed:
        return contracts.Entry(
            **common,
            check=contracts.Check(
                predicate=result.predicate, passed=True, scope_tested=result.scope_tested
            ),
        )
    common["kind"] = "witness"
    return contracts.Entry(
        **common,
        witness=contracts.Witness(
            inputs={
                "source": source_name,
                "line": result.line,
                "path": result.target_path,
                "predicate": result.predicate,
            },
            procedure=method.procedure,
            observed=result.observed,
            artifact_identity=source_digest,
            path=source_name,
        ),
    )


def _explain_writable_then_read(
    entry: contracts.Entry, result: static_checks.StaticResult, *, n_responses: int
) -> contracts.Entry:
    """The headline and the paragraph a reader gets for RL0201, from the numbers measured (A-015).

    The two sentences are not the finding's message repeated. The message says what was read; the
    headline says what it means, and the paragraph says what is still not established, because a
    static read of the source cannot say whether any sampled response actually took the path. The
    response count is the corpus's, and when there is no corpus the sentence does not invent one.
    """
    directory = str(result.target_path or "").rsplit("/", 1)[0]
    where = f"{directory}/" if directory else str(result.target_path or "the path it reads")
    used_it = (
        f"Whether any of the {n_responses} responses used that path"
        if n_responses
        else "Whether any response used that path"
    )
    entry.result = dict(entry.result or {}) | {
        "headline": "This reward reads a file the graded process can write.",
        "explanation": (
            f"{result.source_name} line {result.line} runs the tests in {where}, and {where} sits "
            f"inside the directory the graded response runs in. {used_it} is not established "
            f"here: the executed witness is not in this build."
        ),
    }
    return entry


def _finding(
    code: str,
    *,
    entry_id: str,
    message: str,
    level: str,
    kind: str,
    rationale: str,
    arm: str,
    rule: str | None = None,
    file: str | None = None,
    line: int | None = None,
) -> contracts.Finding:
    """One finding. `rule` is the name it is raised under and defaults to the entry id.

    The two are not the same name. The entry id is where the hole or the witness sits in the
    record, and one measurand can be asked by more than one arm: the rule names the arm, the entry
    id does not move, and `entries` still points at the entry the finding came from (A-025 point 2).
    """
    finding = contracts.Finding(
        id=FINDING_IDS[code],
        rule=rule or entry_id,
        level=level,  # type: ignore[arg-type]
        kind=kind,  # type: ignore[arg-type]
        severity_rationale=rationale,
        scope="evaluator_comparison",
        entries=[entry_id],
        partial_fingerprint=contracts.digest({"code": code, "entry": entry_id, "message": message})
        .removeprefix("sha256:")[:16],
        message=message,
        code=code,
        arm=arm,  # type: ignore[arg-type]
    )
    if file is not None and line is not None:
        finding.location = contracts.models.FindingLocation(file=file, line=int(line))
    return finding


def _panel_package(instrument: Any) -> Any:
    """The instrument package a module-level `findings_for` would be declared on.

    Discovery imports `reward_lens.instruments.<name>` and reads `PANEL` off it, so the package is
    what section 8.1 means by "the package": the module the class is defined in may be a private one
    below it (P-REACH's instrument lives in `reach.probes`), and both packages re-export the
    function, so the harvest asks the package and not the defining module.
    """
    defined_in = str(getattr(type(instrument), "__module__", ""))
    prefix = f"{registry.PACKAGE}."
    if defined_in.startswith(prefix):
        defined_in = prefix + defined_in[len(prefix) :].split(".", 1)[0]
    return sys.modules.get(defined_in)


def _wants_every_entry(harvest: Any) -> bool:
    """Whether a package's `findings_for` takes all the entries at once rather than one of them.

    Section 8.1 names one shape for this form, `findings_for(entry)`, and P-FRAME's is that.
    P-REACH's is `findings_for(entries: Sequence[Entry])`, which is neither of the two forms the
    section allocates; called per entry it would iterate a single `Entry` and the panel's findings
    would be lost behind the exception. What the function declares is read rather than guessed at,
    so the harvest is decided by the signature and not by the package's name, and the disagreement
    is booked against the package rather than silently normalised.
    """
    try:
        parameters = list(inspect.signature(harvest).parameters.values())
    except (TypeError, ValueError):  # a callable that will not introspect: assume the named form
        return False
    positional = [
        one
        for one in parameters
        if one.kind in (one.POSITIONAL_ONLY, one.POSITIONAL_OR_KEYWORD)
    ]
    if not positional:
        return False
    first = positional[0]
    annotation = str(first.annotation).lower()
    return first.name.endswith("entries") or any(
        hint in annotation for hint in ("sequence", "iterable", "list[", "tuple[")
    )


def _panel_findings(instrument: Any, produced: list[contracts.Entry]) -> list[contracts.Finding]:
    """Every finding a discovered instrument raises over the entries it has just produced (8.1).

    A panel that measures something and cannot say what it means is half a panel: the entry carries
    the number and the finding carries the judgement, and without this the judgement never reached
    `Assay.findings`, the verdict's reasons or the text surface. The two forms are the section's:
    `findings(entries)` on the instrument object, or `findings_for` on the package. An instrument
    that exposes neither raises nothing, which is a panel whose entries speak for themselves and not
    an error.
    """
    if not produced:
        return []
    own = getattr(instrument, "findings", None)
    if callable(own):
        return list(own(list(produced)))
    package = _panel_package(instrument)
    harvest = getattr(package, "findings_for", None)
    if not callable(harvest):
        return []
    if _wants_every_entry(harvest):
        return list(harvest(list(produced)))
    raised: list[contracts.Finding] = []
    for entry in produced:
        raised.extend(harvest(entry))
    return raised


def _unallocated_codes(raised: Iterable[contracts.Finding]) -> list[str]:
    """The codes among `raised` that section 6 allocates to no panel, in the order they appear."""
    loose: list[str] = []
    for finding in raised:
        code = getattr(finding, "code", None)
        if code and code not in PANEL_FINDING_CODES and code not in loose:
            loose.append(str(code))
    return loose


def _code_allocation_check(
    ctx: RunContext, *, entry_id: str, codes: list[str], subject_ref: str
) -> contracts.Entry:
    """A failed `check` naming the codes a panel raised that section 6 allocates to no panel."""
    named = ", ".join(codes)
    predicate = "every finding a discovered panel raises declares a code wave 2 allocates to a panel"
    return contracts.Entry(
        entry_id="validity.panel_finding_codes",
        section="validity",
        kind="check",
        measurand="the finding codes a discovered panel declares",
        method=contracts.Method(
            id="validity.panel_finding_codes",
            version="1.0.0",
            params_digest=contracts.digest({"allocated": sorted(PANEL_FINDING_CODES)}),
            procedure=(
                "each finding a discovered instrument raised is read for its code and the code is "
                "looked up in the allocation wave-2 interfaces section 6 writes down; nothing is "
                "executed and the finding is carried either way"
            ),
        ),
        scope="evaluator_comparison",
        subject_ref=subject_ref,
        depends_on=("digest:source",),
        state="complete",
        provenance=ctx.provenance(arm="static"),
        check=contracts.Check(
            predicate=predicate,
            passed=False,
            scope_tested=f"the findings {entry_id} raised in this run",
        ),
        result={
            "instrument": entry_id,
            "unallocated_codes": codes,
            "observed": (
                f"{entry_id} raised a finding under {named}, which wave-2 interfaces section 6 "
                "allocates to no panel; the finding is in the record and `explain` has nothing to "
                "say about the code"
            ),
        },
    )


def run(req: Any, *, project: Any = None, sandbox: Any = None) -> contracts.Assay:
    """Measure one reward system and return the record (section 7.3, D-41)."""
    started_monotonic = time.monotonic()
    started = utc_now()
    _REUSE.set(None)
    subject = resolve(req, project)
    # Before any instrument, and after `resolve`, which is what turns a path into a subject: a
    # grader that is not there is RL0001 whether or not the store holds an old record of it.
    # D-28 and the store's "one subject version, one measurement" say a second measurement of an
    # unchanged subject is not a second measurement; asking first is what makes that true instead
    # of making the second audit an RL0620.
    # Discovery first, because the reuse check is scope-aware: what a stored record has to cover
    # is the plan this request would run, and the plan is not known until the panels are. One
    # discovery pass serves the reuse check, the absences and the panels themselves.
    discovered = registry.discover()
    filled = _panels(discovered)
    standing = _standing_record(project, discovered=discovered)
    if standing is not None:
        return _reuse(standing[0], standing[1], project=project)
    ctx = RunContext(
        sandbox=sandbox,
        project=project,
        budget="0.00",
        clock=time.monotonic,
        offline=bool(getattr(req, "offline", True)),
        started=started,
    )
    source_name = subject.grader_path.name
    source_digest = contracts.digest({"source": subject.source})
    subject_ref = source_digest
    vut = VerifierUnderTest(source_path=subject.grader_path, entrypoint=subject.entrypoint)

    corpus = instruments.build_corpus(
        tasks_path=subject.tasks_path, responses_path=subject.responses_path
    )
    n_responses = len(corpus.rollouts)
    tasks_for_probes = corpus.tasks
    if not corpus.rollouts:
        tasks_for_probes = ()
        if subject.tasks_path is not None and subject.tasks_path.is_file():
            tasks_for_probes = tuple(instruments.read_jsonl(subject.tasks_path, cap=1))
        corpus = instruments.probe_corpus(tasks_for_probes)
        has_bank = False
    else:
        has_bank = True

    # What this run was actually given, which is what an absence names when a reader can close it.
    # A task set is a task set whether or not a response bank was built from it, so these are read
    # off the request rather than off the corpus, which conflates the two.
    has_tasks = subject.tasks_path is not None and subject.tasks_path.is_file()
    has_responses = subject.responses_path is not None and subject.responses_path.is_file()
    has_outcome = project is not None and getattr(project.config, "outcome", None) is not None
    supplied = tuple(
        name
        for name, present in (
            (absences.TASK_SET, has_tasks),
            (absences.SAMPLED_RESPONSES, has_responses),
            (absences.PROTECTED_CHECK, has_outcome),
        )
        if present
    )  # no run record reaches `audit`: that is what `trace` is for, and its absence says so

    entries: dict[str, list[contracts.Entry]] = {section: [] for section in _SECTIONS}
    findings: list[contracts.Finding] = []
    rules: dict[str, contracts.Rule] = {}

    # The grader is loaded once, before any instrument that needs it, and the failure is held.
    # Letting each instrument discover the failure for itself wrote a different hole per entry: the
    # first named the real `ModuleNotFoundError`, and the rest named whatever they hit downstream of
    # it (an adapter holding `None`), which named neither the module nor the import that raised it.
    # One load means one true statement, repeated on every entry that depended on it, while the
    # static checks, which never touch the loaded module, still run and still pass.
    grader_failure: BaseException | None = None
    try:
        vut.load()
    except Exception as failure:  # the hole is the audit's answer; a failed grader is not a crash
        grader_failure = failure

    def guarded(
        entry_id: str,
        section: str,
        measurand: str,
        work,
        remedy: str,
        *,
        needs_grader: bool = True,
    ) -> None:
        if needs_grader and grader_failure is not None:
            entries[section].append(
                absences.could_not_check(
                    ctx,
                    section=section,
                    entry_id=entry_id,
                    measurand=measurand,
                    failure=grader_failure,
                    remedy=remedy,
                    subject_ref=subject_ref,
                    affected_claims=(f"no {section} claim from {entry_id}",),
                )
            )
            return
        try:
            entries[section].append(work())
        except Exception as failure:  # the audit catches it; the instrument does not swallow it
            entries[section].append(
                absences.could_not_check(
                    ctx,
                    section=section,
                    entry_id=entry_id,
                    measurand=measurand,
                    failure=failure,
                    remedy=remedy,
                    subject_ref=subject_ref,
                    affected_claims=(f"no {section} claim from {entry_id}",),
                )
            )

    def guarded_many(
        entry_id: str, section: str, measurand: str, work, remedy: str
    ) -> list[contracts.Entry]:
        """Every entry an instrument returned, not the first of them.

        Returns what the instrument itself produced, which is empty when it failed or returned
        nothing: the absence written in either case is the runner's own entry, and the caller
        harvests findings from the instrument's entries, never from the runner's hole.

        An instrument that returns nothing measured nothing, and that is what goes in the record: an
        absence naming the instrument. Keeping `[0]` here dropped every entry after the first, and
        turned an empty return into an `IndexError` the guard relabelled as the grader failing to
        load, which is a different and untrue statement about the subject.
        """
        try:
            produced = list(work())
        except Exception as failure:  # the audit catches it; the instrument does not swallow it
            entries[section].append(
                absences.could_not_check(
                    ctx,
                    section=section,
                    entry_id=entry_id,
                    measurand=measurand,
                    failure=failure,
                    remedy=remedy,
                    subject_ref=subject_ref,
                    affected_claims=(f"no {section} claim from {entry_id}",),
                )
            )
            return []
        if not produced:
            entries[section].append(
                ctx.absence(
                    section,
                    entry_id,
                    measurand,
                    f"{entry_id} ran and returned no entry",
                    remedy,
                    (f"no {section} claim from {entry_id}",),
                    subject_ref=subject_ref,
                )
            )
            return []
        entries[section].extend(produced)
        return produced

    # --- validity: the three source checks ------------------------------------------------------
    static_results = static_checks.run_static_checks(
        subject.source,
        source_name=source_name,
        entrypoint=subject.entrypoint,
        tasks=list(corpus.tasks) if has_bank else None,
    )
    for result in static_results:
        entries["validity"].append(
            _static_entry(
                result,
                ctx,
                subject_ref=subject_ref,
                source_name=source_name,
                source_digest=source_digest,
            )
        )
        code = result.code
        if code:
            if code == "RL0201":
                _explain_writable_then_read(
                    entries["validity"][-1], result, n_responses=n_responses
                )
            findings.append(
                _finding(
                    code,
                    entry_id=result.entry_id,
                    rule=result.rule,
                    message=result.message,
                    level="error" if code in {"RL0201", "RL0202"} else "warning",
                    kind="fail",
                    rationale=(
                        "a static reading of the grader's own source, with the line named; the "
                        "black-box arm did not run, so this says nothing about whether a sampled "
                        "response used it"
                    ),
                    arm="static",
                    file=source_name,
                    line=result.line,
                )
            )
            rules[result.entry_id] = contracts.Rule(
                id=result.entry_id,
                name=result.entry_id.split(".", 1)[1].replace("_", " "),
                short_description=result.predicate,
            )

    # --- validity: the fourth static check, which executes ---------------------------------------
    guarded(
        "validity.input_handling",
        "validity",
        "what the grader returns on malformed, empty and oversized input",
        lambda: instruments.input_handling_entry(
            subject.grader_path,
            subject.entrypoint,
            corpus,
            ctx,
            subject_ref=subject_ref,
            trainer=subject.trainer,
            project_dir=subject.project_dir,
        ),
        "run the audit where the grader can be imported and called",
    )
    handling = entries["validity"][-1]
    for code in (handling.result or {}).get("codes", []) if handling.result else []:
        if code not in FINDING_IDS:
            continue
        findings.append(
            _finding(
                code,
                entry_id="validity.input_handling",
                message="; ".join(handling.result.get("consequences", [])) or code,
                level="error",
                kind="fail",
                rationale=(
                    "the grader was called on an input a policy can emit and returned a value the "
                    "named trainer silently converts into a score (D-65)"
                ),
                arm="white_box",
                file=source_name,
                line=None,
            )
        )
        rules["validity.input_handling"] = contracts.Rule(
            id="validity.input_handling",
            name="input handling",
            short_description=(
                "the grader returns a real score, or a refusal the trainer cannot mistake for one"
            ),
        )

    # --- validity: D10 replay, D1 coverage -------------------------------------------------------
    guarded(
        "validity.replay_determinism",
        "validity",
        "whether the same input scored twice yields the same score",
        lambda: instruments.replay_entry(vut, corpus, ctx, subject_ref=subject_ref),
        "run the audit where the grader can be imported and called",
    )
    guarded(
        "validity.decision_coverage",
        "validity",
        "the fraction of the grader's branch arcs the response bank has ever taken",
        lambda: instruments.coverage_entry(vut, corpus, ctx, subject_ref=subject_ref)
        if has_bank
        else _no_bank_coverage(ctx, subject_ref),
        "run the audit where the grader can be imported and traced",
    )

    entries["validity"].extend(
        absences.validity_absences(ctx, subject_ref=subject_ref, has_task_set=has_tasks)
    )

    # --- reach: D8, static only ------------------------------------------------------------------
    # The inventory says what the graded process can touch while the tasks are run. With no task
    # set there is no run to inventory, so the panel says that rather than reporting a floor of
    # zero over a corpus that does not exist (A-015, `audit-unfamiliar.txt` line 12).
    if not has_tasks:
        entries["reach"].append(
            absences.needs_note(
                ctx.absence(
                    "reach",
                    "reach.attack_surface",
                    "the surfaces the grader's source exposes to the graded process",
                    "no task set, so nothing to reach into",
                    "pass a task set with `--tasks <file>`, or point the audit at a project whose "
                    "rewardlens.yaml names one",
                    ("no claim about what the graded process can reach",),
                    subject_ref=subject_ref,
                    depends_on=("digest:source",),
                ),
                absences.TASK_SET,
            )
        )
    else:
        guarded(
            "reach.attack_surface",
            "reach",
            "the surfaces the grader's source exposes to the graded process",
            lambda: instruments.attack_surface_entry(
                vut, ctx, subject_ref=subject_ref, source=subject.source
            ),
            "run the audit where the grader's source can be parsed",
            # D8 reads the source, so a grader whose imports are missing is still fully readable.
            needs_grader=False,
        )
        # The inventory is static, and the witness that would execute it is not in this build. The
        # panel says both: the exposure it found, and the check it could not make. Two entries and
        # not one qualified detection, because a reader who takes the count for a measured reach
        # has been told something this run did not do (A-014: the section then reads `!`).
        entries["reach"].append(
            ctx.absence(
                "reach",
                "reach.executed_witness",
                "whether the graded process actually reaches the surfaces the source exposes",
                absences.NOT_IN_BUILD,
                "run a build whose executed reach arm has landed; this one inventories the source "
                "and executes nothing",
                ("no claim that any inventoried surface is reached in a real run",),
                subject_ref=subject_ref,
                state="COULD_NOT_CHECK",
                depends_on=("digest:source",),
            )
        )

    # --- what this build fills, asked rather than assumed -----------------------------------------
    # Discovery runs before the absences, because the absences must skip exactly the panels that are
    # about to fill themselves. A landed panel carrying both its entries and a "this build ships
    # none of it" absence is the one thing the record must never say.
    # --- every panel this build does not fill -----------------------------------------------------
    for section, panel_entries in absences.panel_absences(
        ctx, subject_ref=subject_ref, skip=filled, supplied=supplied
    ).items():
        entries[section].extend(panel_entries)

    # --- panels a wave-2 build has plugged in -----------------------------------------------------
    for panel in discovered:
        section = str(panel.section)
        if section not in entries:
            entries["validity"].append(
                ctx.absence(
                    "validity",
                    "validity.panel_section",
                    "the record section a discovered panel names",
                    f"a panel names the section {section!r}, which is not a record section",
                    "register the panel under one of: " + ", ".join(_SECTIONS),
                    (f"no claim from the {section} panel",),
                    subject_ref=subject_ref,
                    state="COULD_NOT_CHECK",
                )
            )
            continue
        for instrument in panel.instruments:
            entry_id = str(getattr(instrument, "id", f"{section}.instrument"))
            produced = guarded_many(
                entry_id,
                section,
                entry_id,
                lambda instrument=instrument: instrument.run(subject, corpus, ctx),
                "run a build in which this instrument imports",
            )
            # 8.1: the panel's own findings, harvested here so that a discovered panel's judgement
            # reaches `Assay.findings`, the verdict's reasons and the text surface under its own
            # panel. The entries are the instrument's, not the runner's absence, and a panel that
            # raises nothing is carried unchanged.
            try:
                raised = _panel_findings(instrument, produced)
            except Exception as failure:  # a broken harvest is not a broken measurement
                entries[section].append(
                    absences.could_not_check(
                        ctx,
                        section=section,
                        entry_id=f"{entry_id}.findings",
                        measurand=f"the findings {entry_id} raises over its own entries",
                        failure=failure,
                        remedy="run a build in which this panel's findings function returns",
                        subject_ref=subject_ref,
                        affected_claims=(f"no {section} finding from {entry_id}",),
                    )
                )
                continue
            loose = _unallocated_codes(raised)
            if loose:
                entries["validity"].append(
                    _code_allocation_check(
                        ctx, entry_id=entry_id, codes=loose, subject_ref=subject_ref
                    )
                )
            findings.extend(raised)
    for module, detail in registry.import_failures.items():
        entries["validity"].append(
            ctx.absence(
                "validity",
                "validity.panel_import",
                f"the panels {module} would have contributed",
                f"{module} would not import: {detail}",
                "install what that module needs, or remove it from the build",
                ("no claim from the panels that module holds",),
                subject_ref=subject_ref,
                state="COULD_NOT_CHECK",
            )
        )

    # --- the outcome state, which in this wave is always unqualified -----------------------------
    # The unqualified outcome is a hole and a reason for the verdict, and it is not a finding
    # (A-025 point 2). A finding is something measured about the grader: this is something the run
    # did not measure, and the record already carries it twice, as the `soundness.instrument_absent`
    # absence and as the decision's `outcome_unqualified:` reason. Emitting it a third time as a
    # warning-level finding put a hole in the list a reader scans for defects, and the report's own
    # count of findings then disagreed with the list under it. The rule stays in the catalogue,
    # because the entry is still there and the catalogue is what names an entry's question.
    outcome_entry_id = "soundness.instrument_absent"
    rules[outcome_entry_id] = contracts.Rule(
        id=outcome_entry_id,
        name="outcome unqualified",
        short_description="the independent outcome check has not been qualified",
    )

    # Every absence gets its code and its short remedy here, once every entry is in and every
    # `needs` note with it (A-025 point 3).
    absences.stamp_codes(entries)
    record = _assemble(
        req=req,
        subject=subject,
        project=project,
        ctx=ctx,
        started=started,
        started_monotonic=started_monotonic,
        entries=entries,
        findings=findings,
        rules=rules,
        source_digest=source_digest,
        corpus=corpus,
        filled=filled,
    )
    # `_assemble` stops at the statement; `_write` seals it, because the seal needs the digest of
    # the manifest and the manifest needs the bytes of the record. The sealed record is the one
    # returned, so the caller holds what is on disk and what the page carries.
    return _write(record, subject=subject, project=project, req=req)


def _no_bank_coverage(ctx: RunContext, subject_ref: str) -> contracts.Entry:
    """D1 with no response bank is an absence, never a fraction over the audit's own probes."""
    return ctx.absence(
        "validity",
        "validity.decision_coverage",
        "the fraction of the grader's branch arcs the response bank has ever taken",
        "no response bank was supplied, so no corpus has exercised anything",
        "pass sampled responses with `--responses <file>`; D1 measures what a corpus exercised",
        ("no claim about which of the grader's decisions have ever been exercised",),
        subject_ref=subject_ref,
        depends_on=("digest:source", "digest:samples"),
    )


def _assemble(
    *,
    req: Any,
    subject: AuditSubject,
    project: Any,
    ctx: RunContext,
    started: str,
    started_monotonic: float,
    entries: dict[str, list[contracts.Entry]],
    findings: list[contracts.Finding],
    rules: dict[str, contracts.Rule],
    source_digest: str,
    corpus: instruments.Corpus,
    filled: tuple[str, ...] = (),
) -> contracts.Assay:
    method_ids = tuple(
        sorted({entry.method.id for group in entries.values() for entry in group})
    )
    if project is not None:
        version = project.version(methods=method_ids)
        version_ref = version.to_version_ref()
        digests = version.to_subject_digests()
        system_id = _ident(version.id)
        system_name = _system_name(getattr(project.config, "name", None), project.root)
    else:
        digests = contracts.models.Digests(
            source=source_digest,
            environment=None,
            scorer_config=None,
            task_distribution=None,
            samples=None,
            outcome_protocol=None,
            policy=None,
            training_semantics=None,
            instrument_method=contracts.digest({"methods": list(method_ids)}),
        )
        version_ref = contracts.models.VersionRef(
            id="unversioned",
            digest=contracts.digest({"digests": digests.to_dict(), "parents": []}),
            parents=[],
            declared_change=None,
        )
        system_id = _ident(subject.name)
        system_name = _clean(subject.grader_path.name) or DEFAULT_SYSTEM_NAME
    holes_count = sum(
        1 for group in entries.values() for entry in group if entry.kind == "absence"
    )
    record = contracts.Assay(
        schema_url=contracts.SCHEMA_URL,
        assay_id="sha256:" + "0" * 64,
        created=started,
        producer=contracts.Producer(
            tool="reward-lens",
            version=_tool_version(),
            method_set=contracts.digest({"methods": list(method_ids)}),
        ),
        subject=contracts.Subject(
            reward_system=contracts.models.RewardSystemRef(id=system_id, name=system_name),
            version=version_ref,
            context=contracts.models.ContextRef(
                policy=None,
                # A-019: the four wordings the two transcripts choose between are keyed on this
                # field being null, and never on whether the audited path is a file or a
                # directory. So a run that was given a task set names it here, and a bare grader
                # leaves it null, which is the difference the wordings are asking about.
                # The field is a digest, so the task set is named by the one the record already
                # carries for it: null exactly when no task set was supplied.
                task_set=digests.task_distribution
                if getattr(subject, "tasks_path", None) is not None
                and subject.tasks_path.is_file()
                else None,
                configuration=corpus.origin[:200] or None,
            ),
            digests=digests,
        ),
        intent=contracts.Intent(
            success=(getattr(project.config, "success", None) if project is not None else None),
            constraints=list(getattr(project.config, "constraints", []) or [])
            if project is not None
            else [],
            non_equivalent_pairs=[],
            outcome_check=contracts.models.OutcomeCheck(
                state="unqualified", kind=None, provenance=None, measured_disagreement=None
            ),
            attack_budget=contracts.models.AttackBudget(seeker_calls=0, usd="0.00"),
            partitions=[],
        ),
        measurement=contracts.models.Measurement(**entries),
        holes=[],
        rules=sorted(rules.values(), key=lambda rule: rule.id),
        findings=findings,
        join=[],
        experiments=[],
        calibration_links=[],
        tables=[],
        decision=contracts.Decision(
            state="unresolved",
            # The action is the use the project declares for this version, and the verdict is
            # unresolved *for that action*: a decision with no action names no use, and a reader
            # then has a verdict about nothing. A project that declares a trainer is going to
            # train on this version, which is what the config exposes; a bare grader declares no
            # use, so there is none to name and the field stays null.
            action=_declared_action(project),
            policy=None,
            reasons=_reasons(entries, findings, outcome_kind=_outcome_kind(project)),
            tradeoff=None,
            signature=None,
            regression_cases=[],
        ),
        cost=contracts.Cost(
            wall_s=round(max(0.0, time.monotonic() - started_monotonic), 3),
            cpu_s=0.0,
            usd="0.00",
            api_calls=0,
        ),
        embedding=contracts.Embedding(tier="A", omitted_tables=[]),
        attestation=contracts.Attestation(statement_digest=None, backend=None),
        provenance=contracts.Provenance(
            offline=ctx.offline,
            sandbox_tier=ctx.tier(),
            os=_os_name(),
            seed=int(getattr(req, "seed", 20260911)),
            reproduce=[f"reward-lens audit {req.path} --seed {getattr(req, 'seed', 20260911)}"],
        ),
        environment_excluded_from_digest={
            "python": platform.python_version(),
            "platform": sys.platform,
            "holes": holes_count,
        },
    )
    record.holes_from_entries()
    # The two sentences and the plan the run went in under, at the record's top level (A-016).
    # They are composed here, after every entry is in, because both are statements about the whole
    # record: a sentence built while the entries were still arriving would be about a record that
    # never existed. `paid_calls` is the planner's zero for the same reason it is zero there: the
    # seeker is the only arm that costs money and this build does not ship it (D-41).
    record.extensions = dict(record.extensions or {})
    record.extensions[absences.TRANSCRIPT_EXTENSION] = {
        "could_establish": absences.could_establish(record),
        "could_not": absences.could_not(record),
        "subject": _subject_clauses(subject=subject, project=project, corpus=corpus),
        "plan": {
            "panels": len(filled),
            "paid_calls": 0,
            "estimate_s": _estimate_s(filled),
        },
    }
    if project is not None:
        # The project's `instrument_method` is derived from the record's own entries, so the
        # subject is settled from the assembled record rather than from a guess at the method set.
        settled = project.version_of(record)
        record.subject.version = settled.to_version_ref()
        record.subject.digests = settled.to_subject_digests()
    return record


def _subject_clauses(*, subject: AuditSubject, project: Any, corpus: instruments.Corpus) -> dict:
    """The Subject line's clauses the schema's closed `subject` cannot carry (A-019).

    Two shapes, and which one is written is read off the record's own situation rather than off the
    path the reader typed: a project names the grader file and the counts it measured over, and a
    bare grader names that same file as the user gave it, the signature it declares and the shape
    that signature is in. Both name the file (A-022 corrects A-019's "module" here): the module
    name is an import detail the reader never typed, and `my_grader.py` is what they can point at
    again.
    A directory audited without a task set is a bare grader for this purpose, which is the point of
    keying on the record.
    """
    if project is not None:
        clauses = {
            "grader": subject.grader_path.name,
            "tasks": int(getattr(corpus, "n_tasks", 0) or 0),
            "responses": len(corpus.rollouts),
        }
        # The task set's file name, which the record's own `subject` cannot carry: its `task_set`
        # is a digest, and a digest does not tell a reader which file to open again. A project with
        # no task set names none, because there is no file to name (A-025 point 4).
        tasks_path = getattr(subject, "tasks_path", None)
        if tasks_path is not None:
            clauses["task_file"] = tasks_path.name
        return clauses
    # A bare grader declares nothing, so the shape is the one the audit had to assume to call it
    # at all: a plain number back from a plain call. The clause says it so a reader can contradict
    # it, which is the only way an assumption in a record is worth carrying.
    shape = "plain"
    return {
        "grader": subject.grader_path.name,
        "signature": instruments.declared_signature(subject.source, subject.entrypoint)
        or f"{subject.entrypoint}(...)",
        "shape": f"{shape} shape",
    }


def _seal(record: contracts.Assay, *, bundle_manifest_digest: str | None) -> contracts.Assay:
    """D-80: plan the embedding on the statement, then take the id over the planned record.

    The record declares its own embedding tier before it is sealed, so the id is the digest of a
    record that already says how it will be carried, and nothing downstream may edit a sealed
    record. That is why the plan is taken here and not by the renderer, and it is also why the
    bundle manifest's digest has to be in hand this early: the page carries that digest in a
    `<meta>`, so the digest changes the page's length, and the length is `embedding.html_bytes`,
    which is sealed into the record and which the renderer weighs the page against.

    Sealing is idempotent. `plan_embedding` ignores an `embedding` already on the record and
    recomputes it, and `assay_id` is excluded from its own digest, so `_write` may call this more
    than once on the same object while the manifest digest settles.
    """
    from reward_lens.render.report import plan_embedding

    record.embedding = contracts.Embedding.model_validate(
        plan_embedding(record, bundle_manifest_digest=bundle_manifest_digest).to_dict()
    )
    record.assay_id = contracts.digest(record)
    return record


def _all_absent(entries: dict[str, list[contracts.Entry]], section: str) -> bool:
    group = entries.get(section) or []
    return bool(group) and all(entry.kind == "absence" for entry in group)


#: The order the report prints the dark sections in, and so the order the decision's
#: `required_missing:` reasons take (A-025 point 1). It is not the record's own field order: the
#: report puts signal before reward_statistics, and the reasons follow the report, because a reader
#: checks this list against the sections they have just read past. A section the record gains and
#: this tuple does not name still gets its reason, appended in record order, so nothing can fall
#: out of the list by being forgotten here.
_REASON_SECTIONS: tuple[str, ...] = (
    "soundness",
    "exploits",
    "framing",
    "signal",
    "reward_statistics",
    "trace",
    "forecast",
    "calibration",
)

#: What `outcome_unqualified:` names when the subject declares no independent check at all. The
#: reason names the protocol that was not qualified, and a bare grader declares no protocol: the
#: slot says so rather than borrowing a kind the subject never claimed.
_NO_OUTCOME_DECLARED = "none_declared"


def _reason_order() -> tuple[str, ...]:
    """The printed order, then any section the record has that the printed order does not name."""
    return _REASON_SECTIONS + tuple(
        section for section in _SECTIONS if section not in _REASON_SECTIONS
    )


def _outcome_kind(project: Any) -> str | None:
    """The kind of independent check the subject declares, or None when it declares none."""
    outcome = getattr(getattr(project, "config", None), "outcome", None)
    kind = getattr(outcome, "kind", None)
    return str(kind) if kind else None


def _reasons(
    entries: dict[str, list[contracts.Entry]],
    findings: list[contracts.Finding],
    *,
    outcome_kind: str | None = None,
) -> list[str]:
    """Why the decision is what it is, blocking findings first (section 7.3's vocabulary).

    A decision that named only what was missing left out what was found: the demo exits non-zero on
    a finding at level error, and a reader given only `outcome_unqualified` and `required_missing`
    could not tell which finding did it. One error-level finding is one reason. Naming it twice,
    once by rule and once by finding id, put the same finding in the list twice and made the report
    count two where it prints `1 blocking finding`: the text layout prints the count and not the
    name, so a second name bought nothing and cost the count. The name kept is the finding id
    (`RGX-local-0001`), because the envelope golden reads `blocking_finding:RGX-local-0001` and the
    golden is the specification. The rule is the stabler name across records and reads better, and
    it is still one field away in the record's own findings, which carry both; the record the
    golden pins is the one this has to be.

    `outcome_unqualified:` is its own reason kind and is never a blocking finding: it says the
    outcome check was not qualified, which is a limit of the run and not a defect of the grader.
    What it names is the protocol that was not qualified, the kind the subject declares
    (`protected_test_suite` for the demo), and not the id of the entry that stands in for the
    absent instrument: the entry id is where the record puts the hole, and a reader who sees
    `outcome_unqualified:soundness.instrument_absent` reads it as a second name for the hole
    rather than as the check that would have settled the verdict.

    The order is the golden's (A-025 point 1): the blocking findings, then the unqualified
    outcome, then the sections that could not run, in the order the report prints them. The
    outcome comes before the sections because it is about the verdict and they are about the
    measurement, and a reader stops at the first line that explains the state.

    Every dark section the policy requires is one reason, and no section is waived. Calibration
    was held out of the count on the reading that no reader could fetch a certified reference
    material for this substrate, so naming it would send the reader after something that does not
    exist. A-023 withdrew that: a required panel the run could not fill costs the verdict whether
    or not the input is obtainable, and a reason a reader cannot act on is still why the decision
    is unresolved. The remedy row for calibration already says that nothing here can be calibrated
    until such a material is published, which is where that caveat belongs. The count is measured,
    not curated: the demo leaves eight dark and the transcript prints eight.
    """
    reasons: list[str] = []
    for finding in findings:
        if finding.level != "error":
            continue
        name = finding.id or finding.rule
        reason = f"blocking_finding:{name}"
        if name and reason not in reasons:
            reasons.append(reason)
    reasons.append(f"outcome_unqualified:{outcome_kind or _NO_OUTCOME_DECLARED}")
    reasons += [
        f"required_missing:{section}"
        for section in _reason_order()
        if _all_absent(entries, section)
    ]
    return reasons


def _declared_action(project: Any) -> str | None:
    """The use the project declares for the version, or null when it declares none."""
    reward = getattr(getattr(project, "config", None), "reward", None)
    return "train_on_this_version" if getattr(reward, "trainer", None) else None


def _tool_version() -> str:
    try:
        from importlib.metadata import version

        return version("reward-lens")
    except Exception:  # pragma: no cover - an editable tree without metadata
        return "4.0.0"


def record_bytes(record: contracts.Assay) -> bytes:
    """The one serialisation of a record on disk: RFC 8785 JCS over the whole record.

    `canonical_bytes` is the *digest* serialisation and strips the four fields a record cannot be
    hashed over, its own `assay_id` among them. Writing those bytes to a file, as the projectless
    branch of `_write` did, wrote a record that did not carry its own identifier and differed field
    for field from the record the SDK returned and from the record the store writes. The store's
    serialisation is the one both branches use here, so the bytes on disk are the returned record.
    """
    import rfc8785

    return rfc8785.dumps(record.to_dict())


def manifest_bytes(manifest: dict[str, Any]) -> bytes:
    """The one serialisation of a manifest on disk, the record's own: RFC 8785 JCS.

    The page's claim is `contracts.digest` of this mapping, and a reader checks it by loading the
    file and digesting what comes back. JCS makes that hold for any JSON encoding of the same
    mapping, but writing the canonical bytes means the file a reader hashes and the bytes the
    digest was taken over are the same string, with nothing in between to argue about.
    """
    import rfc8785

    return rfc8785.dumps(dict(manifest))


def _manifest(*, record_name: str, payload: bytes, assay_id: str) -> dict[str, Any]:
    """What the bundle is, as the page's `<meta name="bundle-manifest-digest">` will claim it.

    D-16, and section 7's `bundle_manifest_digest`: the page names the bundle it was rendered
    from, so a reader holding a page and a directory can tell whether the two belong together.
    Paths are relative to the manifest, which sits beside them, so a bundle that is copied or
    moved whole still checks. Every artifact this pipeline writes is named here except three: the
    report, which carries this manifest's digest and so cannot also be digested by it; the
    manifest, which cannot name itself; and the store's `index.json`, which is the project's
    rather than this bundle's, spans every assay the project holds and is rewritten by the next
    audit, so naming it would make the manifest stop describing what is on disk as soon as one
    more audit ran.

    `assay_id` is nested under `assay` rather than written at the top level, and the nesting is
    load-bearing. `contracts.digest` removes `EXCLUDED_FROM_DIGEST` before hashing and
    `assay_id` is the first name on that list, so a manifest that put the id at the top level
    would leave the one field a reader most wants pinned outside the manifest's own digest.
    """
    return {
        "schema": "reward-lens/bundle-manifest/1",
        "assay": {"id": assay_id},
        "tool": {"name": "reward-lens", "version": _tool_version()},
        "artifacts": [
            {
                "path": record_name,
                "role": "record",
                "media_type": "application/json",
                "bytes": len(payload),
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            }
        ],
    }


def _name_stem(record: contracts.Assay, subject: AuditSubject) -> str:
    """What a record's name starts at: the version it measures, named as the project declares it.

    A-015: the stem is the subject version's id, not the reward system's name and not the
    directory. A project that declares `version.id: code-reward-v1` gets records called that
    whichever directory it was opened from, which is what makes two copies of one project produce
    the same name rather than two names that agree about nothing. A bare grader has no version to
    declare, so it is named for its file with the suffix dropped: `my_grader.py` is the reward
    system, `my_grader` is the stem of the bundle.
    """
    version_id = str(record.subject.version.id or "")
    if version_id and version_id != "unversioned":
        return _ident(version_id)
    return _ident(subject.grader_path.stem)


def _measured(record: contracts.Assay) -> tuple[str, ...]:
    """Every (entry id, method identity) pair the record holds, sorted: what it measured and how.

    The same key the store's `_check_not_a_rewrite` reads, in the model's spelling rather than the
    dict's, because the name and the refusal have to agree: the name is what keeps two records
    that the store would otherwise see as one out of one file.
    """
    return tuple(
        sorted(f"{entry.entry_id}@{method_identity(entry.method)}" for entry in record.entries())
    )


def _derived_name(
    record: contracts.Assay, *, subject: AuditSubject, stamp: str, project: Any = None
) -> str:
    """`<version id>-<date>`, and the version's short digest after it only on a real collision.

    The date alone was not a name. Two audits of one subject on one day want the same file, and
    the store, which never rewrites a record, refused the second (RL0620) after the instruments
    had already run. Two things answer that now, and the suffix is the second of them. The
    unchanged case never reaches here: `_standing_record` hands the stored record back before any
    instrument runs. What is left is the changed case, where two records of one subject on one day
    genuinely differ, and only there does the name carry what tells them apart.

    So the suffix is conditional, and the condition is asked of the store rather than assumed: the
    name is plain unless the store already holds it for a different subject version. An
    unconditional suffix put eight hex on every bundle a reader ever sees, for a collision almost
    no project has. The store has no name allocator to ask -- `Project` offers `path_for`, `names`
    and `record`, and none of them proposes a next name -- so this is the pipeline's rule. The
    digest is truncated because the name is a label, not the identity; the whole digest is inside
    the record.
    """
    name = f"{_name_stem(record, subject)}-{stamp}"
    if project is None:
        return name
    try:
        if name not in project.names():
            return name
        standing = project.open_record(name)
    except (OSError, ValueError, contracts.RewardLensError):
        # A name the store cannot read as a record is not evidence of a collision, and the write
        # path already steps over such a file rather than raising on it.
        return name
    # A-016: the suffix comes from the digest that differs. The version digest when the version
    # moved, and the instrument-method digest when it did not, which is the wider-plan case: the
    # subject is the same and what was measured of it is not, so the name has to carry the second
    # or the two records want one file. What "the instruments differ" means here is the set of
    # entry ids, for the reason the store's rewrite rule reads the same thing: every absence in a
    # section shares one placeholder method id, so the method digest does not move when a panel
    # lands and finds nothing, and the two records would then want one name.
    if standing.subject.version.digest != record.subject.version.digest:
        differing = record.subject.version.digest
    elif _measured(standing) != _measured(record):
        differing = contracts.digest({"entries": list(_measured(record))})
    else:
        return name
    short = differing.rsplit(":", 1)[-1][:NAME_DIGEST_CHARS]
    return f"{name}-{short}"


def _write(
    record: contracts.Assay, *, subject: AuditSubject, project: Any, req: Any
) -> contracts.Assay:
    """The record through the store, the manifest beside it, the report through the renderer.

    Section 10, and D-16 for the manifest. The order is forced by a circle. The manifest names the
    record file and the digest of its bytes; the record's bytes turn on the manifest's digest,
    because the page carries that digest in a `<meta>` and the page's length is sealed into the
    record as `embedding.html_bytes`. The circle closes on an accident of shape: a digest is
    always `sha256:` and sixty-four hex characters, so the page's length depends on whether there
    is a digest and never on which one. Seal with none, take the digest that produces, seal again
    with it, and the third pass repeats the second.

    The loop settles that for itself rather than trusting the argument, and writes nothing until
    it has, so a manifest and a page that disagree never reach the disk. The record's bytes are
    predicted with `record_bytes` during the settling and the written file is weighed against the
    prediction afterwards, which is one write rather than three and still leaves the manifest
    naming the digest of the file that is actually there.
    """
    from reward_lens.render.report import render_to

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    name = getattr(req, "name", None) or _derived_name(
        record, subject=subject, stamp=stamp, project=project
    )
    record_name = f"{name}.assay.json"

    manifest: dict[str, Any] = {}
    digest: str | None = None
    for _ in range(MANIFEST_PASSES):
        _seal(record, bundle_manifest_digest=digest)
        manifest = _manifest(
            record_name=record_name, payload=record_bytes(record), assay_id=record.assay_id
        )
        settled = contracts.digest(manifest)
        if settled == digest:
            break
        digest = settled
    else:  # pragma: no cover - the fixed point is reached on the third pass
        raise RuntimeError(
            f"the bundle manifest digest did not settle in {MANIFEST_PASSES} passes; the last"
            f" was {digest}"
        )

    if project is not None:
        written = project.record(record, name=name)
    else:
        directory = subject.grader_path.parent / "assays"
        directory.mkdir(parents=True, exist_ok=True)
        written = directory / record_name
        written.write_bytes(record_bytes(record))

    named = manifest["artifacts"][0]
    on_disk = "sha256:" + hashlib.sha256(written.read_bytes()).hexdigest()
    if written.name != named["path"] or on_disk != named["sha256"]:
        raise RuntimeError(
            f"the manifest names {named['path']} at {named['sha256']}, and the record was written"
            f" to {written.name} as {on_disk}: the store's serialisation is no longer the one"
            f" `record_bytes` predicts, so the manifest would name bytes that are not on disk"
        )
    written.with_name(MANIFEST_NAME).write_bytes(manifest_bytes(manifest))
    render_to(
        record, written.with_name(f"{name}.assay.html"), bundle_manifest_digest=digest
    )
    return record
