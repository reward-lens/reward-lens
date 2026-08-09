"""The three real instruments, and the one static check that has to execute the grader.

Each instrument takes a `VerifierUnderTest` and returns a `Reading` that is evidence or a refusal.
A refusal is a value, not an exception, and it becomes an absence naming the refusal's own reason
and remedy. It never becomes a number: an instrument that declined to measure something and a
measurement of zero are the two things D-18 exists to keep apart.

The fourth static check lives here rather than in `static_checks` because it is the one that runs
the grader: three responses a policy can really emit, through the integrated Python adapter and
`Sandbox.run` with the project staged (A-005), paired with what the named trainer does with the
value that comes back (D-65).
"""

from __future__ import annotations

import ast
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from reward_lens import contracts
from reward_lens.core.reading import is_refusal
from reward_lens.instruments.base import Corpus, RunContext
from reward_lens.verifier import ListCorpus, Rollout, VerifierUnderTest

from . import static_checks

__all__ = ["AuditCorpus", "Corpus", "attack_surface_entry", "coverage_entry", "input_handling_entry", "replay_entry"]

#: What each of D-65's three silent failure modes costs, and which finding code names it: the code,
#: the sentence for each trainer whose documentation this build has read, keyed on the name the
#: project declares, the sentence when none is declared, and the sentence when one is declared that
#: this build cannot speak for.
#:
#: A-017. The first sentence is a claim about a trainer, so a run that was told of none may not
#: make it. What is measured either way is the same and it is the grader's half: what came back on
#: that input. The second half is read from the named trainer's documented semantics, and with no
#: name there is nothing to read it from, so the second sentence stops at what was measured and
#: says what would be needed to carry it further. That is why there are whole sentences here rather
#: than one with the trainer's name substituted in: dropping a name out of a sentence built around
#: one leaves a sentence that still claims what it can no longer support.
#:
#: A-025's fresh review adds the third case, and it is the reason the second element is keyed by
#: name at all. `reward.trainer` admits four names, and only trl's semantics were read for D-65, so
#: a build that picked the declared-trainer sentence by whether *a* name was there told a project
#: declaring `verifiers` that "the trainer you named turns None into NaN": trl's documented
#: behaviour, asserted of a trainer nothing was read about. A name this table does not hold is not
#: a name to guess at. It gets the sentence that stops at what the grader returned, and the entry
#: carries a limitation saying which trainer was declared and that this build has no reading of it.
TRAINER_BEHAVIOUR: dict[str, tuple[str, dict[str, str], str, str]] = {
    "unscored": (
        "RL0210",
        {
            "trl": "the grader returns None on malformed input, and the trainer you named turns "
            "None into NaN and drops the row",
        },
        "the grader returns None on malformed input, and no trainer is declared under "
        "`reward.trainer`, so what becomes of that value in training is not known here",
        "the grader returns None on malformed input, and this build has not read the declared "
        "trainer's semantics, so what becomes of that value in training is not known here",
    ),
    "grader_error": (
        "RL0211",
        {
            "trl": "the grader raises, and the trainer you named scores the raise 0.0, which the "
            "optimiser reads as a bad response rather than a broken measurement",
        },
        "the grader raises, and no trainer is declared under `reward.trainer`, so what becomes of "
        "the raise in training is not known here",
        "the grader raises, and this build has not read the declared trainer's semantics, so what "
        "becomes of the raise in training is not known here",
    ),
    "bool_score": (
        "RL0212",
        {
            "trl": "the grader returns a bool, and the trainer you named passes it through "
            "float(), so True becomes the maximum score",
        },
        "the grader returns a bool, and no trainer is declared under `reward.trainer`, so what "
        "becomes of that value in training is not known here",
        "the grader returns a bool, and this build has not read the declared trainer's semantics, "
        "so what becomes of that value in training is not known here",
    ),
}


def trainer_sentence(row: tuple[str, dict[str, str], str, str], trainer: str | None) -> str:
    """The consequence sentence for one behaviour under the trainer this project declared.

    Selected by name, which is the whole point: the documented half of the sentence belongs to one
    trainer, and there is no reading of it to hand for a trainer whose documentation was never
    read. The name is matched case-insensitively because `TRL` and `trl` are the same trainer and a
    reader who capitalised it has not declared a different one.
    """
    _, documented, undeclared, undocumented = row
    if not trainer:
        return undeclared
    return documented.get(trainer.strip().lower(), undocumented)


def trainer_is_documented(trainer: str | None) -> bool:
    """Whether this build has read the named trainer's semantics for any of D-65's three modes."""
    if not trainer:
        return False
    name = trainer.strip().lower()
    return any(name in row[1] for row in TRAINER_BEHAVIOUR.values())

#: The exposure a surface kind is catalogued under, for the id a surface prints beside the count.
#: Only the kinds this static inventory can name are here; a kind with no entry has no id, and the
#: entry says so by carrying none rather than by inventing one.
#: The catalogue's id for each exposure, keyed on what the exposure *is* and not on the access kind
#: that found it. An access is a call site; an exposure is a crossing, a path the graded process can
#: write and the grader then reads, and which exposure a crossing is depends on what sits at that
#: path. Only `tests` is namable from static facts alone in this build: the other three take an
#: executed witness to tell an exposure from a call site, and that witness is P-REACH's.
EXPOSURE_IDS: dict[str, str] = {
    "tests": "REACH-EDIT-TESTS",
    "exec": "REACH-EXEC",
    "network": "REACH-NETWORK",
    "credential": "REACH-CREDENTIAL",
}

#: What a crossing's read path looks like when the thing the graded process can rewrite is the
#: check itself: a `tests/` directory anywhere in the path, or a file named as a test module.
#: `outcome/test_solution.py` is that, which is the demo's crossing.
_TESTS_PATH = re.compile(r"(?:^|/)tests?(?:/|$)|(?:^|/)(?:test_[^/]*|[^/]*_test)\.py$")


def _exposure_id(crossings: Sequence[Any]) -> str | None:
    """The catalogue id for the exposure these crossings are, or null when there is none to name.

    A crossing at a path this build's catalogue has no id for is still counted: the count is what
    the inventory measured, and the id is what it could name.
    """
    for crossing in crossings:
        if _TESTS_PATH.search(str(crossing.read_path).replace("\\", "/")):
            return EXPOSURE_IDS["tests"]
    return None

_OVERSIZED = 1 << 18

#: Verdicts that say the harness failed before the grader spoke. They carry no observation about
#: the grader's input handling, whatever else the envelope holds. Every other verdict does carry
#: one: `scored`, `invalid_input`, `unscored` and `grader_error` are the grader's own answer, and
#: `timeout` and `resource_exhausted` are what the grader did with the input until a limit stopped
#: it, which for the oversized probe is the answer D-65 asks for.
NO_OBSERVATION: frozenset[str] = frozenset({"parse_error", "provider_unavailable", "cancelled"})


@dataclass
class AuditCorpus:
    """The rollouts the instruments run on, and where they came from."""

    rollouts: tuple[Rollout, ...]
    origin: str
    tasks: tuple[dict, ...] = ()
    n_tasks: int = 0

    def as_list(self) -> ListCorpus:
        return ListCorpus.of(self.rollouts)

    def __len__(self) -> int:
        return len(self.rollouts)


def read_jsonl(path: Path, cap: int | None = None) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                out.append(record)
            if cap is not None and len(out) >= cap:
                break
    return out


def _task_id(record: dict) -> str | None:
    for key in ("task_id", "id", "task"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return None


def _response_text(record: dict) -> str | None:
    for key in ("text", "response", "completion", "output"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return None


def normalised_task(record: dict, fallback_id: str) -> dict:
    """The task as the adapter's input contract needs it: `id` and `prompt` present as strings.

    The task set names its identifier `task_id`; the adapter requires `id`. The probe carries both
    and the entry says so, because silently renaming a caller's field is how a probe comes to test
    something the real run never sees.
    """
    task = dict(record)
    task.setdefault("id", _task_id(record) or fallback_id)
    task.setdefault("prompt", str(record.get("prompt", "")) or "a probe of the input contract")
    return task


def build_corpus(
    *, tasks_path: Path | None, responses_path: Path | None, cap: int | None = None
) -> Corpus:
    """The response bank as rollouts, joined to their tasks. Empty when either input is absent."""
    if tasks_path is None or responses_path is None:
        return AuditCorpus(rollouts=(), origin="no task set and no response bank were supplied")
    if not tasks_path.is_file() or not responses_path.is_file():
        return AuditCorpus(rollouts=(), origin=f"{tasks_path.name} or {responses_path.name} is not a file")
    tasks = read_jsonl(tasks_path)
    by_id = {_task_id(task) or f"task-{index}": task for index, task in enumerate(tasks)}
    rollouts: list[Rollout] = []
    for index, record in enumerate(read_jsonl(responses_path, cap=cap)):
        task = by_id.get(_task_id(record) or "")
        text = _response_text(record)
        if task is None or text is None:
            continue
        rollouts.append(
            Rollout(
                id=str(record.get("response_id") or f"r-{index}"),
                inputs={"task": normalised_task(task, f"task-{index}"), "response": text},
            )
        )
    return AuditCorpus(
        rollouts=tuple(rollouts),
        origin=f"{len(rollouts)} sampled responses joined to {len(by_id)} tasks",
        tasks=tuple(tasks),
        n_tasks=len(by_id),
    )


def probe_corpus(tasks: Sequence[dict]) -> Corpus:
    """The corpus a bare grader gets: the audit's own probes, and it says so."""
    task = normalised_task(tasks[0] if tasks else {}, "probe-task")
    rollouts = tuple(
        Rollout(id=f"probe-{name}", inputs={"task": task, "response": text})
        for name, text in _probes()
    )
    return AuditCorpus(
        rollouts=rollouts,
        origin="the audit's own probe inputs; no response bank was supplied",
        tasks=(task,),
        n_tasks=1,
    )


def _seed_scores(vut: VerifierUnderTest, rollouts: Sequence[Rollout]) -> tuple[Rollout, ...]:
    """Grade each input once so that D10 has a baseline. Inputs it cannot score are dropped."""
    entry = vut.load()  # already the entrypoint callable, not the module
    seeded: list[Rollout] = []
    for rollout in rollouts:
        try:
            value = entry(**rollout.inputs)
        except Exception:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            seeded.append(Rollout(id=rollout.id, inputs=rollout.inputs, score=float(value)))
    return tuple(seeded)


def _probes() -> tuple[tuple[str, str], ...]:
    return (
        ("malformed", "```python\ndef (\n```"),
        ("empty", ""),
        ("oversized", "x" * _OVERSIZED),
    )


# --- D1, decision coverage -----------------------------------------------------------------------


def coverage_entry(
    vut: VerifierUnderTest, corpus: Corpus, ctx: RunContext, *, subject_ref: str
) -> contracts.Entry:
    """D1 as an `estimate`, or the refusal as an absence. Never a fraction it did not compute."""
    from reward_lens.verifier.coverage import measure_coverage

    started = ctx.clock()
    if not corpus.rollouts:
        return ctx.absence(
            "validity",
            "validity.decision_coverage",
            "the fraction of the grader's branch arcs the response bank has ever taken",
            f"no rollouts to trace: {corpus.origin}",
            "pass a response bank with `--responses <file>`; D1 measures what a corpus exercised, "
            "and with no corpus there is nothing it could have exercised",
            ("no claim about which of the grader's decisions have ever been exercised",),
            subject_ref=subject_ref,
            depends_on=("digest:source", "digest:samples"),
        )
    reading = measure_coverage(vut, corpus.as_list(), rung=1)
    duration = ctx.clock() - started
    if is_refusal(reading):
        return ctx.absence(
            "validity",
            "validity.decision_coverage",
            "the fraction of the grader's branch arcs the response bank has ever taken",
            f"{reading.instrument} refused: {reading.detail}",
            reading.remedy,
            ("no claim about which of the grader's decisions have ever been exercised",),
            subject_ref=subject_ref,
            state="REFUSED",
            depends_on=("digest:source", "digest:samples"),
            duration_s=duration,
        )
    value = reading.branch_fraction
    entry = contracts.Entry(
        entry_id="validity.decision_coverage",
        section="validity",
        kind="estimate",
        measurand="the fraction of the grader's branch arcs the response bank has ever taken",
        method=contracts.Method(
            id="validity.decision_coverage",
            version="1.0.0",
            params_digest=contracts.digest({"rung": 1, "n_rollouts": reading.n_rollouts}),
            procedure=(
                "every rollout in the response bank is re-run under the standard library's line "
                "tracer and the branch arcs it took are recorded; the estimate is covered arcs "
                "over the arc universe of the grader's own source file"
            ),
            credited_to="reward_lens.verifier.coverage.measure_coverage (D1)",
        ),
        scope="evaluator_comparison",
        subject_ref=subject_ref,
        depends_on=["digest:source", "digest:samples"],
        state="complete",
        provenance=ctx.provenance(duration_s=duration),
        limitations=[
            "an arc the corpus never took is not thereby unreachable; this is a statement about "
            "the corpus, not about the grader",
        ],
        value=round(float(value), 6),
        unit="fraction of branch arcs",
        n=int(reading.n_rollouts),
        sampling_unit="rollout",
        uncertainty=contracts.Uncertainty(
            method=contracts.NO_INTERVAL,
            level=0.95,
            reason=(
                "the sampling unit that would carry an interval here is the task, and the "
                "bootstrap is not run by this panel; no endpoints are invented to fill the field"
            ),
            clusters=int(corpus.n_tasks),
            cluster_unit="task",
        ),
        result={
            "statements_covered": reading.statements_covered,
            "statements_total": reading.statements_total,
            "branch_arcs_covered": reading.branch_arcs_covered,
            "branch_arcs_total": reading.branch_arcs_total,
            "uncovered_clauses": list(reading.uncovered_clauses),
        },
    )
    return entry


# --- D10 with no corpus: the declared signature's own pair (A-020) --------------------------------

#: How many times the synthetic pair is graded. The transcript prints the number, so it lives with
#: the check rather than with the run that asked for it.
SYNTHETIC_REPEATS = 100

#: Parameter names that carry the task, and names that carry the response. A signature that uses
#: neither vocabulary falls back to position, which is what `score(task, response)` declares.
_TASK_NAMES = frozenset({"task", "item", "example", "sample", "row", "record", "problem"})
_RESPONSE_NAMES = frozenset(
    {"response", "completion", "output", "answer", "solution", "text", "prediction"}
)

#: What the synthetic task carries under a key the grader's source reads off it. `prompt` is the
#: probe corpus's wording, so the two synthetic inputs this build makes say the same thing.
_SYNTHETIC_VALUES: dict[str, str] = {
    "prompt": "a probe of the input contract",
    "id": "synthetic-task",
    "task_id": "synthetic-task",
}
_SYNTHETIC_RESPONSE = "a probe of the input contract, answered"


def _task_keys(source: str, entrypoint: str, param: str) -> tuple[str, ...]:
    """The literal keys the entrypoint reads off its task parameter, in source order.

    The signature says a task is wanted; it does not say what shape one is. The source does, in the
    only way a reader can check: the keys the function itself asks the task for. A key reached
    through a variable or a helper is not here, which is why the pair is recorded in the entry.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    keys: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != entrypoint:
            continue
        for inner in ast.walk(node):
            target: ast.expr | None = None
            if isinstance(inner, ast.Subscript) and isinstance(inner.slice, ast.Constant):
                target, key = inner.value, inner.slice.value
            elif (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "get"
                and inner.args
                and isinstance(inner.args[0], ast.Constant)
            ):
                target, key = inner.func.value, inner.args[0].value
            if target is None or not isinstance(key, str):
                continue
            if isinstance(target, ast.Name) and target.id == param and key not in keys:
                keys.append(key)
    return tuple(keys)


def declared_signature(source: str, entrypoint: str) -> str | None:
    """`score(task, response)`: the signature the grader's source declares, read off the source.

    Off the source and not off the imported function, because this is what the subject line calls
    a declaration: what the file says it takes. A grader that could not be imported still has one.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entrypoint:
            args = node.args
            names = [arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
            return f"{entrypoint}({', '.join(names)})"
    return None


def _synthetic_pair(vut: VerifierUnderTest) -> tuple[str, dict[str, Any]] | None:
    """The signature as it is declared, and one call's keyword arguments, or None if undecidable.

    A parameter this cannot name and cannot place is not one to guess at: the pair is then not
    derived from the declaration at all, and the check says so rather than grading something the
    declaration never asked for.
    """
    parameters = inspect.signature(vut.load()).parameters
    names = [
        name
        for name, param in parameters.items()
        if param.kind
        in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    ]
    signature = f"{vut.entrypoint}({', '.join(names)})"
    kwargs: dict[str, Any] = {}
    for index, name in enumerate(names):
        wants_task = name in _TASK_NAMES or (index == 0 and name not in _RESPONSE_NAMES)
        wants_response = name in _RESPONSE_NAMES or (index == 1 and not wants_task)
        if wants_task:
            keys = _task_keys(vut.source(), vut.entrypoint, name) or ("id", "prompt")
            kwargs[name] = {
                key: _SYNTHETIC_VALUES.get(key, f"the audit's synthetic {key}") for key in keys
            }
        elif wants_response:
            kwargs[name] = _SYNTHETIC_RESPONSE
        elif parameters[name].default is not inspect.Parameter.empty:
            continue  # the declaration already says what this one is when no one passes it
        else:
            return None
    return signature, kwargs


def _namespace_fingerprint(namespace: dict[str, Any]) -> str:
    """A digest of the module's own names, as the repeats leave them.

    This is what "no hidden state" is read from besides a repeat that disagrees: a cache, a counter
    or a seed the grader keeps between calls is a name in this namespace whose value moved.
    """
    rows: dict[str, str] = {}
    for key, value in namespace.items():
        if key.startswith("__"):
            continue
        try:
            rows[key] = repr(value)
        except Exception:  # noqa: BLE001 - a repr that raises is state this cannot read
            rows[key] = "<unreadable>"
    return contracts.digest({"namespace": rows})


def _synthetic_entry(
    ctx: RunContext,
    *,
    subject_ref: str,
    duration: float,
    passed: bool,
    repeats: int,
    identical: bool,
    hidden_state: bool,
    summary: str,
    scope_tested: str,
    assumptions: Sequence[str],
    extra: dict[str, Any] | None = None,
) -> contracts.Entry:
    """The one shape the synthetic replay returns, pass or fail. Never an absence: it ran."""
    return contracts.Entry(
        entry_id="validity.replay_determinism",
        section="validity",
        kind="check",
        measurand="whether the same input scored twice yields the same score",
        method=contracts.Method(
            id="validity.replay_determinism",
            version="1.0.0",
            params_digest=contracts.digest({"repeats": repeats, "n": 1, "synthetic": True}),
            procedure=(
                f"one task and response pair derived from the declared signature is graded "
                f"{repeats} times and the outputs are compared with each other; the grader "
                f"module's own names are digested before and after the repeats"
            ),
            credited_to="reward_lens.product.audit.instruments.synthetic_replay_entry (D10)",
        ),
        scope="evaluator_comparison",
        subject_ref=subject_ref,
        depends_on=["digest:source"],
        state="complete",
        provenance=ctx.provenance(duration_s=duration),
        assumptions=list(assumptions),
        limitations=[
            "the pair is the audit's own, not a task this reward system will ever be given: this "
            "is the grader's determinism on one input, and nothing about the tasks it will score",
            "what this reads of hidden state is a repeat that disagrees and the module's names "
            "before and after; state that carries and moves neither is not visible to it",
        ],
        check=contracts.Check(
            predicate="every repeat of one input yields the output the first repeat yielded",
            passed=passed,
            scope_tested=scope_tested,
        ),
        result={
            "repeats": repeats,
            "identical": identical,
            "hidden_state": hidden_state,
            "summary": summary,
            **(extra or {}),
        },
    )


def synthetic_replay_entry(
    vut: VerifierUnderTest,
    ctx: RunContext,
    *,
    subject_ref: str,
    origin: str,
    started: float | None = None,
    repeats: int = SYNTHETIC_REPEATS,
) -> contracts.Entry:
    """D10 on a grader with nothing to replay: the pair its own declaration asks for (A-020).

    A grader with no task set and no response bank was an absence here, and the absence was wrong:
    determinism is a property of the grader, and a grader that can be called can be called twice.
    So the pair is derived from the signature the grader declares, recorded in `assumptions` so a
    reader sees exactly what was graded, and replayed. What this cannot do is make the pair a real
    task, which is the first limitation on the entry.
    """
    started = ctx.clock() if started is None else started
    try:
        derived = _synthetic_pair(vut)
    except (TypeError, ValueError):
        derived = None
    if derived is None:
        return ctx.absence(
            "validity",
            "validity.replay_determinism",
            "whether the same input scored twice yields the same score",
            f"{origin}, and the declared signature does not say what a pair for it would be",
            "pass a response bank with `--responses <file>`, or a task set the audit can probe",
            ("no claim that this grader is deterministic",),
            subject_ref=subject_ref,
            state="COULD_NOT_CHECK",
            depends_on=("digest:source",),
            duration_s=ctx.clock() - started,
        )
    signature, kwargs = derived
    assumptions = [
        f"{origin}, so the pair replayed is the audit's own",
        f"the pair is derived from the declared signature `{signature}`: "
        f"{json.dumps(kwargs, sort_keys=True, default=repr)}",
    ]
    entry = vut.load()
    namespace = getattr(entry, "__globals__", {})
    before = _namespace_fingerprint(namespace)
    outputs: list[str] = []
    for _ in range(repeats):
        try:
            outputs.append(repr(entry(**kwargs)))
        except Exception as exc:  # noqa: BLE001 - the raise is the result, not an error here
            return _synthetic_entry(
                ctx,
                subject_ref=subject_ref,
                duration=ctx.clock() - started,
                passed=False,
                repeats=repeats,
                identical=False,
                hidden_state=False,
                summary=(
                    f"the grader raised {type(exc).__name__} on the pair its own signature asks "
                    f"for, on repeat {len(outputs) + 1} of {repeats}"
                ),
                scope_tested=f"one pair derived from `{signature}`, graded {len(outputs) + 1} times",
                assumptions=assumptions,
                extra={"raised": type(exc).__name__, "repeats_completed": len(outputs)},
            )
    identical = len(set(outputs)) <= 1
    hidden_state = _namespace_fingerprint(namespace) != before
    return _synthetic_entry(
        ctx,
        subject_ref=subject_ref,
        duration=ctx.clock() - started,
        passed=identical and not hidden_state,
        repeats=repeats,
        identical=identical,
        hidden_state=hidden_state,
        summary=", ".join(
            (
                f"{repeats} repeats",
                "identical output" if identical else f"{len(set(outputs))} different outputs",
                "no hidden state" if not hidden_state else "the module's own names moved",
            )
        ),
        scope_tested=f"one pair derived from `{signature}`, graded {repeats} times",
        assumptions=assumptions,
        extra={"output": outputs[0] if outputs else None, "checked": "the module's namespace"},
    )


# --- D10, replay determinism ---------------------------------------------------------------------


def replay_entry(
    vut: VerifierUnderTest, corpus: Corpus, ctx: RunContext, *, subject_ref: str, repeats: int = 2
) -> contracts.Entry:
    """D10 as the fifth dynamic check: does the same input score the same twice."""
    from reward_lens.verifier.replay import replay_fidelity

    started = ctx.clock()
    if not corpus.rollouts:
        return synthetic_replay_entry(
            vut,
            ctx,
            subject_ref=subject_ref,
            origin=f"no rollouts to replay: {corpus.origin}",
            started=started,
        )
    rollouts = corpus.rollouts
    seeded = False
    if all(rollout.score is None for rollout in rollouts):
        # D10 compares against a recorded score. With no training record there is none, so the
        # audit grades each input once and replays against that: this measures the grader's
        # agreement with itself, which is the determinism question, and the entry says so.
        rollouts = _seed_scores(vut, rollouts)
        seeded = True
    if not rollouts:
        return synthetic_replay_entry(
            vut,
            ctx,
            subject_ref=subject_ref,
            origin=(
                "no input in the corpus could be graded even once, so there was nothing in it "
                f"to replay: {corpus.origin}"
            ),
            started=started,
        )
    reading = replay_fidelity(vut, ListCorpus.of(rollouts), repeats=repeats)
    duration = ctx.clock() - started
    if is_refusal(reading):
        return ctx.absence(
            "validity",
            "validity.replay_determinism",
            "whether the same input scored twice yields the same score",
            f"{reading.instrument} refused: {reading.detail}",
            reading.remedy,
            ("no claim that this grader is deterministic",),
            subject_ref=subject_ref,
            state="REFUSED",
            depends_on=("digest:source",),
            duration_s=duration,
        )
    report = getattr(reading, "value", reading)
    nondeterministic = int(getattr(report, "n_nondeterministic", 0) or 0)
    attempted = int(getattr(report, "n_attempted", len(corpus)) or 0)
    unreplayable = int(getattr(report, "n_unreplayable", 0) or 0)
    passed = nondeterministic == 0 and unreplayable == 0
    return contracts.Entry(
        entry_id="validity.replay_determinism",
        section="validity",
        kind="check",
        measurand="whether the same input scored twice yields the same score",
        method=contracts.Method(
            id="validity.replay_determinism",
            version="1.0.0",
            params_digest=contracts.digest({"repeats": repeats, "n": attempted}),
            procedure=(
                f"every rollout is re-graded {repeats} times and the scores are compared with "
                f"each other; a rollout whose repeats disagree is nondeterministic"
            ),
            credited_to="reward_lens.verifier.replay.replay_fidelity (D10)",
        ),
        scope="evaluator_comparison",
        subject_ref=subject_ref,
        depends_on=["digest:source"],
        state="complete",
        provenance=ctx.provenance(duration_s=duration),
        limitations=[
            "what this reads of hidden state is a repeat that disagrees; state that carries "
            "between calls and does not move the score is not visible to it",
            f"the corpus is {corpus.origin}",
        ]
        + (
            [
                "no training record supplied a score to replay against, so the baseline is the "
                "audit's own first grading: this is self-agreement, not fidelity to a record"
            ]
            if seeded
            else []
        ),
        check=contracts.Check(
            predicate="every input re-graded yields the score it yielded the first time",
            passed=passed,
            scope_tested=f"{attempted} inputs, each graded {repeats} times",
        ),
        result={
            "n_attempted": attempted,
            "repeats": repeats,
            "n_nondeterministic": nondeterministic,
            "n_unreplayable": unreplayable,
            "n_reproduced": int(getattr(report, "n_reproduced", 0) or 0),
            # A-015's three fields, and the sentence a surface prints without recomposing them.
            # `hidden_state` is what this check can see of state carried between calls: a repeat
            # that disagrees. State that carries and does not move the score is invisible here,
            # which is what the limitation below says.
            "identical": passed,
            "hidden_state": nondeterministic > 0,
            "summary": ", ".join(
                (
                    f"{repeats} repeats",
                    "identical output"
                    if passed
                    else f"{nondeterministic} of {attempted} inputs disagreed across repeats",
                    "no hidden state" if nondeterministic == 0 else "state carried between calls",
                )
            ),
        },
    )


# --- D8, the exposure inventory, static only -----------------------------------------------------


def attack_surface_entry(
    vut: VerifierUnderTest, ctx: RunContext, *, subject_ref: str, source: str = ""
) -> contracts.Entry:
    """D8 as a static exposure inventory. Never an executed reach finding: that is P-REACH's."""
    from reward_lens.verifier.attack import attack_surface

    started = ctx.clock()
    reading = attack_surface(vut)
    duration = ctx.clock() - started
    if is_refusal(reading):
        return ctx.absence(
            "reach",
            "reach.attack_surface",
            "what the graded process can touch that the score depends on",
            f"{reading.instrument} refused: {reading.detail}",
            reading.remedy,
            ("no claim about what the graded process can reach",),
            subject_ref=subject_ref,
            state="REFUSED",
            depends_on=("digest:source",),
            duration_s=duration,
        )
    surface = getattr(reading, "value", reading)
    accesses = tuple(getattr(surface, "accesses", ()) or ())
    by_kind: dict[str, int] = {}
    for access in accesses:
        key = str(getattr(access, "kind", "unknown"))
        by_kind[key] = by_kind.get(key, 0) + 1
    # An exposure is a crossing and not an access. The census of call sites by kind is a fact about
    # the source, and it stays under `by_kind`; the surfaces *exposed* are the paths the graded
    # process can write and the grader then reads, which is the same count the isolation witness
    # measures in this record, off the same walk (`static_checks.crossings`). Counting call sites
    # here was how one record came to say 12 surfaces exposed and 0 crossings in the same panel.
    try:
        crossings = static_checks.crossings(ast.parse(source))
    except (SyntaxError, OSError, ValueError):
        crossings = ()
    exposed = len(crossings)
    return contracts.Entry(
        entry_id="reach.attack_surface",
        section="reach",
        # A detection and not a check: what this reads off the source is the presence of an
        # exposure, and a count of them. A failed check would say the grader failed a bar this
        # inventory never set, and it would take the section's glyph to `✘` when what the
        # transcript prints is `!`: the panel ran, statically, and its executed witness did not.
        kind="detection",
        measurand="the surfaces the grader's source exposes to the graded process",
        method=contracts.Method(
            id="reach.attack_surface",
            version="1.0.0",
            params_digest=contracts.digest({"rung": 0, "arm": "static"}),
            procedure=(
                "the grader's source is parsed and every read, write, execution and credential "
                "site it matches a pattern for is inventoried; nothing is executed"
            ),
            credited_to="reward_lens.verifier.attack.attack_surface (D8)",
        ),
        scope="evaluator_comparison",
        subject_ref=subject_ref,
        depends_on=["digest:source"],
        state="partial",
        provenance=ctx.provenance(duration_s=duration, arm="white_box"),
        limitations=[
            "static: no executed witness",
            "the inventory is a floor: a surface reached through a helper, through getattr or "
            "through a library with no pattern here is not counted",
        ],
        n=exposed,
        detection=contracts.models.Detection(
            target="a surface the grader's source exposes to the graded process",
            material=None,
            unqualified=True,
            unqualified_reason=(
                "static analysis with no executed witness and no reference material for this "
                "substrate: nothing here establishes a limit of detection, so the count is a "
                "floor and not a qualified detection"
            ),
        ),
        result={
            # A-015: the exposure the surface names, the one-line count and the note that says
            # what the count is not. `exposure_id` is the catalogue's id for the exposure this
            # inventory found, and is absent when it found none.
            "exposure_id": _exposure_id(crossings),
            "summary": (
                f"{exposed} surface{'' if exposed == 1 else 's'} exposed, by static analysis only"
            ),
            "note": "no executed witness: not in this build",
            "surfaces_exposed": exposed,
            "by_kind": by_kind,
            "accesses": len(accesses),
            "taints": len(tuple(getattr(surface, "taints", ()) or ())),
            "crossings": exposed,
        },
    )


# --- the fourth static check: what the grader returns on malformed, empty and oversized input -----


def input_handling_entry(
    grader_path: Path,
    entrypoint: str,
    corpus: Corpus,
    ctx: RunContext,
    *,
    subject_ref: str,
    trainer: str | None,
    project_dir: Path | None,
) -> contracts.Entry:
    """Three responses a policy can really emit, through the adapter, paired with D-65."""
    from reward_lens.execution import Limits
    from reward_lens.graders.python_grader import PythonGrader

    started = ctx.clock()
    task = normalised_task(corpus.tasks[0] if corpus.tasks else {}, "probe-task")
    grader = PythonGrader.from_path(grader_path, entrypoint)
    limits = Limits()
    observed: list[dict[str, Any]] = []
    codes: list[str] = []
    adapter_errors: list[str] = []
    for name, text in _probes():
        envelope = grader.score(
            task, text, sandbox=ctx.sandbox, limits=limits, project_dir=project_dir
        )
        verdict = str(envelope.verdict)
        adapter_errors.extend(str(err) for err in (envelope.errors or ()))
        seen = {str(getattr(finding, "code", "")) for finding in (envelope.findings or ())}
        seen |= {code for code in ("RL0210", "RL0211", "RL0212") if code in " ".join(envelope.errors or ())}
        if verdict == "unscored":
            seen.add("RL0210")
        elif verdict == "grader_error":
            seen.add("RL0211")
        if isinstance(envelope.score, bool):
            seen.add("RL0212")
        seen.discard("")
        codes.extend(sorted(seen))
        observed.append(
            {
                "probe": name,
                "bytes": len(text),
                "verdict": verdict,
                "score": envelope.score,
                "codes": sorted(seen),
            }
        )
    duration = ctx.clock() - started
    unique = sorted(set(codes))
    unreached = [probe for probe in observed if probe["verdict"] in NO_OBSERVATION]
    if len(unreached) == len(observed):
        # The adapter never reached the grader, so nothing was observed about its input handling.
        # A check that passed because it could not run is exactly what D-18 refuses.
        return ctx.absence(
            "validity",
            "validity.input_handling",
            "what the grader returns on malformed, empty and oversized input",
            "the adapter returned no result for any probe (verdict "
            + "; ".join(str(probe["verdict"]) for probe in observed)
            + f") at sandbox tier {getattr(ctx.sandbox, 'tier', 'unknown')}"
            + (f": {adapter_errors[0]}" if adapter_errors else ""),
            "run the audit in a build whose Python adapter can reach the grader inside the "
            "sandbox; until it can, nothing is known about this grader's input handling",
            ("no claim about what this grader returns on malformed, empty or oversized input",),
            subject_ref=subject_ref,
            state="COULD_NOT_CHECK",
            depends_on=("digest:source",),
            duration_s=duration,
        )
    passed = not unique
    # Kept by code as well as in a list: a finding's message is one sentence about one code
    # (A-015), and a run that fires two codes joined them into a sentence neither code made.
    by_code = {
        row[0]: trainer_sentence(row, trainer)
        for row in TRAINER_BEHAVIOUR.values()
        if row[0] in unique
    }
    consequences = list(by_code.values())
    # Some probes reached the grader and some did not: the entry says which, and says partial, so
    # that a pass is never read as covering an input nothing was learned about.
    partial_notes = (
        [
            "the adapter returned no result for "
            + ", ".join(str(probe["probe"]) for probe in unreached)
            + f" at sandbox tier {getattr(ctx.sandbox, 'tier', 'unknown')}"
            + (f" ({adapter_errors[0]})" if adapter_errors else "")
            + "; nothing is claimed about those inputs"
        ]
        if unreached
        else []
    )
    reached = len(observed) - len(unreached)
    return contracts.Entry(
        entry_id="validity.input_handling",
        section="validity",
        kind="check",
        measurand=(
            "what the grader returns on malformed, empty and oversized input, and what the named "
            "trainer does with that value"
        ),
        method=contracts.Method(
            id="validity.input_handling",
            version="1.0.0",
            params_digest=contracts.digest(
                {"probes": [name for name, _ in _probes()], "oversized_bytes": _OVERSIZED}
            ),
            procedure=(
                "three responses a policy can emit (a malformed code fence, an empty string and "
                "an oversized one) are graded through the Python adapter inside the sandbox, and "
                "each returned value is paired with what the named trainer does with it (D-65)"
            ),
            credited_to="reward_lens.graders.python_grader.PythonGrader",
        ),
        scope="evaluator_comparison",
        subject_ref=subject_ref,
        depends_on=["digest:source"],
        state="partial" if unreached else "complete",
        provenance=ctx.provenance(duration_s=duration),
        limitations=[
            (
                f"the trainer's behaviour is what {trainer} does with the returned value; it is "
                "read from that trainer's documented semantics, not measured here"
                if trainer_is_documented(trainer)
                else f"`{trainer}` is declared under `reward.trainer` and this build has read no "
                "semantics for it, so nothing here is claimed about what it does with the "
                "returned value; what the grader returned is measured either way"
                if trainer
                else "no trainer is declared under `reward.trainer`, so nothing here is claimed "
                "about what a trainer does with the returned value; declare one to have its "
                "documented handling read against what the grader returned"
            ),
            "the probes vary the response, which is what a policy controls; a malformed task is "
            "the harness's own defect and is not in scope here",
            *partial_notes,
        ],
        assumptions=[
            "the adapter's input contract requires task.id, and the task set names it task_id; "
            "the probe task carries both"
        ],
        check=contracts.Check(
            predicate=(
                "the grader returns a real score, or a refusal the trainer cannot mistake for "
                "one, on malformed, empty and oversized input"
            ),
            passed=passed,
            scope_tested=(
                f"{reached} of {len(observed)} probe responses against one task from the "
                "subject's own task set"
            ),
        ),
        result={
            "probes": observed,
            "codes": unique,
            "consequences": consequences,
            "consequence_by_code": by_code,
        },
    )
