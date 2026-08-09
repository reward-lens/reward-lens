"""Recognising a reward system's shape by running `inspect.signature` over it, and nothing else.

Three rules hold this module together.

Nothing here imports `trl`, `verifiers`, `transformers` or `verl`. A shape is established from the
parameter names, the parameter kinds, the return annotation and `inspect.iscoroutinefunction`, all
of which are readable without the framework that defines the convention being installed. A test
runs detection in a clean interpreter and asserts that none of those four modules arrived.

Nothing here asks the user which framework they have. D-65 forbids that surface, and the reason is
that the answer is already in the file.

The three silent failure modes are probed rather than described. `detect` calls the entry once on
a canonical row and reports what came back, so the finding carries a witness that was executed.
The probe runs the callable in this process, which is the one thing here that runs adopter code:
`probe=False` turns it off, and `detect_project`, which is what `init --detect` uses, passes it.

Owned by P-CONNECT.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .errors import SHAPE_UNSUPPORTED_PAGE
from .shapes import DECLARED_SHAPES, Detection, Finding, Shape, shape_for_declared
from .verifiers_discriminator import GROUP_INDICATORS, bound_names, is_group_rubric

__all__ = [
    "ProjectDetection",
    "SILENT_MODES",
    "detect",
    "detect_project",
    "render_detect_transcript",
]

#: verl's reward entry point, by the parameter names it has had since the repository opened.
_VERL_NAMES = frozenset({"data_source", "solution_str", "ground_truth"})

#: The three arguments TRL passes by keyword on every call (`grpo_trainer.py:1683-1685`, 1.13.0).
_TRL_NAMES = frozenset({"prompts", "completions", "completion_ids"})

#: Names only `verifiers` puts in a reward signature. TRL never passes any of them.
_VERIFIERS_NAMES = frozenset({"state", "states", "answer", "answers", "info", "infos", "task", "tasks"})

#: The three modes of D-65, in the order the transcript's `watch` list prints them.
SILENT_MODES: tuple[tuple[str, str], ...] = (
    ("RL0213", "none_to_nan"),
    ("RL0214", "exception_to_zero"),
    ("RL0215", "bool_as_score"),
)

_BATCH_PROBE: dict[str, Any] = {
    "prompts": ["probe prompt one", "probe prompt two"],
    "completions": ["probe completion one```x```", "probe completion two"],
    "completion_ids": [[1, 2], [3, 4]],
    "answer": ["42", "42"],
    "answers": ["42", "42"],
    "states": [{}, {}],
    "tasks": ["probe", "probe"],
    "infos": [{}, {}],
}

_SINGLE_PROBE: dict[str, Any] = {
    "prompt": "probe prompt",
    "completion": "probe completion```x```",
    #: inspect's second positional argument. Present so that a scorer this module recognises is a
    #: scorer this module can call: a missing one would come back as a fabricated RL0214.
    "target": "42",
    "answer": "42",
    "state": {},
    "task": "probe",
    "info": {},
}


# --- rendering a signature the way the transcript prints it --------------------------------------


def _annotation_text(annotation: Any) -> str:
    if annotation is inspect.Signature.empty:
        return "Any"
    if hasattr(annotation, "__origin__"):
        return str(annotation)
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation)


def signature_text(fn: Callable[..., Any], *, name: str | None = None) -> str:
    """`name(params) -> return`, with parameter names and kinds only. No types on the parameters."""
    signature = inspect.signature(fn)
    parts = []
    for parameter in signature.parameters.values():
        if parameter.kind is parameter.VAR_KEYWORD:
            parts.append("**" + parameter.name)
        elif parameter.kind is parameter.VAR_POSITIONAL:
            parts.append("*" + parameter.name)
        else:
            parts.append(parameter.name)
    label = name or getattr(fn, "__name__", type(fn).__name__)
    return f"{label}({', '.join(parts)}) -> {_annotation_text(signature.return_annotation)}"


# --- resolving what was handed in ----------------------------------------------------------------


def _module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(f"_rl_connect_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"{path} is not importable as a Python module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def module_callables(module) -> list[tuple[str, Callable[..., Any]]]:
    """Every callable the module defines itself, in the order the file defines them."""
    found = []
    for name, value in vars(module).items():
        if name.startswith("_") or not callable(value):
            continue
        if getattr(value, "__module__", None) != module.__name__:
            continue
        found.append((name, value))
    found.sort(key=lambda pair: getattr(pair[1], "__code__", None).co_firstlineno if hasattr(pair[1], "__code__") else 0)
    return found


def _resolve(target: Any) -> tuple[Any, str]:
    if isinstance(target, Path):
        module = _module_from_path(target)
        callables = module_callables(module)
        if not callables:
            raise ValueError(f"{target} defines no callable to connect")
        name, fn = callables[0]
        return fn, f"{target.stem}:{name}"
    if isinstance(target, str):
        module_name, separator, attribute = target.partition(":")
        if not separator or not attribute:
            raise ValueError(f"{target!r} is not module:callable or a path")
        fn = getattr(importlib.import_module(module_name), attribute)
        return fn, target
    module_name = getattr(target, "__module__", type(target).__module__)
    label = getattr(target, "__qualname__", type(target).__qualname__)
    return target, f"{module_name}:{label}"


# --- classification, all of it by signature -------------------------------------------------------


def _is_step_environment(target: Any) -> bool:
    return (
        not inspect.isfunction(target)
        and not inspect.ismethod(target)
        and callable(getattr(target, "step", None))
        and callable(getattr(target, "reset", None))
    )


def _is_pretrained_model(target: Any) -> bool:
    forward = getattr(target, "forward", None)
    if forward is None or getattr(target, "config", None) is None:
        return False
    try:
        parameters = inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return False
    return "input_ids" in parameters


def _is_inspect_scorer(signature: inspect.Signature) -> bool:
    """inspect's scorer: two positional parameters, `state` then `target`.

    D-65 says this family is discriminated by type rather than by name, so the annotations decide
    where they are written down, and the parameter pair decides where they are not. Either way the
    test runs before the `verifiers` name list, because `state` is a name both conventions use and
    only inspect puts a `target` beside it.
    """
    positional = [
        p
        for p in signature.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) != 2 or len(signature.parameters) != 2:
        return False
    annotations = [_annotation_text(p.annotation) for p in positional]
    by_type = annotations[0].endswith("TaskState") and annotations[1].endswith("Target")
    return by_type or [p.name for p in positional] == ["state", "target"]


def _declared_family(fn: Callable[..., Any]) -> str:
    """D-65's family, from the signature alone, in the order the conventions can be told apart.

    TRL is the one that has to be got right, because it is the one an adopter already has. It binds
    a reward function entirely by keyword: `reward_func(prompts=..., completions=...,
    completion_ids=..., **reward_kwargs)` at `trl/trainer/grpo_trainer.py:1683-1685` on TRL 1.13.0.
    So a function that declares only the arguments it uses needs a var-keyword to absorb the rest,
    and `completions` plus `**kwargs` is as much the TRL convention as all three names written out.
    Parameter order does not enter into it and neither does keyword-only, because nothing is ever
    passed positionally.
    """
    signature = inspect.signature(fn)
    names = set(signature.parameters)
    var_keyword = any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values())
    if _is_inspect_scorer(signature):
        return "inspect"
    if names & _VERIFIERS_NAMES:
        return "verifiers"
    if names & _TRL_NAMES and (var_keyword or _TRL_NAMES <= names):
        return "trl"
    if names & GROUP_INDICATORS and is_group_rubric(fn):
        # Plural names and no var-keyword: TRL's extra arguments would not bind, so what is left
        # that calls this is a rubric, which binds the subset of what it has by name.
        return "verifiers"
    return "plain"


def _classify(target: Any, entry: str) -> Detection:
    evidence: list[str] = []
    if _is_step_environment(target):
        detail = (
            "a step() environment returns its reward inside a transition, so there is no call "
            "that scores one response against one task"
        )
        return Detection(
            shape=Shape.UNSUPPORTED,
            entry=entry,
            evidence=[
                f"signature: {signature_text(target.step, name='step')}",
                "inspect.signature: step() and reset() present, so this is an environment",
            ],
            declared="unsupported",
            reason=detail,
            docs_page=SHAPE_UNSUPPORTED_PAGE,
            target=target,
        )
    if _is_pretrained_model(target):
        forward_text = signature_text(target.forward, name="forward")
        output = _annotation_text(inspect.signature(target.forward).return_annotation)
        return Detection(
            shape=Shape.PRETRAINED_MODEL,
            entry=entry,
            evidence=[
                f"signature: {forward_text}",
                f"inspect.signature: forward returns {output}, read through .logits[:, 0]",
                "config present, so the scoring head is the one the trainer already loaded",
            ],
            declared="trl",
            target=target,
        )

    fn = target if (inspect.isfunction(target) or inspect.ismethod(target)) else target
    signature = inspect.signature(fn)
    names = set(signature.parameters)
    text = signature_text(fn, name=entry.rpartition(":")[2].rpartition(".")[2])
    evidence.append(f"signature: {text}")

    if len(names & _VERL_NAMES) >= 2:
        detail = (
            "verl's compute_score takes a data source and a ground truth rather than a task and "
            "a response, and reward-lens would have to guess how its extra_info reaches a score"
        )
        return Detection(
            shape=Shape.UNSUPPORTED,
            entry=entry,
            evidence=evidence + [f"inspect.signature: {sorted(names & _VERL_NAMES)} is verl"],
            declared="unsupported",
            reason=detail,
            docs_page=SHAPE_UNSUPPORTED_PAGE,
            target=target,
        )

    awaits = inspect.iscoroutinefunction(fn)
    declared = _declared_family(fn)
    group = is_group_rubric(fn)
    shape = shape_for_declared(declared, awaits=awaits, group=group)
    evidence.append(f"inspect.signature: parameters {sorted(names)}, family {declared}")
    if awaits:
        evidence.append("inspect.iscoroutinefunction: true, so the entry is awaited")
    if declared == "verifiers":
        evidence.append(f"verifiers discriminator: group={group}, by the rule Rubric uses")
    return Detection(
        shape=shape, entry=entry, evidence=evidence, declared=declared, awaits=awaits, target=target
    )


# --- the probe, and the three findings it can produce ---------------------------------------------


def _finding(code: str, rule: str, trainer: str, behaviour: str, message: str, witness: dict) -> Finding:
    return Finding(
        code=code,
        rule=rule,
        level="warning",
        trainer=trainer,
        trainer_behaviour=behaviour,
        message=message,
        witness=witness,
    )


def _call(detection: Detection, row: dict[str, Any]) -> Any:
    fn = detection.target
    arguments = {name: row[name] for name in bound_names(fn, row)}
    if detection.awaits:
        return asyncio.run(fn(**arguments))
    return fn(**arguments)


def probe(detection: Detection) -> list[Finding]:
    """Call the entry once and report what D-65 says the frameworks would have swallowed."""
    if detection.shape in (Shape.UNSUPPORTED, Shape.PRETRAINED_MODEL):
        return []
    row = _BATCH_PROBE if detection.shape in (Shape.BATCH_FN, Shape.ASYNC_TRL) else _SINGLE_PROBE
    try:
        returned = _call(detection, row)
    except Exception as caught:  # the mode being reported is that this is swallowed elsewhere
        return [
            _finding(
                "RL0214",
                "exception_to_zero",
                "verifiers",
                "Rubric logs the exception and records 0.0 for that reward function",
                f"{detection.entry} raised {type(caught).__name__} on a well formed row",
                {
                    "called": True,
                    "entry": detection.entry,
                    "exception": type(caught).__name__,
                    "detail": str(caught),
                    "would_have_scored": 0.0,
                },
            )
        ]

    values = returned if isinstance(returned, list) else [returned]
    findings: list[Finding] = []
    for index, value in enumerate(values):
        if value is None:
            findings.append(
                _finding(
                    "RL0213",
                    "none_to_nan",
                    "trl",
                    "GRPOTrainer turns the None into NaN and drops that reward function for the row",
                    f"{detection.entry} returned None for row {index}",
                    {
                        "called": True,
                        "entry": detection.entry,
                        "returned": list(values),
                        "index": index,
                        "would_have_become": "nan",
                    },
                )
            )
            break
    for index, value in enumerate(values):
        if isinstance(value, bool):
            findings.append(
                _finding(
                    "RL0215",
                    "bool_as_score",
                    "trl",
                    "float() accepts the boolean, so 1.0 and 0.0 are recorded as though measured",
                    f"{detection.entry} returned a bool for row {index}",
                    {
                        "called": True,
                        "entry": detection.entry,
                        "returned": value,
                        "index": index,
                        "float_of_it": float(value),
                    },
                )
            )
            break
    return findings


# --- the public entry point -----------------------------------------------------------------------


def detect(target: str | Callable[..., Any] | Path, *, probe_entry: bool = True) -> Detection:
    """Recognise a reward system's shape from its signature. Nothing is asked and nothing is guessed.

    `target` is a live callable, a `module:callable` string, or a path to a reward file. The call
    is interfaces section 1's, so it returns rather than raises: a shape 4.0 will not adapt comes
    back as the `Shape.UNSUPPORTED` detection, carrying its `reason` and its docs page, and RL0710
    is raised by whoever tries to use it. That is `bind`, and it is `init --detect`. Pass
    `probe_entry=False` to skip the call that produces the silent-mode witnesses.
    """
    resolved, entry = _resolve(target)
    detection = _classify(resolved, entry)
    if probe_entry:
        detection.warnings = probe(detection)
    return detection


# --- a whole project, which is what `init --detect` looks at --------------------------------------


@dataclass
class ProjectDetection:
    """Everything the connect transcript prints, read off one directory."""

    root: Path
    reward_file: str
    callables: list[Detection] = field(default_factory=list)
    task_path: str = ""
    task_rows: int = 0
    task_fields: list[str] = field(default_factory=list)
    outcome: str | None = None

    def project_file_text(self, *, displayed: str) -> str:
        """The `rewardlens.yaml` D-65 writes, and the same bytes the transcript shows indented."""
        entries = [f'"{d.entry}",' for d in self.callables]
        shapes = [f"{d.declared}," for d in self.callables]
        entry_width = max(len(e) for e in entries)
        shape_width = max(len(s) for s in shapes)
        lines = ["reward:", "  kind: composite", "  components:"]
        for entry, shape in zip(entries, shapes):
            lines.append(
                f"    - {{entry: {entry:<{entry_width}} shape: {shape:<{shape_width}} weight: 1.0}}"
            )
        watch = ", ".join(rule for _, rule in SILENT_MODES)
        lines.append(f"  watch: [{watch}]")
        reference = self.task_fields[1] if len(self.task_fields) > 1 else self.task_fields[0]
        lines.append(
            f"tasks:   {{path: {self.task_path}, prompt: {self.task_fields[0]}, "
            f"reference: {reference}}}"
        )
        lines.append(
            f"outcome: null      # declare one with: reward-lens init {displayed} --outcome ./tests"
        )
        return "\n".join(lines) + "\n"

    def write_project_file(self, *, displayed: str) -> Path:
        """Write `rewardlens.yaml` into the project root and return where it landed."""
        target = self.root / "rewardlens.yaml"
        target.write_text(self.project_file_text(displayed=displayed))
        return target


def detect_project(root: str | Path) -> ProjectDetection:
    """Scan a directory the way `init --detect` does: the reward file, then the task file.

    The reward module is imported, because that is what a trainer does with it, and every shape is
    then read off `inspect.signature`. The entries are not called here: a project scan produces the
    `watch` list, and the witnesses behind it are produced when the audit runs.
    """
    base = Path(root)
    reward_file = next(
        (name for name in ("rewards.py", "reward.py", "rewards/__init__.py") if (base / name).is_file()),
        "",
    )
    if not reward_file:
        raise ValueError(f"{base} holds no reward file to connect")
    module = _module_from_path(base / reward_file)
    stem = Path(reward_file).stem
    callables = [_classify(fn, f"{stem}:{name}") for name, fn in module_callables(module)]

    task_path, rows, fields = _tasks(base)
    return ProjectDetection(
        root=base,
        reward_file=reward_file,
        callables=callables,
        task_path=task_path,
        task_rows=rows,
        task_fields=fields,
    )


def _tasks(base: Path) -> tuple[str, int, list[str]]:
    for relative in ("data/train.jsonl", "train.jsonl", "data/tasks.jsonl"):
        path = base / relative
        if not path.is_file():
            continue
        fields: list[str] = []
        rows = 0
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows += 1
                if not fields:
                    fields = list(json.loads(line))
        return relative, rows, fields
    return "", 0, []


# --- the transcript -------------------------------------------------------------------------------


def render_detect_transcript(project: ProjectDetection, *, displayed: str) -> str:
    """Section 5.9's transcript, rendered from one scan. Every count in it came off the fixture.

    The shape column is what the connector has to show: it was filled without the user being asked
    which framework they run. The `watch` list names D-65's three silent modes before the first
    audit, and `outcome: null` is printed rather than omitted, with the command that fills it.
    """
    signatures = [d.signature_line for d in project.callables]
    width = max(len(s) for s in signatures) + 4
    lines = [
        f"  Found      {project.reward_file}  ·  {len(project.callables)} callables recognised"
    ]
    for detection, text in zip(project.callables, signatures):
        lines.append(f"    {text:<{width}}{detection.declared}")
    lines.append(
        f"  Tasks      {project.task_path}  ·  {project.task_rows} rows  ·  "
        f"fields: {', '.join(project.task_fields)}"
    )
    lines.append(
        "  Outcome    none declared. A protected check is what turns a score into a claim."
    )
    lines.append("")
    lines.append(f"  Wrote {displayed}/rewardlens.yaml")
    for line in project.project_file_text(displayed=displayed).splitlines():
        lines.append("    " + line)
    lines += [
        "",
        "  Weights default to 1.0 until you declare them. The audit reads what your run recorded",
        "  and reports any disagreement with this file as a finding.",
        "",
        f"  Next   reward-lens audit {displayed}",
    ]
    return "\n".join(lines) + "\n"


def _unused() -> None:  # pragma: no cover - keeps the import list honest for linters
    _ = (DECLARED_SHAPES, sys)
