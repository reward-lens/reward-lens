"""The four static validity checks, over the grader's own source with the standard library's `ast`.

Section 7.3 names them and says what they are not: a generic lint command does not satisfy any of
them. Each one answers a question about this grader and returns what it looked at, so the entry it
becomes can name a line and a path rather than a verdict on its own.

Every check is a floor, never a ceiling. A grader that reaches a file through `getattr`, through a
helper this does not follow, or through a library with no pattern here is a grader whose exposure
this under-reports, and each result says so in its own limitations.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = ["CHECK_IDS", "STATIC_ARM", "StaticResult", "rule_name", "run_static_checks"]

#: The arm these checks run on: they read the grader's source and never execute it.
STATIC_ARM = "static"

#: Rule name -> the finding code it raises when it fails (A-025 point 2). The key is the rule and
#: not the entry id, because the rule carries the arm the check ran on and the entry id does not.
#: One measurand can be asked twice, once by reading the source and once by executing it, and the
#: two are the same hole in the record and two different claims in the findings: a reader who is
#: told a path is writable wants to know whether that was read off the source or watched happening.
#: The entry id stays `validity.writable_then_read` so the hole does not move.
CHECK_IDS: dict[str, str] = {
    "validity.static.writable_then_read": "RL0201",
    "validity.static.input_leakage": "RL0202",
    "validity.static.unreachable_score_branch": "RL0203",
}


def rule_name(entry_id: str, arm: str) -> str:
    """The rule an entry's finding is raised under: the entry id with the arm named in it.

    `validity.writable_then_read` measured statically is `validity.static.writable_then_read`. An
    entry id that already names an arm is handed back unchanged, so calling this twice is safe.
    """
    section, _, rest = entry_id.partition(".")
    if not rest or not arm or rest.startswith(f"{arm}."):
        return entry_id
    return f"{section}.{arm}.{rest}"

#: Names that, read out of a task, are the task's own solution rather than a declared output shape.
ANSWER_KEYS = frozenset(
    {
        "answer",
        "answers",
        "solution",
        "solutions",
        "canonical_solution",
        "gold",
        "gold_answer",
        "ground_truth",
        "reference_solution",
        "label",
        "labels",
    }
)

#: Calls through which a graded response is executed by the grader itself.
_EXEC_NAMES = frozenset({"exec", "eval", "execfile"})
_EXEC_DOTTED = frozenset(
    {"os.system", "os.popen", "subprocess.run", "subprocess.call", "subprocess.Popen",
     "subprocess.check_output", "subprocess.check_call", "runpy.run_path", "runpy.run_module"}
)

#: Calls through which the grader reads a file back.
_READ_ATTRS = frozenset({"read_text", "read_bytes", "read", "readlines", "readline"})
_READ_DOTTED = frozenset({"json.load", "yaml.safe_load", "yaml.load", "pickle.load", "tomllib.load"})


@dataclass
class StaticResult:
    """One static check: what it asked, what it saw, and where."""

    entry_id: str
    measurand: str
    predicate: str
    scope_tested: str
    passed: bool
    observed: str
    result: dict[str, Any] = field(default_factory=dict)
    line: int | None = None
    target_path: str | None = None
    limitations: tuple[str, ...] = ()
    source_name: str = ""
    #: The sentence a finding carries, which is not the observation. `observed` is the full reading,
    #: with the execution site and the reason the two together matter; A-015 freezes the finding's
    #: message as the shorter sentence the transcript prints. A check that does not set one has no
    #: finding to carry it, and falls back to the observation.
    message: str = ""

    def __post_init__(self) -> None:
        if not self.message:
            self.message = self.observed

    @property
    def rule(self) -> str:
        """The name the finding is raised under, with the static arm in it (A-025 point 2)."""
        return rule_name(self.entry_id, STATIC_ARM)

    @property
    def code(self) -> str | None:
        return None if self.passed else CHECK_IDS.get(self.rule)


# --- small ast helpers ---------------------------------------------------------------------------


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _literal_path(node: ast.AST, constants: dict[str, str]) -> str | None:
    """The relative path a node names, or None when it is not a literal relative path.

    `Path("outcome") / "test_solution.py"` resolves; `Path(__file__).parent / "x"` does not, and
    that distinction is the whole check: only a path the graded process's working directory can
    reach is a path the graded process can write.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _literal_path(node.left, constants)
        right = _literal_path(node.right, constants)
        return None if left is None or right is None else f"{left}/{right}"
    if isinstance(node, ast.Call):
        dotted = _dotted(node.func)
        if dotted in {"Path", "pathlib.Path", "os.path.join"} and node.args:
            parts = [_literal_path(arg, constants) for arg in node.args]
            return None if any(part is None for part in parts) else "/".join(parts)  # type: ignore[arg-type]
    return None


def _module_constants(tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for statement in tree.body:
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target] if isinstance(statement, ast.AnnAssign) and statement.value
            else []
        )
        value = getattr(statement, "value", None)
        for target in targets:
            if isinstance(target, ast.Name) and value is not None:
                resolved = _literal_path(value, constants)
                if resolved is not None:
                    constants[target.id] = resolved
    return constants


def _entrypoint_params(tree: ast.Module, entrypoint: str) -> tuple[str | None, str | None]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entrypoint:
            names = [argument.arg for argument in node.args.args]
            return (names[0] if names else None, names[1] if len(names) > 1 else None)
    return (None, None)


def _names_in(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


# --- RL0201: the graded process can write what the grader later reads ----------------------------


@dataclass(frozen=True)
class Crossing:
    """A path the graded process can write and the grader then reads.

    It takes both facts. A read of a literal relative path is only in the graded process's reach
    when the grader runs that process in the same working directory, and an execution site with
    nothing read afterwards crosses nothing. The isolation check and the exposure inventory read
    the two off one walk, so they cannot come to disagree about what the source holds.
    """

    read_line: int
    read_path: str
    execution_line: int
    execution_call: str


def _relative_reads_and_execs(
    tree: ast.Module,
) -> tuple[list[tuple[int, str]], list[tuple[int, str]], int]:
    """Every literal relative read site, every execution site, and the reads with no literal path.

    The third number is what this walk could not resolve: a read whose path is built at runtime is
    counted but not located, which is why the inventory below it is a floor and says so.
    """
    constants = _module_constants(tree)
    execs: list[tuple[int, str]] = []
    reads: list[tuple[int, str]] = []
    absolute_reads = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted(node.func)
        bare = node.func.id if isinstance(node.func, ast.Name) else ""
        if bare in _EXEC_NAMES or dotted in _EXEC_DOTTED:
            execs.append((node.lineno, bare or dotted))
            continue
        target: ast.AST | None = None
        if bare == "open" and node.args:
            target = node.args[0]
        elif isinstance(node.func, ast.Attribute) and node.func.attr in _READ_ATTRS:
            target = node.func.value
        elif dotted in _READ_DOTTED and node.args:
            target = node.args[0]
        if target is None:
            continue
        path = _literal_path(target, constants)
        if path is None:
            absolute_reads += 1
        elif not Path(path).is_absolute():
            reads.append((node.lineno, path))
    return reads, execs, absolute_reads


def crossings(tree: ast.Module) -> tuple[Crossing, ...]:
    """The crossings this source holds, which is what an exposure inventory counts.

    Empty when the grader never runs the graded response, because nothing a grader reads is
    writable by a process that never ran. The execution site named is the first, which is the one
    the isolation check reports; each read that site puts in reach is one crossing.
    """
    reads, execs, _ = _relative_reads_and_execs(tree)
    if not execs or not reads:
        return ()
    exec_line, exec_call = execs[0]
    return tuple(
        Crossing(
            read_line=line,
            read_path=path,
            execution_line=exec_line,
            execution_call=exec_call,
        )
        for line, path in reads
    )


def _writable_then_read(tree: ast.Module, source_name: str) -> StaticResult:
    reads, execs, absolute_reads = _relative_reads_and_execs(tree)
    limitations = (
        "static: the reads this follows are literal relative paths reachable from module "
        "constants; a path built at runtime, reached through getattr or opened by a library "
        "with no pattern here is not counted, so the inventory is a floor",
    )
    if execs and reads:
        line, path = reads[0]
        return StaticResult(
            entry_id="validity.writable_then_read",
            measurand="whether the graded process can write a path the grader later reads",
            predicate=(
                "no path the grader reads is reachable by the graded process's own working "
                "directory"
            ),
            scope_tested=f"every call site in {source_name} that reads a literal relative path",
            passed=False,
            observed=(
                f"{source_name} line {line} reads {path}, and line {execs[0][0]} runs the graded "
                f"response in the same working directory, so the response can write what the "
                f"grader then reads"
            ),
            message=(
                f"{source_name} line {line} reads {path}, which the graded process can write"
            ),
            source_name=source_name,
            result={
                "read_line": line,
                "read_path": path,
                "execution_line": execs[0][0],
                "execution_call": execs[0][1],
                "relative_reads": len(reads),
                "unresolved_reads": absolute_reads,
            },
            line=line,
            target_path=path,
            limitations=limitations,
        )
    why = (
        "the grader never executes the graded response"
        if not execs
        else "the grader reads no literal relative path"
    )
    return StaticResult(
        entry_id="validity.writable_then_read",
        measurand="whether the graded process can write a path the grader later reads",
        predicate="no path the grader reads is reachable by the graded process's working directory",
        scope_tested=f"every call site in {source_name} that reads a literal relative path",
        passed=True,
        observed=f"{why} ({len(reads)} relative reads, {len(execs)} execution sites)",
        result={
            "relative_reads": len(reads),
            "execution_sites": len(execs),
            "unresolved_reads": absolute_reads,
        },
        limitations=limitations,
    )


# --- RL0202: the task input carries the answer, or a path to it ----------------------------------


def _task_keys(tree: ast.Module, task_param: str | None) -> dict[str, int]:
    """Every key the grader pulls out of the task, and the line it read it on."""
    keys: dict[str, int] = {}
    if task_param is None:
        return keys
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id == task_param and isinstance(node.slice, ast.Constant):
                if isinstance(node.slice.value, str):
                    keys.setdefault(node.slice.value, node.lineno)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "get" and isinstance(node.func.value, ast.Name):
                if node.func.value.id == task_param and node.args:
                    first = node.args[0]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        keys.setdefault(first.value, node.lineno)
    return keys


def _answerish(key: str) -> bool:
    lowered = key.lower()
    if lowered in ANSWER_KEYS:
        return True
    if lowered.endswith(("_path", "_file", "_dir")):
        stem = lowered.rsplit("_", 1)[0]
        return stem in ANSWER_KEYS or any(word in stem for word in ("answer", "solution", "gold"))
    return False


def _input_leakage(
    tree: ast.Module, source_name: str, entrypoint: str, tasks: Sequence[dict] | None
) -> StaticResult:
    task_param, _ = _entrypoint_params(tree, entrypoint)
    keys = _task_keys(tree, task_param)
    leaking = sorted(key for key in keys if _answerish(key))
    scanned = 0
    task_keys: set[str] = set()
    if tasks is not None:
        for record in tasks:
            scanned += 1
            task_keys |= {str(key) for key in record}
        leaking = sorted(set(leaking) | {key for key in task_keys if _answerish(key)})
    scope = (
        f"the keys {entrypoint} reads out of its task argument in {source_name}"
        if tasks is None
        else f"those keys and the keys of {scanned} task records"
    )
    limitations = (
        "static: a key is judged by its name, so an answer stored under a name this set does "
        "not carry is not counted; and with no task set only the grader's own reads are in scope",
    )
    if leaking:
        line = min((keys[key] for key in leaking if key in keys), default=None)
        return StaticResult(
            entry_id="validity.input_leakage",
            measurand="whether the task input carries the answer or a path to it",
            predicate="no key the grader reads out of the task names the task's own solution",
            scope_tested=scope,
            passed=False,
            observed=(
                f"the task input carries {', '.join(leaking)}, which names the answer rather than "
                f"the question; anything that can see the task can see the answer"
            ),
            result={"leaking_keys": leaking, "task_keys_read": sorted(keys), "tasks_scanned": scanned},
            line=line,
            target_path=None,
            limitations=limitations,
        )
    return StaticResult(
        entry_id="validity.input_leakage",
        measurand="whether the task input carries the answer or a path to it",
        predicate="no key the grader reads out of the task names the task's own solution",
        scope_tested=scope,
        passed=True,
        observed=f"the grader reads {', '.join(sorted(keys)) or 'no task key'}; none names an answer",
        result={"task_keys_read": sorted(keys), "tasks_scanned": scanned},
        limitations=limitations,
    )


# --- RL0203: a score branch unreachable, or reachable from the input alone ------------------------


def _numeric(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if not isinstance(node.value, bool):
            return float(node.value)
    return None


def _unreachable_score_branch(tree: ast.Module, source_name: str, entrypoint: str) -> StaticResult:
    task_param, response_param = _entrypoint_params(tree, entrypoint)
    task_alone: list[dict[str, Any]] = []
    dead: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test_names = _names_in(node.test)
            constant = _numeric(node.test) if isinstance(node.test, ast.Constant) else None
            if isinstance(node.test, ast.Constant) and not node.test.value:
                dead.append({"line": node.lineno, "why": "the guard is a constant that is never true"})
            elif constant is not None and constant:
                pass
            returns = [
                child
                for child in node.body
                if isinstance(child, ast.Return) and _numeric(child.value) is not None
            ]
            for statement in returns:
                value = _numeric(statement.value)
                if value is None or value <= 0.0:
                    continue  # the score-domain floor is how a grader declines to pay, not a branch
                if task_param and task_param in test_names and (
                    not response_param or response_param not in test_names
                ):
                    task_alone.append(
                        {"line": statement.lineno, "value": value, "guard_line": node.lineno}
                    )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.If, ast.For, ast.While)):
            body = getattr(node, "body", [])
            for index, statement in enumerate(body[:-1]):
                if isinstance(statement, (ast.Return, ast.Raise)):
                    following = body[index + 1]
                    dead.append(
                        {"line": following.lineno, "why": "the statement before it always returns"}
                    )
    limitations = (
        "static: reachability is judged inside one function from literal guards, so a branch "
        "unreachable only through a caller's argument domain is not found here",
        "D1's decision coverage says which branches the response bank never took; a branch no "
        "corpus reached is not thereby unreachable, and it is not read as a failure of this check",
    )
    if task_alone or dead:
        first = (task_alone or dead)[0]
        observed = (
            f"line {first['line']} returns {first.get('value')} under a guard that reads the task "
            f"alone, so the score is decided before the response is looked at"
            if task_alone
            else f"line {first['line']} is a score branch no input can reach: {first['why']}"
        )
        return StaticResult(
            entry_id="validity.unreachable_score_branch",
            measurand=(
                "whether a score branch is unreachable from any legitimate input, or reachable "
                "from the input alone"
            ),
            predicate=(
                "every branch that returns a score is reachable, and none of them is decided by "
                "the task alone"
            ),
            scope_tested=f"every conditional return of a numeric score in {source_name}",
            passed=False,
            observed=observed,
            result={"task_only_branches": task_alone, "unreachable_branches": dead},
            line=int(first["line"]),
            target_path=None,
            limitations=limitations,
        )
    return StaticResult(
        entry_id="validity.unreachable_score_branch",
        measurand=(
            "whether a score branch is unreachable from any legitimate input, or reachable from "
            "the input alone"
        ),
        predicate=(
            "every branch that returns a score is reachable, and none of them is decided by the "
            "task alone"
        ),
        scope_tested=f"every conditional return of a numeric score in {source_name}",
        passed=True,
        observed="no score branch is decided by the task alone and none is statically unreachable",
        result={"task_only_branches": [], "unreachable_branches": []},
        limitations=limitations,
    )


def run_static_checks(
    source: str,
    *,
    source_name: str,
    entrypoint: str = "score",
    tasks: Sequence[dict] | None = None,
) -> list[StaticResult]:
    """The three source checks. The fourth, input handling, executes and lives in `instruments`."""
    tree = ast.parse(source, filename=source_name)
    return [
        _writable_then_read(tree, source_name),
        _input_leakage(tree, source_name, entrypoint, tasks),
        _unreachable_score_branch(tree, source_name, entrypoint),
    ]
