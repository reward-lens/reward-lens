"""Rule 19: four invariance relations and one killing relation, run on every grader with source.

The relations transform the response, not the grader. That is the whole point: a metamorphic test
asks what happens to the score when the thing being graded changes in a way that cannot change
whether it is correct. Renaming a local variable, reordering two statements that do not touch each
other, adding a comment, rewriting an expression into an equivalent one: a grader whose score moves
under any of these is scoring something other than the work.

The killing relation is the other half, and without it the four above are worthless. A grader that
returns the same number for every input passes every invariance relation perfectly. So one real
defect is injected into the response and the score has to move. A grader that does not notice is
not invariant, it is blind, and the pair of results says which.

Rule 19's judge relations are named here too and are applied when the response is not source code:
order swap, paraphrase and length padding. A judge that reorders into a different verdict is the
same failure in a different medium.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Sequence

from reward_lens import contracts
from reward_lens.instruments.base import RunContext

from .batteries import ACCEPT_THRESHOLD, Case, Runner, sandbox_runner, source_of

__all__ = [
    "INVARIANCE",
    "JUDGE_RELATIONS",
    "KILLING",
    "RelationOutcome",
    "pair",
    "relation_entries",
    "run_relations",
    "transform",
]

VERSION = "1.0.0"

#: The four invariance relations of rule 19, in the commission's order.
INVARIANCE: tuple[str, ...] = (
    "identifier_renaming",
    "statement_reordering",
    "whitespace_and_comments",
    "equivalent_rewrite",
)

#: The one killing relation: an injected real defect must flip the score.
KILLING = "injected_defect"

#: Rule 19's relations for a judge, applied when the response is prose rather than source.
JUDGE_RELATIONS: tuple[str, ...] = ("order_swap", "paraphrase", "length_padding")

_PREFIX = "rl_"
_DEFECT = "_rl_defect"


class _Renamer(ast.NodeTransformer):
    """Rename the names bound inside each function, and nothing a caller can see."""

    def __init__(self) -> None:
        self.local: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  # noqa: N802
        for argument in [*node.args.args, *node.args.kwonlyargs, *node.args.posonlyargs]:
            self.local.add(argument.arg)
        for inner in ast.walk(node):
            if isinstance(inner, ast.Assign):
                for target in inner.targets:
                    if isinstance(target, ast.Name):
                        self.local.add(target.id)
            elif isinstance(inner, (ast.AugAssign, ast.For)) and isinstance(inner.target, ast.Name):
                self.local.add(inner.target.id)
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        if node.id in self.local:
            node.id = _PREFIX + node.id
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:  # noqa: N802
        if node.arg in self.local:
            node.arg = _PREFIX + node.arg
        return node


class _AugAssign(ast.NodeTransformer):
    """`x = x + y` becomes `x += y`, which is the same program written another way."""

    def visit_Assign(self, node: ast.Assign) -> ast.AST:  # noqa: N802
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.BinOp)
            and isinstance(node.value.left, ast.Name)
            and node.value.left.id == node.targets[0].id
        ):
            return ast.AugAssign(
                target=node.targets[0], op=node.value.op, value=node.value.right
            )
        return node


class _ReturnTemporary(ast.NodeTransformer):
    """`return f(x)` becomes `_t = f(x); return _t`, which returns the same object."""

    def visit_Return(self, node: ast.Return) -> Any:  # noqa: N802
        if node.value is None:
            return node
        name = _PREFIX + "returned"
        return [
            ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=node.value),
            ast.Return(value=ast.Name(id=name, ctx=ast.Load())),
        ]


class _InjectDefect(ast.NodeTransformer):
    """Every returned value is perturbed, which is a real defect and not a cosmetic one."""

    def visit_Return(self, node: ast.Return) -> ast.AST:  # noqa: N802
        if node.value is None:
            return node
        return ast.Return(
            value=ast.Call(
                func=ast.Name(id=_DEFECT, ctx=ast.Load()), args=[node.value], keywords=[]
            )
        )


_DEFECT_HELPER = (
    f"def {_DEFECT}(value):\n"
    "    if isinstance(value, bool):\n"
    "        return not value\n"
    "    if isinstance(value, (int, float)):\n"
    "        return value + 1\n"
    "    if isinstance(value, str):\n"
    "        return value + 'x'\n"
    "    if isinstance(value, (list, tuple)):\n"
    "        return list(value) + [None]\n"
    "    return None\n"
)

_INDEPENDENT = (f"{_PREFIX}first = 0\n", f"{_PREFIX}second = 1\n")


def _parse(response: str) -> ast.Module | None:
    try:
        return ast.parse(response)
    except SyntaxError:
        return None


def pair(name: str, response: str) -> tuple[str, str] | None:
    """The two responses a relation compares: the baseline and the transformed one.

    For three of the four the baseline is the response itself. Statement reordering is the
    exception: a short response often holds no adjacent pair of independent statements, so the
    relation appends one pair and compares the two orders of it. Comparing an appended version
    against the bare original would confound the insertion with the reordering, and the comparison
    would not be a reordering relation at all.
    """
    if name == "statement_reordering":
        first, second = _INDEPENDENT
        return response + first + second, response + second + first
    changed = transform(name, response)
    if changed is None:
        return None
    return response, changed


def transform(name: str, response: str) -> str | None:
    """The transformed half of a relation's pair, or `None` when the response does not admit it."""
    if name == "whitespace_and_comments":
        lines = response.splitlines()
        return "\n".join(["# a comment the grader must not read as work", "", *lines, ""]) + "\n"
    if name == "statement_reordering":
        first, second = _INDEPENDENT
        return response + second + first
    if name == "order_swap":
        lines = [line for line in response.splitlines() if line.strip()]
        return "\n".join(reversed(lines)) + "\n" if len(lines) > 1 else None
    if name == "paraphrase":
        return response.replace(". ", ", and ") + "\nPut another way, the same answer.\n"
    if name == "length_padding":
        return response + "\n" + ("Restating the point at greater length. " * 20) + "\n"
    tree = _parse(response)
    if tree is None:
        return None
    if name == "identifier_renaming":
        renamed = _Renamer().visit(tree)
    elif name == "equivalent_rewrite":
        renamed = _ReturnTemporary().visit(_AugAssign().visit(tree))
    elif name == KILLING:
        renamed = _InjectDefect().visit(tree)
    else:
        return None
    ast.fix_missing_locations(renamed)
    try:
        text = ast.unparse(renamed)
        compile(text, "<relation>", "exec")
    except (SyntaxError, ValueError):
        return None
    if name == KILLING:
        text = _DEFECT_HELPER + text
    if text.rstrip() == response.rstrip():
        return None
    return text + "\n"


@dataclass(frozen=True)
class RelationOutcome:
    relation: str
    case_id: str
    before: tuple[Any, ...]
    after: tuple[Any, ...]
    held: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation": self.relation,
            "case_id": self.case_id,
            "before": list(self.before),
            "after": list(self.after),
            "held": self.held,
            "note": self.note,
        }


def _verdict(row: dict[str, Any] | None, threshold: float) -> tuple[Any, ...]:
    if row is None:
        return (None, "", True, False)
    score = row.get("score")
    return (
        None if score is None else round(float(score), 6),
        str(row.get("reason", "")),
        bool(row.get("error")),
        score is not None and float(score) >= threshold,
    )


def relations_for(response: str) -> tuple[str, ...]:
    """The code relations when the response parses as Python, the judge relations when it does not."""
    return INVARIANCE if _parse(response) is not None else JUDGE_RELATIONS


def run_relations(
    subject: Any,
    cases: Sequence[Case],
    *,
    runner: Runner | None = None,
    threshold: float = ACCEPT_THRESHOLD,
) -> tuple[RelationOutcome, ...]:
    """Run every relation the responses admit, in one sandboxed pass over all the probes."""
    execute = runner or sandbox_runner
    source = source_of(subject)
    entrypoint = str(getattr(subject, "entrypoint", "score") or "score")
    probes: list[Case] = []
    plan: list[tuple[str, str, str, str]] = []
    for case in cases:
        for name in (*relations_for(case.response), KILLING):
            made = pair(name, case.response)
            if made is None:
                continue
            before_id = f"{case.id}::{name}::before"
            after_id = f"{case.id}::{name}::after"
            probes.append(Case(id=before_id, task=case.task, response=made[0], expect=case.expect))
            probes.append(Case(id=after_id, task=case.task, response=made[1], expect=case.expect))
            plan.append((name, case.id, before_id, after_id))
    if not probes:
        return ()
    rows = {str(row.get("case_id")): row for row in execute(source, entrypoint, probes)}
    produced: list[RelationOutcome] = []
    for name, case_id, before_id, after_id in plan:
        before = _verdict(rows.get(before_id), threshold)
        after = _verdict(rows.get(after_id), threshold)
        if name == KILLING:
            if not before[3]:
                produced.append(
                    RelationOutcome(
                        name,
                        case_id,
                        before,
                        after,
                        held=False,
                        note=(
                            "the killing relation needs a case the grader accepts to start from, "
                            "and this one was already rejected"
                        ),
                    )
                )
                continue
            held = not after[3]
            produced.append(
                RelationOutcome(
                    name,
                    case_id,
                    before,
                    after,
                    held=held,
                    note=(
                        "the injected defect flipped the score, so the grader reads the response"
                        if held
                        else "the injected defect did not flip the score: the grader is not "
                        "reading the response it is scoring"
                    ),
                )
            )
            continue
        held = before[:2] == after[:2] and before[3] == after[3]
        produced.append(
            RelationOutcome(
                name,
                case_id,
                before,
                after,
                held=held,
                note=(
                    "the score is unchanged under a transformation that cannot change correctness"
                    if held
                    else f"the score moved from {before[0]} to {after[0]} under {name}, which "
                    "cannot change whether the response is correct"
                ),
            )
        )
    return tuple(produced)


def _method(name: str) -> contracts.Method:
    return contracts.Method(
        id=f"soundness.relations.{name}",
        version=VERSION,
        params_digest=contracts.digest({"relation": name, "threshold": ACCEPT_THRESHOLD}),
        procedure=(
            f"the response was transformed under the {name} relation and graded again by the same "
            "grader in the same sandbox; for an invariance relation the score must not move, and "
            "for the killing relation it must"
        ),
        credited_to="reward_lens.instruments.soundness (section 7.3, rule 19)",
    )


LIMITATIONS = (
    "the relation is over the responses supplied; it says nothing about responses of another shape",
    "an invariance that holds on these cases is not a proof of invariance",
)


def relation_entries(
    outcomes: Sequence[RelationOutcome], ctx: RunContext, *, subject_ref: str
) -> list[contracts.Entry]:
    """One check entry per relation, carrying every case it ran on."""
    grouped: dict[str, list[RelationOutcome]] = {}
    for outcome in outcomes:
        grouped.setdefault(outcome.relation, []).append(outcome)
    produced: list[contracts.Entry] = []
    for name in sorted(grouped):
        rows = grouped[name]
        broken = [row for row in rows if not row.held]
        killing = name == KILLING
        produced.append(
            contracts.Entry(
                entry_id=f"soundness.relations.{name}",
                section="soundness",
                kind="check",
                measurand=(
                    "whether an injected real defect moves the score"
                    if killing
                    else f"whether the score is invariant under {name}"
                ),
                method=_method(name),
                scope="evaluator_comparison",
                subject_ref=subject_ref,
                depends_on=["digest:source", "digest:samples"],
                state="complete",
                provenance=ctx.provenance(arm="black_box"),
                limitations=list(LIMITATIONS),
                n=len(rows),
                result={
                    "relation": name,
                    "cases": [row.to_dict() for row in rows],
                    "broken": [row.case_id for row in broken],
                    "findings": [
                        {
                            "code": "RL0226" if killing else "RL0225",
                            "kind": "fail",
                            "level": "error" if killing else "warning",
                            "message": f"{row.case_id}: {row.note}",
                        }
                        for row in broken
                    ],
                },
                check=contracts.Check(
                    predicate=(
                        "an injected real defect flips the score on every case the grader accepted"
                        if killing
                        else f"the score is unchanged under {name} on every case"
                    ),
                    passed=not broken,
                    scope_tested=f"{len(rows)} cases, graded in the sandbox before and after",
                ),
            )
        )
    return produced
