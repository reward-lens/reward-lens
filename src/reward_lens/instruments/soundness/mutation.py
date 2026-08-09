"""Mutation testing over the grader source, with the commission's eight bookkeeping states.

A-007 and section 7.3 name the states, and the reason there are eight rather than two is that
"killed" and "survived" hide four different ways a mutant can fail to be evidence. A mutant that
does not compile was never a test of anything. One that compiles but is never reached tells you
about coverage, not about the suite. One that is reached but behaves identically on every case is
the equivalence question, and the equivalence question is undecidable in general, so a mutant that
was not distinguished is recorded as `unresolved_equivalence` and is never subtracted from the
denominator. Only a survivor whose equivalence was established by a procedure the record names is
`confirmed_equivalent`, and only those come out of the second denominator.

That is why rule 18 asks for the score twice. Killed over generated counts every mutant that was
ever proposed and is therefore a lower bound on the suite's real strength. Killed over generated
minus confirmed-equivalent is the number a reader wants, and it is only honest beside the first
one and beside the sampling fraction, because a score over six sampled mutants and a score over
six hundred are not the same claim.

The operators work on the AST and one site at a time, so every mutant differs from the original in
exactly one place and a survivor names one thing the suite does not test. Evaluation goes through
the same `Runner` seam the batteries use, so a mutant is graded by the same sandboxed process shape
as the grader itself.
"""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from reward_lens import contracts
from reward_lens.contracts.errors import InternalFailure
from reward_lens.instruments.base import RunContext

from . import intervals
from .batteries import ACCEPT_THRESHOLD, Case, Runner, sandbox_runner, source_of

__all__ = [
    "OPERATOR_NAMES",
    "STATES",
    "EquivalenceNotFromTheStates",
    "Mutant",
    "MutantRecord",
    "MutationLedger",
    "Reproducer",
    "Score",
    "confirm_equivalent",
    "demonstration_ledger",
    "propose",
    "run",
    "score_entry",
    "survival_entry",
]

VERSION = "1.0.0"

#: The eight states, in the commission's own order and with the commission's own names (A-007).
STATES: tuple[str, ...] = (
    "proposed",
    "buildable",
    "exercised",
    "behaviourally_distinct",
    "eligible",
    "killed",
    "surviving",
    "unresolved_equivalence",
)

#: The five operators. Each changes one node, so a survivor names one untested thing.
OPERATOR_NAMES: tuple[str, ...] = (
    "comparison_swap",
    "boolean_flip",
    "constant_bump",
    "condition_true",
    "statement_deletion",
)

_COMPARISONS: dict[type, type] = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}


class EquivalenceNotFromTheStates(InternalFailure):
    """A denominator was asked for that the eight states as recorded cannot produce.

    The commission's instruction is to stop and report rather than invent a ninth state, so this
    is raised rather than worked around. It is reached by confirming a mutant equivalent that the
    ledger does not hold as surviving, which would put a mutant into the second denominator's
    subtraction that the first denominator never counted as eligible.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(
            code="RL0900",
            message=f"the mutation ledger cannot produce this denominator from the eight states: {detail}",
            remediation=(
                "record the mutant's equivalence against the state it actually holds, or leave it "
                "in unresolved_equivalence, which is never subtracted"
            ),
            context={"detail": detail},
        )


@dataclass(frozen=True)
class Mutant:
    id: str
    operator: str
    line: int
    source: str
    description: str


@dataclass(frozen=True)
class Reproducer:
    """The smallest case on which a survivor is still undistinguished from the original."""

    case_id: str
    original: str
    response: str
    shrink_steps: int
    observed: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "response": self.response,
            "original_length": len(self.original),
            "minimised_length": len(self.response),
            "shrink_steps": self.shrink_steps,
            "observed": self.observed,
        }


@dataclass
class MutantRecord:
    mutant: Mutant
    states: frozenset[str]
    killed_by: str | None = None
    reproducer: Reproducer | None = None
    equivalence_procedure: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.mutant.id,
            "operator": self.mutant.operator,
            "line": self.mutant.line,
            "description": self.mutant.description,
            "states": [state for state in STATES if state in self.states],
            "killed_by": self.killed_by,
            "equivalence_procedure": self.equivalence_procedure,
            "reproducer": None if self.reproducer is None else self.reproducer.to_dict(),
        }


@dataclass(frozen=True)
class Score:
    """One of rule 18's two scores, with the denominator it was taken over."""

    killed: int
    denominator: int
    label: str
    lower_bound: bool
    value: float
    uncertainty: contracts.Uncertainty
    sampling_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "killed": self.killed,
            "denominator": self.denominator,
            "denominator_label": self.label,
            "is_lower_bound": self.lower_bound,
            "value": self.value,
            "interval": None if not self.uncertainty.bounded else list(self.uncertainty.interval),
            "interval_method": self.uncertainty.method,
            "sampling_fraction": self.sampling_fraction,
        }


EQUIVALENCE_STATEMENT = (
    "whether a mutant is equivalent to the original is undecidable in general, so a mutant is "
    "subtracted from the second denominator only when a named procedure established its "
    "equivalence; a mutant in unresolved_equivalence was not distinguished and is never subtracted"
)


@dataclass
class MutationLedger:
    records: tuple[MutantRecord, ...]
    sites_available: int
    operators: tuple[str, ...] = OPERATOR_NAMES
    cases_run: int = 0

    def counts(self) -> dict[str, int]:
        return {
            state: sum(1 for record in self.records if state in record.states) for state in STATES
        }

    @property
    def sampling_fraction(self) -> float:
        if self.sites_available <= 0:
            return 0.0
        return round(len(self.records) / self.sites_available, 6)

    def confirmed_equivalent(self) -> tuple[MutantRecord, ...]:
        found = tuple(
            record
            for record in self.records
            if record.equivalence_procedure is not None
        )
        for record in found:
            if "surviving" not in record.states:
                raise EquivalenceNotFromTheStates(
                    f"{record.mutant.id} carries an equivalence procedure but is not surviving"
                )
        return found

    def confirmed_equivalent_count(self) -> int:
        return len(self.confirmed_equivalent())

    def equivalence_statement(self) -> str:
        return EQUIVALENCE_STATEMENT

    def _score(self, denominator: int, label: str, lower_bound: bool) -> Score:
        killed = self.counts()["killed"]
        if denominator < killed:
            raise EquivalenceNotFromTheStates(
                f"{label} is {denominator}, below the {killed} mutants recorded as killed"
            )
        return Score(
            killed=killed,
            denominator=denominator,
            label=label,
            lower_bound=lower_bound,
            value=0.0 if denominator == 0 else round(killed / denominator, 6),
            uncertainty=intervals.uncertainty_for(killed, denominator),
            sampling_fraction=self.sampling_fraction,
        )

    def score_over_generated(self) -> Score:
        """Rule 18's lower bound: every mutant proposed is in the denominator."""
        return self._score(self.counts()["proposed"], "killed over generated", True)

    def score_over_non_equivalent(self) -> Score:
        """Rule 18's second score: generated minus the mutants a named procedure confirmed equivalent."""
        return self._score(
            self.counts()["proposed"] - self.confirmed_equivalent_count(),
            "killed over generated minus confirmed-equivalent",
            False,
        )

    def survival_by_operator(self) -> dict[str, tuple[MutantRecord, ...]]:
        grouped: dict[str, list[MutantRecord]] = {}
        for record in self.records:
            if "surviving" in record.states:
                grouped.setdefault(record.mutant.operator, []).append(record)
        return {name: tuple(rows) for name, rows in sorted(grouped.items())}


def confirm_equivalent(ledger: MutationLedger, mutant_id: str, *, procedure: str) -> MutantRecord:
    """Record that a named procedure established this survivor's equivalence to the original."""
    for record in ledger.records:
        if record.mutant.id != mutant_id:
            continue
        if "surviving" not in record.states:
            raise EquivalenceNotFromTheStates(
                f"{mutant_id} is not surviving, and only a surviving mutant can be confirmed "
                "equivalent; its states are "
                + ", ".join(state for state in STATES if state in record.states)
            )
        record.equivalence_procedure = procedure
        return record
    raise EquivalenceNotFromTheStates(f"{mutant_id} is not in this ledger")


# --- proposing mutants ------------------------------------------------------------------------------


class _Site:
    __slots__ = ("operator", "index", "line", "description")

    def __init__(self, operator: str, index: int, line: int, description: str) -> None:
        self.operator = operator
        self.index = index
        self.line = line
        self.description = description


def _sites(tree: ast.AST, operators: Sequence[str]) -> list[_Site]:
    found: list[_Site] = []
    counters = {name: 0 for name in OPERATOR_NAMES}
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if "comparison_swap" in operators and isinstance(node, ast.Compare):
            for position, op in enumerate(node.ops):
                if type(op) in _COMPARISONS:
                    found.append(
                        _Site(
                            "comparison_swap",
                            counters["comparison_swap"],
                            line,
                            f"{type(op).__name__} became {_COMPARISONS[type(op)].__name__}",
                        )
                    )
                    counters["comparison_swap"] += 1
                    del position
        if "boolean_flip" in operators and isinstance(node, ast.BoolOp):
            found.append(
                _Site(
                    "boolean_flip",
                    counters["boolean_flip"],
                    line,
                    f"{type(node.op).__name__} became its opposite",
                )
            )
            counters["boolean_flip"] += 1
        if (
            "constant_bump" in operators
            and isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            found.append(
                _Site(
                    "constant_bump",
                    counters["constant_bump"],
                    line,
                    f"the constant {node.value!r} became {node.value + 1!r}",
                )
            )
            counters["constant_bump"] += 1
        if "condition_true" in operators and isinstance(node, ast.If):
            found.append(
                _Site("condition_true", counters["condition_true"], line, "the test became True")
            )
            counters["condition_true"] += 1
        if "statement_deletion" in operators and isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module, ast.For, ast.While)
        ):
            body = list(getattr(node, "body", []))
            for position in range(len(body)):
                if len(body) > 1 and not isinstance(body[position], ast.Return):
                    found.append(
                        _Site(
                            "statement_deletion",
                            counters["statement_deletion"],
                            getattr(body[position], "lineno", line),
                            "one statement was removed",
                        )
                    )
                    counters["statement_deletion"] += 1
    return found


class _Mutator(ast.NodeTransformer):
    def __init__(self, operator: str, index: int) -> None:
        self.operator = operator
        self.index = index
        self.seen = 0
        self.applied = False

    def _take(self) -> bool:
        hit = self.seen == self.index
        self.seen += 1
        if hit:
            self.applied = True
        return hit

    def visit_Compare(self, node: ast.Compare) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        if self.operator == "comparison_swap":
            for position, op in enumerate(node.ops):
                if type(op) in _COMPARISONS and self._take():
                    node.ops[position] = _COMPARISONS[type(op)]()
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        if self.operator == "boolean_flip" and self._take():
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
        if (
            self.operator == "constant_bump"
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
            and self._take()
        ):
            return ast.Constant(value=node.value + 1)
        return node

    def visit_If(self, node: ast.If) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        if self.operator == "condition_true" and self._take():
            node.test = ast.Constant(value=True)
        return node

    def _drop(self, node: ast.AST) -> ast.AST:
        body = list(getattr(node, "body", []))
        if self.operator != "statement_deletion" or len(body) <= 1:
            return node
        kept: list[ast.stmt] = []
        for statement in body:
            if not isinstance(statement, ast.Return) and len(body) > 1 and self._take():
                continue
            kept.append(statement)
        if kept:
            node.body = kept  # type: ignore[attr-defined]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        return self._drop(node)

    def visit_For(self, node: ast.For) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        return self._drop(node)

    def visit_While(self, node: ast.While) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        return self._drop(node)

    def visit_Module(self, node: ast.Module) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        return self._drop(node)


def propose(
    source: str,
    *,
    operators: Sequence[str] = OPERATOR_NAMES,
    limit: int | None = None,
    seed: int = 0,
) -> tuple[tuple[Mutant, ...], int]:
    """Every one-site mutant this source admits, and how many sites there were to sample from.

    `sites_available` is the count of distinct buildable mutants, which is the denominator the
    sampling fraction is taken over. A site whose mutation unparses back to the original is not a
    mutant and is not counted: it would inflate the fraction with changes that changed nothing.
    """
    tree = ast.parse(source)
    baseline = ast.unparse(tree)
    built: list[Mutant] = []
    for site in _sites(tree, operators):
        mutated = ast.parse(source)
        mutator = _Mutator(site.operator, site.index)
        mutated = mutator.visit(mutated)
        if not mutator.applied:
            continue
        ast.fix_missing_locations(mutated)
        try:
            text = ast.unparse(mutated)
            compile(text, "<mutant>", "exec")
        except (SyntaxError, ValueError):
            continue
        if text == baseline:
            continue
        built.append(
            Mutant(
                id=f"m{len(built):03d}",
                operator=site.operator,
                line=site.line,
                source=text,
                description=site.description,
            )
        )
    sites_available = len(built)
    if limit is not None and limit < sites_available:
        rng = random.Random(seed)
        built = sorted(rng.sample(built, limit), key=lambda one: one.id)
    return tuple(built), sites_available


# --- evaluating mutants -----------------------------------------------------------------------------


def _verdicts(rows: Sequence[dict[str, Any]], threshold: float) -> dict[str, tuple[Any, ...]]:
    out: dict[str, tuple[Any, ...]] = {}
    for row in rows:
        score = row.get("score")
        accepted = score is not None and float(score) >= threshold
        out[str(row.get("case_id"))] = (
            None if score is None else round(float(score), 6),
            str(row.get("reason", "")),
            bool(row.get("error")),
            accepted,
        )
    return out


def _as_expected(case: Case, verdict: tuple[Any, ...]) -> bool:
    accepted = bool(verdict[3])
    return accepted if case.expect == "accept" else not accepted


def _minimise(
    mutant: Mutant,
    entrypoint: str,
    case: Case,
    baseline: tuple[Any, ...],
    execute: Runner,
    threshold: float,
) -> Reproducer:
    """Shrink the response to the shortest prefix on which the survivor is still undistinguished."""
    lines = case.response.splitlines(keepends=True)
    candidates: list[tuple[int, str]] = []
    for drop in range(1, min(len(lines), 8)):
        candidates.append((drop, "".join(lines[: len(lines) - drop])))
    best = (0, case.response)
    if candidates:
        probes = [
            Case(id=f"{case.id}#s{drop}", task=case.task, response=text, expect=case.expect)
            for drop, text in candidates
        ]
        mutant_rows = _verdicts(execute(mutant.source, entrypoint, probes), threshold)
        for drop, text in candidates:
            key = f"{case.id}#s{drop}"
            if key in mutant_rows and mutant_rows[key] == baseline:
                best = (drop, text)
    return Reproducer(
        case_id=case.id,
        original=case.response,
        response=best[1],
        shrink_steps=best[0],
        observed=(
            f"the mutant and the original agree on this input, so the suite does not tell them "
            f"apart: {mutant.description} at line {mutant.line}"
        ),
    )


def run(
    subject: Any,
    cases: Sequence[Case],
    *,
    limit: int | None = None,
    seed: int = 0,
    operators: Sequence[str] = OPERATOR_NAMES,
    runner: Runner | None = None,
    threshold: float = ACCEPT_THRESHOLD,
) -> MutationLedger:
    """Propose, build, exercise, compare and judge every mutant, filling the eight states."""
    execute = runner or sandbox_runner
    source = source_of(subject)
    entrypoint = str(getattr(subject, "entrypoint", "score") or "score")
    mutants, sites_available = propose(source, operators=operators, limit=limit, seed=seed)
    original = _verdicts(execute(source, entrypoint, list(cases)), threshold)
    by_case = {case.id: case for case in cases}
    records: list[MutantRecord] = []
    for mutant in mutants:
        states = {"proposed"}
        try:
            compile(mutant.source, "<mutant>", "exec")
            states.add("buildable")
        except (SyntaxError, ValueError):
            records.append(MutantRecord(mutant=mutant, states=frozenset(states)))
            continue
        verdicts = _verdicts(execute(mutant.source, entrypoint, list(cases)), threshold)
        if any(verdict[0] is not None for verdict in verdicts.values()):
            states.add("exercised")
        distinct = [
            case_id
            for case_id, verdict in verdicts.items()
            if case_id in original and verdict != original[case_id]
        ]
        if distinct:
            states.add("behaviourally_distinct")
        if {"buildable", "exercised", "behaviourally_distinct"} <= states:
            states.add("eligible")
            killed_by = next(
                (
                    case_id
                    for case_id in sorted(distinct)
                    if case_id in by_case
                    and _as_expected(by_case[case_id], original[case_id])
                    and not _as_expected(by_case[case_id], verdicts[case_id])
                ),
                None,
            )
            if killed_by is not None:
                states.add("killed")
                records.append(
                    MutantRecord(mutant=mutant, states=frozenset(states), killed_by=killed_by)
                )
                continue
            states.add("surviving")
            undistinguished = next(
                (
                    case_id
                    for case_id in sorted(original)
                    if case_id in verdicts and verdicts[case_id] == original[case_id]
                ),
                None,
            )
            reproducer = (
                None
                if undistinguished is None
                else _minimise(
                    mutant,
                    entrypoint,
                    by_case[undistinguished],
                    original[undistinguished],
                    execute,
                    threshold,
                )
            )
            records.append(
                MutantRecord(mutant=mutant, states=frozenset(states), reproducer=reproducer)
            )
            continue
        if {"buildable", "exercised"} <= states:
            states.add("unresolved_equivalence")
        records.append(MutantRecord(mutant=mutant, states=frozenset(states)))
    return MutationLedger(
        records=tuple(records),
        sites_available=sites_available,
        operators=tuple(operators),
        cases_run=len(cases),
    )


def demonstration_ledger() -> MutationLedger:
    """A worked ledger in which each of the eight states holds at least one mutant.

    It is a constructed example and not a measurement: nothing here was run, and no record ever
    carries it. It exists because the eight states are the part of this module most easily got
    wrong, and a ledger that reaches every one of them is what makes the bookkeeping checkable
    without waiting for a grader whose suite happens to leave a mutant in each state.
    """

    def mutant(index: int, operator: str) -> Mutant:
        return Mutant(
            id=f"d{index:03d}",
            operator=operator,
            line=index,
            source=f"# worked example {index}\n",
            description=f"the worked example's {operator} at line {index}",
        )

    def reproducer(index: int) -> Reproducer:
        return Reproducer(
            case_id=f"case-{index}",
            original="line one\nline two\nline three\n",
            response="line one\n",
            shrink_steps=2,
            observed="the mutant and the original agree on this input",
        )

    rows: list[MutantRecord] = [
        MutantRecord(mutant(1, "comparison_swap"), frozenset({"proposed"})),
        MutantRecord(mutant(2, "boolean_flip"), frozenset({"proposed", "buildable"})),
        MutantRecord(
            mutant(3, "constant_bump"),
            frozenset({"proposed", "buildable", "exercised", "unresolved_equivalence"}),
        ),
        MutantRecord(
            mutant(4, "condition_true"),
            frozenset(
                {
                    "proposed",
                    "buildable",
                    "exercised",
                    "behaviourally_distinct",
                    "eligible",
                    "killed",
                }
            ),
            killed_by="case-4",
        ),
        MutantRecord(
            mutant(5, "statement_deletion"),
            frozenset(
                {
                    "proposed",
                    "buildable",
                    "exercised",
                    "behaviourally_distinct",
                    "eligible",
                    "killed",
                }
            ),
            killed_by="case-5",
        ),
        MutantRecord(
            mutant(6, "comparison_swap"),
            frozenset(
                {
                    "proposed",
                    "buildable",
                    "exercised",
                    "behaviourally_distinct",
                    "eligible",
                    "surviving",
                }
            ),
            reproducer=reproducer(6),
            equivalence_procedure=(
                "a hand proof that the swapped comparison is unreachable, checked against the "
                "grader's own guard clause"
            ),
        ),
        MutantRecord(
            mutant(7, "boolean_flip"),
            frozenset(
                {
                    "proposed",
                    "buildable",
                    "exercised",
                    "behaviourally_distinct",
                    "eligible",
                    "surviving",
                }
            ),
            reproducer=reproducer(7),
        ),
        MutantRecord(
            mutant(8, "constant_bump"),
            frozenset({"proposed", "buildable", "exercised", "unresolved_equivalence"}),
        ),
        MutantRecord(
            mutant(9, "statement_deletion"),
            frozenset(
                {
                    "proposed",
                    "buildable",
                    "exercised",
                    "behaviourally_distinct",
                    "eligible",
                    "killed",
                }
            ),
            killed_by="case-9",
        ),
    ]
    return MutationLedger(records=tuple(rows), sites_available=14, cases_run=9)


# --- the entries --------------------------------------------------------------------------------------

LIMITATIONS = (
    "the score is over the mutants that were sampled, and the sampling fraction is on the entry",
    EQUIVALENCE_STATEMENT,
    "a surviving mutant names something the suite does not test; it is not proof the grader is "
    "wrong",
)


def _method(name: str, params: dict[str, Any]) -> contracts.Method:
    return contracts.Method(
        id=f"soundness.mutation.{name}",
        version=VERSION,
        params_digest=contracts.digest(params),
        procedure=(
            "one-site AST mutants of the grader source were built, run against the same cases as "
            "the grader itself in a sandboxed process, and sorted into the commission's eight "
            "states; the score is reported over generated and over generated minus the mutants a "
            "named procedure confirmed equivalent"
        ),
        credited_to="reward_lens.instruments.soundness (section 7.3, rule 18)",
    )


def score_entry(
    ledger: MutationLedger, ctx: RunContext, *, subject_ref: str, duration_s: float = 0.0
) -> contracts.Entry:
    """The mutation score, both ways, with the sampling fraction and the eight states (rule 18)."""
    over_generated = ledger.score_over_generated()
    over_non_equivalent = ledger.score_over_non_equivalent()
    counts = ledger.counts()
    survivors = ledger.survival_by_operator()
    return contracts.estimate(
        "soundness",
        "soundness.mutation.score",
        "the fraction of mutants of the grader source that the protected suite kills",
        over_generated.value,
        "fraction",
        counts["proposed"],
        "mutant",
        over_generated.uncertainty,
        subject_ref=subject_ref,
        provenance=ctx.provenance(duration_s=duration_s, arm="white_box"),
        method=_method(
            "score",
            {
                "operators": list(ledger.operators),
                "sites_available": ledger.sites_available,
                "sampled": len(ledger.records),
            },
        ),
        depends_on=("digest:source", "digest:samples"),
        limitations=LIMITATIONS,
        denominator="mutants generated",
        result={
            "is_rate": True,
            "mutation_score": True,
            "score_over_generated": over_generated.to_dict(),
            "score_over_non_equivalent": over_non_equivalent.to_dict(),
            "sampling_fraction": ledger.sampling_fraction,
            "sites_available": ledger.sites_available,
            "sampled": len(ledger.records),
            "states": counts,
            "equivalence_statement": ledger.equivalence_statement(),
            "confirmed_equivalent": [
                record.mutant.id for record in ledger.confirmed_equivalent()
            ],
            "mutants": [record.to_dict() for record in ledger.records],
            "findings": [
                {
                    "code": "RL0224",
                    "kind": "fail",
                    "level": "warning",
                    "message": (
                        f"{record.mutant.id} ({record.mutant.operator} at line "
                        f"{record.mutant.line}) survived the suite: "
                        f"{record.mutant.description}"
                    ),
                }
                for rows in survivors.values()
                for record in rows
                if record.equivalence_procedure is None
            ],
        },
    )


def survival_entry(
    ledger: MutationLedger, ctx: RunContext, *, subject_ref: str, duration_s: float = 0.0
) -> contracts.Entry:
    """Survival grouped by operator, each survivor with its minimised reproducer."""
    survivors = ledger.survival_by_operator()
    return contracts.Entry(
        entry_id="soundness.mutation.survival",
        section="soundness",
        kind="check",
        measurand="which mutation operators produce mutants the protected suite does not kill",
        method=_method("survival", {"operators": list(ledger.operators)}),
        scope="evaluator_comparison",
        subject_ref=subject_ref,
        depends_on=["digest:source"],
        state="complete",
        provenance=ctx.provenance(duration_s=duration_s, arm="white_box"),
        limitations=list(LIMITATIONS),
        n=sum(len(rows) for rows in survivors.values()),
        result={
            "by_operator": {
                operator: [record.to_dict() for record in rows]
                for operator, rows in survivors.items()
            },
            "operators_with_survivors": sorted(survivors),
            "minimised_reproducers": sum(
                1 for rows in survivors.values() for record in rows if record.reproducer is not None
            ),
        },
        check=contracts.Check(
            predicate="no mutation operator produces a mutant the suite leaves alive",
            passed=not survivors,
            scope_tested=(
                f"{len(ledger.records)} mutants sampled from {ledger.sites_available} sites, over "
                f"{len(ledger.operators)} operators"
            ),
        ),
    )


_ = (field, replace)
