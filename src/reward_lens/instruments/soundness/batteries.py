"""The batteries: known-good accepted, known-wrong rejected for the expected reason.

Four things live here, and they are four because each one catches a different failure. The
known-good battery catches a grader that rejects correct work. The known-wrong battery catches a
grader that accepts wrong work, and it checks the *reason* as well as the verdict, because a
rejection for the wrong reason is a grader that happened to be right and will not be right again.
The gold-sanity gate catches the case where the whole battery is meaningless because the reference
implementations do not pass, and it says that is an evaluator defect: what it must never say is
that the task is impossible, which is the reading the commission forbids. The differential catches
what the subject accepts and a stricter reference verifier rejects, where the owner supplies one.

Every rate goes out through `intervals`, so no number leaves this module without its interval and
without the sentence saying what its n could not have established.

The grader runs under P-EXEC (`reward_lens.execution.run_python`), one staged process per battery
and not one per case: the program takes the whole case list, loops it, and prints one JSON array
behind a nonce. A `runner` can be injected, which is what lets mutation and the metamorphic
relations reuse this seam and what lets a test drive the bookkeeping deterministically.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from reward_lens import contracts
from reward_lens.errors import make
from reward_lens.execution import DEFAULT_LIMITS, Limits, run_python
from reward_lens.instruments.base import RunContext

from . import intervals

__all__ = [
    "ACCEPT_THRESHOLD",
    "BatteryReport",
    "Case",
    "CaseOutcome",
    "DifferentialReport",
    "Disagreement",
    "GoldSanity",
    "battery_entry",
    "cases_from_partition",
    "differential",
    "differential_entry",
    "gold_sanity",
    "gold_sanity_entry",
    "outcome_suite_empty",
    "run_battery",
    "sandbox_runner",
    "source_of",
]

VERSION = "1.0.0"

#: A score at or above this is an acceptance. It is a parameter of every battery and is written
#: into the entry, because "accepted" with no threshold beside it is not a reproducible verdict.
ACCEPT_THRESHOLD = 1.0


@dataclass(frozen=True)
class Case:
    """One graded case: the task, the response, and what a sound grader must say about it."""

    id: str
    task: dict[str, Any]
    response: str
    expect: str = "accept"
    expected_reason: str = ""
    kind: str = "known_good"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "response": self.response,
            "expect": self.expect,
            "expected_reason": self.expected_reason,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Case":
        return cls(
            id=str(row["id"]),
            task=dict(row.get("task") or {}),
            response=str(row.get("response", "")),
            expect=str(row.get("expect", "accept")),
            expected_reason=str(row.get("expected_reason", "")),
            kind=str(row.get("kind", "known_good")),
        )


@dataclass(frozen=True)
class CaseOutcome:
    """What the grader said about one case, and whether that is what a sound grader says."""

    case_id: str
    score: float | None
    reason: str
    error: str
    accepted: bool
    as_expected: bool
    reason_matched: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "score": self.score,
            "reason": self.reason,
            "error": self.error,
            "accepted": self.accepted,
            "as_expected": self.as_expected,
            "reason_matched": self.reason_matched,
        }


@dataclass(frozen=True)
class BatteryReport:
    kind: str
    threshold: float
    outcomes: tuple[CaseOutcome, ...] = ()

    @property
    def n(self) -> int:
        return len(self.outcomes)

    @property
    def k(self) -> int:
        """A case counts only when the verdict and the reason are both what was expected."""
        return sum(1 for one in self.outcomes if one.as_expected and one.reason_matched)

    @property
    def rate(self) -> float:
        return 0.0 if self.n == 0 else round(self.k / self.n, 6)

    @property
    def failures(self) -> tuple[CaseOutcome, ...]:
        return tuple(one for one in self.outcomes if not (one.as_expected and one.reason_matched))


@dataclass(frozen=True)
class GoldSanity:
    """The gate: if the known-good solutions do not pass, nothing downstream of them means anything."""

    passed: bool
    failures: tuple[str, ...]
    statement: str
    n: int


@dataclass(frozen=True)
class Disagreement:
    case_id: str
    subject_score: float | None
    reference_score: float | None
    direction: str


@dataclass(frozen=True)
class DifferentialReport:
    n: int
    over_accepted: int
    not_stricter: int
    disagreements: tuple[Disagreement, ...] = field(default=())


Runner = Callable[[str, str, Sequence[Case]], list[dict[str, Any]]]


# --- the execution seam ---------------------------------------------------------------------------

_PROGRAM = '''"""Run one grader source over a list of cases and report each verdict."""
import json
import sys
import types

NONCE = __NONCE__
SOURCE = __SOURCE__
ENTRYPOINT = __ENTRYPOINT__
CASES = json.loads(__CASES__)

rows = []


def _failed(case_id, detail):
    return {"case_id": case_id, "score": None, "reason": "", "error": detail}


def _brief(failure):
    return type(failure).__name__ + ": " + str(failure)[:200]


target = None
try:
    module = types.ModuleType("rl_soundness_subject")
    module.__dict__["__name__"] = "rl_soundness_subject"
    sys.modules["rl_soundness_subject"] = module
    exec(compile(SOURCE, "rl_soundness_subject.py", "exec"), module.__dict__)
    target = module.__dict__[ENTRYPOINT]
except Exception as failure:
    for case in CASES:
        rows.append(_failed(case["id"], "the grader source did not load: " + _brief(failure)))

if target is not None:
    for case in CASES:
        try:
            got = target(case["task"], case["response"])
        except Exception as failure:
            rows.append(_failed(case["id"], _brief(failure)))
            continue
        if isinstance(got, dict):
            try:
                value = float(got.get("score", 0.0))
            except (TypeError, ValueError) as failure:
                rows.append(_failed(case["id"], "the score field is not a number: " + _brief(failure)))
                continue
            rows.append({
                "case_id": case["id"],
                "score": value,
                "reason": str(got.get("reason", "")),
                "error": "",
            })
        elif isinstance(got, bool):
            rows.append({"case_id": case["id"], "score": float(got), "reason": "", "error": ""})
        elif isinstance(got, (int, float)):
            rows.append({"case_id": case["id"], "score": float(got), "reason": "", "error": ""})
        else:
            rows.append(_failed(case["id"], "the grader returned a " + type(got).__name__))

sys.stdout.write(json.dumps({"nonce": NONCE, "rows": rows}))
'''


def _decode(stream: bytes, nonce: str) -> list[dict[str, Any]] | None:
    """The last JSON object on the stream that carries this run's nonce, and nothing else."""
    for line in reversed(stream.decode("utf-8", errors="replace").splitlines()):
        start = line.find('{"nonce"')
        if start < 0:
            continue
        try:
            payload = json.loads(line[start:])
        except ValueError:
            continue
        if payload.get("nonce") == nonce:
            rows = payload.get("rows")
            return list(rows) if isinstance(rows, list) else None
    return None


def sandbox_runner(
    source: str,
    entrypoint: str,
    cases: Sequence[Case],
    *,
    limits: Limits = DEFAULT_LIMITS,
    cwd: Path | None = None,
) -> list[dict[str, Any]]:
    """Run one grader source over every case in one sandboxed process (D-37)."""
    if not cases:
        return []
    nonce = "rlsound" + os.urandom(8).hex()
    program = (
        _PROGRAM.replace("__NONCE__", repr(nonce))
        .replace("__SOURCE__", repr(source))
        .replace("__ENTRYPOINT__", repr(entrypoint))
        .replace("__CASES__", repr(json.dumps([case.to_dict() for case in cases])))
    )
    root = Path(cwd) if cwd is not None else Path(os.environ.get("TMPDIR", "/tmp"))
    result = run_python(program, cwd=root, limits=limits)
    rows = _decode(result.stdout, nonce) or _decode(result.stderr, nonce)
    if rows is None:
        detail = result.stderr.decode("utf-8", errors="replace")[-400:].strip()
        return [
            {
                "case_id": case.id,
                "score": None,
                "reason": "",
                "error": f"the graded process returned nothing readable (exit {result.exit_code})"
                + (f": {detail}" if detail else ""),
            }
            for case in cases
        ]
    return rows


def source_of(subject: Any) -> str:
    """The grader source as an instrument may read it: the field first, the file behind it second."""
    declared = getattr(subject, "source", "") or ""
    if declared:
        return str(declared)
    path = getattr(subject, "grader_path", None)
    if path is None:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""


# --- the batteries --------------------------------------------------------------------------------


def _judge(case: Case, row: dict[str, Any], threshold: float) -> CaseOutcome:
    score = row.get("score")
    value = None if score is None else float(score)
    reason = str(row.get("reason", ""))
    error = str(row.get("error", ""))
    accepted = value is not None and value >= threshold
    as_expected = accepted if case.expect == "accept" else not accepted
    wanted = case.expected_reason.strip().lower()
    if not wanted:
        matched = True
    else:
        matched = wanted in reason.lower() or wanted in error.lower()
    return CaseOutcome(
        case_id=case.id,
        score=value,
        reason=reason,
        error=error,
        accepted=accepted,
        as_expected=as_expected,
        reason_matched=matched,
    )


def run_battery(
    subject: Any,
    cases: Sequence[Case],
    *,
    kind: str,
    runner: Runner | None = None,
    threshold: float = ACCEPT_THRESHOLD,
) -> BatteryReport:
    """Run one battery and judge every case on the verdict and on the reason."""
    execute = runner or sandbox_runner
    source = source_of(subject)
    entrypoint = str(getattr(subject, "entrypoint", "score") or "score")
    rows = {str(row.get("case_id")): row for row in execute(source, entrypoint, list(cases))}
    outcomes = tuple(
        _judge(case, rows.get(case.id, {"score": None, "error": "no verdict was returned"}), threshold)
        for case in cases
    )
    return BatteryReport(kind=kind, threshold=threshold, outcomes=outcomes)


def gold_sanity(report: BatteryReport) -> GoldSanity:
    """The gate over the known-good battery.

    A rejected gold solution is an evaluator defect and the statement says so in those words. What
    it never says is that the task is impossible: a reference implementation that does not pass is
    evidence about the grader, and reading it as evidence about the task is the inversion the
    commission forbids.
    """
    failures = tuple(one.case_id for one in report.outcomes if not one.accepted)
    if not report.outcomes:
        return GoldSanity(
            passed=False,
            failures=(),
            statement=(
                "the gold-sanity gate had no known-good solution to run, so it establishes nothing "
                "about the grader"
            ),
            n=0,
        )
    if not failures:
        return GoldSanity(
            passed=True,
            failures=(),
            statement=(
                f"every one of the {len(report.outcomes)} known-good solutions was accepted, so "
                "the batteries downstream of this gate are measuring a grader that can say yes"
            ),
            n=len(report.outcomes),
        )
    return GoldSanity(
        passed=False,
        failures=failures,
        statement=(
            f"{len(failures)} of {len(report.outcomes)} known-good solutions were rejected, which "
            "is an evaluator defect: these solutions are correct by construction, so the rejection "
            "is a fact about the grader and not about the task"
        ),
        n=len(report.outcomes),
    )


def differential(
    subject: Any,
    cases: Sequence[Case],
    *,
    reference: Any,
    runner: Runner | None = None,
    threshold: float = ACCEPT_THRESHOLD,
) -> DifferentialReport:
    """What the subject accepts and a stricter reference verifier rejects, case by case."""
    execute = runner or sandbox_runner
    entrypoint = str(getattr(subject, "entrypoint", "score") or "score")
    mine = {str(row.get("case_id")): row for row in execute(source_of(subject), entrypoint, list(cases))}
    theirs = {
        str(row.get("case_id")): row
        for row in execute(
            source_of(reference),
            str(getattr(reference, "entrypoint", entrypoint) or entrypoint),
            list(cases),
        )
    }
    disagreements: list[Disagreement] = []
    over, loose = 0, 0
    for case in cases:
        mine_score = mine.get(case.id, {}).get("score")
        their_score = theirs.get(case.id, {}).get("score")
        mine_ok = mine_score is not None and float(mine_score) >= threshold
        their_ok = their_score is not None and float(their_score) >= threshold
        if mine_ok and not their_ok:
            over += 1
            disagreements.append(
                Disagreement(case.id, mine_score, their_score, "subject_accepts_reference_rejects")
            )
        elif their_ok and not mine_ok:
            loose += 1
            disagreements.append(
                Disagreement(case.id, mine_score, their_score, "reference_accepts_subject_rejects")
            )
    return DifferentialReport(
        n=len(cases),
        over_accepted=over,
        not_stricter=loose,
        disagreements=tuple(disagreements),
    )


# --- the protected suite ----------------------------------------------------------------------------


def outcome_suite_empty(partition: Any) -> Exception:
    """RL0341, wave 1's code, for a protected suite that holds nothing (interfaces section 2).

    An empty suite is not a suite with no findings. Scoring it would report a rate over an empty
    denominator and pass, which is exactly the failure the code exists to stop.
    """
    root = getattr(partition, "root", None)
    return make(
        "RL0341",
        path=str(root) if root is not None else "",
        partition_id=str(getattr(partition, "id", "")),
    )


def cases_from_partition(partition: Any) -> tuple[Case, ...]:
    """Read the protected suite's live items as battery cases; an empty suite refuses with RL0341."""
    if partition is None or getattr(partition, "is_empty", True):
        raise outcome_suite_empty(partition)
    root = Path(getattr(partition, "root"))
    found: list[Case] = []
    for name, _digest in partition.manifest():
        if not name.endswith(".json"):
            continue
        payload = json.loads((root / name).read_text(encoding="utf-8"))
        rows: Iterable[dict[str, Any]] = payload if isinstance(payload, list) else [payload]
        found.extend(Case.from_dict(row) for row in rows)
    if not found:
        raise outcome_suite_empty(partition)
    return tuple(found)


# --- the entries ------------------------------------------------------------------------------------


def _method(name: str, params: dict[str, Any]) -> contracts.Method:
    return contracts.Method(
        id=f"soundness.{name}",
        version=VERSION,
        params_digest=contracts.digest(params),
        procedure=_PROCEDURES[name],
        credited_to="reward_lens.instruments.soundness (section 7.3, rules 1 to 4)",
    )


_PROCEDURES = {
    "batteries": (
        "each case was graded by running the grader source in a sandboxed process on the case's "
        "own task and response; a case counts only when the verdict matched what a sound grader "
        "must say and, for a rejection, when the stated reason carried the expected one"
    ),
    "gold_sanity": (
        "the known-good solutions were graded first, and the gate asks only whether every one of "
        "them was accepted; a rejection here is read as an evaluator defect and never as evidence "
        "that the task cannot be solved"
    ),
    "differential": (
        "the same cases were graded by the subject and by the stricter reference verifier the "
        "owner supplied, and every case the subject accepted and the reference rejected was kept "
        "with both scores"
    ),
}

LIMITATIONS = (
    "the battery measures the grader on the cases it was given; a rate is over those cases and "
    "not over the task distribution",
    "acceptance is a score at or above the stated threshold and nothing else",
)


def battery_entry(
    report: BatteryReport,
    subject: Any,
    ctx: RunContext,
    *,
    subject_ref: str,
    duration_s: float = 0.0,
    load_bearing: bool = False,
) -> contracts.Entry:
    """One battery as an estimate: the rate, the interval, and what this n could not establish."""
    said = intervals.what_n_can_establish(report.k, report.n)
    return contracts.estimate(
        "soundness",
        f"soundness.batteries.{report.kind}",
        _MEASURANDS[report.kind],
        report.rate,
        "fraction",
        report.n,
        "graded case",
        intervals.uncertainty_for(report.k, report.n, load_bearing=load_bearing),
        subject_ref=subject_ref,
        provenance=ctx.provenance(duration_s=duration_s, arm="black_box"),
        method=_method("batteries", {"kind": report.kind, "threshold": report.threshold}),
        depends_on=("digest:source", "digest:samples"),
        limitations=LIMITATIONS,
        denominator=f"{report.kind} cases run",
        rate_definition="false_reject" if report.kind == "known_good" else "false_accept",
        result={
            "is_rate": True,
            "kind": report.kind,
            "k": report.k,
            "n": report.n,
            "threshold": report.threshold,
            "what_n_can_establish": said,
            "outcomes": [one.to_dict() for one in report.outcomes],
            "failures": [one.case_id for one in report.failures],
            "findings": [
                {
                    "code": "RL0221" if report.kind == "known_wrong" else "RL0222",
                    "kind": "fail",
                    "level": "error" if report.kind == "known_wrong" else "warning",
                    "message": (
                        f"{one.case_id}: expected the grader to {one.expected()} and it "
                        f"{one.observed()}"
                    ),
                }
                for one in _failure_views(report)
            ],
        },
    )


_MEASURANDS = {
    "known_good": "the fraction of known-good solutions the grader accepts",
    "known_wrong": (
        "the fraction of known-wrong solutions the grader rejects for the expected reason"
    ),
}


@dataclass(frozen=True)
class _FailureView:
    case_id: str
    kind: str
    outcome: CaseOutcome

    def expected(self) -> str:
        return "accept it" if self.kind == "known_good" else "reject it for the expected reason"

    def observed(self) -> str:
        if self.outcome.error:
            return f"failed with {self.outcome.error}"
        verdict = "accepted it" if self.outcome.accepted else "rejected it"
        if not self.outcome.accepted and not self.outcome.reason_matched:
            return f"{verdict} for another reason: {self.outcome.reason or '(no reason given)'}"
        return verdict


def _failure_views(report: BatteryReport) -> tuple[_FailureView, ...]:
    return tuple(_FailureView(one.case_id, report.kind, one) for one in report.failures)


def gold_sanity_entry(
    gate: GoldSanity, ctx: RunContext, *, subject_ref: str, duration_s: float = 0.0
) -> contracts.Entry:
    """The gate as a check: it passed or it did not, and the statement says what that means."""
    return contracts.Entry(
        entry_id="soundness.batteries.gold_sanity",
        section="soundness",
        kind="check",
        measurand="whether every known-good solution is accepted by the grader under audit",
        method=_method("gold_sanity", {"n": gate.n}),
        scope="evaluator_comparison",
        subject_ref=subject_ref,
        depends_on=["digest:source", "digest:samples"],
        state="complete",
        provenance=ctx.provenance(duration_s=duration_s, arm="black_box"),
        limitations=list(LIMITATIONS),
        n=gate.n,
        result={
            "passed": gate.passed,
            "failures": list(gate.failures),
            "statement": gate.statement,
            "findings": (
                []
                if gate.passed
                else [
                    {
                        "code": "RL0222",
                        "kind": "fail",
                        "level": "error",
                        "message": gate.statement,
                    }
                ]
            ),
        },
        check=contracts.Check(
            predicate="every known-good solution supplied to this panel is accepted by the grader",
            passed=gate.passed,
            scope_tested=f"{gate.n} known-good solutions, graded in the sandbox",
        ),
    )


def differential_entry(
    report: DifferentialReport,
    subject: Any,
    ctx: RunContext,
    *,
    subject_ref: str,
    duration_s: float = 0.0,
) -> contracts.Entry:
    """The differential as a rate: how often the subject accepts what the stricter reference rejects."""
    return contracts.estimate(
        "soundness",
        "soundness.batteries.differential",
        (
            "the fraction of cases the grader accepts and the owner's stricter reference verifier "
            "rejects"
        ),
        0.0 if report.n == 0 else round(report.over_accepted / report.n, 6),
        "fraction",
        report.n,
        "graded case",
        intervals.uncertainty_for(report.over_accepted, report.n),
        subject_ref=subject_ref,
        provenance=ctx.provenance(duration_s=duration_s, arm="black_box"),
        method=_method("differential", {"n": report.n}),
        depends_on=("digest:source", "digest:samples"),
        limitations=(
            *LIMITATIONS,
            "the reference is the one the owner supplied, and the comparison says nothing about "
            "any other verifier",
        ),
        denominator="cases graded by both the subject and the reference",
        rate_definition="false_accept",
        result={
            "is_rate": True,
            "n": report.n,
            "over_accepted": report.over_accepted,
            "reference_accepts_subject_rejects": report.not_stricter,
            "what_n_can_establish": intervals.what_n_can_establish(report.over_accepted, report.n),
            "disagreements": [
                {
                    "case_id": row.case_id,
                    "subject_score": row.subject_score,
                    "reference_score": row.reference_score,
                    "direction": row.direction,
                }
                for row in report.disagreements
            ],
            "findings": [
                {
                    "code": "RL0223",
                    "kind": "fail",
                    "level": "warning",
                    "message": (
                        f"{row.case_id}: the grader accepted what the stricter reference verifier "
                        f"rejected ({row.subject_score} against {row.reference_score})"
                    ),
                }
                for row in report.disagreements
                if row.direction == "subject_accepts_reference_rejects"
            ],
        },
    )
