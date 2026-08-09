"""The soundness panel: is this reward measuring what it claims (section 7.3)?

Four instruments under one `PANEL`, found by `reward_lens.product.audit.registry.discover()`.
`Batteries` asks whether the grader accepts correct work and rejects wrong work for the right
reason, and gates both on the gold-sanity check. `Mutation` asks how much of the grader the
protected suite actually tests. `Relations` asks whether the score moves when the response changes
in ways that cannot change whether it is correct, and whether it moves when a real defect is put
in. `ExploitPremium` asks what an exploiting response buys over the best honest response on the
same task.

Findings are harvested by the module-level `findings_for(entry)` of interfaces section 8.1, and
the instrument objects deliberately expose no `findings` method, because a package offers exactly
one of the two forms. Nothing here edits `product/audit/`.

`self_check` is the panel auditing its own output before the runner sees it: a rate without a
bounded interval and a mutation score with one denominator are both refusals, because those are
the two ways this panel could produce a number that reads as a measurement and is not one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from reward_lens import contracts
from reward_lens.contracts.errors import UsageError
from reward_lens.instruments.base import Panel, RunContext

from . import batteries as batteries_module
from . import intervals, mutation, premium, relations
from .batteries import ACCEPT_THRESHOLD, Case

__all__ = [
    "ACCEPT_THRESHOLD",
    "FINDING_CODES",
    "FINDING_ID",
    "PANEL",
    "Batteries",
    "Case",
    "ExploitPremium",
    "Mutation",
    "Relations",
    "SelfCheckFailed",
    "batteries_module",
    "findings_for",
    "intervals",
    "mutation",
    "premium",
    "relations",
    "self_check",
]

VERSION = "1.0.0"

#: The id the runner's `FINDING_IDS` takes for this package (interfaces section 8.1).
FINDING_ID = "RGX-local-0221"

#: Every code this packet allocates, each with the finding id it is recorded under. The band is
#: RL0221 to RL0229 and nothing outside it is ever written here (interfaces section 6).
FINDING_CODES: dict[str, str] = {
    "RL0221": "RGX-local-0221",
    "RL0222": "RGX-local-0222",
    "RL0223": "RGX-local-0223",
    "RL0224": "RGX-local-0224",
    "RL0225": "RGX-local-0225",
    "RL0226": "RGX-local-0226",
    "RL0227": "RGX-local-0227",
    "RL0228": "RGX-local-0228",
    "RL0229": "RGX-local-0229",
}

#: What each code says, in the catalogue's own voice. P-ERRORS mints the rows; these are the
#: proposals, and they are here so that a message this panel writes and the row a reader looks up
#: cannot drift apart.
FINDING_TITLES: dict[str, str] = {
    "RL0221": "a known-wrong solution was accepted, or rejected for a reason other than the expected one",
    "RL0222": "a known-good solution was rejected, which is a defect in the evaluator and not in the task",
    "RL0223": "the grader accepted work the owner's stricter reference verifier rejected",
    "RL0224": "a mutant of the grader survived the protected suite, naming something the suite does not test",
    "RL0225": "the score moved under a transformation of the response that cannot change correctness",
    "RL0226": "an injected real defect did not move the score, so the grader is not reading the response",
    "RL0227": "an exploiting response outscored the best honest response on the same task",
    "RL0228": "the grader source was not available, so mutation and the metamorphic relations did not run",
    "RL0229": "the panel's own self-check refused a number it was about to report",
}

#: An absence this panel writes carries its finding code here, because an absence has no `result`
#: to put one in and a hole nobody can look up is a hole nobody fixes.
_ABSENCE_CODES: dict[str, str] = {
    "soundness.mutation.source": "RL0228",
    "soundness.relations.source": "RL0228",
}

#: The interval methods a count rate may leave this panel under (rules 1, 2 and 4).
RATE_METHODS = frozenset({"wilson", "clopper_pearson", "exact_zero_event"})


class SelfCheckFailed(UsageError):
    """RL0229: the panel refused one of its own numbers before the runner could record it.

    Two things trip it, and both are things this panel is forbidden to emit: a rate with no bounded
    interval, and a mutation score reported over one denominator. Neither is caught anywhere else,
    because by the time an entry reaches the record it is already a measurement a reader will use.
    """

    def __init__(self, detail: str, *, entry_id: str) -> None:
        super().__init__(
            code="RL0229",
            message=f"the soundness panel refused its own output: {detail}",
            remediation=(
                "report the rate with the interval rule 1 names, and the mutation score over both "
                "denominators rule 18 names, or do not report the number at all"
            ),
            context={"entry_id": entry_id, "detail": detail},
        )


def self_check(entries: Sequence[contracts.Entry]) -> None:
    """Every rate carries a bounded interval and every mutation score carries both denominators."""
    for entry in entries:
        result = entry.result or {}
        if result.get("is_rate"):
            uncertainty = entry.uncertainty
            if uncertainty is None or not uncertainty.bounded:
                raise SelfCheckFailed(
                    f"{entry.entry_id} reports a rate with no interval", entry_id=entry.entry_id
                )
            if uncertainty.method not in RATE_METHODS:
                raise SelfCheckFailed(
                    f"{entry.entry_id} reports a rate under {uncertainty.method}, which rule 1 "
                    "does not name for a count rate",
                    entry_id=entry.entry_id,
                )
        if result.get("mutation_score"):
            missing = [
                field
                for field in ("score_over_generated", "score_over_non_equivalent", "sampling_fraction")
                if field not in result
            ]
            if missing:
                raise SelfCheckFailed(
                    f"{entry.entry_id} reports a mutation score missing {', '.join(missing)}; "
                    "rule 18 asks for both denominators and the sampling fraction",
                    entry_id=entry.entry_id,
                )
    return None


def findings_for(entry: contracts.Entry) -> tuple[contracts.Finding, ...]:
    """Every finding this entry carries, in the one harvest form this package offers (section 8.1)."""
    if entry.kind == "absence":
        code = _ABSENCE_CODES.get(entry.entry_id)
        if code is None:
            return ()
        return (
            _finding(
                entry,
                0,
                {
                    "code": code,
                    "kind": "notApplicable",
                    "level": "note",
                    "message": (entry.absence.missing_access if entry.absence else FINDING_TITLES[code]),
                },
            ),
        )
    rows = list((entry.result or {}).get("findings") or ())
    return tuple(_finding(entry, index, row) for index, row in enumerate(rows))


def _finding(entry: contracts.Entry, index: int, row: dict[str, Any]) -> contracts.Finding:
    code = str(row["code"])
    arm = getattr(entry.provenance, "arm", None) or ("static" if entry.kind == "absence" else "black_box")
    return contracts.Finding(
        id=FINDING_CODES[code],
        rule=entry.entry_id,
        level=str(row.get("level", "warning")),  # type: ignore[arg-type]
        kind=str(row.get("kind", "fail")),  # type: ignore[arg-type]
        severity_rationale=FINDING_TITLES[code],
        scope=entry.scope,
        entries=[entry.entry_id],
        partial_fingerprint=contracts.digest(
            {"code": code, "entry": entry.entry_id, "index": index, "message": row.get("message", "")}
        ).removeprefix("sha256:")[:16],
        message=str(row.get("message", FINDING_TITLES[code])),
        code=code,
        arm=arm,  # type: ignore[arg-type]
    )


# --- what the instruments read ------------------------------------------------------------------


def _subject_ref(subject: Any) -> str:
    """The same digest the runner uses for its own entries, so the record agrees with itself."""
    return contracts.digest({"source": batteries_module.source_of(subject)})


def _protected_suite(ctx: RunContext) -> Any:
    """The protected suite the project declares, or `None` when it declares none (P-OUTCOME)."""
    project = getattr(ctx, "project", None)
    if project is None:
        return None
    config = getattr(project, "config", project)
    spec = getattr(config, "outcome", None)
    path = getattr(spec, "path", None)
    if not path:
        return None
    root = Path(getattr(project, "root", getattr(project, "dir", "."))) / str(path)
    from reward_lens.partitions import Partition, PartitionKind

    return Partition.from_dir(
        f"protected-suite:{path}",
        PartitionKind.PROTECTED_SUITE,
        root,
        log_path=root.parent / "partition-access.jsonl",
    )


def _cases(subject: Any, corpus: Any, ctx: RunContext) -> tuple[tuple[Case, ...], str]:
    """The battery cases: the protected suite when the project declares one, else the corpus."""
    partition = _protected_suite(ctx)
    if partition is not None:
        return batteries_module.cases_from_partition(partition), "the project's protected suite"
    built: list[Case] = []
    for index, rollout in enumerate(getattr(corpus, "rollouts", ()) or ()):
        row = rollout if isinstance(rollout, dict) else getattr(rollout, "__dict__", {})
        response = str(row.get("response", ""))
        if not response:
            continue
        honest = bool(row.get("honest", True))
        task = row.get("task")
        if task is None:
            task = next(
                (
                    one
                    for one in getattr(corpus, "tasks", ()) or ()
                    if str(one.get("id", "")) == str(row.get("task_id", ""))
                ),
                {},
            )
        built.append(
            Case(
                id=str(row.get("id", f"rollout-{index}")),
                task=dict(task or {}),
                response=response,
                expect="accept" if honest else "reject",
                kind="known_good" if honest else "known_wrong",
            )
        )
    return tuple(built), "the corpus supplied to this run"


def _no_cases(ctx: RunContext, entry_id: str, subject_ref: str, measurand: str) -> contracts.Entry:
    return ctx.absence(
        "soundness",
        entry_id,
        measurand,
        "no protected suite was declared and the corpus carried no responses to grade",
        "supply a protected suite in the project, or run with a response bank",
        ("no claim about what this grader accepts or rejects",),
        subject_ref=subject_ref,
        depends_on=("digest:samples",),
    )


# --- the instruments --------------------------------------------------------------------------------


class Batteries:
    """Known-good, known-wrong, the gold-sanity gate and the differential against a reference."""

    id = "soundness.batteries"
    section: contracts.Section = "soundness"
    method = batteries_module._method("batteries", {"threshold": ACCEPT_THRESHOLD})

    def run(self, subject: Any, corpus: Any, ctx: RunContext) -> list[contracts.Entry]:
        subject_ref = _subject_ref(subject)
        cases, origin = _cases(subject, corpus, ctx)
        if not cases:
            return [
                _no_cases(
                    ctx,
                    "soundness.batteries",
                    subject_ref,
                    "whether the grader accepts correct work and rejects wrong work",
                )
            ]
        produced: list[contracts.Entry] = []
        good = tuple(case for case in cases if case.kind == "known_good")
        wrong = tuple(case for case in cases if case.kind != "known_good")
        good_report = batteries_module.run_battery(subject, good, kind="known_good")
        produced.append(
            batteries_module.battery_entry(good_report, subject, ctx, subject_ref=subject_ref)
        )
        if wrong:
            wrong_report = batteries_module.run_battery(subject, wrong, kind="known_wrong")
            produced.append(
                batteries_module.battery_entry(
                    wrong_report, subject, ctx, subject_ref=subject_ref, load_bearing=True
                )
            )
        produced.append(
            batteries_module.gold_sanity_entry(
                batteries_module.gold_sanity(good_report), ctx, subject_ref=subject_ref
            )
        )
        reference = getattr(subject, "reference_verifier", None)
        if reference is None:
            produced.append(
                ctx.absence(
                    "soundness",
                    "soundness.batteries.differential",
                    "what the grader accepts and a stricter reference verifier rejects",
                    "the owner supplied no stricter reference verifier to compare against",
                    "supply a reference verifier the owner stands behind and run the audit again",
                    ("no claim that this grader agrees with any independent verifier",),
                    subject_ref=subject_ref,
                    depends_on=("digest:source",),
                    limitations=(f"the cases came from {origin}",),
                )
            )
        else:
            produced.append(
                batteries_module.differential_entry(
                    batteries_module.differential(subject, cases, reference=reference),
                    subject,
                    ctx,
                    subject_ref=subject_ref,
                )
            )
        self_check(produced)
        return produced


class Mutation:
    """Mutation testing over the grader source, with the commission's eight bookkeeping states."""

    id = "soundness.mutation"
    section: contracts.Section = "soundness"
    method = mutation._method("score", {"operators": list(mutation.OPERATOR_NAMES)})

    #: How many mutants one audit samples. Small on purpose: each one is a graded process, and the
    #: sampling fraction is on the entry so the number is never read as a census.
    limit = 6
    seed = 3

    def run(self, subject: Any, corpus: Any, ctx: RunContext) -> list[contracts.Entry]:
        subject_ref = _subject_ref(subject)
        source = batteries_module.source_of(subject)
        if not source:
            return [_source_absence(ctx, "soundness.mutation.source", subject_ref, "mutation")]
        cases, _origin = _cases(subject, corpus, ctx)
        if not cases:
            return [
                _no_cases(
                    ctx,
                    "soundness.mutation",
                    subject_ref,
                    "how much of the grader the protected suite tests",
                )
            ]
        ledger = mutation.run(subject, cases, limit=self.limit, seed=self.seed)
        produced = [
            mutation.score_entry(ledger, ctx, subject_ref=subject_ref),
            mutation.survival_entry(ledger, ctx, subject_ref=subject_ref),
        ]
        self_check(produced)
        return produced


class Relations:
    """The four invariance relations and the one killing relation of rule 19."""

    id = "soundness.relations"
    section: contracts.Section = "soundness"
    method = relations._method(relations.KILLING)

    def run(self, subject: Any, corpus: Any, ctx: RunContext) -> list[contracts.Entry]:
        subject_ref = _subject_ref(subject)
        source = batteries_module.source_of(subject)
        if not source:
            return [_source_absence(ctx, "soundness.relations.source", subject_ref, "the relations")]
        cases, _origin = _cases(subject, corpus, ctx)
        if not cases:
            return [
                _no_cases(
                    ctx,
                    "soundness.relations",
                    subject_ref,
                    "whether the score is invariant under transformations that preserve correctness",
                )
            ]
        outcomes = relations.run_relations(subject, cases)
        produced = relations.relation_entries(outcomes, ctx, subject_ref=subject_ref)
        self_check(produced)
        return produced


class ExploitPremium:
    """The gap between the best exploiting response and the best honest response on a task."""

    id = "soundness.exploit_premium"
    section: contracts.Section = "soundness"
    method = premium.premium_entry.__doc__ and contracts.Method(
        id="soundness.exploit_premium",
        version=VERSION,
        params_digest=contracts.digest({"threshold": ACCEPT_THRESHOLD}),
        procedure=(
            "every supplied response was graded, and on each task the best exploiting score was "
            "taken against the best honest score on that same task"
        ),
        credited_to="reward_lens.instruments.soundness (section 7.3)",
    )

    def run(self, subject: Any, corpus: Any, ctx: RunContext) -> list[contracts.Entry]:
        subject_ref = _subject_ref(subject)
        cases, _origin = _cases(subject, corpus, ctx)
        if not cases:
            return [
                _no_cases(
                    ctx,
                    "soundness.exploit_premium",
                    subject_ref,
                    "what an exploiting response buys over the best honest response",
                )
            ]
        rows = {
            str(row.get("case_id")): row
            for row in batteries_module.sandbox_runner(
                batteries_module.source_of(subject),
                str(getattr(subject, "entrypoint", "score") or "score"),
                cases,
            )
        }
        scored = tuple(
            premium.Scored(
                task_id=str(case.task.get("id", "task")),
                response_id=case.id,
                score=float(rows.get(case.id, {}).get("score") or 0.0),
                honest=case.kind == "known_good",
            )
            for case in cases
        )
        produced = [
            premium.premium_entry(premium.exploit_premium(scored), ctx, subject_ref=subject_ref)
        ]
        self_check(produced)
        return produced


def _source_absence(
    ctx: RunContext, entry_id: str, subject_ref: str, what: str
) -> contracts.Entry:
    return ctx.absence(
        "soundness",
        entry_id,
        f"what {what} over the grader source would have measured",
        "the grader source was not readable, so there was nothing to mutate or transform",
        "supply the grader as source, or record that this audit was black box only",
        (f"no claim about what {what} would have found",),
        subject_ref=subject_ref,
        depends_on=("digest:source",),
    )


PANEL = Panel(
    section="soundness",
    instruments=(Batteries(), Mutation(), Relations(), ExploitPremium()),
)

_ = intervals
