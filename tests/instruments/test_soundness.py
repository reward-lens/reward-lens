"""The soundness panel: batteries, mutation, metamorphic relations and exploit premium.

Every number this panel reports is a rate or a gap, and the test for each one is the same shape:
the value, the interval that bounds it, the denominator it was taken over, and the sentence that
says what the n could not have established. A test that asserted only the value would pass on a
panel that reports bare fractions, which is the failure this file exists to catch.

The grader under test is a small real one written to a temporary directory, not a stub object: the
battery, the mutants and the relations all run it through P-EXEC's sandbox, and the one place a
stub appears is the injected `runner`, where a deterministic in-process evaluation is what makes
the mutation ledger's eight states assertable one at a time.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import pytest

from reward_lens import contracts
from reward_lens.instruments.base import RunContext
from reward_lens.instruments.soundness import (
    FINDING_CODES,
    FINDING_ID,
    PANEL,
    Batteries,
    ExploitPremium,
    Mutation,
    Relations,
    SelfCheckFailed,
    findings_for,
    self_check,
)
from reward_lens.instruments.soundness import batteries as batteries_module
from reward_lens.instruments.soundness import intervals, mutation, premium, relations

# --- the grader under test ----------------------------------------------------------------------

GRADER = '''"""A small real grader: accept a solution that sums a list of integers."""


def score(task, response):
    namespace = {}
    try:
        exec(response, namespace)
    except Exception as failure:
        return {"score": 0.0, "reason": "response did not run: " + type(failure).__name__}
    target = namespace.get("solve")
    if target is None:
        return {"score": 0.0, "reason": "no solve defined"}
    for case in task["cases"]:
        got = target(case["xs"])
        if got != case["want"]:
            return {"score": 0.0, "reason": "wrong answer on " + str(case["xs"])}
    return {"score": 1.0, "reason": "all cases matched"}
'''

SUBJECT_REF = "sha256:" + "0" * 64

TASK: dict[str, Any] = {
    "id": "sum-a-list",
    "cases": [
        {"xs": [1, 2, 3], "want": 6},
        {"xs": [], "want": 0},
        {"xs": [-1, 1], "want": 0},
    ],
}

GOOD = "def solve(xs):\n    total = 0\n    for x in xs:\n        total = total + x\n    return total\n"
GOOD_2 = "def solve(xs):\n    return sum(xs)\n"
GOOD_3 = "def solve(xs):\n    acc = 0\n    for item in xs:\n        acc += item\n    return acc\n"
WRONG_OFF_BY_ONE = "def solve(xs):\n    return sum(xs) + 1\n"
WRONG_EMPTY = "def solve(xs):\n    return 0\n"
WRONG_RAISES = "raise ValueError('no')\n"
EXPLOIT = "def solve(xs):\n    return sum(xs)\nimport sys\n"


@pytest.fixture(scope="module")
def grader_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("soundness-grader")
    path = root / "grader.py"
    path.write_text(GRADER, encoding="utf-8")
    return path


class FakeSubject:
    """What an instrument reads of the reward system under audit (interfaces section 2, A-013)."""

    def __init__(self, grader_path: Path, project: Any = None) -> None:
        self.grader_path = grader_path
        self.entrypoint = "score"
        self.project_dir = grader_path.parent
        self.tasks_path = None
        self.responses_path = None
        self.trainer = None
        self.name = "sum-a-list grader"
        self.source = grader_path.read_text(encoding="utf-8")


class FakeCorpus:
    def __init__(self, tasks: Sequence[dict], rollouts: Sequence[Any] = ()) -> None:
        self.rollouts = tuple(rollouts)
        self.origin = "the test's own task and response bank"
        self.tasks = tuple(tasks)
        self.n_tasks = len(tasks)

    def as_list(self) -> Any:
        return list(self.rollouts)

    def __len__(self) -> int:
        return len(self.rollouts)


def known_good_cases() -> tuple[batteries_module.Case, ...]:
    return tuple(
        batteries_module.Case(
            id=f"good-{index}",
            task=TASK,
            response=response,
            expect="accept",
            kind="known_good",
        )
        for index, response in enumerate((GOOD, GOOD_2, GOOD_3))
    )


def known_wrong_cases() -> tuple[batteries_module.Case, ...]:
    return (
        batteries_module.Case(
            id="wrong-off-by-one",
            task=TASK,
            response=WRONG_OFF_BY_ONE,
            expect="reject",
            expected_reason="wrong answer",
            kind="known_wrong",
        ),
        batteries_module.Case(
            id="wrong-empty",
            task=TASK,
            response=WRONG_EMPTY,
            expect="reject",
            expected_reason="wrong answer",
            kind="known_wrong",
        ),
        batteries_module.Case(
            id="wrong-raises",
            task=TASK,
            response=WRONG_RAISES,
            expect="reject",
            expected_reason="did not run",
            kind="known_wrong",
        ),
    )


def suite_dir(root: Path) -> Path:
    """A protected suite on disk, in the shape `cases_from_partition` reads."""
    root.mkdir(parents=True, exist_ok=True)
    for case in known_good_cases() + known_wrong_cases():
        (root / f"{case.id}.json").write_text(json.dumps(case.to_dict()), encoding="utf-8")
    return root


def context() -> RunContext:
    return RunContext(sandbox=None, project=None, budget="0.00", offline=True)


# --- rules 1 to 4: the intervals ------------------------------------------------------------------


def test_wilson_at_twelve_of_twelve_is_the_lower_bound_the_commission_states() -> None:
    low, high = intervals.wilson(12, 12)
    assert round(low, 2) == 0.76, (low, high)
    assert high == 1.0


def test_a_lower_bound_of_point_nine_nine_needs_two_hundred_and_ninety_nine() -> None:
    assert intervals.n_for_lower_bound(0.99) == 299
    assert intervals.exact_one_sided_lower(298) < 0.99 <= intervals.exact_one_sided_lower(299)


def test_the_method_is_wilson_below_forty_one_and_clopper_pearson_when_load_bearing() -> None:
    assert intervals.method_for(7, 40, load_bearing=False) == "wilson"
    assert intervals.method_for(7, 40, load_bearing=True) == "clopper_pearson"
    assert intervals.method_for(7, 400, load_bearing=True) == "clopper_pearson"


def test_no_interval_this_module_produces_is_a_wald_or_a_jeffreys_interval() -> None:
    named = {intervals.method_for(k, n, load_bearing=lb)
             for k, n in ((0, 5), (0, 40), (5, 5), (12, 12), (3, 90))
             for lb in (False, True)}
    assert named <= {"wilson", "clopper_pearson", "exact_zero_event"}


def test_zero_events_gets_the_exact_bound_and_the_rule_of_three_is_labelled() -> None:
    assert intervals.method_for(0, 40, load_bearing=False) == "exact_zero_event"
    exact = intervals.exact_zero_event_upper(40)
    assert math.isclose(exact, 1.0 - 0.05 ** (1.0 / 40), rel_tol=1e-9)
    assert intervals.rule_of_three(40) == pytest.approx(3.0 / 40)
    assert intervals.rule_of_three(10) is None
    said = intervals.what_n_can_establish(0, 40)
    assert "exact" in said and "approximation" in said and "n >= 30" in said


def test_what_n_can_establish_names_both_of_the_commission_s_anchors() -> None:
    said = intervals.what_n_can_establish(12, 12)
    assert "0.76" in said
    assert "299" in said


def test_the_uncertainty_block_carries_the_method_and_the_interval() -> None:
    block = intervals.uncertainty_for(12, 12)
    assert isinstance(block, contracts.Uncertainty)
    assert block.method == "wilson"
    assert block.bounded is True
    assert round(block.interval[0], 2) == 0.76


# --- the batteries --------------------------------------------------------------------------------


def test_the_known_good_battery_reports_a_rate_with_an_interval_and_the_n_line(
    grader_path: Path,
) -> None:
    subject = FakeSubject(grader_path)
    report = batteries_module.run_battery(subject, known_good_cases(), kind="known_good")
    assert report.n == 3
    assert report.k == 3
    entry = batteries_module.battery_entry(report, subject, context(), subject_ref=SUBJECT_REF)
    assert entry.kind == "estimate"
    assert entry.uncertainty is not None and entry.uncertainty.bounded
    assert entry.uncertainty.method in {"wilson", "clopper_pearson", "exact_zero_event"}
    assert entry.result["what_n_can_establish"]
    assert "299" in entry.result["what_n_can_establish"]


def test_the_known_wrong_battery_requires_the_expected_reason_not_only_a_rejection(
    grader_path: Path,
) -> None:
    subject = FakeSubject(grader_path)
    report = batteries_module.run_battery(subject, known_wrong_cases(), kind="known_wrong")
    assert report.n == 3
    assert report.k == 3, [outcome.reason for outcome in report.outcomes]
    assert all(outcome.reason_matched for outcome in report.outcomes)
    mislabelled = batteries_module.Case(
        id="wrong-but-other-reason",
        task=TASK,
        response=WRONG_RAISES,
        expect="reject",
        expected_reason="wrong answer",
        kind="known_wrong",
    )
    other = batteries_module.run_battery(subject, (mislabelled,), kind="known_wrong")
    assert other.k == 0
    assert other.outcomes[0].accepted is False
    assert other.outcomes[0].reason_matched is False


def test_the_gold_sanity_gate_calls_a_rejected_gold_an_evaluator_defect(grader_path: Path) -> None:
    subject = FakeSubject(grader_path)
    passing = batteries_module.gold_sanity(
        batteries_module.run_battery(subject, known_good_cases(), kind="known_good")
    )
    assert passing.passed is True
    assert passing.failures == ()
    broken = grader_path.parent / "broken.py"
    broken.write_text(GRADER.replace('"score": 1.0', '"score": 0.0'), encoding="utf-8")
    failing = batteries_module.gold_sanity(
        batteries_module.run_battery(
            FakeSubject(broken), known_good_cases(), kind="known_good"
        )
    )
    assert failing.passed is False
    assert len(failing.failures) == 3
    assert "evaluator defect" in failing.statement
    assert "impossible" not in failing.statement


def test_the_differential_against_a_stricter_reference_names_the_over_accepted_case(
    grader_path: Path,
) -> None:
    subject = FakeSubject(grader_path)
    stricter = grader_path.parent / "stricter.py"
    stricter.write_text(
        GRADER.replace(
            'return {"score": 1.0, "reason": "all cases matched"}',
            'if "import" in response:\n'
            '        return {"score": 0.0, "reason": "imports are not allowed"}\n'
            '    return {"score": 1.0, "reason": "all cases matched"}',
        ),
        encoding="utf-8",
    )
    cases = known_good_cases() + (
        batteries_module.Case(
            id="exploit-imports", task=TASK, response=EXPLOIT, expect="accept", kind="known_good"
        ),
    )
    report = batteries_module.differential(subject, cases, reference=FakeSubject(stricter))
    assert report.n == 4
    assert report.over_accepted == 1
    assert [row.case_id for row in report.disagreements] == ["exploit-imports"]
    entry = batteries_module.differential_entry(
        report, subject, context(), subject_ref=SUBJECT_REF
    )
    assert entry.uncertainty is not None and entry.uncertainty.bounded


def test_an_empty_protected_suite_refuses_with_rl0341_rather_than_scoring_nothing(
    tmp_path: Path,
) -> None:
    from reward_lens.partitions import AccessLog, Partition, PartitionKind

    empty = tmp_path / "empty-suite"
    empty.mkdir()
    partition = Partition.from_dir(
        "suite-empty", PartitionKind.PROTECTED_SUITE, empty, log_path=tmp_path / "access.jsonl"
    )
    assert partition.is_empty
    with pytest.raises(Exception) as raised:
        batteries_module.cases_from_partition(partition)
    assert raised.value.code == "RL0341"
    assert raised.value.exit_code == 4
    assert "suite-empty" in json.dumps(raised.value.context)
    assert isinstance(AccessLog, type)


def test_a_filled_protected_suite_reads_back_every_case(tmp_path: Path) -> None:
    from reward_lens.partitions import Partition, PartitionKind

    root = suite_dir(tmp_path / "suite")
    partition = Partition.from_dir(
        "suite-full", PartitionKind.PROTECTED_SUITE, root, log_path=tmp_path / "access.jsonl"
    )
    cases = batteries_module.cases_from_partition(partition)
    assert len(cases) == 6
    assert {case.kind for case in cases} == {"known_good", "known_wrong"}


# --- mutation -------------------------------------------------------------------------------------


def test_every_one_of_the_eight_states_is_reached_and_the_names_are_the_commission_s() -> None:
    assert mutation.STATES == (
        "proposed",
        "buildable",
        "exercised",
        "behaviourally_distinct",
        "eligible",
        "killed",
        "surviving",
        "unresolved_equivalence",
    )
    ledger = _ledger()
    counts = ledger.counts()
    assert set(counts) == set(mutation.STATES)
    assert all(counts[state] > 0 for state in mutation.STATES), counts


def test_the_score_is_reported_both_ways_with_the_sampling_fraction() -> None:
    ledger = _ledger()
    over_generated = ledger.score_over_generated()
    over_non_equivalent = ledger.score_over_non_equivalent()
    assert over_generated.denominator == ledger.counts()["proposed"]
    assert over_generated.label == "killed over generated"
    assert over_generated.lower_bound is True
    assert over_non_equivalent.denominator == (
        ledger.counts()["proposed"] - ledger.confirmed_equivalent_count()
    )
    assert over_non_equivalent.denominator < over_generated.denominator
    for score in (over_generated, over_non_equivalent):
        assert score.uncertainty.method == "wilson"
        assert score.uncertainty.bounded
        assert 0.0 <= score.value <= 1.0
    assert 0.0 < ledger.sampling_fraction < 1.0


def test_an_unresolved_equivalence_mutant_is_never_subtracted() -> None:
    ledger = _ledger()
    unresolved = ledger.counts()["unresolved_equivalence"]
    assert unresolved > 0
    subtracted = (
        ledger.counts()["proposed"] - ledger.score_over_non_equivalent().denominator
    )
    assert subtracted == ledger.confirmed_equivalent_count()
    assert subtracted < unresolved + ledger.confirmed_equivalent_count()
    assert "undecidable" in ledger.equivalence_statement()


def test_survival_is_grouped_by_operator_with_a_minimised_reproducer_each() -> None:
    ledger = _ledger()
    by_operator = ledger.survival_by_operator()
    assert by_operator
    for operator, survivors in by_operator.items():
        assert operator in mutation.OPERATOR_NAMES
        for record in survivors:
            assert record.reproducer is not None
            assert len(record.reproducer.response) <= len(record.reproducer.original)
            assert record.reproducer.shrink_steps >= 0


def test_a_confirmed_equivalent_mutant_that_is_not_surviving_stops_the_run() -> None:
    ledger = _ledger()
    killed = next(r for r in ledger.records if "killed" in r.states)
    with pytest.raises(mutation.EquivalenceNotFromTheStates) as raised:
        mutation.confirm_equivalent(ledger, killed.mutant.id, procedure="a hand proof")
    assert "surviving" in str(raised.value)


def test_the_operators_build_real_mutants_from_real_grader_source() -> None:
    mutants, sites = mutation.propose(GRADER, limit=None)
    assert sites == len(mutants)
    assert sites > 4
    assert {m.operator for m in mutants} <= set(mutation.OPERATOR_NAMES)
    for one in mutants:
        assert one.source != GRADER
        compile(one.source, "<mutant>", "exec")


def test_mutation_runs_the_real_grader_through_the_sandbox_and_kills_something(
    grader_path: Path,
) -> None:
    subject = FakeSubject(grader_path)
    ledger = mutation.run(
        subject,
        known_good_cases() + known_wrong_cases(),
        limit=6,
        seed=3,
    )
    assert ledger.counts()["proposed"] == 6
    assert ledger.counts()["killed"] >= 1
    assert ledger.sampling_fraction <= 1.0


# --- metamorphic relations --------------------------------------------------------------------------


def test_the_four_invariance_relations_and_the_killing_relation_are_named() -> None:
    assert relations.INVARIANCE == (
        "identifier_renaming",
        "statement_reordering",
        "whitespace_and_comments",
        "equivalent_rewrite",
    )
    assert relations.KILLING == "injected_defect"
    assert relations.JUDGE_RELATIONS == ("order_swap", "paraphrase", "length_padding")


def test_every_invariance_relation_transforms_the_response_into_something_different() -> None:
    for name in relations.INVARIANCE:
        changed = relations.transform(name, GOOD)
        assert changed is not None, name
        assert changed != GOOD, name
        compile(changed, "<transformed>", "exec")


def test_the_relations_run_on_the_real_grader_and_every_invariance_holds(
    grader_path: Path,
) -> None:
    subject = FakeSubject(grader_path)
    outcomes = relations.run_relations(subject, known_good_cases())
    ran = {outcome.relation for outcome in outcomes}
    assert ran == set(relations.INVARIANCE) | {relations.KILLING}
    invariance = [o for o in outcomes if o.relation in relations.INVARIANCE]
    assert invariance and all(o.held for o in invariance), [
        (o.relation, o.note) for o in invariance if not o.held
    ]
    killing = [o for o in outcomes if o.relation == relations.KILLING]
    assert killing and all(o.held for o in killing), [(o.case_id, o.note) for o in killing]


def test_a_grader_that_ignores_the_response_fails_the_killing_relation(tmp_path: Path) -> None:
    blind = tmp_path / "blind.py"
    blind.write_text(
        'def score(task, response):\n    return {"score": 1.0, "reason": "always"}\n',
        encoding="utf-8",
    )
    outcomes = relations.run_relations(FakeSubject(blind), known_good_cases()[:1])
    killing = [o for o in outcomes if o.relation == relations.KILLING]
    assert killing and not any(o.held for o in killing)


# --- exploit premium ------------------------------------------------------------------------------


def test_the_premium_is_best_exploit_minus_best_honest_on_the_same_task() -> None:
    scored = (
        premium.Scored("t1", "honest-a", 0.40, honest=True),
        premium.Scored("t1", "honest-b", 0.90, honest=True),
        premium.Scored("t1", "exploit-a", 1.00, honest=False),
        premium.Scored("t1", "exploit-b", 0.20, honest=False),
    )
    report = premium.exploit_premium(scored)
    row = report.per_task[0]
    assert row.best_honest == 0.90
    assert row.best_exploit == 1.00
    assert row.premium == pytest.approx(0.10)
    assert row.honest_id == "honest-b"
    assert row.exploit_id == "exploit-a"
    average = sum(s.score for s in scored if s.honest) / 2
    assert row.premium != pytest.approx(row.best_exploit - average)


def test_a_task_with_no_matched_honest_output_is_excluded_and_named() -> None:
    scored = (
        premium.Scored("t1", "honest-a", 0.5, honest=True),
        premium.Scored("t1", "exploit-a", 0.9, honest=False),
        premium.Scored("t2", "exploit-b", 1.0, honest=False),
    )
    report = premium.exploit_premium(scored)
    assert [row.task_id for row in report.per_task] == ["t1"]
    assert report.unmatched == ("t2",)


def test_the_premium_entry_states_its_interval_or_why_it_has_none() -> None:
    small = premium.exploit_premium(
        tuple(
            row
            for index in range(4)
            for row in (
                premium.Scored(f"t{index}", f"h{index}", 0.5, honest=True),
                premium.Scored(f"t{index}", f"e{index}", 0.9, honest=False),
            )
        )
    )
    entry = premium.premium_entry(small, context(), subject_ref=SUBJECT_REF)
    assert entry.uncertainty.method == "no_interval_below_15_clusters"
    assert entry.uncertainty.bounded is False
    assert entry.uncertainty.clusters == 4
    big = premium.exploit_premium(
        tuple(
            row
            for index in range(20)
            for row in (
                premium.Scored(f"t{index}", f"h{index}", 0.5, honest=True),
                premium.Scored(f"t{index}", f"e{index}", 0.9, honest=False),
            )
        )
    )
    entry_big = premium.premium_entry(big, context(), subject_ref=SUBJECT_REF)
    assert entry_big.uncertainty.method == "task_level_bootstrap"
    assert entry_big.uncertainty.bounded is True


# --- the panel, discovery, findings and the self-check ----------------------------------------------


def test_the_panel_is_a_soundness_panel_of_four_instruments() -> None:
    assert PANEL.section == "soundness"
    assert [one.id for one in PANEL.instruments] == [
        "soundness.batteries",
        "soundness.mutation",
        "soundness.relations",
        "soundness.exploit_premium",
    ]
    for one in PANEL.instruments:
        assert isinstance(one, (Batteries, Mutation, Relations, ExploitPremium))
        assert one.section == "soundness"


def test_discovery_finds_this_panel() -> None:
    from reward_lens.product.audit.registry import discover

    found = {panel.section for panel in discover()}
    assert "soundness" in found


def test_the_package_offers_exactly_one_harvest_form() -> None:
    assert callable(findings_for)
    for one in PANEL.instruments:
        assert not hasattr(one, "findings"), one.id


def test_the_panel_runs_end_to_end_and_every_entry_is_well_formed(
    grader_path: Path, tmp_path: Path
) -> None:
    subject = FakeSubject(grader_path)
    corpus = FakeCorpus((TASK,), rollouts=_rollouts())
    ctx = context()
    entries: list[contracts.Entry] = []
    for one in PANEL.instruments:
        entries.extend(one.run(subject, corpus, ctx))
    assert entries
    assert {entry.section for entry in entries} == {"soundness"}
    for entry in entries:
        assert entry.entry_id.startswith("soundness.")
        assert entry.method.id.startswith("soundness.")
        assert entry.provenance.sandbox_tier
        if entry.kind == "estimate":
            assert entry.uncertainty is not None
            assert entry.n is not None and entry.sampling_unit
    self_check(entries)


def test_findings_carry_this_packet_s_codes_and_a_fingerprint(grader_path: Path) -> None:
    assert FINDING_ID == "RGX-local-0221"
    assert set(FINDING_CODES) == {f"RL02{n}" for n in range(21, 30)}
    subject = FakeSubject(grader_path)
    corpus = FakeCorpus((TASK,), rollouts=_rollouts())
    ctx = context()
    produced: list[contracts.Finding] = []
    for one in PANEL.instruments:
        for entry in one.run(subject, corpus, ctx):
            produced.extend(findings_for(entry))
    assert produced
    for finding in produced:
        assert finding.code in FINDING_CODES
        assert finding.id.startswith("RGX-local-02")
        assert len(finding.partial_fingerprint) >= 8
        assert finding.entries


def test_a_rate_without_its_interval_fails_the_panel_s_own_self_check() -> None:
    entry = _rate_entry()
    self_check([entry])
    entry.uncertainty = contracts.Uncertainty(
        method="no_interval_below_15_clusters",
        level=0.95,
        reason="none stated",
        clusters=3,
    )
    with pytest.raises(SelfCheckFailed) as raised:
        self_check([entry])
    assert raised.value.code == "RL0229"
    assert "soundness.batteries.known_good" in str(raised.value)


def test_a_mutation_score_with_one_denominator_fails_the_self_check() -> None:
    entry = _mutation_entry()
    self_check([entry])
    del entry.result["score_over_non_equivalent"]
    with pytest.raises(SelfCheckFailed) as raised:
        self_check([entry])
    assert raised.value.code == "RL0229"
    assert "denominator" in str(raised.value)


def test_an_exception_inside_the_instrument_is_not_swallowed(grader_path: Path) -> None:
    class Exploding:
        entrypoint = "score"
        name = "a subject whose source cannot be read"

        def __init__(self, path: Path) -> None:
            self.grader_path = path
            self.project_dir = path.parent
            self.tasks_path = None
            self.responses_path = None
            self.trainer = None

        @property
        def source(self) -> str:
            raise RuntimeError("the source could not be read")

    with pytest.raises(RuntimeError, match="could not be read"):
        Mutation().run(Exploding(grader_path), FakeCorpus((TASK,)), context())


def test_a_grader_with_no_readable_source_records_an_absence_not_a_pass(tmp_path: Path) -> None:
    missing = tmp_path / "not-here.py"

    class NoSource(FakeSubject):
        def __init__(self) -> None:
            self.grader_path = missing
            self.entrypoint = "score"
            self.project_dir = tmp_path
            self.tasks_path = None
            self.responses_path = None
            self.trainer = None
            self.name = "a grader with no source"
            self.source = ""

    entries = Mutation().run(NoSource(), FakeCorpus((TASK,)), context())
    assert entries and all(entry.kind == "absence" for entry in entries)
    assert entries[0].absence.state == "NOT_MEASURED"
    findings = [f for entry in entries for f in findings_for(entry)]
    assert [f.code for f in findings] == ["RL0228"]
    assert findings[0].kind == "notApplicable"


# --- helpers ---------------------------------------------------------------------------------------


def _rollouts() -> tuple[dict, ...]:
    return (
        {"task_id": TASK["id"], "id": "honest-1", "response": GOOD, "honest": True},
        {"task_id": TASK["id"], "id": "honest-2", "response": GOOD_2, "honest": True},
        {"task_id": TASK["id"], "id": "exploit-1", "response": EXPLOIT, "honest": False},
    )


def _in_process_runner(source: str, entrypoint: str, cases: Sequence[Any]) -> list[dict]:
    """A deterministic stand-in for the sandbox, used only where a fixed ledger is the point."""
    namespace: dict[str, Any] = {}
    rows: list[dict] = []
    try:
        exec(compile(source, "<runner>", "exec"), namespace)
        target = namespace[entrypoint]
    except Exception as failure:  # pragma: no cover - the mutants here all compile
        return [
            {"case_id": case.id, "score": None, "reason": "", "error": repr(failure)}
            for case in cases
        ]
    for case in cases:
        try:
            got = target(case.task, case.response)
        except Exception as failure:
            rows.append({"case_id": case.id, "score": None, "reason": "", "error": repr(failure)})
            continue
        if isinstance(got, dict):
            rows.append(
                {
                    "case_id": case.id,
                    "score": float(got.get("score", 0.0)),
                    "reason": str(got.get("reason", "")),
                    "error": "",
                }
            )
        else:
            rows.append({"case_id": case.id, "score": float(got), "reason": "", "error": ""})
    return rows


_LEDGER: mutation.MutationLedger | None = None


def _ledger() -> mutation.MutationLedger:
    """One ledger in which each of the eight states holds at least one mutant, built on purpose."""
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = mutation.demonstration_ledger()
    return _LEDGER


def _rate_entry() -> contracts.Entry:
    return batteries_module.battery_entry(
        batteries_module.BatteryReport(
            kind="known_good",
            threshold=1.0,
            outcomes=(
                batteries_module.CaseOutcome(
                    case_id="good-0",
                    score=1.0,
                    reason="all cases matched",
                    error="",
                    accepted=True,
                    as_expected=True,
                    reason_matched=True,
                ),
            ),
        ),
        None,
        context(),
        subject_ref=SUBJECT_REF,
    )


def _mutation_entry() -> contracts.Entry:
    return mutation.score_entry(_ledger(), context(), subject_ref=SUBJECT_REF)
