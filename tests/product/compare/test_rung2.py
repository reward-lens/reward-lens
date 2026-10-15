"""Rung 2 reports finite-bank reward-selected outcome, not oracle availability."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import runpy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from reward_lens import contracts
from reward_lens.bank import BankProvenance, Response, ResponseBank, TaskId, TieBreak
from reward_lens.contracts import EntryProvenance, UsageError
from reward_lens.execution.limits import Limits
from reward_lens.execution.linux import LinuxSandbox
from reward_lens.outcome import (
    OutcomeCheck,
    TaskValidity,
    UnqualifiedReason,
    qualified_protocol,
    qualify,
    resolved_fail,
    resolved_pass,
    unqualified,
)
from reward_lens.outcome.qualify import SuiteResult
from reward_lens.partitions import Partition, PartitionKind
from reward_lens.product.compare.rung2 import (
    Calibration,
    OutcomeProvider,
    Susceptibility,
    canonical_assay_entry,
    expected_max_order_statistic,
    load_selection_stress_execution,
    pass_at_k,
    selection_stress,
)


class RecordedSelector:
    """Selector whose reward is the recorded score, through P-BANK's real context."""

    def score(self, _task, response, *, sandbox, limits):
        assert isinstance(sandbox, LinuxSandbox)
        assert isinstance(limits, Limits)
        return {
            "correct-low-a": Decimal("1"),
            "correct-low-b": Decimal("2"),
            "wrong-high-a": Decimal("3"),
            "wrong-high-b": Decimal("4"),
        }[response]


def _bank() -> ResponseBank:
    task = TaskId("task")
    # Two correct low-reward candidates and two wrong high-reward candidates.
    return ResponseBank(
        id="anti-correlated",
        digest="bank",
        provenance=BankProvenance("fixture", None, None, "source"),
        tasks=(task,),
        responses={
            task: (
                Response("01", task, "correct-low-a", {"reward": Decimal("1")}),
                Response("02", task, "correct-low-b", {"reward": Decimal("2")}),
                Response("03", task, "wrong-high-a", {"reward": Decimal("3")}),
                Response("04", task, "wrong-high-b", {"reward": Decimal("4")}),
            )
        },
        task_payloads={task: {"id": "task", "prompt": "fixture task"}},
    )


def _provider(tmp_path: Path) -> OutcomeProvider:
    root = tmp_path / "qualified"
    root.mkdir()
    (root / "task").write_text("control", encoding="utf-8")
    suite = Partition.from_dir(
        "qualified-suite", PartitionKind.PROTECTED_SUITE, root, log_path=tmp_path / "access.jsonl"
    )
    provenance = EntryProvenance(
        started="2026-09-26T00:00:00Z", duration_s=0.0, sandbox_tier="T0", offline=True
    )
    check = OutcomeCheck("independent", "sha256:" + "c" * 64, "fixture", suite.id)
    protocol = qualified_protocol(
        qualify(
            check=check,
            suite=suite,
            grader_owner="reward",
            grader_source_digest="sha256:" + "b" * 64,
            subject_ref="sha256:" + "a" * 64,
            provenance=provenance,
            result=SuiteResult(1, 1, 1, 1),
        )
    )

    def resolve(_task: TaskId, response: Response):
        if response.text.startswith("correct"):
            return resolved_pass(
                item_id=response.digest,
                reason="independent success",
                protocol=protocol,
                task_validity=TaskValidity.VALID,
            )
        return resolved_fail(
            item_id=response.digest,
            reason="independent failure",
            protocol=protocol,
            task_validity=TaskValidity.VALID,
        )

    return OutcomeProvider(check=check, protocol=protocol, resolve=resolve)


def _contexts() -> tuple[LinuxSandbox, Limits]:
    return LinuxSandbox(tier="T0"), Limits(wall_s=5, cpu_s=5, network=False)


def _write_project_local_selection_config(root: Path) -> Path:
    """Actual supervised controls and target evaluation, without JSON labels/counts."""
    (root / "controls").mkdir(parents=True, exist_ok=True)
    (root / "controls" / "known-good").write_text("control", encoding="utf-8")
    (root / "controls" / "known-wrong").write_text("control", encoding="utf-8")
    (root / "targets").mkdir(parents=True, exist_ok=True)
    for digest in ("01", "02", "03", "04"):
        (root / "targets" / digest).write_text(digest, encoding="utf-8")
    selector = root / "selector.py"
    selector.write_text(
        "def score(task, response):\n"
        "    return {'correct-low-a': 1, 'correct-low-b': 2, 'wrong-high-a': 3, 'wrong-high-b': 4}[response]\n",
        encoding="utf-8",
    )
    evaluator = root / "outcome_evaluator.py"
    evaluator.write_text(
        "def score(task, response):\n"
        "    return 1.0 if response in ('control-good', 'correct-low-a', 'correct-low-b') else 0.0\n",
        encoding="utf-8",
    )
    evaluator_digest = contracts.digest(
        {"source": evaluator.read_text(encoding="utf-8"), "entrypoint": "score"}
    )
    config = root / "selection-stress.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "selector": {"path": "selector.py", "entrypoint": "score"},
                "outcome": {
                    "check": {
                        "owner": "independent",
                        "source_digest": evaluator_digest,
                        "interface": "reward_lens.graders.python_grader.PythonGrader.score/v1",
                        "partition_id": "control-suite",
                    },
                    "evaluator": {"path": "outcome_evaluator.py", "entrypoint": "score"},
                    "controls": {
                        "partition_path": "controls",
                        "partition_id": "control-suite",
                        "known_good": [
                            {
                                "item_id": "known-good",
                                "task": {"id": "control-good", "prompt": "control"},
                                "response": "control-good",
                            }
                        ],
                        "known_wrong": [
                            {
                                "item_id": "known-wrong",
                                "task": {"id": "control-wrong", "prompt": "control"},
                                "response": "control-wrong",
                            }
                        ],
                    },
                    "target": {
                        "partition_path": "targets",
                        "partition_id": "target-suite",
                        "response_items": {"01": "01", "02": "02", "03": "03", "04": "04"},
                        "tasks": {"task": {"id": "task", "prompt": "fixture task"}},
                    },
                    "candidate_owner": "candidate",
                    "audited_grader": {"owner": "reward", "source_digest": "sha256:" + "b" * 64},
                    "subject_ref": "sha256:" + "a" * 64,
                },
                "ladder": [2],
                "bootstrap_resamples": 10,
                "sandbox_tier": "L1",
                "limits": {
                    "wall_s": 5,
                    "cpu_s": 5,
                    "memory_bytes": 1073741824,
                    "processes": 1,
                    "stdout_bytes": 4096,
                    "artifact_bytes": 4096,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return config


def test_project_local_factory_executes_controls_and_resolver_without_json_outcome_labels(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    config = _write_project_local_selection_config(project)
    subject = "sha256:" + "a" * 64

    execution = load_selection_stress_execution(project, config, expected_subject_ref=subject)
    assert execution.outcome_provider is not None
    assert execution.outcome_provider.qualified
    assert execution.validated_subject == subject
    assert execution.ladder == (2,)
    assert execution.bootstrap_resamples == 10
    assert isinstance(execution.selector, object)
    assert isinstance(execution.sandbox, LinuxSandbox)
    assert execution.sandbox.tier == "L1"
    assert execution.limits.network is False
    assert execution.canonical_declaration["outcome"]["qualification"] == "supervised_receipts"
    assert execution.canonical_declaration["validated_subject"] == subject
    assert execution.canonical_declaration["outcome"]["subject_ref"] == subject
    assert (
        execution.canonical_declaration["outcome"]["qualification_entry"]["subject_ref"] == subject
    )

    result = selection_stress(
        _bank(),
        execution.selector,
        execution.outcome_provider,
        ladder=execution.ladder,
        bootstrap_resamples=execution.bootstrap_resamples,
        sandbox=execution.sandbox,
        limits=execution.limits,
    )
    assert result.entries[0].kind == "estimate"
    assert result.entries[0].result["curve"][0]["selected_success"] == "0.166667"
    assert result.entries[0].result["supervised_target_receipts"]
    assert result.entries[0].subject_ref == subject
    assert result.entries[0].result["source_bank"]["digest"] == "bank"
    assert result.entries[0].result["subject_ref"] == subject
    assert all(
        receipt["subject_ref"] == subject
        for receipt in result.entries[0].result["supervised_target_receipts"]
    )


def test_factory_preflights_missing_wrong_and_substituted_subjects_before_execution(
    tmp_path: Path,
) -> None:
    expected = "sha256:" + "a" * 64
    project = tmp_path / "project"
    config = _write_project_local_selection_config(project)

    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["outcome"].pop("subject_ref")
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UsageError, match="outcome.subject_ref") as missing:
        load_selection_stress_execution(project, config, expected_subject_ref=expected)
    assert missing.value.context["field"] == "outcome.subject_ref"
    assert not (project / ".reward-lens").exists()

    config = _write_project_local_selection_config(project)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["outcome"]["subject_ref"] = "sha256:not-a-digest"
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UsageError, match="sha256 digest") as wrong:
        load_selection_stress_execution(project, config, expected_subject_ref=expected)
    assert wrong.value.context["field"] == "outcome.subject_ref"
    assert not (project / ".reward-lens").exists()

    config = _write_project_local_selection_config(project)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["outcome"]["subject_ref"] = "sha256:" + "d" * 64
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UsageError, match="must match expected_subject_ref") as substituted:
        load_selection_stress_execution(project, config, expected_subject_ref=expected)
    assert substituted.value.context["field"] == "outcome.subject_ref"
    assert not (project / ".reward-lens").exists()


def test_project_local_factory_requires_explicit_outcome_absence_and_rejects_outside_paths(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    config = _write_project_local_selection_config(project)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["outcome"] = None
    config.write_text(json.dumps(payload), encoding="utf-8")
    absent = load_selection_stress_execution(project, config)
    assert absent.outcome_provider is None
    result = selection_stress(
        _bank(),
        absent.selector,
        absent.outcome_provider,
        ladder=absent.ladder,
        bootstrap_resamples=absent.bootstrap_resamples,
        sandbox=absent.sandbox,
        limits=absent.limits,
    )
    assert canonical_assay_entry(result).kind == "absence"

    payload["outcome"] = json.loads(
        _write_project_local_selection_config(project).read_text(encoding="utf-8")
    )["outcome"]
    payload["selector"]["path"] = "../outside.py"
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UsageError, match="project-local"):
        load_selection_stress_execution(project, config)


def test_supervised_factory_refuses_fabricated_counts_paths_network_and_partition_overlap(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    config = _write_project_local_selection_config(project)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["outcome"]["controls"]["counts"] = {"known_good_passed": 99}
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UsageError, match="exactly"):
        load_selection_stress_execution(project, config)

    payload = json.loads(_write_project_local_selection_config(project).read_text(encoding="utf-8"))
    for field in ("evaluator", "controls", "target"):
        nested = payload["outcome"][field]
        nested["path" if field == "evaluator" else "partition_path"] = "../escape"
        config.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(UsageError, match="project-local"):
            load_selection_stress_execution(project, config)
        payload = json.loads(
            _write_project_local_selection_config(project).read_text(encoding="utf-8")
        )

    payload["outcome"]["target"]["partition_path"] = "controls"
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UsageError, match="disjoint"):
        load_selection_stress_execution(project, config)

    payload = json.loads(_write_project_local_selection_config(project).read_text(encoding="utf-8"))
    payload["sandbox_tier"] = "T0"
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UsageError, match="egress"):
        load_selection_stress_execution(project, config)


def test_supervised_factory_control_failure_and_ambiguous_target_are_canonical_absences(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    config = _write_project_local_selection_config(project)
    evaluator = project / "outcome_evaluator.py"
    evaluator.write_text("def score(task, response):\n    return 1.0\n", encoding="utf-8")
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["outcome"]["check"]["source_digest"] = contracts.digest(
        {"source": evaluator.read_text(encoding="utf-8"), "entrypoint": "score"}
    )
    config.write_text(json.dumps(payload), encoding="utf-8")
    unqualified = load_selection_stress_execution(project, config)
    assert unqualified.outcome_provider is not None
    assert unqualified.outcome_provider.protocol is None
    assert (
        canonical_assay_entry(
            selection_stress(
                _bank(),
                unqualified.selector,
                unqualified.outcome_provider,
                ladder=unqualified.ladder,
                bootstrap_resamples=unqualified.bootstrap_resamples,
                sandbox=unqualified.sandbox,
                limits=unqualified.limits,
            )
        ).kind
        == "absence"
    )

    config = _write_project_local_selection_config(project)
    evaluator.write_text(
        "def score(task, response):\n    return 1.0 if response == 'control-good' else 0.0 if response == 'control-wrong' else 0.5\n",
        encoding="utf-8",
    )
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["outcome"]["check"]["source_digest"] = contracts.digest(
        {"source": evaluator.read_text(encoding="utf-8"), "entrypoint": "score"}
    )
    config.write_text(json.dumps(payload), encoding="utf-8")
    ambiguous = load_selection_stress_execution(project, config)
    result = selection_stress(
        _bank(),
        ambiguous.selector,
        ambiguous.outcome_provider,
        ladder=ambiguous.ladder,
        bootstrap_resamples=ambiguous.bootstrap_resamples,
        sandbox=ambiguous.sandbox,
        limits=ambiguous.limits,
    )
    assert canonical_assay_entry(result).kind == "absence"


def test_exact_finite_control_distinguishes_selected_winner_success_from_oracle_pass_at_k(
    tmp_path: Path,
) -> None:
    sandbox, limits = _contexts()
    provider = _provider(tmp_path)
    result = selection_stress(
        _bank(),
        RecordedSelector(),
        provider,
        ladder=(2,),
        bootstrap_resamples=20,
        sandbox=sandbox,
        limits=limits,
    )

    reading = result.reading
    assert result.scope == "selection_stress"
    assert reading.points[0].selected_success.value == Decimal("0.166667")
    assert reading.points[0].oracle_pass_at_k == Decimal("0.833333")
    assert reading.points[0].tie_break is TieBreak.FIRST_BY_DIGEST
    assert pass_at_k(4, 2, 2) == Decimal("0.833333")
    actual_outcomes = tuple(
        provider.resolve(TaskId("task"), response)
        for response in _bank().responses_for(TaskId("task"))
    )
    assert expected_max_order_statistic(actual_outcomes, 2) == Decimal("0.166667")
    with pytest.raises(TypeError, match="accepted Outcome"):
        expected_max_order_statistic((True, True, False, False), 2)  # type: ignore[arg-type]
    # The capability target is a separate witness; its absence cannot erase this curve.
    assert (
        result.scorecard.rows[1].value.reason == "not measured: no held-out capability set declared"
    )
    entry = canonical_assay_entry(result)
    assert entry.entry_id == "reward_statistics.selection_stress"
    assert entry.result["curve"][0]["selected_success"] == "0.166667"
    assert entry.result["curve"][0]["oracle_pass_at_k"] == "0.833333"
    assert entry.result["calibration"]["rho"] == "0.964286"
    assert entry.result["qualification"]["protocol_digest"] == provider.protocol.protocol_digest
    assert entry.result["finite_bank_population"]["N"] == 4
    assert entry.result["draw_law"] == "uniform finite subsets without replacement"
    assert entry.result["ties"]["policy"] == "first_by_digest"
    assert entry.result["selection_seed"] == 0
    assert entry.result["task_bootstrap"]["method"] == "task_level_bootstrap"
    assert entry.result["source_bank"]["digest"] == "bank"
    reopened = contracts.Entry.model_validate_json(contracts.canonical_bytes(entry))
    assert reopened.entry_id == entry.entry_id
    assert reopened.result == entry.result


class TiedSelector:
    def score(self, _task, _response, *, sandbox, limits):
        assert isinstance(sandbox, LinuxSandbox)
        assert isinstance(limits, Limits)
        return Decimal("1")


def test_ties_are_declared_and_resolved_by_lowest_digest_deterministically(tmp_path: Path) -> None:
    sandbox, limits = _contexts()
    provider = _provider(tmp_path)
    first = selection_stress(
        _bank(),
        TiedSelector(),
        provider,
        ladder=(2,),
        bootstrap_resamples=10,
        sandbox=sandbox,
        limits=limits,
    )
    second = selection_stress(
        _bank(),
        TiedSelector(),
        provider,
        ladder=(2,),
        bootstrap_resamples=10,
        sandbox=sandbox,
        limits=limits,
    )
    assert first.reading.points[0].tie_break is TieBreak.FIRST_BY_DIGEST
    assert first.reading.points[0].selected_success.value == Decimal("0.833333")
    assert second.reading.points[0].selected_success == first.reading.points[0].selected_success


def test_unknown_outcomes_are_excluded_and_a_protocol_without_a_real_qualification_is_not_a_curve(
    tmp_path: Path,
) -> None:
    provider = _provider(tmp_path)
    original = provider.resolve

    def unknown(task: TaskId, response: Response):
        if response.digest == "01":
            return unqualified(
                item_id=response.digest,
                reason=UnqualifiedReason.AMBIGUITY,
                protocol=provider.protocol,
                task_validity=TaskValidity.UNKNOWN,
            )
        return original(task, response)

    sandbox, limits = _contexts()
    result = selection_stress(
        _bank(),
        RecordedSelector(),
        OutcomeProvider(provider.check, provider.protocol, unknown),
        ladder=(1,),
        bootstrap_resamples=10,
        sandbox=sandbox,
        limits=limits,
    )
    assert result.reading.points[0].unqualified_outcomes > 0


def test_all_unqualified_actual_outcomes_produce_one_canonical_absence_entry(
    tmp_path: Path,
) -> None:
    provider = _provider(tmp_path)

    def every_outcome_unknown(_task: TaskId, response: Response):
        return unqualified(
            item_id=response.digest,
            reason=UnqualifiedReason.AMBIGUITY,
            protocol=provider.protocol,
            task_validity=TaskValidity.UNKNOWN,
        )

    sandbox, limits = _contexts()
    result = selection_stress(
        _bank(),
        RecordedSelector(),
        OutcomeProvider(provider.check, provider.protocol, every_outcome_unknown),
        ladder=(2,),
        bootstrap_resamples=10,
        sandbox=sandbox,
        limits=limits,
    )

    assert (
        result.scorecard.rows[0].value.reason
        == "not measured: no resolved selected independent outcomes"
    )
    assert len(result.entries) == 1
    entry = canonical_assay_entry(result)
    assert entry.entry_id == "reward_statistics.selection_stress"
    assert entry.kind == "absence"
    payload = entry.extensions["https://reward-lens.github.io/assay/selection-stress-absence/v1"]
    assert (
        payload["curve"][0]["selected_success_absence"]["reason"]
        == "not measured: no resolved selected independent outcomes"
    )
    assert payload["calibration"]["rho"] == "0.964286"
    assert payload["qualification"]["protocol_digest"] == provider.protocol.protocol_digest
    assert payload["source_bank"]["digest"] == "bank"
    reopened = contracts.Entry.model_validate_json(contracts.canonical_bytes(entry))
    assert reopened.extensions == entry.extensions


def test_missing_protocol_is_a_named_absence_and_budget_past_half_refuses(tmp_path: Path) -> None:
    sandbox, limits = _contexts()
    provider = _provider(tmp_path)
    absent = selection_stress(
        _bank(),
        RecordedSelector(),
        None,
        ladder=(1,),
        bootstrap_resamples=10,
        sandbox=sandbox,
        limits=limits,
    )
    assert (
        absent.scorecard.rows[0].value.reason
        == "not measured: no independent outcome check for this bank"
    )
    assert canonical_assay_entry(absent).kind == "absence"
    declared_but_unqualified = OutcomeProvider(
        check=OutcomeCheck(
            "independent", "sha256:" + "d" * 64, "fixture", "not-the-qualified-suite"
        ),
        protocol=provider.protocol,
        resolve=lambda _task, _response: pytest.fail("unqualified protocol must not run a curve"),
    )
    protocol_absent = selection_stress(
        _bank(),
        RecordedSelector(),
        declared_but_unqualified,
        ladder=(1,),
        bootstrap_resamples=10,
        sandbox=sandbox,
        limits=limits,
    )
    assert (
        protocol_absent.scorecard.rows[0].value.reason
        == "not measured: independent outcome protocol is unqualified"
    )
    foreign_same_partition = OutcomeProvider(
        check=OutcomeCheck(
            "foreign evaluator",
            "sha256:" + "e" * 64,
            "foreign-interface",
            provider.check.partition_id,
        ),
        protocol=provider.protocol,
        resolve=lambda _task, _response: pytest.fail(
            "foreign check must be refused before resolver runs"
        ),
    )
    foreign_absent = selection_stress(
        _bank(),
        RecordedSelector(),
        foreign_same_partition,
        ladder=(1,),
        bootstrap_resamples=10,
        sandbox=sandbox,
        limits=limits,
    )
    assert (
        foreign_absent.scorecard.rows[0].value.reason
        == "not measured: independent outcome protocol is unqualified"
    )
    with pytest.raises(UsageError) as caught:
        selection_stress(
            _bank(),
            RecordedSelector(),
            provider,
            ladder=(3,),
            bootstrap_resamples=10,
            sandbox=sandbox,
            limits=limits,
        )
    assert caught.value.code == "RL0132"
    assert caught.value.exit_code == 4


def test_susceptibility_refuses_when_its_conditions_are_stripped() -> None:
    calibration = Calibration(
        Decimal("0.964286"),
        Decimal("0.13"),
        "K-meta fired",
        "ev:c978e3958769ecf93dcf079771a3aedb",
        "campaign-results/RESULTS.md:99",
    )
    assert Susceptibility(Decimal("0.1"), calibration).calibration == calibration
    with pytest.raises(UsageError) as caught:
        Susceptibility(Decimal("0.1"), None)
    assert caught.value.code == "RL0133"
    assert caught.value.exit_code == 4


def test_module_has_no_trainer_or_replacement_estimator() -> None:
    import reward_lens.product.compare.rung2 as module

    source = inspect.getsource(module)
    assert "trainer" not in source.lower()
    assert "load_entrypoint" not in source
    assert "SuiteResult" not in source
    tree = ast.parse(source)
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "choices" for node in ast.walk(tree)
    )
    assert "(1 - c/n)" not in source


@pytest.mark.parametrize("mode", ["estimate", "missing", "unqualified", "unresolved"])
def test_measurement_start_precedes_execution_and_is_preserved_on_reopen(
    tmp_path, monkeypatch, mode
):
    import reward_lens.product.compare.rung2 as module

    captured = datetime(2026, 10, 5, 8, 1, 2, tzinfo=timezone.utc)
    calls = []

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            calls.append("clock")
            return captured if len(calls) == 1 else captured + timedelta(hours=2)

    monkeypatch.setattr(module, "datetime", Clock, raising=False)
    provider = _provider(tmp_path)
    original = provider.resolve

    def resolve(task, response):
        assert calls == ["clock"], "start must already be captured before resolution"
        if mode == "unresolved":
            return unqualified(
                item_id=response.digest,
                reason=UnqualifiedReason.AMBIGUITY,
                protocol=provider.protocol,
                task_validity=TaskValidity.UNKNOWN,
            )
        return original(task, response)

    if mode == "missing":
        provider = None
    elif mode == "unqualified":
        provider = OutcomeProvider(provider.check, None, resolve)
    else:
        provider = OutcomeProvider(provider.check, provider.protocol, resolve)

    class Selector(RecordedSelector):
        def score(self, *args, **kwargs):
            assert calls == ["clock"], "start must already be captured before selection"
            return super().score(*args, **kwargs)

    sandbox, limits = _contexts()
    result = selection_stress(
        _bank(),
        Selector(),
        provider,
        ladder=(2,),
        bootstrap_resamples=10,
        sandbox=sandbox,
        limits=limits,
    )
    reopened = canonical_assay_entry(result)
    assert calls == ["clock"], "entry assembly must not recapture completion as start"
    assert reopened.provenance.started == "2026-10-05T08:01:02Z"
    assert reopened.kind == ("estimate" if mode == "estimate" else "absence")
    assert contracts.canonical_bytes(reopened) == contracts.canonical_bytes(result.entries[0])


def test_identical_provenance_is_byte_deterministic_and_only_start_varies(tmp_path, monkeypatch):
    import reward_lens.product.compare.rung2 as module

    instant = datetime(2026, 10, 5, 8, 1, 2, tzinfo=timezone.utc)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return instant

    monkeypatch.setattr(module, "datetime", Clock, raising=False)
    sandbox, limits = _contexts()
    provider = _provider(tmp_path)

    def measure():
        return canonical_assay_entry(
            selection_stress(
                _bank(),
                TiedSelector(),
                provider,
                ladder=(2,),
                bootstrap_resamples=10,
                sandbox=sandbox,
                limits=limits,
            )
        )

    first, same = measure(), measure()
    assert contracts.canonical_bytes(first) == contracts.canonical_bytes(same)
    instant += timedelta(days=1)
    later = measure()
    one, two = first.to_dict(), later.to_dict()
    assert one["provenance"].pop("started") == "2026-10-05T08:01:02Z"
    assert two["provenance"].pop("started") == "2026-10-06T08:01:02Z"
    assert one == two


def test_public_rung2_freshness_expires_from_actual_start_without_relabelling(
    tmp_path, monkeypatch
):
    import reward_lens.product.compare.rung2 as module
    from reward_lens import api
    from reward_lens.acceptance import decide, load_policy

    captured = datetime(2026, 10, 5, 8, 1, 2, tzinfo=timezone.utc)

    class ProducerClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return captured

    class PolicyClock(datetime):
        instant = captured + timedelta(seconds=604800)

        @classmethod
        def now(cls, tz=None):
            return cls.instant

    monkeypatch.setattr(module, "datetime", ProducerClock, raising=False)
    monkeypatch.setattr(
        importlib.import_module("reward_lens.acceptance.decide"), "datetime", PolicyClock
    )
    helpers = runpy.run_path(str(Path(__file__).with_name("test_public_execution.py")))
    root = helpers["_project"](tmp_path / "project")
    bank, config = helpers["_write_selection_stress_inputs"](root)
    _record = api.compare(
        api.CompareRequest(
            project=root,
            bank=bank,
            selection_config=config,
            baseline="before",
            candidate="after",
            rungs=(2,),
            name="freshness",
        )
    )
    saved = root / "assays" / "freshness.assay.json"
    reopened = api.open_record(saved)
    entry = next(
        entry
        for entry in reopened.entries()
        if entry.entry_id == "reward_statistics.selection_stress"
    )
    assert entry.provenance.started == "2026-10-05T08:01:02Z"
    policy_path = root / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "id": "rung2-start-freshness",
                "version": "1.0.0",
                "actions": ["use_for_selection"],
                "default_interval": "0.95",
                "multiplicity": {"rule": "benjamini_hochberg_fdr", "family": "rung2"},
                "checks": [
                    {
                        "key": "rung2.complete",
                        "source": "entry_complete",
                        "section": "reward_statistics",
                        "entry_id": "reward_statistics.selection_stress",
                        "fresh_for_seconds": 604800,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    policy = load_policy(policy_path)
    assert decide(reopened, policy).state == "qualified"
    before = contracts.canonical_bytes(reopened)
    PolicyClock.instant += timedelta(seconds=1)
    stale = decide(reopened, policy)
    assert stale.state == "unresolved"
    assert stale.reasons == ["required_stale:rung2.complete"]
    assert contracts.canonical_bytes(api.open_record(saved)) == before
