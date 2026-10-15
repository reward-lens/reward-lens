"""Acceptance-policy contract tests (D-13, D-72, O-06)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from reward_lens.acceptance import (
    CheckResult,
    PolicyDigestMismatch,
    PolicyInvalid,
    RegressionCase,
    apply_and_persist,
    apply_policy,
    decide,
    load_policy,
    register_regression_runner,
    rerun_changed_regressions,
)
from reward_lens.api._record import absent_record

POLICIES = Path(__file__).parents[2] / "src/reward_lens/acceptance/policies"
GOLDEN = Path(__file__).parents[2] / "fleet/golden/wave-3/policy/absent-assay.json"


def _fresh_producer_timestamp() -> str:
    """Date a live producer fixture relative to the policy's seven-day clock."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_d72_policies_are_files_with_required_evidence_and_defaults() -> None:
    policies = [
        load_policy(POLICIES / f"{name}.json")
        for name in ("default-evaluator-v1", "default-selection-v1", "default-learning-v1")
    ]
    assert [policy.id for policy in policies] == [
        "default-evaluator-v1",
        "default-selection-v1",
        "default-learning-v1",
    ]
    assert all(policy.default_interval == Decimal("0.95") for policy in policies)
    assert all(policy.required_evidence for policy in policies)
    assert all(policy.multiplicity.rule for policy in policies)


def test_team_can_substitute_policy_by_file(tmp_path: Path) -> None:
    source = POLICIES / "default-evaluator-v1.json"
    replacement = tmp_path / "team-policy.json"
    replacement.write_bytes(source.read_bytes())
    assert load_policy(replacement).digest == load_policy(source).digest


def test_every_required_check_is_evaluated_even_after_rejection() -> None:
    policy = load_policy(POLICIES / "default-evaluator-v1.json")
    record = absent_record(command="audit", reproduce=["reward-lens audit"])
    decision = decide(record, policy)
    assert set(decision.evaluated) == set(policy.required_evidence)
    assert decision.state == "rejected"
    assert any(result.state == "missing" for result in decision.evaluated.values())


@pytest.mark.parametrize("state", ["missing", "stale", "invalid", "inconclusive"])
def test_unusable_required_evidence_is_unresolved(state: str) -> None:
    policy = load_policy(POLICIES / "default-evaluator-v1.json")
    record = absent_record(command="audit", reproduce=["reward-lens audit"])
    overrides = {key: CheckResult("qualified", "control") for key in policy.required_evidence}
    overrides[policy.required_evidence[0]] = CheckResult(state, state)
    decision = decide(record, policy, overrides=overrides)
    assert decision.state == "unresolved"
    assert decision.evaluated[policy.required_evidence[0]].state == state


def test_policy_digest_mismatch_refuses() -> None:
    policy = load_policy(POLICIES / "default-evaluator-v1.json")
    record = absent_record(command="audit", reproduce=["reward-lens audit"])
    with pytest.raises(PolicyDigestMismatch) as raised:
        decide(record, policy, recorded_policy_digest="sha256:" + "0" * 64)
    assert raised.value.code == "RL0311"
    assert raised.value.exit_code == 4


def test_invalid_policy_refuses_with_reserved_code(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not json", encoding="utf-8")
    with pytest.raises(PolicyInvalid) as raised:
        load_policy(invalid)
    assert raised.value.code == "RL0310"
    assert raised.value.exit_code == 4


def test_policy_with_an_unknown_evidence_key_refuses(tmp_path: Path) -> None:
    payload = json.loads((POLICIES / "default-evaluator-v1.json").read_text(encoding="utf-8"))
    payload["checks"][0]["section"] = "not_a_record_section"
    invalid = tmp_path / "unknown-evidence.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PolicyInvalid) as raised:
        load_policy(invalid)
    assert raised.value.code == "RL0310"


def test_policy_check_must_match_its_registered_canonical_producer(tmp_path: Path) -> None:
    """A section-shaped spelling is insufficient: the producer binding is exact."""
    payload = json.loads((POLICIES / "default-selection-v1.json").read_text(encoding="utf-8"))
    payload["checks"][0]["entry_id"] = "reward_statistics.looks_plausible"
    invalid = tmp_path / "nearby-producer.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PolicyInvalid, match="registered producer"):
        load_policy(invalid)


def test_retained_regression_reruns_only_when_dependencies_change() -> None:
    case = RegressionCase(
        id="regression.signal",
        from_version="1.0.0",
        evidence_ref="signal.contrast_fraction",
        depends_on=("digest:samples",),
    )
    assert not case.needs_rerun({"digest:source"})
    assert case.needs_rerun({"digest:samples"})


def test_retained_regression_reacts_to_actual_saved_dependency_digest() -> None:
    from reward_lens.contracts import Assay

    policy = load_policy(POLICIES / "default-evaluator-v1.json")
    previous = absent_record(command="audit", reproduce=["reward-lens audit"])
    previous.subject.digests.samples = "sha256:" + "1" * 64
    case = RegressionCase(
        "regression.actual-samples", policy.version, "test.actual-samples", ("digest:samples",)
    )
    apply_policy(previous, policy, regression_cases=(case,))
    saved = Assay.model_validate(previous.to_dict())
    candidate = Assay.model_validate(saved.to_dict())
    candidate.subject.digests.samples = "sha256:" + "2" * 64
    register_regression_runner(
        "test.actual-samples", lambda rerun_case, record: CheckResult("qualified", rerun_case.id)
    )
    assert rerun_changed_regressions(saved, candidate) == {
        "regression.actual-samples": CheckResult("qualified", "regression.actual-samples")
    }


def test_apply_policy_attaches_retained_decision_and_round_trips_through_assay() -> None:
    from reward_lens.contracts import Assay

    policy = load_policy(POLICIES / "default-evaluator-v1.json")
    record = absent_record(command="audit", reproduce=["reward-lens audit"])
    record.subject.digests.samples = "sha256:" + "1" * 64
    decision = apply_policy(
        record,
        policy,
        regression_cases=(
            RegressionCase(
                id="regression.signal",
                from_version=policy.version,
                evidence_ref="signal.contrast_fraction",
                depends_on=("digest:samples",),
            ),
        ),
    )
    reloaded = Assay.model_validate(record.to_dict())
    assert decision.policy_digest == policy.digest
    assert reloaded.decision.policy.id == policy.id
    assert reloaded.decision.policy.digest == policy.digest
    assert reloaded.decision.regression_cases == ["regression.signal"]


def test_public_finalizer_uses_audit_writer_and_reopens_saved_assay(tmp_path: Path) -> None:
    """The exported API, rather than a test assignment, owns save/reopen."""
    policy = load_policy(POLICIES / "default-evaluator-v1.json")
    record = absent_record(command="audit", reproduce=["reward-lens audit"])
    record.subject.digests.samples = "sha256:" + "4" * 64
    persisted = apply_and_persist(
        record,
        policy,
        project=None,
        name="policy-public-finalizer",
        fallback_dir=tmp_path,
        regression_cases=(
            RegressionCase(
                "regression.persisted-samples",
                policy.version,
                "test.persisted-samples",
                ("digest:samples",),
            ),
        ),
    )
    assert persisted.paths["assay"].is_file()
    assert persisted.paths["manifest"].is_file()
    assert persisted.paths["report"].is_file()
    assert persisted.record.decision.policy.digest == policy.digest
    assert (
        persisted.record.extensions["https://reward-lens.org/acceptance-policy/v1"][
            "policy_version"
        ]
        == policy.version
    )
    assert persisted.record.extensions["https://reward-lens.org/acceptance-policy/v1"][
        "regression_cases"
    ] == [
        {
            "id": "regression.persisted-samples",
            "from_version": policy.version,
            "evidence_ref": "test.persisted-samples",
            "depends_on": ["digest:samples"],
            "dependency_digests": {"digest:samples": "sha256:" + "4" * 64},
        }
    ]


def test_actual_rung2_entry_drives_saved_selection_policy_without_a_synthetic_entry(
    tmp_path: Path,
) -> None:
    """Rung 2's actual engine entry, not a policy-built dict, survives finalization."""
    from decimal import Decimal

    from reward_lens import contracts
    from reward_lens.bank import BankProvenance, Response, ResponseBank, TaskId
    from reward_lens.contracts import Assay, EntryProvenance
    from reward_lens.execution.limits import Limits
    from reward_lens.execution.linux import LinuxSandbox
    from reward_lens.outcome import (
        OutcomeCheck,
        TaskValidity,
        qualified_protocol,
        qualify,
        resolved_fail,
        resolved_pass,
    )
    from reward_lens.outcome.qualify import SuiteResult
    from reward_lens.partitions import Partition, PartitionKind
    from reward_lens.product.compare.rung2 import (
        OutcomeProvider,
        canonical_assay_entry,
        selection_stress,
    )

    class RecordedSelector:
        def score(self, _task, response, *, sandbox, limits):
            return {"correct-low": Decimal("1"), "wrong-high": Decimal("2")}[response]

    task = TaskId("policy-task")
    bank = ResponseBank(
        id="policy-bank",
        digest="policy-bank",
        provenance=BankProvenance("fixture", None, None, "source"),
        tasks=(task,),
        responses={
            task: (
                Response("01", task, "correct-low", {"reward": Decimal("1")}),
                Response("02", task, "wrong-high", {"reward": Decimal("2")}),
            )
        },
        task_payloads={task: {"id": "policy-task"}},
    )
    qualified_root = tmp_path / "qualified"
    qualified_root.mkdir()
    (qualified_root / "task").write_text("control", encoding="utf-8")
    suite = Partition.from_dir(
        "policy-qualified",
        PartitionKind.PROTECTED_SUITE,
        qualified_root,
        log_path=tmp_path / "access.jsonl",
    )
    provenance = EntryProvenance(
        started=_fresh_producer_timestamp(), duration_s=0.0, sandbox_tier="T0", offline=True
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

    def resolve(_task, response):
        outcome = resolved_pass if response.text == "correct-low" else resolved_fail
        return outcome(
            item_id=response.digest,
            reason="actual independent outcome",
            protocol=protocol,
            task_validity=TaskValidity.VALID,
        )

    produced = selection_stress(
        bank,
        RecordedSelector(),
        OutcomeProvider(check=check, protocol=protocol, resolve=resolve),
        ladder=(1,),
        bootstrap_resamples=4,
        sandbox=LinuxSandbox(tier="T0"),
        limits=Limits(wall_s=5, cpu_s=5, network=False),
    )
    entry = canonical_assay_entry(produced)
    base = absent_record(command="audit", reproduce=["reward-lens audit"])
    payload = base.to_dict()
    payload["measurement"]["reward_statistics"] = [
        entry.model_dump(mode="json", by_alias=True, exclude_unset=True)
    ]
    payload["holes"] = [hole for hole in payload["holes"] if hole["section"] != "reward_statistics"]
    payload["assay_id"] = contracts.digest(payload)
    canonical = Assay.model_validate(payload)
    policy = load_policy(POLICIES / "default-selection-v1.json")
    persisted = apply_and_persist(
        canonical, policy, project=None, name="actual-rung2-policy", fallback_dir=tmp_path
    )
    assert (
        persisted.record.measurement.reward_statistics[0].result["calibration"]
        == entry.result["calibration"]
    )
    assert persisted.decision.evaluated["rung2.complete"].state == "qualified"
    assert (
        persisted.decision.evaluated["susceptibility.calibration_conditions"].state == "qualified"
    )
    assert persisted.decision.evaluated["capability_floor.held"].state == "missing"
    assert persisted.decision.state == "unresolved"


def test_actual_capability_floor_true_false_and_unknown_survive_policy_persistence(
    tmp_path: Path,
) -> None:
    """Consume only P-SCORECARD's typed entry and preserve its measured/absence distinction."""
    from decimal import Decimal

    from reward_lens import contracts
    from reward_lens.contracts import Assay, EntryProvenance, ProjectConfig
    from reward_lens.outcome import (
        OutcomeCheck,
        SuiteResult,
        TaskValidity,
        UnqualifiedReason,
        qualified_protocol,
        qualify,
        resolved_fail,
        resolved_pass,
        unqualified,
    )
    from reward_lens.partitions import Partition, PartitionKind
    from reward_lens.product.compare.scorecard import capability_floor_assay_entry

    root = tmp_path / "floor-project"
    held_out_root = root / "heldout"
    held_out_root.mkdir(parents=True)
    for index in range(4):
        (held_out_root / f"task-{index}.txt").write_text("held-out task", encoding="utf-8")
    control_root = root / "controls"
    control_root.mkdir()
    (control_root / "control.py").write_text("qualification control", encoding="utf-8")
    target = Partition.from_dir(
        "heldout", PartitionKind.HELD_OUT_TASKS, held_out_root, log_path=root / "heldout.jsonl"
    )
    controls = Partition.from_dir(
        "controls", PartitionKind.PROTECTED_SUITE, control_root, log_path=root / "controls.jsonl"
    )
    project = ProjectConfig.model_validate(
        {
            "reward": {"entry": "grader.py"},
            "tasks": {"path": "tasks.jsonl", "prompt": "prompt"},
            "outcome": {
                "kind": "protected_test_suite",
                "path": "outcome-check",
                "held_out_capability": {"partition_id": "heldout", "path": "heldout"},
            },
        }
    )
    provenance = EntryProvenance(
        started=_fresh_producer_timestamp(), duration_s=0.25, sandbox_tier="L3", offline=True
    )

    def measured_rows(attempt: EntryProvenance):
        protocol = qualified_protocol(
            qualify(
                check=OutcomeCheck(
                    "independent evaluator", "sha256:" + "c" * 64, "pytest", controls.id
                ),
                suite=controls,
                grader_owner="audited grader",
                grader_source_digest="sha256:" + "b" * 64,
                subject_ref="sha256:" + "a" * 64,
                provenance=attempt,
                result=SuiteResult(1, 1, 1, 1),
            )
        )
        rows = tuple(
            (resolved_pass if index < 3 else resolved_fail)(
                item_id=f"task-{index}.txt",
                reason="actual held-out outcome",
                protocol=protocol,
                task_validity=TaskValidity.VALID,
                evaluation_partition=target,
            )
            for index in range(4)
        )
        return protocol, rows

    protocol, outcomes = measured_rows(provenance)

    def persist_floor(
        *, floor: str, rows, row_protocol=protocol, drop_check: bool = False
    ) -> tuple[object, object]:
        base = absent_record(command="audit", reproduce=["reward-lens audit"])
        entry = capability_floor_assay_entry(
            project=project,
            project_root=root,
            partition=target,
            outcomes=rows,
            protocol=row_protocol,
            floor=Decimal(floor),
            subject_ref=base.subject.version.digest,
            provenance=row_protocol.provenance,
            scope="evaluator_comparison",
        )
        if drop_check:
            entry = entry.model_copy(update={"check": None})
        payload = base.to_dict()
        payload["measurement"]["soundness"] = [
            entry.model_dump(mode="json", by_alias=True, exclude_unset=True)
        ]
        payload["holes"] = [hole for hole in payload["holes"] if hole["section"] != "soundness"]
        payload["assay_id"] = contracts.digest(payload)
        saved = apply_and_persist(
            Assay.model_validate(payload),
            load_policy(POLICIES / "default-selection-v1.json"),
            project=None,
            name=f"floor-{floor}",
            fallback_dir=tmp_path,
        )
        return entry, saved

    true_entry, true_saved = persist_floor(floor="0.700000", rows=outcomes)
    assert true_entry.result["type"] == "held_out_capability_floor/v1"
    assert true_entry.result["target"]["partition_digest"] == target.digest
    assert true_saved.decision.evaluated["capability_floor.held"].state == "qualified"
    assert true_saved.decision.state == "unresolved"

    stale_protocol, stale_outcomes = measured_rows(
        EntryProvenance(
            started="2000-01-01T00:00:00Z",
            duration_s=0.25,
            sandbox_tier="L3",
            offline=True,
        )
    )
    _, stale_saved = persist_floor(
        floor="0.700000", rows=stale_outcomes, row_protocol=stale_protocol
    )
    assert stale_saved.decision.evaluated["capability_floor.held"].state == "stale"
    assert stale_saved.decision.state == "unresolved"

    false_entry, false_saved = persist_floor(floor="0.800000", rows=outcomes)
    assert false_entry.check is not None and false_entry.check.passed is False
    assert false_saved.decision.evaluated["capability_floor.held"].state == "rejected"
    assert false_saved.decision.state == "rejected"

    unchecked_entry, unchecked_saved = persist_floor(
        floor="0.700000", rows=outcomes, drop_check=True
    )
    assert unchecked_entry.check is None
    assert unchecked_saved.decision.evaluated["capability_floor.held"].state == "missing"
    assert unchecked_saved.decision.state == "unresolved"

    unknown = tuple(
        unqualified(
            item_id=row.item_id,
            reason=UnqualifiedReason.TIMEOUT,
            protocol=row._protocol,
            task_validity=TaskValidity.UNKNOWN,
            evaluation_partition=target,
        )
        for row in outcomes
    )
    unknown_entry, unknown_saved = persist_floor(floor="0.700000", rows=unknown)
    assert unknown_entry.kind == "absence"
    assert unknown_saved.decision.evaluated["capability_floor.held"].state == "missing"
    assert unknown_saved.decision.state == "unresolved"


def test_actual_rung5_controls_drive_learning_policy_but_not_capability_floor(
    tmp_path: Path,
) -> None:
    """The R5 engine derives useful-task controls; policy only consumes that retained evidence."""
    import json

    import torch

    from reward_lens import contracts
    from reward_lens.learning.contracts import EIGHT_OBJECTS, TrainerProfile
    from reward_lens.learning.intervention.clamp import (
        ClampSpecification,
        ShortForkArm,
        build_clamp_callback,
        run_short_fork,
    )
    from reward_lens.product.compare.rung5 import (
        Rung5Input,
        append_short_fork_assay,
        compare_verified_short_fork,
        write_cpu_conformance_record,
    )

    base = absent_record(command="audit", reproduce=["reward-lens audit"])
    source_fixture = Path(__file__).parents[1] / "fixtures" / "learning" / "cpu_toy" / "fork"
    fixture = tmp_path / "bound-r5-fixture"
    fixture.mkdir()
    inputs = json.loads((source_fixture / "fork.json").read_text(encoding="utf-8"))
    inputs["subject_ref"] = base.subject.version.digest
    (fixture / "fork.json").write_text(json.dumps(inputs), encoding="utf-8")
    profile = TrainerProfile("1.10.0", EIGHT_OBJECTS, "cpu-toy-fork-v2")
    model = torch.nn.Linear(1, 1, bias=False)
    callback = build_clamp_callback(
        profile,
        ClampSpecification(
            target_name=inputs["target"]["name"],
            target=lambda _current: model.weight.mean(),
            target_value=0.5,
            tolerance=1e-6,
            max_correction=2.0,
            max_iterations=4,
        ),
    )
    result = run_short_fork(
        (
            ShortForkArm(
                "null", lambda: None, lambda: 0.5, {inputs["useful_task"]["name"]: lambda: 0.8}
            ),
            ShortForkArm(
                "null", lambda: None, lambda: 0.5, {inputs["useful_task"]["name"]: lambda: 0.7}
            ),
            ShortForkArm(
                "candidate",
                lambda: None,
                lambda: 0.5,
                {inputs["useful_task"]["name"]: lambda: 0.75},
            ),
            ShortForkArm(
                "placebo", lambda: None, lambda: 0.5, {inputs["useful_task"]["name"]: lambda: 0.76}
            ),
        ),
        profile=profile,
        clamp=callback,
        fixture_root=fixture,
        limits=inputs["limits"],
    )
    retained = tmp_path / "actual-r5.json"
    write_cpu_conformance_record(result, fixture_root=fixture, path=retained)
    comparison = compare_verified_short_fork(
        Rung5Input(
            result=result,
            record_path=retained,
            fixture_root=fixture,
            subject_ref=base.subject.version.digest,
            provenance=contracts.EntryProvenance(
                started=_fresh_producer_timestamp(), duration_s=0.0, sandbox_tier="T0", offline=True
            ),
        )
    )
    policy = load_policy(POLICIES / "default-learning-v1.json")
    persisted = apply_and_persist(
        append_short_fork_assay(base, comparison),
        policy,
        project=None,
        name="actual-rung5-policy",
        fallback_dir=tmp_path,
    )
    assert persisted.decision.evaluated["rung4_or_5.complete"].state == "qualified"
    assert persisted.decision.evaluated["null_replay.passed"].state == "qualified"
    assert persisted.decision.evaluated["placebo.passed"].state == "qualified"
    assert persisted.decision.evaluated["capability_floor.held"].state == "missing"
    assert persisted.decision.state == "unresolved"


def test_absent_assay_golden_is_derived_from_policy_files_and_producer_entries() -> None:
    record = absent_record(command="audit", reproduce=["reward-lens audit"])
    actual = {}
    for path in sorted(POLICIES.glob("*.json")):
        policy = load_policy(path)
        decision = decide(record, policy)
        actual[policy.id] = {
            "policy_id": decision.policy_id,
            "policy_digest": decision.policy_digest,
            "state": decision.state,
            "action": decision.action,
            "evaluated": {
                key: {"state": result.state, "reason": result.reason}
                for key, result in decision.evaluated.items()
            },
            "reasons": decision.reasons,
        }
    assert actual == json.loads(GOLDEN.read_text(encoding="utf-8"))
