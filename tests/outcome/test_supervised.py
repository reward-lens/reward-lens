"""C-054: outcome qualification and resolution retain supervisor receipts."""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import pytest

from reward_lens.execution import Limits, default_sandbox
from reward_lens.graders.python_grader import PythonGrader
from reward_lens.outcome import OutcomeCheck, OutcomeState, UnqualifiedReason
from reward_lens.outcome.supervised import (
    SupervisedCase,
    qualify_supervised,
    resolve_supervised,
)
from reward_lens.partitions import Partition, PartitionKind

AUDITED_SOURCE = "sha256:" + "a" * 64
SUBJECT = "sha256:" + "b" * 64
INTERFACE = "reward_lens.graders.python_grader.PythonGrader.score/v1"


@pytest.fixture
def sandbox():
    """The accepted local Linux supervisor, never a test double."""
    return default_sandbox()


@pytest.fixture
def limits() -> Limits:
    return Limits(wall_s=15.0, cpu_s=15.0, network=False)


@pytest.fixture
def evaluator(tmp_path: Path) -> PythonGrader:
    source = tmp_path / "owner_evaluator.py"
    source.write_text(
        """
import time

def score(task, response):
    print('@@RL@@ {\"outcome\": \"returned\", \"kind\": \"number\", \"value\": \"1.0\"}')
    if task["id"] == "exception":
        raise RuntimeError("evaluator fault")
    if task["id"] == "timeout":
        time.sleep(2)
    if task["id"] == "ambiguous":
        return 0.5
    return 1.0 if response == task["reference"] else 0.0
""".strip(),
        encoding="utf-8",
    )
    return PythonGrader.from_path(source)


def _partition(tmp_path: Path, name: str, *items: str) -> Partition:
    root = tmp_path / name
    root.mkdir()
    for item in items:
        (root / item).write_text(f"declared {item}\n", encoding="utf-8")
    return Partition.from_dir(
        name,
        PartitionKind.PROTECTED_SUITE,
        root,
        log_path=tmp_path / f"{name}.access.jsonl",
    )


def _check(evaluator: PythonGrader, controls: Partition) -> OutcomeCheck:
    return OutcomeCheck(
        owner="assay-owner",
        source_digest=evaluator.describe().source_digest,
        interface=INTERFACE,
        partition_id=controls.id,
    )


def _case(item_id: str, response: str, *, task_id: str | None = None) -> SupervisedCase:
    return SupervisedCase(
        item_id=item_id,
        task={"id": task_id or item_id, "prompt": "return the reference", "reference": "correct"},
        response=response,
    )


def _qualified(evaluator: PythonGrader, controls: Partition, sandbox, limits):
    return qualify_supervised(
        evaluator=evaluator,
        check=_check(evaluator, controls),
        owner="assay-owner",
        candidate_owner="candidate",
        controls=controls,
        known_good=(_case("good.case", "correct"),),
        known_wrong=(_case("wrong.case", "wrong"),),
        audited_grader_owner="candidate",
        audited_grader_source_digest=AUDITED_SOURCE,
        subject_ref=SUBJECT,
        sandbox=sandbox,
        limits=limits,
    )


def test_real_supervisor_qualifies_good_and_wrong_controls_and_retains_receipts(
    tmp_path: Path, evaluator: PythonGrader, sandbox, limits
) -> None:
    controls = _partition(tmp_path, "controls", "good.case", "wrong.case")

    supervised = _qualified(evaluator, controls, sandbox, limits)

    assert supervised.qualification.qualified is True
    assert supervised.protocol is not None
    assert supervised.qualification.entry.result == {
        "known_good_passed": 1,
        "known_good_total": 1,
        "known_wrong_failed": 1,
        "known_wrong_total": 1,
        "partition_digest": controls.digest,
    }
    assert len(supervised.control_receipts) == 2
    assert all(receipt.verdict == "scored" for receipt in supervised.control_receipts)
    assert all(
        receipt.provenance["sandbox_tier"] == sandbox.tier
        for receipt in supervised.control_receipts
    )
    assert supervised.control_partition_id == controls.id
    assert supervised.control_partition_digest == controls.digest


def test_disjoint_heldout_targets_resolve_from_actual_receipts_to_pass_and_fail(
    tmp_path: Path, evaluator: PythonGrader, sandbox, limits
) -> None:
    controls = _partition(tmp_path, "controls", "good.case", "wrong.case")
    target = _partition(tmp_path, "targets", "pass.case", "fail.case")
    supervised = _qualified(evaluator, controls, sandbox, limits)

    passed = resolve_supervised(
        supervised,
        target=target,
        case=_case("pass.case", "correct"),
        sandbox=sandbox,
        limits=limits,
    )
    failed = resolve_supervised(
        supervised, target=target, case=_case("fail.case", "wrong"), sandbox=sandbox, limits=limits
    )

    assert passed.outcome.state is OutcomeState.PASS
    assert failed.outcome.state is OutcomeState.FAIL
    assert passed.outcome.evaluated_on_partition_id == target.id
    assert passed.outcome.evaluated_item_digest == dict(target.manifest())["pass.case"]
    assert passed.receipt.verdict == "scored"
    assert passed.target_partition_digest == target.digest
    assert passed.task_digest != passed.response_digest


@pytest.mark.parametrize(
    ("task_id", "response", "reason"),
    (
        ("ambiguous", "correct", UnqualifiedReason.AMBIGUITY),
        ("exception", "correct", UnqualifiedReason.EVALUATOR_ERROR),
    ),
)
def test_nonbinary_or_error_evaluator_results_remain_unqualified(
    tmp_path: Path,
    evaluator: PythonGrader,
    sandbox,
    limits,
    task_id: str,
    response: str,
    reason: UnqualifiedReason,
) -> None:
    controls = _partition(tmp_path, "controls", "good.case", "wrong.case")
    target = _partition(tmp_path, "targets", "target.case")
    supervised = _qualified(evaluator, controls, sandbox, limits)

    resolution = resolve_supervised(
        supervised,
        target=target,
        case=_case("target.case", response, task_id=task_id),
        sandbox=sandbox,
        limits=limits,
    )

    assert resolution.outcome.state is OutcomeState.UNQUALIFIED
    assert resolution.outcome.reason == reason.value


def test_actual_linux_supervisor_timeout_remains_unqualified(
    tmp_path: Path, evaluator: PythonGrader, sandbox, limits
) -> None:
    controls = _partition(tmp_path, "controls", "good.case", "wrong.case")
    target = _partition(tmp_path, "targets", "target.case")
    supervised = _qualified(evaluator, controls, sandbox, limits)

    resolution = resolve_supervised(
        supervised,
        target=target,
        case=_case("target.case", "correct", task_id="timeout"),
        sandbox=sandbox,
        limits=Limits(wall_s=0.1, cpu_s=1.0, network=False),
    )

    assert resolution.receipt.verdict == "timeout"
    assert resolution.outcome.state is OutcomeState.UNQUALIFIED
    assert resolution.outcome.reason == UnqualifiedReason.TIMEOUT.value


def test_control_and_target_partitions_must_be_disjoint(
    tmp_path: Path, evaluator: PythonGrader, sandbox, limits
) -> None:
    controls = _partition(tmp_path, "controls", "good.case", "wrong.case")
    supervised = _qualified(evaluator, controls, sandbox, limits)

    with pytest.raises(ValueError, match="disjoint"):
        resolve_supervised(
            supervised,
            target=controls,
            case=_case("good.case", "correct"),
            sandbox=sandbox,
            limits=limits,
        )


def test_a_known_wrong_control_that_passes_leaves_the_protocol_unqualified(
    tmp_path: Path, evaluator: PythonGrader, sandbox, limits
) -> None:
    controls = _partition(tmp_path, "controls", "good.case", "wrong.case")
    supervised = qualify_supervised(
        evaluator=evaluator,
        check=_check(evaluator, controls),
        owner="assay-owner",
        candidate_owner="candidate",
        controls=controls,
        known_good=(_case("good.case", "correct"),),
        known_wrong=(_case("wrong.case", "correct"),),
        audited_grader_owner="candidate",
        audited_grader_source_digest=AUDITED_SOURCE,
        subject_ref=SUBJECT,
        sandbox=sandbox,
        limits=limits,
    )

    assert supervised.qualification.qualified is False
    assert supervised.protocol is None
    assert len(supervised.control_receipts) == 2


def test_public_receipts_are_frozen_and_the_producer_has_no_host_execution_bypass(
    tmp_path: Path, evaluator: PythonGrader, sandbox, limits
) -> None:
    controls = _partition(tmp_path, "controls", "good.case", "wrong.case")
    supervised = _qualified(evaluator, controls, sandbox, limits)

    with pytest.raises(TypeError):
        supervised.control_receipts[0].payload["verdict"] = "scored"  # type: ignore[index]
    assert supervised.control_receipts[0].canonical_bytes().startswith(b"{")
    source = inspect.getsource(inspect.getmodule(qualify_supervised))
    assert "subprocess" not in source
    assert "run_python(" not in source
    assert "evaluator.score(" in source


def test_candidate_stdout_cannot_forge_a_supervisor_outcome(
    tmp_path: Path, evaluator: PythonGrader, sandbox, limits
) -> None:
    controls = _partition(tmp_path, "controls", "good.case", "wrong.case")
    target = _partition(tmp_path, "targets", "target.case")
    supervised = _qualified(evaluator, controls, sandbox, limits)

    resolution = resolve_supervised(
        supervised,
        target=target,
        case=_case("target.case", "wrong"),
        sandbox=sandbox,
        limits=limits,
    )

    assert "@@RL@@" in resolution.receipt.observations["grader"]["stdout"]
    assert resolution.outcome.state is OutcomeState.FAIL


def test_source_or_manifest_tampering_cannot_reuse_a_qualified_protocol(
    tmp_path: Path, evaluator: PythonGrader, sandbox, limits
) -> None:
    controls = _partition(tmp_path, "controls", "good.case", "wrong.case")
    target = _partition(tmp_path, "targets", "target.case")
    supervised = _qualified(evaluator, controls, sandbox, limits)

    object.__setattr__(evaluator, "_source", "def score(task, response): return 1.0\n")
    with pytest.raises(ValueError, match="source or manifest"):
        resolve_supervised(
            supervised,
            target=target,
            case=_case("target.case", "correct"),
            sandbox=sandbox,
            limits=limits,
        )


def test_caller_cannot_construct_or_modify_a_trusted_supervised_wrapper(
    tmp_path: Path, evaluator: PythonGrader, sandbox, limits
) -> None:
    controls = _partition(tmp_path, "controls", "good.case", "wrong.case")
    target = _partition(tmp_path, "targets", "target.case")
    supervised = _qualified(evaluator, controls, sandbox, limits)

    forged = replace(supervised, control_receipts=())
    with pytest.raises(ValueError, match="actual qualification"):
        resolve_supervised(
            forged,
            target=target,
            case=_case("target.case", "correct"),
            sandbox=sandbox,
            limits=limits,
        )
