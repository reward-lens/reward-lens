"""The separation is measured against a control, not asserted (D-40, D-37).

One sandboxed run does two reads. The canary, staged in the root the evaluator bound, must come
back; the protected path, outside every bound root, must not, and the failure must name it. The
control is the same run with the protected material moved inside the staged root: it must be read,
which is what rules out a sandbox that simply fails every open. A run that crashed measured
nothing and certifies nothing.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from reward_lens.contracts.errors import RewardLensError
from reward_lens.execution import probe
from reward_lens.outcome import (
    NON_CONFINING_TIERS,
    SeparationProof,
    attempt_protected_read,
    certify_separation,
    confines,
)
from reward_lens.outcome import separation as separation_module
from reward_lens.partitions import Partition, PartitionKind

#: A candidate-side program that dies before it opens anything, which is the shape of every run
#: that produces no observation: a crash, a kill, an interpreter that never started.
CRASHING_PROGRAM = """
raise RuntimeError("the candidate-side process died before it read anything")
"""


@pytest.fixture(scope="module")
def held_tier() -> str:
    return probe().tier_held


@pytest.fixture
def proof(protected: Partition, tmp_path: Path) -> SeparationProof:
    """The measurement itself, made once and looked at from several angles."""
    return attempt_protected_read(protected, "test_case_0.py", cwd=tmp_path / "work")


# --- the run ------------------------------------------------------------------------------------


def test_the_attempt_runs_at_the_tier_this_machine_probed(
    protected: Partition, proof: SeparationProof, held_tier: str
) -> None:
    assert isinstance(proof, SeparationProof)
    assert proof.tier == held_tier
    assert proof.sandbox_os == probe().os
    assert proof.attempted_path == str(protected.root / "test_case_0.py")
    assert proof.staged_root == str(protected.public_root)


def test_the_canary_staged_for_the_process_is_read(proof: SeparationProof) -> None:
    """Without this the run measures a sandbox that reads nothing and calls it containment."""
    assert proof.completed is True
    assert proof.canary_read is True
    assert proof.measured is True


def test_the_candidate_side_process_cannot_read_the_protected_path(
    proof: SeparationProof, held_tier: str
) -> None:
    assert proof.protected_inside_staged is False
    if confines(held_tier):
        assert proof.read_succeeded is False
        assert proof.error_name in {"FileNotFoundError", "PermissionError", "IsADirectoryError"}
        assert "PROTECTED-CANARY" not in proof.observed
        assert proof.contained is True
    else:
        assert proof.contained is False
        assert held_tier in NON_CONFINING_TIERS


def test_the_failure_names_the_path_that_was_refused(
    proof: SeparationProof, held_tier: str
) -> None:
    if confines(held_tier):
        assert proof.names_the_path is True
        assert proof.attempted_path in proof.observed


def test_the_control_reads_the_same_material_when_it_is_staged(
    staged: Partition, tmp_path: Path
) -> None:
    """One variable: the protected material sits inside the staged root and nothing else changes.

    A sandbox that failed every read would fail here too, and this is what says it did not.
    """
    proof = attempt_protected_read(
        staged,
        "test_case_0.py",
        cwd=tmp_path / "work-control",
        public_root=tmp_path / "control",
    )
    assert proof.measured is True
    assert proof.protected_inside_staged is True
    assert proof.read_succeeded is True
    assert "PROTECTED-CANARY" in proof.observed
    assert proof.contained is False


def test_the_control_and_the_measurement_differ_only_in_where_the_material_sits(
    protected: Partition, staged: Partition, proof: SeparationProof, tmp_path: Path, held_tier: str
) -> None:
    control = attempt_protected_read(
        staged,
        "test_case_0.py",
        cwd=tmp_path / "work-control",
        public_root=tmp_path / "control",
    )
    assert control.tier == proof.tier == held_tier
    assert control.sandbox_os == proof.sandbox_os
    assert Path(control.attempted_path).name == Path(proof.attempted_path).name
    assert control.canary_read == proof.canary_read is True
    assert control.protected_inside_staged is not proof.protected_inside_staged
    assert control.read_succeeded is not proof.read_succeeded


def test_the_attempt_is_logged_against_the_partition(
    protected: Partition, proof: SeparationProof
) -> None:
    event = protected.access_log.events[0]
    assert event.granted is False
    assert event.capability == "candidate-process"
    assert "separation" in event.purpose


def test_a_partition_whose_roots_overlap_cannot_be_built(tmp_path: Path) -> None:
    """The proof's premise is a property of the object: the items are outside the staged root."""
    root = tmp_path / "overlap" / "acceptance"
    root.mkdir(parents=True)
    (root / "test_case_0.py").write_bytes(b"assert True\n")
    with pytest.raises(RewardLensError) as caught:
        Partition.from_dir(
            "overlapping-1",
            PartitionKind.PROTECTED_SUITE,
            root,
            log_path=tmp_path / "logs" / "overlapping-1.jsonl",
            public_root=tmp_path / "overlap",
        )
    assert caught.value.code == "RL0001"
    assert "overlap" in caught.value.message


def test_a_partition_with_no_public_root_has_nothing_to_measure_against(
    reference: Partition, tmp_path: Path
) -> None:
    with pytest.raises(RewardLensError) as caught:
        attempt_protected_read(reference, "test_case_0.py", cwd=tmp_path / "work-none")
    assert caught.value.code == "RL0001"
    assert "public root" in caught.value.message


# --- certification ------------------------------------------------------------------------------


def test_certification_of_the_real_attempt_matches_what_the_probe_established(
    proof: SeparationProof, held_tier: str
) -> None:
    if confines(held_tier):
        record = certify_separation(proof)
        assert record["contained"] is True
        assert record["tier"] == held_tier
        assert record["mechanism"] == "process_isolation"
        assert record["canary_read"] is True
        assert record["protected_inside_staged"] is False
    else:
        with pytest.raises(RewardLensError):
            certify_separation(proof)


def test_certification_refuses_a_tier_that_confines_nothing(proof: SeparationProof) -> None:
    unconfined = replace_proof(proof, tier="T0", read_succeeded=True, error_name="", observed="")
    assert unconfined.contained is False
    with pytest.raises(RewardLensError) as caught:
        certify_separation(unconfined)
    assert caught.value.code == "RL0402"
    assert "T0" in caught.value.message


def test_certification_refuses_a_run_that_did_not_complete(proof: SeparationProof) -> None:
    crashed = replace_proof(
        proof,
        tier="L3",
        completed=False,
        canary_read=False,
        observed="",
        error_name="SandboxRunFailed",
        failure_detail="exit 1: RuntimeError",
    )
    assert crashed.contained is False
    with pytest.raises(RewardLensError) as caught:
        certify_separation(crashed)
    assert caught.value.code == "RL0401"
    assert "did not complete" in caught.value.message
    assert "nothing was measured" in caught.value.message


def test_a_candidate_script_that_crashes_certifies_nothing(
    protected: Partition, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, held_tier: str
) -> None:
    """The whole of answer 3: a real run that raises before reading anything, put to the certifier.

    Attempt 1 mapped every non-zero exit to `contained=True`, so a sandbox that never ran the
    program certified as containment. This is that run.
    """
    monkeypatch.setattr(separation_module, "PROGRAM", CRASHING_PROGRAM)
    crashed = attempt_protected_read(protected, "test_case_0.py", cwd=tmp_path / "work-crash")
    assert crashed.tier == held_tier
    assert crashed.completed is False
    assert crashed.canary_read is False
    assert crashed.measured is False
    assert crashed.contained is False
    assert "RuntimeError" in crashed.failure_detail
    with pytest.raises(RewardLensError) as caught:
        certify_separation(crashed)
    assert caught.value.code == "RL0401"
    assert "did not complete" in caught.value.message


def test_certification_refuses_a_run_whose_canary_never_came_back(proof: SeparationProof) -> None:
    blind = replace_proof(proof, tier="L3", canary_read=False)
    assert blind.contained is False
    with pytest.raises(RewardLensError) as caught:
        certify_separation(blind)
    assert caught.value.code == "RL0401"
    assert "canary" in caught.value.message
    assert proof.canary_path in caught.value.message


def test_certification_refuses_the_control(staged: Partition, tmp_path: Path) -> None:
    control = attempt_protected_read(
        staged,
        "test_case_0.py",
        cwd=tmp_path / "work-control",
        public_root=tmp_path / "control",
    )
    with pytest.raises(RewardLensError) as caught:
        certify_separation(control)
    assert caught.value.code == "RL0401"
    assert control.attempted_path in caught.value.remediation


def test_certification_refuses_a_confining_tier_that_let_the_read_through(
    proof: SeparationProof,
) -> None:
    leaked = replace_proof(
        proof, tier="L3", read_succeeded=True, error_name="", observed="PROTECTED-CANARY"
    )
    assert leaked.contained is False
    with pytest.raises(RewardLensError) as caught:
        certify_separation(leaked)
    assert caught.value.code == "RL0401"
    assert proof.attempted_path in caught.value.remediation
    assert "did not establish the separation" in caught.value.message


def test_certification_refuses_a_failure_about_some_other_path(proof: SeparationProof) -> None:
    elsewhere = replace_proof(
        proof,
        tier="L3",
        error_name="FileNotFoundError",
        observed="[Errno 2] No such file or directory: '/nowhere/else.py'",
    )
    assert elsewhere.names_the_path is False
    assert elsewhere.contained is False
    with pytest.raises(RewardLensError) as caught:
        certify_separation(elsewhere)
    assert caught.value.code == "RL0401"
    assert "does not name" in caught.value.message


def test_the_proof_states_the_real_threat_model(proof: SeparationProof) -> None:
    assert "operating-system user" in proof.caveat
    assert "process isolation" in proof.caveat
    assert "concealed" not in proof.caveat


def test_a_reference_partition_is_readable_from_the_same_sandbox(
    reference: Partition, tmp_path: Path
) -> None:
    """The proof measures the boundary, so it has to be able to observe a read that is allowed."""
    proof = attempt_protected_read(
        reference,
        "test_case_0.py",
        cwd=tmp_path / "work-reference",
        public_root=reference.root,
    )
    assert proof.read_succeeded is True
    assert "PROTECTED-CANARY" in proof.observed


def replace_proof(proof: SeparationProof, **changes: object) -> SeparationProof:
    """A copy of a real proof with named fields changed, so every case starts from a real run."""
    return dataclasses.replace(proof, **changes)  # type: ignore[arg-type]
