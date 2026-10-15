"""Named values emitted at actual protected-read boundary outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reward_lens.outcome import attempt_protected_read, certify_separation
from reward_lens.partitions import AccessLog, Partition, PartitionKind
from reward_lens.partitions.access import AccessEventCode


def _partition(
    tmp_path: Path, name: str, kind: PartitionKind, *, public_root: Path | None = None
) -> Partition:
    root = tmp_path / name
    root.mkdir()
    (root / "case.py").write_text("PROTECTED-CANARY\n", encoding="utf-8")
    return Partition.from_dir(
        name, kind, root, log_path=tmp_path / f"{name}.jsonl", public_root=public_root
    )


def test_actual_contained_read_persists_the_named_denial(tmp_path: Path) -> None:
    public = tmp_path / "candidate-visible"
    public.mkdir()
    protected = _partition(
        tmp_path, "acceptance", PartitionKind.PROTECTED_SUITE, public_root=public
    )
    proof = attempt_protected_read(protected, "case.py", cwd=tmp_path / "work")
    if not proof.contained:
        pytest.skip(f"this host did not establish an actual contained read: {proof.to_record()!r}")
    certified = certify_separation(proof)
    event = protected.access_log.events[-1]
    assert event.granted is False
    assert event.code is AccessEventCode.ACCEPTANCE_ACCESS_DENIED
    assert certified["access_code"] == AccessEventCode.ACCEPTANCE_ACCESS_DENIED.value
    persisted = json.loads(protected.access_log.path.read_text(encoding="utf-8").splitlines()[-1])
    assert persisted["code"] == "ACCEPTANCE_ACCESS_DENIED"
    reloaded = AccessLog(protected.access_log.path, partition_id=protected.id).events[-1]
    assert reloaded.code is AccessEventCode.ACCEPTANCE_ACCESS_DENIED


def test_actual_allowed_read_uses_a_distinct_named_access_outcome(tmp_path: Path) -> None:
    reference = _partition(tmp_path, "reference", PartitionKind.REFERENCE_MATERIAL)
    proof = attempt_protected_read(
        reference, "case.py", cwd=tmp_path / "work", public_root=reference.root
    )
    assert proof.read_succeeded is True
    event = reference.access_log.events[-1]
    assert event.granted is True
    assert event.code is AccessEventCode.ACCEPTANCE_ACCESS_ALLOWED
    assert event.code is not AccessEventCode.ACCEPTANCE_ACCESS_DENIED


def test_legacy_jsonl_event_without_a_code_remains_readable(tmp_path: Path) -> None:
    path = tmp_path / "legacy.jsonl"
    path.write_text(
        json.dumps(
            {
                "at": "2026-09-29T00:00:00Z",
                "partition_id": "legacy",
                "item_id": "case.py",
                "capability": "candidate-process",
                "purpose": "old record",
                "granted": False,
                "reason": "old denied event",
                "use_count": 0,
                "action": "read",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    event = AccessLog(path, partition_id="legacy").events[0]
    assert event.code is None
