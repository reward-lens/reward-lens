"""Partition capabilities retain the candidate and seeker boundary used by outcome resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from reward_lens.contracts.errors import UsageError
from reward_lens.partitions import (
    ALLOWED_CAPABILITIES,
    CANDIDATE_AUTHOR,
    CANDIDATE_PROCESS,
    SEEKER,
    Partition,
    PartitionKind,
)


def _protected(tmp_path: Path) -> Partition:
    root = tmp_path / "acceptance"
    root.mkdir()
    (root / "test_case.py").write_text("protected", encoding="utf-8")
    return Partition.from_dir(
        "acceptance-suite-1",
        PartitionKind.PROTECTED_SUITE,
        root,
        log_path=tmp_path / "access.jsonl",
    )


def test_candidate_capabilities_never_open_a_protected_partition(tmp_path: Path) -> None:
    protected = _protected(tmp_path)
    allowed = ALLOWED_CAPABILITIES[PartitionKind.PROTECTED_SUITE]
    assert allowed is not None
    assert CANDIDATE_AUTHOR not in allowed
    assert CANDIDATE_PROCESS not in allowed
    for capability in (CANDIDATE_AUTHOR, CANDIDATE_PROCESS):
        with pytest.raises(UsageError):
            protected.read("test_case_0.py", capability=capability, purpose="forge a success")


def test_seeker_is_not_an_allowed_partition_reader(tmp_path: Path) -> None:
    protected = _protected(tmp_path)
    assert all(
        allowed is None or SEEKER not in allowed for allowed in ALLOWED_CAPABILITIES.values()
    )
    with pytest.raises(UsageError):
        protected.read("test_case_0.py", capability=SEEKER, purpose="forge a success")
