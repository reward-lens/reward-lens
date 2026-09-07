"""Fixtures for the outcome and partition tests.

Every partition here is written to a real directory under `tmp_path`, because the separation proof
runs a real process against a real path and a fake filesystem would prove nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reward_lens.partitions import AccessLog, Partition, PartitionKind

CANARY = b"assert candidate_output == 42  # PROTECTED-CANARY\n"


def write_suite(root: Path, *, items: int = 3) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(items):
        (root / f"test_case_{index}.py").write_bytes(CANARY)
    return root


@pytest.fixture
def protected(tmp_path: Path) -> Partition:
    """A protected suite of three cases, with its access log beside it but outside its root.

    The public root is a sibling directory, not a parent and not a child: that is the layout the
    separation proof needs, and `Partition` refuses any other.
    """
    root = write_suite(tmp_path / "acceptance")
    (tmp_path / "delivery").mkdir(parents=True, exist_ok=True)
    return Partition.from_dir(
        "acceptance-suite-1",
        PartitionKind.PROTECTED_SUITE,
        root,
        log_path=tmp_path / "logs" / "acceptance-suite-1.jsonl",
        public_root=tmp_path / "delivery",
    )


@pytest.fixture
def staged(tmp_path: Path) -> Partition:
    """The control's partition: the same three cases, sitting inside a root that will be staged.

    Nothing else differs from `protected`. `Partition` will not hold this layout, because the
    separation proof's premise is that the items are outside the staged root, so the control names
    its staged root at the call instead: one variable, changed deliberately.
    """
    root = write_suite(tmp_path / "control" / "acceptance")
    return Partition.from_dir(
        "acceptance-suite-1-control",
        PartitionKind.PROTECTED_SUITE,
        root,
        log_path=tmp_path / "logs" / "acceptance-suite-1-control.jsonl",
    )


@pytest.fixture
def reference(tmp_path: Path) -> Partition:
    root = write_suite(tmp_path / "reference", items=2)
    return Partition.from_dir(
        "reference-material-1",
        PartitionKind.REFERENCE_MATERIAL,
        root,
        log_path=tmp_path / "logs" / "reference-material-1.jsonl",
    )


@pytest.fixture
def log(tmp_path: Path) -> AccessLog:
    return AccessLog(tmp_path / "logs" / "plain.jsonl", partition_id="plain")
