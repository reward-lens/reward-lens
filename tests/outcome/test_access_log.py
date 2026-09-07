"""Every read of a partition is appended to its access log at the read, and it survives a restart."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reward_lens.contracts.errors import UsageError
from reward_lens.partitions import (
    ACCEPTANCE_EVALUATOR,
    CANDIDATE_AUTHOR,
    AccessLog,
    Partition,
    PartitionKind,
)


def test_a_granted_read_is_on_disk_before_the_bytes_come_back(protected: Partition) -> None:
    path = protected.access_log.path
    assert not path.exists()
    protected.read("test_case_0.py", capability=ACCEPTANCE_EVALUATOR, purpose="score")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["item_id"] == "test_case_0.py"
    assert rows[0]["granted"] is True
    assert rows[0]["capability"] == ACCEPTANCE_EVALUATOR
    assert rows[0]["purpose"] == "score"


def test_the_event_is_written_even_when_the_read_itself_raises(protected: Partition) -> None:
    with pytest.raises(UsageError):
        protected.read("no_such_item.py", capability=ACCEPTANCE_EVALUATOR, purpose="score")
    events = protected.access_log.events
    assert len(events) == 1
    assert events[0].granted is False
    assert "no_such_item.py" in events[0].reason


def test_a_refused_read_is_logged_with_the_capability_that_asked(protected: Partition) -> None:
    with pytest.raises(UsageError):
        protected.read("test_case_1.py", capability=CANDIDATE_AUTHOR, purpose="repair")
    event = protected.access_log.events[-1]
    assert event.granted is False
    assert event.capability == CANDIDATE_AUTHOR
    assert event.purpose == "repair"
    assert event.item_id == "test_case_1.py"


def test_the_use_count_rises_with_each_granted_read(protected: Partition) -> None:
    for _ in range(3):
        protected.read("test_case_2.py", capability=ACCEPTANCE_EVALUATOR, purpose="score")
    assert protected.access_log.uses("test_case_2.py") == 3
    assert [event.use_count for event in protected.access_log.events] == [1, 2, 3]


def test_the_log_survives_a_restart_and_is_never_rewritten(protected: Partition, tmp_path: Path) -> None:
    protected.read("test_case_0.py", capability=ACCEPTANCE_EVALUATOR, purpose="score")
    reopened = AccessLog(protected.access_log.path, partition_id=protected.id)
    assert len(reopened.events) == 1
    reopened.record(
        item_id="test_case_1.py", capability=ACCEPTANCE_EVALUATOR, purpose="score", granted=True
    )
    assert len(AccessLog(protected.access_log.path, partition_id=protected.id).events) == 2


def test_the_log_has_no_way_to_drop_an_event(log: AccessLog) -> None:
    log.record(item_id="x", capability=ACCEPTANCE_EVALUATOR, purpose="score", granted=True)
    assert isinstance(log.events, tuple)
    for verb in ("remove", "pop", "clear", "truncate", "delete"):
        assert not hasattr(log, verb)


def test_a_reference_partition_logs_its_reads_too(reference: Partition) -> None:
    reference.read("test_case_0.py", capability=CANDIDATE_AUTHOR, purpose="calibrate")
    assert reference.access_log.events[-1].granted is True
    assert reference.access_log.events[-1].capability == CANDIDATE_AUTHOR


def test_the_event_carries_a_utc_timestamp_the_record_accepts(protected: Partition) -> None:
    from reward_lens.contracts.models import Timestamp  # noqa: F401
    import re

    protected.read("test_case_0.py", capability=ACCEPTANCE_EVALUATOR, purpose="score")
    at = protected.access_log.events[-1].at
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", at)


def test_partition_kind_is_on_every_event(protected: Partition) -> None:
    protected.read("test_case_0.py", capability=ACCEPTANCE_EVALUATOR, purpose="score")
    assert protected.access_log.events[-1].partition_id == protected.id
    assert PartitionKind(protected.kind) is PartitionKind.PROTECTED_SUITE
