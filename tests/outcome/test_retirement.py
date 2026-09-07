"""A disclosed acceptance item is retired, refused afterwards, and the partition's digest moves."""

from __future__ import annotations

import pytest

from reward_lens.contracts.errors import UsageError
from reward_lens.partitions import ACCEPTANCE_EVALUATOR, Partition, PartitionKind


def test_a_disclosed_item_is_retired_and_the_digest_changes(protected: Partition) -> None:
    before = protected.digest
    after = protected.retire("test_case_0.py", disclosed_to="repair-agent", reason="shown in a diff")
    assert after != before
    assert protected.digest == after
    assert protected.retired == frozenset({"test_case_0.py"})


def test_a_retired_item_is_refused_on_reuse(protected: Partition) -> None:
    protected.retire("test_case_0.py", disclosed_to="repair-agent", reason="shown in a diff")
    with pytest.raises(UsageError) as caught:
        protected.read("test_case_0.py", capability=ACCEPTANCE_EVALUATOR, purpose="score")
    assert caught.value.code == "RL0001"
    assert caught.value.exit_code == 4
    assert "retired" in caught.value.message
    assert "test_case_0.py" in caught.value.message


def test_the_rest_of_the_partition_is_still_usable(protected: Partition) -> None:
    protected.retire("test_case_0.py", disclosed_to="repair-agent", reason="shown in a diff")
    assert protected.read("test_case_1.py", capability=ACCEPTANCE_EVALUATOR, purpose="score")
    assert [name for name, _ in protected.manifest()] == ["test_case_1.py", "test_case_2.py"]


def test_the_retirement_is_in_the_access_log_with_who_saw_it(protected: Partition) -> None:
    protected.retire("test_case_0.py", disclosed_to="repair-agent", reason="shown in a diff")
    event = protected.access_log.events[-1]
    assert event.item_id == "test_case_0.py"
    assert event.granted is False
    assert event.capability == "repair-agent"
    assert "retired" in event.purpose
    assert "shown in a diff" in event.reason


def test_retirement_survives_a_restart(protected: Partition) -> None:
    protected.retire("test_case_0.py", disclosed_to="repair-agent", reason="shown in a diff")
    reopened = Partition.from_dir(
        protected.id,
        PartitionKind.PROTECTED_SUITE,
        protected.root,
        log_path=protected.access_log.path,
    )
    assert reopened.retired == frozenset({"test_case_0.py"})
    assert reopened.digest == protected.digest
    with pytest.raises(UsageError):
        reopened.read("test_case_0.py", capability=ACCEPTANCE_EVALUATOR, purpose="score")


def test_retiring_twice_refuses_rather_than_logging_a_second_disclosure(protected: Partition) -> None:
    protected.retire("test_case_0.py", disclosed_to="repair-agent", reason="shown in a diff")
    with pytest.raises(UsageError) as caught:
        protected.retire("test_case_0.py", disclosed_to="repair-agent", reason="again")
    assert "already retired" in caught.value.message


def test_retiring_an_unknown_item_refuses(protected: Partition) -> None:
    with pytest.raises(UsageError) as caught:
        protected.retire("no_such_item.py", disclosed_to="repair-agent", reason="typo")
    assert "no_such_item.py" in caught.value.message


def test_an_empty_partition_is_an_empty_partition_not_a_zero(protected: Partition) -> None:
    for name in ("test_case_0.py", "test_case_1.py", "test_case_2.py"):
        protected.retire(name, disclosed_to="repair-agent", reason="shown in a diff")
    assert protected.manifest() == ()
    assert protected.is_empty is True


def test_retire_takes_the_item_alone_as_the_interface_freezes_it(protected: Partition) -> None:
    """Section 4 freezes `retire(item)`; section 8.4 makes the other two keyword-only and optional.

    A caller who knows only that an item leaked has to be able to retire it. Requiring a recipient
    and a cause meant the honest answer "it is out and I do not know to whom" left the item in the
    set, still being scored, still not measuring anything.
    """
    before = protected.digest
    after = protected.retire("test_case_0.py")
    assert after != before
    assert protected.retired == frozenset({"test_case_0.py"})
    event = protected.access_log.events[-1]
    assert event.action == "retire"
    assert event.capability == "unstated"
    assert event.reason == "retired with neither the recipient nor the cause stated"


def test_what_the_caller_does_state_is_what_the_log_carries(protected: Partition) -> None:
    protected.retire("test_case_0.py", disclosed_to="repair-agent")
    protected.retire("test_case_1.py", reason="pasted into a public issue")
    protected.retire("test_case_2.py", disclosed_to="a reviewer", reason="quoted in a review")
    reasons = [event.reason for event in protected.access_log.events if event.action == "retire"]
    assert reasons == [
        "disclosed to repair-agent, with no cause stated",
        "retired with no recipient stated: pasted into a public issue",
        "disclosed to a reviewer: quoted in a review",
    ]
    capabilities = [
        event.capability for event in protected.access_log.events if event.action == "retire"
    ]
    assert capabilities == ["repair-agent", "unstated", "a reviewer"]


def test_a_retirement_with_nothing_stated_still_replays_across_a_restart(
    protected: Partition, tmp_path
) -> None:
    protected.retire("test_case_0.py")
    reopened = Partition.from_dir(
        protected.id,
        PartitionKind.PROTECTED_SUITE,
        protected.root,
        log_path=protected.access_log.path,
        public_root=protected.public_root,
    )
    assert reopened.retired == frozenset({"test_case_0.py"})
    with pytest.raises(UsageError):
        reopened.read("test_case_0.py", capability=ACCEPTANCE_EVALUATOR, purpose="score")
