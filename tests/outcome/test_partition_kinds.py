"""Five kinds, each an object with a stable id, a content digest and an access log (D-40)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from reward_lens.partitions import (
    ACCEPTANCE_EVALUATOR,
    ALLOWED_CAPABILITIES,
    CANDIDATE_AUTHOR,
    SEEKER,
    AccessLog,
    Partition,
    PartitionKind,
)

DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def test_there_are_exactly_five_kinds() -> None:
    assert [kind.name for kind in PartitionKind] == [
        "PROTECTED_SUITE",
        "HELD_OUT_TASKS",
        "PANEL_LABELS",
        "REFERENCE_MATERIAL",
        "SEALED_ROUND",
    ]


@pytest.mark.parametrize("kind", list(PartitionKind))
def test_every_kind_carries_an_id_a_digest_and_an_access_log(kind: PartitionKind, tmp_path: Path) -> None:
    root = tmp_path / kind.value
    root.mkdir()
    (root / "item.txt").write_bytes(b"one item\n")
    part = Partition.from_dir(f"{kind.value}-1", kind, root, log_path=tmp_path / "log.jsonl")

    assert part.id == f"{kind.value}-1"
    assert part.kind is kind
    assert DIGEST.match(part.digest)
    assert isinstance(part.access_log, AccessLog)
    assert part.access_log.events == ()


def test_only_reference_material_is_visible_to_the_candidate_side() -> None:
    visible = {kind for kind in PartitionKind if kind.candidate_visible}
    assert visible == {PartitionKind.REFERENCE_MATERIAL}


def test_the_digest_is_over_sorted_paths_and_bytes_not_traversal_order(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    for root, order in ((first, ("z.py", "a.py")), (second, ("a.py", "z.py"))):
        root.mkdir()
        for name in order:
            (root / name).write_bytes(b"same bytes\n")
    one = Partition.from_dir("p", PartitionKind.HELD_OUT_TASKS, first, log_path=tmp_path / "1.jsonl")
    two = Partition.from_dir("p", PartitionKind.HELD_OUT_TASKS, second, log_path=tmp_path / "2.jsonl")
    assert one.digest == two.digest

    (second / "a.py").write_bytes(b"one byte more\n")
    three = Partition.from_dir("p", PartitionKind.HELD_OUT_TASKS, second, log_path=tmp_path / "3.jsonl")
    assert three.digest != one.digest


def test_the_manifest_names_every_item_with_its_own_digest(protected: Partition) -> None:
    manifest = protected.manifest()
    assert [name for name, _ in manifest] == ["test_case_0.py", "test_case_1.py", "test_case_2.py"]
    assert all(DIGEST.match(item) for _, item in manifest)


def test_a_protected_read_by_the_candidate_capability_refuses(protected: Partition) -> None:
    from reward_lens.contracts.errors import UsageError

    with pytest.raises(UsageError) as caught:
        protected.read("test_case_0.py", capability=CANDIDATE_AUTHOR, purpose="write a candidate")
    assert caught.value.code == "RL0001"
    assert caught.value.exit_code == 4
    assert CANDIDATE_AUTHOR in caught.value.message
    assert protected.id in caught.value.message


def test_the_acceptance_evaluator_may_read_what_the_candidate_may_not(protected: Partition) -> None:
    body = protected.read("test_case_0.py", capability=ACCEPTANCE_EVALUATOR, purpose="score")
    assert b"PROTECTED-CANARY" in body


def test_no_partition_field_is_named_hidden(protected: Partition) -> None:
    assert not hasattr(protected, "hidden")
    assert "hidden" not in protected.to_record()


def test_the_seeker_may_not_read_any_partition_kind_because_it_spends_a_sealed_round(
    tmp_path: Path,
) -> None:
    """8.4: the seeker capability has no row in the table on purpose, and this is the purpose.

    Reference material is the case that matters. Its row is `None`, which admits every capability
    the table governs, so a refusal that came only from the table would let the seeker through
    here. `Partition.read` refuses ahead of the table instead, on every kind.
    """
    from reward_lens.contracts.errors import UsageError

    assert all(
        row is None or SEEKER not in row for row in ALLOWED_CAPABILITIES.values()
    ), "no ALLOWED_CAPABILITIES row names SEEKER"
    assert SEEKER not in set().union(
        *(row for row in ALLOWED_CAPABILITIES.values() if row is not None)
    )

    for kind in PartitionKind:
        root = tmp_path / f"seeker-{kind.value}"
        root.mkdir()
        (root / "item.txt").write_bytes(b"one item\n")
        part = Partition.from_dir(
            f"{kind.value}-seeker", kind, root, log_path=tmp_path / f"{kind.value}.jsonl"
        )
        with pytest.raises(UsageError) as caught:
            part.read("item.txt", capability=SEEKER, purpose="attack development")
        assert caught.value.code == "RL0001"
        assert caught.value.exit_code == 4
        assert "sealed round" in caught.value.message
        assert part.access_log.events[-1].granted is False
        assert part.access_log.events[-1].capability == SEEKER


def test_the_reference_row_still_admits_every_capability_the_table_governs(
    reference: Partition,
) -> None:
    """The seeker refusal is not a change to the table: reference material stays open otherwise."""
    assert ALLOWED_CAPABILITIES[PartitionKind.REFERENCE_MATERIAL] is None
    body = reference.read("test_case_0.py", capability=CANDIDATE_AUTHOR, purpose="build against")
    assert b"PROTECTED-CANARY" in body
