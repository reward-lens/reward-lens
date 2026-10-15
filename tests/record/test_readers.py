"""Secondary trace readers bind an artifact to the provenance that produced it."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from reward_lens.product.trace import open_trace
from reward_lens.product.trace.errors import RecordIncomplete, RecordIntegrity

FIXTURES = Path(__file__).parents[1] / "fixtures" / "readers"


@pytest.mark.parametrize(
    ("name", "artifact"),
    [
        ("verifiers_results", FIXTURES / "verifiers"),
        ("inspect_eval_wire", FIXTURES / "inspect" / "offline.eval"),
        ("minimal", FIXTURES / "minimal" / "records.jsonl"),
    ],
)
def test_each_supported_secondary_reader_is_reached_through_the_public_trace_route(
    name: str, artifact: Path
) -> None:
    record = open_trace(artifact, reader=name)
    assert record.reader == name
    assert tuple(record.series) == (
        "reward_recorded",
        "reward_applied",
        "advantage",
        "selection",
    )
    assert "reward" not in record.series
    assert record.integrity


def test_verifiers_fixture_is_bound_to_its_writer_provenance(tmp_path: Path) -> None:
    fixture = FIXTURES / "verifiers"
    copied = tmp_path / "verifiers"
    shutil.copytree(fixture, copied)
    record = open_trace(copied, reader="verifiers_results")
    assert record.rows
    provenance = json.loads((copied / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert provenance["writer"]["source_revision"]
    assert provenance["artifact"]["sha256"]
    (copied / "PROVENANCE.json").unlink()
    with pytest.raises(RecordIntegrity) as caught:
        open_trace(copied, reader="verifiers_results")
    assert caught.value.code == "RL0611"
    assert "PROVENANCE.json" in caught.value.message


def test_tampered_inspect_artifact_refuses_its_provenance_digest(tmp_path: Path) -> None:
    fixture = FIXTURES / "inspect"
    copied = tmp_path / "inspect"
    shutil.copytree(fixture, copied)
    with (copied / "offline.eval").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(RecordIntegrity) as caught:
        open_trace(copied / "offline.eval", reader="inspect_eval_wire")
    assert caught.value.code == "RL0611"
    assert "digest" in caught.value.message


def test_inspect_fixture_loads_header_first_through_the_public_library() -> None:
    """The real ``.eval`` is readable before the trace wrapper consumes it."""
    from inspect_ai.log import read_eval_log

    artifact = FIXTURES / "inspect" / "offline.eval"
    header = read_eval_log(artifact, header_only=True)
    complete = read_eval_log(artifact)
    assert header.status == "success"
    assert complete.status == "success"
    assert len(complete.samples or []) == 1


def test_minimal_reader_preserves_declared_groups_without_adjacency_inference() -> None:
    record = open_trace(FIXTURES / "minimal" / "records.jsonl", reader="minimal")
    assert {row.group_id for row in record.rows} == {"g-a", "g-b"}
    assert all(row.advantage is not None for row in record.rows)


def test_verifiers_reader_names_missing_measurements_with_reused_rl0610() -> None:
    record = open_trace(FIXTURES / "verifiers", reader="verifiers_results")
    with pytest.raises(RecordIncomplete) as caught:
        record.require_measurements("reward_applied")
    assert caught.value.code == "RL0610"
    assert caught.value.exit_code == 4
    assert caught.value.context["series"] == ["reward_applied"]


def test_unread_optional_frameworks_are_documented_without_advertised_readers() -> None:
    from reward_lens.product.trace.readers import discover

    names = {reader.name for reader in discover()}
    assert "verl" not in names
    assert "prime_rl" not in names
    docs = (Path(__file__).parents[2] / "docs/content/reference/trace-readers.md").read_text(
        encoding="utf-8"
    )
    assert "verl" in docs
    assert "score" in docs
    assert "prime-rl" in docs
    assert "traces.jsonl" in docs
