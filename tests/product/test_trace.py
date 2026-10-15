"""P-TRACE-1: the trace spine, the primary readers and the reader registry.

Every acceptance row of the packet is a test here, and every refusal names its code, its exit
status and the field the message has to carry.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import rfc8785

from reward_lens.contracts import validate_record
from reward_lens.core.extras import ExtraRequiredError
from reward_lens.core.types import Component
from reward_lens.product.trace import (
    RULES,
    SERIES_NAMES,
    CheckInput,
    Chunk,
    Reader,
    RecordIncomplete,
    RecordIntegrity,
    Series,
    TraceRecord,
    TraceRow,
    discover,
    follow,
    reader_for,
    run,
    seal_chunk,
)
from reward_lens.product.trace.record import RESPONSE_TEXT_CODEC, response_text_identity
from reward_lens.product.trace.run_record import (
    _declared_weights,
    _response_identity,
    _series,
    _subject_ref,
)
from reward_lens.record import scores as score_tree
from reward_lens.record.convert import trl_parquet
from reward_lens.record.reader import open_run
from reward_lens.record.schema import ComponentRef, decode_foreign, encode_foreign

REPO = Path(__file__).resolve().parents[2]
SHORT = REPO / "tests" / "fixtures" / "grpo_run" / "short"
LONG = REPO / "tests" / "fixtures" / "grpo_run" / "long"
SHORT_ID = "run:8a8c7e29274db0a681313b48dbd1eb63"
LONG_ID = "run:f77bf75940ab982bbc35407af99cc094"
WAVE2_TRACE_GOLDEN = REPO / "fleet" / "golden" / "wave-2" / "trace"
CURRENT_TRACE_GOLDEN = REPO / "fleet" / "golden" / "current" / "trace-fixture.txt"

#: The one place this test file spells the forbidden name, so the grep below can exempt itself.
FORBIDDEN = "rew" + "ard"


# --- the registry and the protocol ---------------------------------------------------------------


def test_discover_finds_both_primary_readers_by_module_name() -> None:
    names = [reader.name for reader in discover()]
    assert "run_record" in names
    assert "trl_completions" in names
    assert names == sorted(names), "discovery is in module-name order"


def test_every_discovered_reader_satisfies_the_protocol() -> None:
    for reader in discover():
        assert isinstance(reader, Reader)
        assert isinstance(reader.name, str) and reader.name
        assert callable(reader.can_read)
        assert callable(reader.read)


def test_no_reader_is_registered_from_outside_the_registry_module() -> None:
    """A reader is a module-level `READER`; nothing mutates a list from elsewhere (D-42)."""
    package = importlib.import_module("reward_lens.product.trace")
    root = Path(package.__file__).parent
    offenders = []
    for path in sorted(root.glob("*.py")):
        if path.name in {"readers.py", "__init__.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"\bregister\s*\(", text):
            offenders.append(path.name)
    assert offenders == []


def test_reader_for_picks_the_reader_that_can_read_the_path() -> None:
    assert reader_for(SHORT).name == "run_record"


# --- both GRPO fixtures ---------------------------------------------------------------------------


@pytest.mark.parametrize(("root", "run_id"), [(SHORT, SHORT_ID), (LONG, LONG_ID)])
def test_reads_both_grpo_fixtures(root: Path, run_id: str) -> None:
    reader = reader_for(root)
    assert reader.can_read(root)
    record = reader.read(root, policy="strict")
    assert isinstance(record, TraceRecord)
    assert record.run_id == run_id
    assert record.rows
    assert all(isinstance(row, TraceRow) for row in record.rows)


def test_reading_a_finished_run_never_imports_trl() -> None:
    before = set(sys.modules)
    reader_for(SHORT).read(SHORT, policy="strict")
    after = set(sys.modules) - before
    assert not any(name == "trl" or name.startswith("trl.") for name in after)


def test_the_short_fixture_row_count_matches_its_manifest() -> None:
    manifest = json.loads(
        (SHORT / "runs" / SHORT_ID.replace(":", "_") / "manifest.json").read_text(encoding="utf-8")
    )
    record = reader_for(SHORT).read(SHORT, policy="strict")
    assert len(record.rows) == manifest["counts"]["trajectories"]


# --- the TRL completions parquet, verified against 1.13.0 -----------------------------------------


def test_the_trl_layout_is_verified_against_the_1_13_0_tag_and_the_command_is_recorded() -> None:
    assert trl_parquet.TRL_TAG == "v1.13.0"
    assert trl_parquet.TRL_VERSION == "1.13.0"
    assert "1.13.0" in trl_parquet.TRL_VERIFIED_COMMAND
    assert trl_parquet.WRITER_LOCATION == "trl/trainer/grpo_trainer.py:3443-3450"
    assert trl_parquet.RESERVED_COLUMNS == ("step", "prompt", "completion", "advantage")


def _write_completions(
    directory: Path,
    *,
    step: int = 0,
    functions: tuple[str, ...] = ("length_reward", "format_reward"),
    extra: dict[str, list] | None = None,
    rows: int = 8,
    advantages: list[float] | None = None,
) -> Path:
    """A completions parquet with exactly the columns the 1.13.0 writer writes."""
    pq = pytest.importorskip("pyarrow.parquet")
    pa = importlib.import_module("pyarrow")
    table: dict[str, list] = {
        "step": [step] * rows,
        "prompt": [f"p{i // 4}" for i in range(rows)],
        "completion": [f"c{i}" for i in range(rows)],
    }
    for offset, name in enumerate(functions):
        table[name] = [float(i + offset) for i in range(rows)]
    for name, values in (extra or {}).items():
        table[name] = values
    table["advantage"] = advantages or [float(i) - 3.5 for i in range(rows)]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"completions_{step:05d}.parquet"
    pq.write_table(pa.table(table), path)
    return path


def test_reads_a_trl_completions_parquet_with_the_1_13_0_columns(tmp_path: Path) -> None:
    path = _write_completions(tmp_path / "completions", extra={"group": ["g0"] * 4 + ["g1"] * 4})
    table = trl_parquet.read_completions(
        path, reward_functions=("length_reward", "format_reward"), extra_columns=("group",)
    )
    assert table.columns[:3] == ("step", "prompt", "completion")
    assert table.columns[-1] == "advantage"
    assert table.reward_functions == ("length_reward", "format_reward")
    assert table.extra_columns == ("group",)
    assert len(table.rows) == 8


def test_the_total_is_a_weighted_sum_and_never_a_column(tmp_path: Path) -> None:
    path = _write_completions(tmp_path / "completions")
    table = trl_parquet.read_completions(
        path, reward_functions=("length_reward", "format_reward"), weights={"format_reward": 2.0}
    )
    totals = table.totals()
    # row 0: length 0.0 * 1.0 + format 1.0 * 2.0
    assert totals[0] == pytest.approx(2.0)
    assert FORBIDDEN not in table.columns


def test_a_raw_reward_column_is_preserved_only_as_a_convenience(tmp_path: Path) -> None:
    path = _write_completions(tmp_path / "completions", functions=(FORBIDDEN, "format_reward"))
    table = trl_parquet.read_completions(path, reward_functions=("format_reward",))
    assert table.raw_reward() == tuple(float(index) for index in range(8))


def test_matrix_row14_raw_reward_is_a_provenance_convenience_not_a_trace_series(
    tmp_path: Path,
) -> None:
    _write_completions(tmp_path / "completions", functions=(FORBIDDEN, "format_reward"))
    reader = next(item for item in discover() if item.name == "trl_completions")
    record = reader.read(tmp_path, policy="permissive", reward_functions=("format_reward",))
    assert FORBIDDEN not in record.series
    assert record.convenience["recorded_raw_reward"].provenance == {
        "kind": "raw_imported_reward_convenience",
        "source": "TRL completions raw reward column",
        "reader": "trl_completions",
    }
    finding = next(
        item for item in record.entries() if item.entry_id == "trace.raw_reward_convenience"
    )
    assert (
        finding.result["finding"]
        == "raw reward is a convenience aggregate, not an optimised composite"
    )


def test_trl_reader_does_not_make_run_identity_from_the_caller_path(tmp_path: Path) -> None:
    original_root = tmp_path / "run-a"
    _write_completions(original_root / "completions")
    copied_root = tmp_path / "run-b"
    shutil.copytree(original_root, copied_root)
    reader = next(item for item in discover() if item.name == "trl_completions")
    original = reader.read(
        original_root, policy="permissive", reward_functions=("length_reward", "format_reward")
    )
    copied = reader.read(
        copied_root, policy="permissive", reward_functions=("length_reward", "format_reward")
    )
    assert original.run_identity is None
    assert copied.run_identity is None
    entry = next(item for item in original.entries() if item.entry_id == "trace.row.0")
    assert entry.kind == "absence"
    assert entry.absence is not None
    assert "recorded run identity" in entry.absence.missing_access


def test_mixed_trl_raw_reward_schema_keeps_convenience_aligned_to_trace_rows(
    tmp_path: Path,
) -> None:
    _write_completions(
        tmp_path / "completions",
        step=0,
        functions=(FORBIDDEN, "format_reward"),
        rows=3,
    )
    _write_completions(
        tmp_path / "completions",
        step=1,
        functions=("format_reward",),
        rows=2,
    )
    reader = next(item for item in discover() if item.name == "trl_completions")
    record = reader.read(tmp_path, policy="permissive", reward_functions=("format_reward",))
    raw = record.convenience["recorded_raw_reward"]
    assert len(raw.values) == len(record.rows) == 5
    assert raw.values == (0.0, 1.0, 2.0, None, None)


def test_an_undeclared_column_partition_refuses_naming_the_series(tmp_path: Path) -> None:
    path = _write_completions(tmp_path / "completions")
    with pytest.raises(RecordIncomplete) as caught:
        trl_parquet.read_completions(path)
    assert caught.value.code == "RL0610"
    assert caught.value.exit_code == 4
    assert "reward_recorded" in caught.value.message
    assert caught.value.context["series"] == ["reward_recorded"]


def test_the_trl_reader_reads_a_run_directory(tmp_path: Path) -> None:
    _write_completions(tmp_path / "completions", step=0, extra={"group": ["g0"] * 4 + ["g1"] * 4})
    _write_completions(tmp_path / "completions", step=1, extra={"group": ["g2"] * 4 + ["g3"] * 4})
    reader = next(r for r in discover() if r.name == "trl_completions")
    assert reader.can_read(tmp_path)
    record = reader.read(
        tmp_path,
        policy="strict",
        reward_functions=("length_reward", "format_reward"),
        extra_columns=("group",),
        group_column="group",
    )
    assert len(record.rows) == 16
    assert sorted({row.step for row in record.rows}) == [0, 1]
    assert record.metadata.declared_reward is not None
    assert record.metadata.declared_reward.partition == ("length_reward", "format_reward")
    assert record.metadata.declared_reward.weights is None
    assert record.metadata.declared_reward.grouping["column"] == "group"
    assert record.rows[0].task_id is not None
    assert record.rows[0].response_digest is not None
    assert record.rows[0].grader_source is None
    assert record.rows[0].score_tree is None
    assert record.rows[0].accepted is None


def test_a_trl_reader_refuses_an_incomplete_source_declared_group(tmp_path: Path) -> None:
    """The production reader forwards caller-declared cardinality into RL0611."""
    _write_completions(
        tmp_path / "completions",
        rows=7,
        extra={"group": ["g0"] * 3 + ["g1"] * 4},
    )
    reader = next(r for r in discover() if r.name == "trl_completions")
    with pytest.raises(RecordIntegrity) as caught:
        reader.read(
            tmp_path,
            policy="strict",
            reward_functions=("length_reward", "format_reward"),
            extra_columns=("group",),
            group_column="group",
            group_size=4,
        )
    assert caught.value.code == "RL0611"
    assert caught.value.context["rule"] == "trace.incomplete_group"
    assert caught.value.context["row"] == "completions_00000.parquet:0"


# --- the four series, and no column named the forbidden name ---------------------------------------


def test_the_four_series_are_named_separately() -> None:
    assert SERIES_NAMES == ("reward_recorded", "reward_applied", "advantage", "selection")
    record = reader_for(SHORT).read(SHORT, policy="strict")
    assert tuple(record.series) == SERIES_NAMES
    for name in SERIES_NAMES:
        assert isinstance(record.series[name], Series)
        assert record.series[name].name == name


def test_run_record_keeps_panel_source_facts_immutable_and_scoped() -> None:
    """The shared bridge retains logged facts; it does not infer selection or loss evidence."""
    from types import MappingProxyType

    record = reader_for(SHORT).read(SHORT, policy="strict")
    first = record.rows[0]
    assert first.task_id is not None
    assert first.score_tree is not None
    assert first.accepted is None
    assert record.metadata.declared_reward is not None
    assert record.metadata.declared_reward.partition == tuple(record.components)
    assert record.metadata.applied_weights is None
    assert record.metadata.group_estimators[first.group_id] is not None
    zero_std = [
        metric for metric in record.metadata.metrics if metric.name == "frac_reward_zero_std"
    ]
    assert zero_std and {metric.scope for metric in zero_std} == {"optimizer_step"}
    assert all(
        metric.provenance["source_field"] == "optimizer.extra.frac_reward_zero_std"
        for metric in zero_std
    )
    assert isinstance(record.metadata.group_estimators, MappingProxyType)
    with pytest.raises(TypeError):
        record.metadata.group_estimators["invented"] = "not allowed"  # type: ignore[index]


def test_planned_contrast_design_is_declared_immutable_input_not_observed_trace_data() -> None:
    from reward_lens.product.trace import PlannedContrastDesign, TraceMetadata

    design = PlannedContrastDesign(
        standard_error=0.25,
        alpha=0.05,
        power=0.8,
        n=48,
        clusters=12,
        covariance={"kind": "cluster_robust"},
        design_effect=1.4,
        provenance={"source": "declared protocol"},
    )
    metadata = TraceMetadata(planned_contrast=design)
    assert metadata.planned_contrast is design
    with pytest.raises(TypeError):
        design.covariance["kind"] = "mutated"  # type: ignore[index]
    with pytest.raises(ValueError, match="standard_error"):
        PlannedContrastDesign(0.0, 0.05, 0.8, 48, 12)


def test_trace_row_snapshots_nested_grader_payloads_without_changing_score_tree_wire_shape() -> (
    None
):
    """A consumer cannot modify another panel's source tree through a nested call payload."""
    source = score_tree.Leaf(
        "score",
        1.0,
        score_tree.GraderCallRef(
            "grader",
            facets={"origin": {"seq": 1}},
            raw={"nested": ["kept"]},
        ),
    )
    row = TraceRow(index=0, step=0, group_id=None, row_id="snapshot", score_tree=source)
    assert row.score_tree is not source
    call = row.score_tree.grader_call
    assert call is not None
    with pytest.raises(TypeError):
        call.facets["origin"]["seq"] = 2
    with pytest.raises((AttributeError, TypeError)):
        call.raw["nested"].append("changed")
    source.grader_call.facets["origin"]["seq"] = 7
    assert call.facets["origin"]["seq"] == 1
    restored = decode_foreign(encode_foreign(row.score_tree))
    assert isinstance(restored, score_tree.Leaf)
    assert restored.grader_call.facets["origin"]["seq"] == 1


def test_real_run_reader_tree_snapshot_refuses_panel_mutation_and_preserves_other_view() -> None:
    first = reader_for(SHORT).read(SHORT, policy="strict")
    leaf = next(one for one in score_tree.leaves(first.rows[0].score_tree) if one.grader_call)
    with pytest.raises(TypeError):
        leaf.grader_call.facets["panel"] = "mutation"
    facets = leaf.grader_call.facets
    with pytest.raises(TypeError):
        facets |= {"panel": "union-mutation"}
    second = reader_for(SHORT).read(SHORT, policy="strict")
    same_leaf = next(one for one in score_tree.leaves(second.rows[0].score_tree) if one.grader_call)
    assert "panel" not in same_leaf.grader_call.facets


def test_only_a_source_declared_composite_carries_a_weight_vector(tmp_path: Path) -> None:
    record = reader_for(SHORT).read(SHORT, policy="strict")
    recorded = record.series["reward_recorded"]
    applied = record.series["reward_applied"]
    assert recorded.weights is None
    assert applied.weights is None
    assert recorded.definition != applied.definition

    # The recorded score partition, not the unrelated top-level component map, aligns this vector.
    copied = _copy_short_fixture(tmp_path)
    run_path = next(copied.glob("runs/*/run.json"))
    document = json.loads(run_path.read_text(encoding="utf-8"))
    document["lineage"]["extra"]["config"]["reward_weights"] = [2.0]
    run_path.write_text(json.dumps(document), encoding="utf-8")
    aligned = reader_for(copied).read(copied, policy="strict")
    assert aligned.series["reward_applied"].present is True
    assert aligned.series["reward_applied"].weights == (2.0,)
    assert aligned.rows[0].reward_applied == pytest.approx(
        aligned.rows[0].components["length_reward"] * 2.0
    )

    # This is the review's counterexample: a partial vector cannot gain an invented unit weight.
    partial_run = SimpleNamespace(
        lineage=SimpleNamespace(extra={"config": {"reward_weights": [2.0]}}),
        components={
            Component.POLICY: ComponentRef(name="policy", version="1"),
            Component.GRADER: ComponentRef(name="score_a", kind="reward_functions"),
        },
    )
    weights, absent_reason = _declared_weights(partial_run, ["score_a", "score_b"])
    assert weights is None
    assert "exactly" in absent_reason
    rows = [
        TraceRow(
            index=0,
            step=0,
            group_id="g",
            row_id="r",
            components={"score_a": 3.0, "score_b": 4.0},
            reward_recorded=7.0,
        )
    ]
    series = _series(rows, ["score_a", "score_b"], weights, absent_reason)
    assert series["reward_recorded"].values == (7.0,)
    assert series["reward_applied"].present is False
    assert series["reward_applied"].weights is None
    assert series["reward_applied"].values == (None,)


def test_a_missing_component_does_not_produce_a_partial_trl_composite(tmp_path: Path) -> None:
    """A weighted aggregate is unavailable for a row missing any required component."""
    _write_completions(
        tmp_path / "completions",
        extra={"group": ["g0"] * 4 + ["g1"] * 4},
    )
    parquet = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "completions" / "completions_00000.parquet"
    table = parquet.read_table(path).to_pydict()
    table["format_reward"][0] = None
    importlib.import_module("pyarrow.parquet").write_table(
        importlib.import_module("pyarrow").table(table), path
    )
    reader = next(r for r in discover() if r.name == "trl_completions")
    record = reader.read(
        tmp_path,
        policy="strict",
        reward_functions=("length_reward", "format_reward"),
        extra_columns=("group",),
        weights={"format_reward": 2.0},
        group_column="group",
        group_size=4,
    )
    assert record.rows[0].reward_applied is None
    assert record.series["reward_applied"].values[0] is None
    assert record.convenience["declared_weighted_component_sum"].values[0] is None


def _copy_short_fixture(tmp_path: Path) -> Path:
    copied = tmp_path / "short"
    shutil.copytree(SHORT, copied)
    return copied


def test_a_serialized_recorded_applied_weight_reopens_into_the_trace_row(tmp_path: Path) -> None:
    """A retained raw trajectory field reaches the canonical Trace row without synthesis."""
    root = _copy_short_fixture(tmp_path)
    run_dir = root / "runs" / SHORT_ID.replace(":", "_")
    trajectories_path = run_dir / "steps" / "step_range=000000-000099" / "trajectories.jsonl"
    rows = [json.loads(line) for line in trajectories_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["applied_weight"] = 0.25
    rows[0]["applied_weight_source"] = "recorded"
    rows[1]["applied_weight"] = 0.75
    rows[1]["applied_weight_source"] = "reconstructed"
    trajectories_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    reopened = open_run(root, SHORT_ID)
    trajectory = next(
        item for step in reopened.steps for group in step.groups for item in group.trajectories
    )
    assert trajectory.applied_weight == 0.25
    assert trajectory.applied_weight_source == "recorded"

    trace = reader_for(root).read(root, policy="strict")
    assert trace.rows[0].applied_weight == 0.25
    assert trace.rows[0].applied_weight_source == "recorded"
    assert trace.rows[1].applied_weight is None
    assert trace.rows[1].applied_weight_source is None
    entry = next(item for item in trace.entries() if item.entry_id == "trace.row.0")
    assert entry.result["applied_weight"] == 0.25
    assert entry.result["applied_weight_source"] == "recorded"


@pytest.mark.parametrize(
    ("value", "source"),
    [(0.25, None), ("not-a-finite-scalar", "recorded")],
)
def test_a_serialized_applied_weight_without_a_valid_recorded_pair_stays_absent(
    tmp_path: Path, value: object, source: str | None
) -> None:
    root = _copy_short_fixture(tmp_path)
    trajectories_path = next(root.glob("runs/*/steps/*/trajectories.jsonl"))
    rows = [json.loads(line) for line in trajectories_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["applied_weight"] = value
    if source is not None:
        rows[0]["applied_weight_source"] = source
    trajectories_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    reopened = open_run(root, SHORT_ID)
    trajectory = next(
        item for step in reopened.steps for group in step.groups for item in group.trajectories
    )
    assert trajectory.applied_weight is None
    assert trajectory.applied_weight_source is None
    trace = reader_for(root).read(root, policy="strict")
    assert trace.rows[0].applied_weight is None
    assert trace.rows[0].applied_weight_source is None


def test_a_run_record_reader_refuses_a_short_source_declared_group(tmp_path: Path) -> None:
    """The run-record reader uses each group's recorded k, not a detached test input."""
    root = _copy_short_fixture(tmp_path)
    run_dir = root / "runs" / SHORT_ID.replace(":", "_")
    groups_path = run_dir / "steps" / "step_range=000000-000099" / "groups.jsonl"
    first_group = json.loads(groups_path.read_text(encoding="utf-8").splitlines()[0])["group_id"]
    trajectories_path = run_dir / "steps" / "step_range=000000-000099" / "trajectories.jsonl"
    original = trajectories_path.read_text(encoding="utf-8").splitlines()
    retained: list[str] = []
    removed = False
    for line in original:
        row = json.loads(line)
        if row["group_id"] == first_group and not removed:
            removed = True
            continue
        retained.append(line)
    assert removed
    trajectories_path.write_text("\n".join(retained) + "\n", encoding="utf-8")
    with pytest.raises(RecordIntegrity) as caught:
        reader_for(root).read(root, policy="strict")
    assert caught.value.code == "RL0611"
    assert caught.value.context["rule"] == "trace.incomplete_group"
    assert caught.value.context["row"].startswith("0:" + first_group)


def test_a_degenerate_group_is_not_selection_or_gradient_evidence(tmp_path: Path) -> None:
    root = _copy_short_fixture(tmp_path)
    run_dir = root / "runs" / SHORT_ID.replace(":", "_")
    groups_path = run_dir / "steps" / "step_range=000000-000099" / "groups.jsonl"
    lines = groups_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    payload = json.loads(first["payload"])
    first["degenerate"] = True
    payload["group_stats"]["degenerate"] = True
    first["payload"] = json.dumps(payload)
    groups_path.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n", encoding="utf-8")
    record = reader_for(root).read(root, policy="strict")
    selection = record.series["selection"]
    assert selection.present is False
    assert all(row.selection is None for row in record.rows)
    assert "gradient" not in selection.definition.lower()


def test_the_native_component_scores_are_separate_named_series() -> None:
    record = reader_for(SHORT).read(SHORT, policy="strict")
    assert "length_reward" in record.components
    assert record.components["length_reward"].name == "length_reward"


def test_native_component_scores_keep_their_raw_reward_provenance() -> None:
    """A component is observed only when the record says where its raw score came from."""
    record = reader_for(SHORT).read(SHORT, policy="strict")
    component = record.components["length_reward"]
    assert component.provenance["kind"] == "raw_native_component_score"
    assert component.provenance["source"] == "recorded score tree"
    assert component.provenance["reader"] == "run_record"


def test_no_key_anywhere_in_a_produced_record_is_the_forbidden_name() -> None:
    record = reader_for(SHORT).read(SHORT, policy="strict")
    assert FORBIDDEN not in record.series
    assert FORBIDDEN not in record.components
    for row in record.rows:
        assert FORBIDDEN not in row.components


def test_no_source_file_of_this_packet_names_a_series_the_forbidden_name() -> None:
    """Trace carriers must not declare an ambiguous generic series name.

    Fiveway may read an external Seeker witness payload under ``result["reward"]``. That
    payload is an evaluator-specific input, not a Trace row, series, or component declaration.
    Keep this scan on the modules that define Trace carriers so it guards the public Trace
    namespace without rejecting that external producer contract.
    """
    package = Path(importlib.import_module("reward_lens.product.trace").__file__).parent
    files = [
        package / "entries.py",
        package / "record.py",
        package / "run_record.py",
        package / "trl_completions.py",
        Path(trl_parquet.__file__),
        package / "live_follow.py",
    ]
    pattern = re.compile(rf"""['"]{FORBIDDEN}['"]""")
    offenders: list[str] = []
    for path in files:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line) and "raw-reward-convenience" not in line:
                offenders.append(f"{path.name}:{number}")
    assert offenders == []


def test_external_seeker_reward_payload_is_not_a_trace_series_or_row_field() -> None:
    """The Seeker producer's payload key never widens the Trace carrier namespace."""
    answer = Path(importlib.import_module("reward_lens.product.trace.answer").__file__)
    assert 'result["reward"]' in answer.read_text(encoding="utf-8")
    assert FORBIDDEN not in SERIES_NAMES
    assert FORBIDDEN not in TraceRow.__dataclass_fields__


def test_the_abstention_channel_survives_as_a_missing_value() -> None:
    """The fixture's grader abstains on every seventh completion; that is preserved, not zeroed."""
    record = reader_for(SHORT).read(SHORT, policy="permissive")
    values = record.series["reward_recorded"].values
    assert any(value is None for value in values)


def test_an_unrecorded_applied_composite_is_incomplete_without_discarding_raw_measurements() -> (
    None
):
    """Missing weights do not license a unit-weight objective, but raw values still survive."""
    record = reader_for(SHORT).read(SHORT, policy="strict")
    assert record.components["length_reward"].present is True
    assert record.series["advantage"].present is True
    assert record.series["reward_applied"].present is False
    assert "weight" in record.series["reward_applied"].absent_reason
    with pytest.raises(RecordIncomplete) as caught:
        record.require_measurements("reward_applied")
    assert caught.value.code == "RL0610"
    assert caught.value.exit_code == 4
    assert caught.value.context["series"] == ["reward_applied"]


# --- the integrity rules ----------------------------------------------------------------------------


def _row(
    index: int, *, step: int = 0, group: str | None = "g0", row_id: str | None = None
) -> TraceRow:
    return TraceRow(
        index=index,
        step=step,
        group_id=group,
        row_id=row_id or f"t{index}",
        components={"length_reward": float(index)},
        reward_recorded=float(index),
        reward_applied=float(index),
        advantage=float(index) - 1.0,
        selection=1.0,
    )


def _input(rows: tuple[TraceRow, ...], **kwargs) -> CheckInput:
    return CheckInput(rows=rows, **kwargs)


RULE_IDS = (
    "trace.duplicate_row_id",
    "trace.incomplete_group",
    "trace.out_of_order_steps",
    "trace.series_length_mismatch",
    "trace.component_length_mismatch",
    "trace.checksum_drift",
    "trace.group_not_declared",
)


def test_every_rule_is_named_and_the_set_is_exactly_the_documented_one() -> None:
    assert tuple(rule.id for rule in RULES) == RULE_IDS
    for rule in RULES:
        assert rule.name and rule.description


def _violating(rule_id: str) -> CheckInput:
    """One input built to violate exactly one named rule."""
    if rule_id == "trace.duplicate_row_id":
        return _input((_row(0, row_id="t0"), _row(1, row_id="t0")))
    if rule_id == "trace.incomplete_group":
        rows = (_row(0, group="g0"), _row(1, group="g0"), _row(2, group="g1"))
        return _input(rows, declared_group_size=2)
    if rule_id == "trace.out_of_order_steps":
        return _input((_row(0, step=3), _row(1, step=1)))
    if rule_id == "trace.series_length_mismatch":
        return _input(
            (_row(0), _row(1)),
            series={"advantage": Series(name="advantage", values=(0.0,))},
        )
    if rule_id == "trace.component_length_mismatch":
        return _input(
            (_row(0), _row(1)),
            components={"length_reward": Series(name="length_reward", values=(0.0,))},
        )
    if rule_id == "trace.checksum_drift":
        return _input((_row(0), _row(1)), declared_digest="sha256:" + "0" * 64)
    if rule_id == "trace.group_not_declared":
        return _input((_row(0, group=None), _row(1, group=None)))
    raise AssertionError(rule_id)


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_each_integrity_rule_fails_on_a_fixture_built_to_violate_it(rule_id: str) -> None:
    from reward_lens.product.trace import check

    results = {result.rule: result for result in check(_violating(rule_id))}
    assert not results[rule_id].passed, f"{rule_id} passed on a fixture built to violate it"
    assert results[rule_id].violations
    assert results[rule_id].violations[0].row is not None
    others = [rid for rid, result in results.items() if rid != rule_id and not result.passed]
    assert others == [], f"{rule_id}'s fixture also tripped {others}"


def test_no_rule_infers_a_group_from_row_adjacency() -> None:
    """Four rows a reader would be tempted to chunk into two groups of two. It refuses."""
    from reward_lens.product.trace import check

    rows = tuple(_row(index, group=None) for index in range(4))
    results = {result.rule: result for result in check(_input(rows, declared_group_size=2))}
    assert not results["trace.group_not_declared"].passed
    assert results["trace.incomplete_group"].passed, (
        "a group must never be reconstructed from adjacency, so the group rule has nothing to say"
    )
    assert all(row.group_id is None for row in rows)


def test_declared_group_with_no_rows_is_incomplete() -> None:
    from reward_lens.product.trace import check, enforce

    data = _input((_row(0, group="g0"),), declared_group_sizes={"g0": 1, "lost": 2})
    result = {item.rule: item for item in check(data)}["trace.incomplete_group"]
    assert not result.passed
    assert result.violations[0].row is None  # no row exists to name for this group
    assert "group 'lost' holds 0 rows" in result.violations[0].message
    with pytest.raises(RecordIntegrity, match="group 'lost' holds 0 rows"):
        enforce(data)


def test_an_integrity_violation_refuses_naming_the_row() -> None:
    from reward_lens.product.trace import enforce

    with pytest.raises(RecordIntegrity) as caught:
        enforce(_violating("trace.duplicate_row_id"))
    assert caught.value.code == "RL0611"
    assert caught.value.exit_code == 4
    assert caught.value.context["row"] == "t0"
    assert caught.value.context["rule"] == "trace.duplicate_row_id"
    assert "t0" in caught.value.message


def test_a_missing_series_refuses_naming_the_series() -> None:
    from reward_lens.product.trace import require_series

    with pytest.raises(RecordIncomplete) as caught:
        require_series({"advantage": Series(name="advantage", values=())}, source="a run record")
    assert caught.value.code == "RL0610"
    assert caught.value.exit_code == 4
    assert caught.value.context["series"] == ["reward_recorded", "reward_applied", "selection"]
    assert "reward_recorded" in caught.value.message


# --- pyarrow and the [trace] extra -------------------------------------------------------------------


def test_pyarrow_is_used_when_it_is_present(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    path = _write_completions(tmp_path / "completions")
    table = trl_parquet.read_completions(path, reward_functions=("length_reward", "format_reward"))
    assert table.engine == "pyarrow"


def test_the_trace_extra_is_named_when_pyarrow_is_absent(tmp_path: Path, monkeypatch) -> None:
    path = _write_completions(tmp_path / "completions")
    monkeypatch.setattr(trl_parquet, "_pyarrow_present", lambda: False)
    with pytest.raises(ExtraRequiredError) as caught:
        trl_parquet.read_completions(path, reward_functions=("length_reward",))
    assert "reward-lens[trace]" in str(caught.value)


def test_doctor_reports_the_trace_extra_and_drops_it_when_pyarrow_is_absent(monkeypatch) -> None:
    ladder = importlib.import_module("reward_lens.product.access.ladder")
    doctor = importlib.import_module("reward_lens.product.access.doctor")
    assert "trace" in ladder.EXTRAS
    assert "trace" in doctor.doctor().install.extras
    real = ladder._importable
    monkeypatch.setattr(
        ladder, "_importable", lambda module: False if module == "pyarrow" else real(module)
    )
    assert "trace" not in doctor.doctor().install.extras


# --- the access level, the subject and the dependencies -----------------------------------------------


def test_the_record_states_the_access_level_its_input_reached() -> None:
    record = reader_for(SHORT).read(SHORT, policy="strict")
    assert record.access_level == "RECORD"
    assert record.access_statement
    assert "RECORD" in record.access_statement


def test_a_parquet_read_states_the_weaker_access_level(tmp_path: Path) -> None:
    _write_completions(tmp_path / "completions", extra={"group": ["g0"] * 4 + ["g1"] * 4})
    reader = next(r for r in discover() if r.name == "trl_completions")
    record = reader.read(
        tmp_path,
        policy="strict",
        reward_functions=("length_reward", "format_reward"),
        extra_columns=("group",),
        group_column="group",
    )
    assert record.access_level == "RECORD"


def test_every_trace_entry_carries_subject_ref_and_depends_on() -> None:
    record = reader_for(SHORT).read(SHORT, policy="strict")
    entries = record.entries()
    assert entries
    for entry in entries:
        assert entry.section == "trace"
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", entry.subject_ref)
        assert entry.subject_ref == record.subject_ref
        assert entry.depends_on
        assert "digest:training_semantics" in entry.depends_on


def test_trace_entries_preserve_verified_measurements_and_render_the_loss_boundary() -> None:
    record = reader_for(SHORT).read(SHORT, policy="strict")
    entries = {entry.entry_id: entry for entry in record.entries()}
    reconstruction = entries["trace.reconstruction"]
    access = entries["trace.access_level"]
    incomplete = entries["trace.optimized_composite"]
    assert reconstruction.kind == "witness"
    assert reconstruction.result["component_provenance"]["length_reward"]["kind"] == (
        "raw_native_component_score"
    )
    assert access.result["measurement_boundary"] == "observed_loss_level"
    assert access.result["gradient_observed"] is False
    assert access.result["optimizer_transform_observed"] is False
    assert access.result["parameter_update_observed"] is False
    assert incomplete.kind == "absence"
    assert incomplete.absence.state == "COULD_NOT_CHECK"
    assert "reward_applied" in incomplete.absence.affected_claims


def test_the_subject_ref_names_the_audited_version_not_the_reader() -> None:
    one = reader_for(SHORT).read(SHORT, policy="strict")
    two = reader_for(LONG).read(LONG, policy="strict")
    assert one.subject_ref == two.subject_ref
    assert one.run_identity != two.run_identity
    assert reader_for(SHORT).read(SHORT, policy="strict").subject_ref == one.subject_ref

    lineage = SimpleNamespace(
        framework="trl", framework_version="1", git_sha="same", config_hash="same"
    )
    version_one = SimpleNamespace(
        kind="train",
        lineage=lineage,
        components={Component.POLICY: ComponentRef(name="policy", version="1")},
    )
    version_two = SimpleNamespace(
        kind="train",
        lineage=lineage,
        components={Component.POLICY: ComponentRef(name="policy", version="2")},
    )
    assert _subject_ref(version_one, "run:one") == _subject_ref(version_one, "run:two")
    assert _subject_ref(version_one, "run:one") != _subject_ref(version_two, "run:one")


def test_subject_ref_survives_copying_the_same_record_to_a_new_path(tmp_path: Path) -> None:
    original = reader_for(SHORT).read(SHORT, policy="strict")
    copied_root = _copy_short_fixture(tmp_path)
    copied = reader_for(copied_root).read(copied_root, policy="strict")
    assert copied.run_id == original.run_id
    assert copied.subject_ref == original.subject_ref
    assert copied.run_identity == original.run_identity


def test_trace_rows_emit_canonical_recorded_task_response_and_source_evidence() -> None:
    record = reader_for(SHORT).read(SHORT, policy="strict")
    source = open_run(SHORT, record.run_id)
    trajectory = next(
        item
        for step in source.steps
        for group in step.groups
        for item in group.trajectories
        if str(item.id) == record.rows[0].row_id.split(":", 4)[4]
    )
    assistant_turns = [turn for turn in trajectory.turns if turn.role == "assistant"]
    assert len(assistant_turns) == 1
    expected_digest, expected_codec = response_text_identity(assistant_turns[0].text)
    entry = next(item for item in record.entries() if item.entry_id == "trace.row.0")
    assert entry.kind == "witness"
    assert entry.result["run_identity"] == record.run_identity
    assert entry.result["task_id"] == record.rows[0].task_id
    assert expected_codec == RESPONSE_TEXT_CODEC == "rfc8785-response-text/1"
    assert entry.result["response_digest"] == record.rows[0].response_digest == expected_digest
    assert entry.result["response_codec"] == record.rows[0].response_codec == expected_codec
    assert entry.result["response_provenance"] == {"assistant_turn_index": assistant_turns[0].index}
    assert entry.result["grader_source"]["component_fingerprint"].startswith("cmp:")


def test_response_text_codec_refuses_ambiguous_or_non_text_response_identity() -> None:
    first = SimpleNamespace(index=3, role="assistant", text="same response")
    second = SimpleNamespace(index=9, role="assistant", text="same response")
    assert _response_identity(SimpleNamespace(turns=(first, second))) == (None, None, None)
    with pytest.raises(TypeError, match="exact str"):
        response_text_identity(None)  # type: ignore[arg-type]


def test_response_text_identity_keeps_turn_index_as_separate_provenance() -> None:
    early = SimpleNamespace(index=1, role="assistant", text="the exact response")
    late = SimpleNamespace(index=8, role="assistant", text="the exact response")
    early_identity = _response_identity(SimpleNamespace(turns=(early,)))
    late_identity = _response_identity(SimpleNamespace(turns=(late,)))
    assert early_identity[:2] == late_identity[:2]
    assert early_identity[2] == {"assistant_turn_index": 1}
    assert late_identity[2] == {"assistant_turn_index": 8}


def test_response_text_identity_hashes_the_exact_rfc8785_json_string() -> None:
    text = "line one\\nline two"
    digest, codec = response_text_identity(text)
    assert codec == RESPONSE_TEXT_CODEC
    assert digest == "sha256:" + hashlib.sha256(rfc8785.dumps(text)).hexdigest()
    assert digest != response_text_identity(text + "\\n")[0]


def test_trace_row_with_missing_recorded_source_or_response_is_an_absence() -> None:
    source = reader_for(SHORT).read(SHORT, policy="strict")
    row = replace(source.rows[0], response_digest=None, response_codec=None, grader_source=None)
    record = replace(source, rows=(row,))
    entry = next(item for item in record.entries() if item.entry_id == "trace.row.0")
    assert entry.kind == "absence"
    assert entry.absence is not None
    assert entry.absence.state == "COULD_NOT_CHECK"
    assert entry.absence.affected_claims == ["trace_row_join"]


def test_the_verb_produces_a_record_that_validates_against_the_frozen_schema() -> None:
    """The canonical serialisation of what the verb returns is a record the schema admits.

    `to_dict()` is the rule `contracts.serialise` states: a property the schema lists under
    `required` is always written, `null` when its value is None, and only the properties the
    schema does not require are dropped. `model_dump_json(exclude_none=True)` is a different rule,
    and it fails here for a reason that has nothing to do with this verb: the attestation every
    verb is given carries `statement_digest=None` until something signs the record, and that
    property is required and nullable, so excluding it by its value removes it from the record.
    """
    from reward_lens.api import TraceRequest

    assay = run(TraceRequest(run=SHORT_ID, project=SHORT))
    payload = assay.to_dict()
    assert json.loads(json.dumps(payload)) == payload  # it is JSON, not a dict of model objects
    validate_record(payload)
    assert payload["measurement"]["trace"]


# --- live following -----------------------------------------------------------------------------------


def _seal(directory: Path, seq: int, rows: list[dict]) -> None:
    seal_chunk(directory, seq, rows)


def test_following_yields_chunks_with_sequence_numbers(tmp_path: Path) -> None:
    for seq in (1, 2, 3):
        _seal(tmp_path, seq, [{"index": seq, "step": seq}])
    chunks = list(follow(tmp_path, policy="strict"))
    assert [chunk.seq for chunk in chunks] == [1, 2, 3]
    assert all(isinstance(chunk, Chunk) and chunk.sealed for chunk in chunks)


def test_a_seal_is_written_atomically(tmp_path: Path) -> None:
    _seal(tmp_path, 1, [{"index": 0, "step": 0}])
    seals = list(tmp_path.glob("*.seal"))
    assert len(seals) == 1
    assert not list(tmp_path.glob("*.tmp"))
    manifest = json.loads(seals[0].read_text(encoding="utf-8"))
    assert manifest["seq"] == 1
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", manifest["digest"])


def test_a_torn_tail_is_visible_under_permissive(tmp_path: Path) -> None:
    _seal(tmp_path, 1, [{"index": 0, "step": 0}])
    (tmp_path / "chunk_00002.jsonl").write_text('{"index": 1, "step": 1}\n{"ind', encoding="utf-8")
    chunks = list(follow(tmp_path, policy="permissive"))
    assert chunks[-1].torn is True
    assert chunks[-1].sealed is False
    assert chunks[-1].torn_bytes == '{"ind'


def test_a_torn_tail_refuses_under_strict(tmp_path: Path) -> None:
    _seal(tmp_path, 1, [{"index": 0, "step": 0}])
    (tmp_path / "chunk_00002.jsonl").write_text('{"index": 1, "step": 1}\n{"ind', encoding="utf-8")
    with pytest.raises(RecordIntegrity) as caught:
        list(follow(tmp_path, policy="strict"))
    assert caught.value.code == "RL0611"
    assert caught.value.exit_code == 4
    assert caught.value.context["rule"] == "trace.torn_tail"
    assert caught.value.context["row"]


def test_a_gap_in_the_sequence_refuses_under_strict(tmp_path: Path) -> None:
    _seal(tmp_path, 1, [{"index": 0, "step": 0}])
    _seal(tmp_path, 3, [{"index": 2, "step": 2}])
    with pytest.raises(RecordIntegrity) as caught:
        list(follow(tmp_path, policy="strict"))
    assert caught.value.context["rule"] == "trace.sequence_gap"
    assert "2" in caught.value.message


def test_a_gap_is_recorded_under_permissive(tmp_path: Path) -> None:
    _seal(tmp_path, 1, [{"index": 0, "step": 0}])
    _seal(tmp_path, 3, [{"index": 2, "step": 2}])
    chunks = list(follow(tmp_path, policy="permissive"))
    assert [chunk.seq for chunk in chunks] == [1, 3]
    assert chunks[1].gap_before == (2,)


def test_following_never_attaches_to_the_training_process(tmp_path: Path) -> None:
    source = Path(importlib.import_module("reward_lens.product.trace.follow").__file__)
    text = source.read_text(encoding="utf-8")
    for forbidden in ("subprocess", "signal.", "psutil", "ptrace"):
        assert forbidden not in text


def test_the_policy_is_declared_in_the_record(tmp_path: Path) -> None:
    _seal(tmp_path, 1, [{"index": 0, "step": 0}])
    (tmp_path / "chunk_00002.jsonl").write_text('{"index": 1, "step": 1}\n{"ind', encoding="utf-8")
    reader = next(r for r in discover() if r.name == "live_follow")
    record = reader.read(tmp_path, policy="permissive")
    assert record.policy == "permissive"
    assert record.complete is False
    assert record.torn_tail is not None
    assert "permissive" in record.torn_tail.allowed_by


def test_the_record_says_the_run_was_incomplete_when_read(tmp_path: Path) -> None:
    _seal(tmp_path, 1, [{"index": 0, "step": 0}])
    reader = next(r for r in discover() if r.name == "live_follow")
    record = reader.read(tmp_path, policy="strict")
    assert record.complete is False
    assert "incomplete" in record.completeness_statement.lower()


# --- the golden transcript -------------------------------------------------------------------------


def test_the_trace_verb_is_no_longer_marked_not_in_this_build() -> None:
    from reward_lens.cli import availability

    assert "trace" not in availability.marked_absent()
    assert availability.absent("trace") is False


def test_the_golden_trace_transcript_is_what_the_cli_prints(tmp_path: Path) -> None:
    expected = CURRENT_TRACE_GOLDEN.read_text(encoding="utf-8")
    command = expected.splitlines()[0]
    assert command.startswith("$ ")
    argv = command[2:].split()
    fixture = tmp_path / "tests" / "fixtures" / "grpo_run" / "short"
    shutil.copytree(SHORT, fixture)
    finished = subprocess.run(  # noqa: S603 - a fixed argv, no shell
        # The copied fixture keeps the command and its text output literal, while trace writes
        # its assay/report beside that disposable source rather than into checked-in test data.
        [sys.executable, "-m", "reward_lens.cli.main", *argv[1:]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    actual = command + "\n" + finished.stdout
    assert finished.returncode == 2, finished.stderr
    # Keep the historical transcript unchanged. The current frontend bundle can grow while the
    # command, record and evidence stay the same; its printed size must match the actual report.
    report_line = re.search(r"^  Report    (.+\.assay\.html)   (\d+) KB$", actual, re.MULTILINE)
    assert report_line is not None
    report_path = tmp_path / report_line.group(1)
    assert int(report_line.group(2)) == report_path.stat().st_size // 1024
    expected = re.sub(
        r"(?m)(^  Report    .+\.assay\.html   )\d+( KB)$",
        lambda match: match.group(1) + report_line.group(2) + match.group(2),
        expected,
    )
    # The wall clock on the last line is volatile, and `fleet/golden/VOLATILE.md` declares it so:
    # rule "duration", for `*.txt`, with this pattern. `fleet/golden/run_demo.py` masks both sides
    # before it diffs, and so does this, for the same reason and with the same rule.
    duration = re.compile(r"\b\d+m \d+s\b|\b\d+(\.\d+)? ?(ms|s)\b(?! per)")
    assay_id = re.compile(r"trace-[0-9a-f]{64}")
    assert assay_id.sub("trace-<assay-id>", duration.sub("<dur>", actual)) == assay_id.sub(
        "trace-<assay-id>", duration.sub("<dur>", expected)
    )


def test_the_golden_root_help_no_longer_marks_trace_absent() -> None:
    text = (WAVE2_TRACE_GOLDEN / "root-help.txt").read_text(encoding="utf-8")
    trace_line = next(line for line in text.splitlines() if line.strip().startswith("trace"))
    assert "(not in this build)" not in trace_line


def test_a_finding_prints_as_continuation_lines_under_its_panel() -> None:
    """A-028: never a second glyph line."""
    text = CURRENT_TRACE_GOLDEN.read_text(encoding="utf-8")
    glyphs = ("✔", "✘", "○", "·")
    lines = text.splitlines()
    for number, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped[0] not in glyphs:
            continue
        following = lines[number + 1].strip() if number + 1 < len(lines) else ""
        if following and following[0] in glyphs:
            indent_here = len(line) - len(line.lstrip())
            indent_next = len(lines[number + 1]) - len(lines[number + 1].lstrip())
            assert indent_next <= indent_here, f"line {number + 1} is a second glyph line"
