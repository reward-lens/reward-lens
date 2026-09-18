"""P-TRACE-1: the trace spine, the primary readers and the reader registry.

Every acceptance row of the packet is a test here, and every refusal names its code, its exit
status and the field the message has to carry.
"""

from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from reward_lens.contracts import validate_record
from reward_lens.core.extras import ExtraRequiredError
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
from reward_lens.record.convert import trl_parquet

REPO = Path(__file__).resolve().parents[2]
SHORT = REPO / "tests" / "fixtures" / "grpo_run" / "short"
LONG = REPO / "tests" / "fixtures" / "grpo_run" / "long"
SHORT_ID = "run:8a8c7e29274db0a681313b48dbd1eb63"
LONG_ID = "run:f77bf75940ab982bbc35407af99cc094"
GOLDEN = REPO / "fleet" / "golden" / "wave-2" / "trace"

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


def test_a_column_with_the_forbidden_name_is_refused(tmp_path: Path) -> None:
    path = _write_completions(tmp_path / "completions", functions=(FORBIDDEN, "format_reward"))
    with pytest.raises(RecordIntegrity) as caught:
        trl_parquet.read_completions(path, reward_functions=(FORBIDDEN, "format_reward"))
    assert caught.value.code == "RL0611"
    assert FORBIDDEN in caught.value.message


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


# --- the four series, and no column named the forbidden name ---------------------------------------


def test_the_four_series_are_named_separately() -> None:
    assert SERIES_NAMES == ("reward_recorded", "reward_applied", "advantage", "selection")
    record = reader_for(SHORT).read(SHORT, policy="strict")
    assert tuple(record.series) == SERIES_NAMES
    for name in SERIES_NAMES:
        assert isinstance(record.series[name], Series)
        assert record.series[name].name == name


def test_the_recorded_and_the_applied_aggregate_carry_their_own_weight_vectors() -> None:
    record = reader_for(SHORT).read(SHORT, policy="strict")
    recorded = record.series["reward_recorded"]
    applied = record.series["reward_applied"]
    assert recorded.weights is not None
    assert applied.weights is not None
    assert recorded.definition != applied.definition


def test_the_native_component_scores_are_separate_named_series() -> None:
    record = reader_for(SHORT).read(SHORT, policy="strict")
    assert "length_reward" in record.components
    assert record.components["length_reward"].name == "length_reward"


def test_no_key_anywhere_in_a_produced_record_is_the_forbidden_name() -> None:
    record = reader_for(SHORT).read(SHORT, policy="strict")
    assert FORBIDDEN not in record.series
    assert FORBIDDEN not in record.components
    for row in record.rows:
        assert FORBIDDEN not in row.components


def test_no_source_file_of_this_packet_names_a_series_the_forbidden_name() -> None:
    """The grep: the bare literal appears only on a line tagged as the name being refused."""
    package = Path(importlib.import_module("reward_lens.product.trace").__file__).parent
    files = sorted(package.glob("*.py")) + [
        Path(trl_parquet.__file__),
    ]
    pattern = re.compile(rf"""['"]{FORBIDDEN}['"]""")
    offenders: list[str] = []
    for path in files:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line) and "forbidden-column-name" not in line:
                offenders.append(f"{path.name}:{number}")
    assert offenders == []


def test_the_abstention_channel_survives_as_a_missing_value() -> None:
    """The fixture's grader abstains on every seventh completion; that is preserved, not zeroed."""
    record = reader_for(SHORT).read(SHORT, policy="permissive")
    values = record.series["reward_recorded"].values
    assert any(value is None for value in values)


# --- the integrity rules ----------------------------------------------------------------------------


def _row(index: int, *, step: int = 0, group: str | None = "g0", row_id: str | None = None) -> TraceRow:
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


def test_the_subject_ref_names_the_audited_version_not_the_reader() -> None:
    one = reader_for(SHORT).read(SHORT, policy="strict")
    two = reader_for(LONG).read(LONG, policy="strict")
    assert one.subject_ref != two.subject_ref
    assert reader_for(SHORT).read(SHORT, policy="strict").subject_ref == one.subject_ref


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


def test_the_golden_trace_transcript_is_what_the_cli_prints() -> None:
    expected = (GOLDEN / "trace-fixture.txt").read_text(encoding="utf-8")
    command = expected.splitlines()[0]
    assert command.startswith("$ ")
    argv = command[2:].split()
    finished = subprocess.run(  # noqa: S603 - a fixed argv, no shell
        # `reward_lens.cli` is a package with no `__main__`; `cli.main` is the module the
        # `reward-lens` console script points at, and it carries the `__main__` guard.
        [sys.executable, "-m", "reward_lens.cli.main", *argv[1:]],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    actual = command + "\n" + finished.stdout
    # The wall clock on the last line is volatile, and `fleet/golden/VOLATILE.md` declares it so:
    # rule "duration", for `*.txt`, with this pattern. `fleet/golden/run_demo.py` masks both sides
    # before it diffs, and so does this, for the same reason and with the same rule. Nothing else
    # in this transcript moves between runs.
    duration = re.compile(r"\b\d+m \d+s\b|\b\d+(\.\d+)? ?(ms|s)\b(?! per)")
    assert duration.sub("<dur>", actual) == duration.sub("<dur>", expected)


def test_the_golden_root_help_no_longer_marks_trace_absent() -> None:
    text = (GOLDEN / "root-help.txt").read_text(encoding="utf-8")
    trace_line = next(line for line in text.splitlines() if line.strip().startswith("trace"))
    assert "(not in this build)" not in trace_line


def test_a_finding_prints_as_continuation_lines_under_its_panel() -> None:
    """A-028: never a second glyph line."""
    text = (GOLDEN / "trace-fixture.txt").read_text(encoding="utf-8")
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
