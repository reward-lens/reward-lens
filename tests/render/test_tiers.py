"""The three bands of D-15, measured on the two objects they are actually limits on.

Nothing here monkeypatches a size constant. The boundary records are built until the plan itself
puts them on either side of the line, so the tier these records fall in is the tier a reader's
record would fall in, and the byte sizes in the assertions are the byte sizes of files that were
written to disk.
"""

from __future__ import annotations

import copy
import sys

import pytest
from conftest import (
    block_rows,
    declare_tables,
    embedded_record,
    has_block,
    inflate,
    plan_and_seal,
    sample_rows,
    seal,
)

from reward_lens.contracts import canonical_bytes
from reward_lens.contracts.errors import RecordInvalid
from reward_lens.render.report import plan_embedding, render, render_to, tiers
from reward_lens.store.project import digest

MB = 1024 * 1024


def test_the_constants_are_the_ones_d15_names() -> None:
    assert tiers.SOFT_A == 2 * MB
    assert tiers.HARD_A == 5 * MB
    assert tiers.CEILING_B == 26_214_400


def test_the_two_limits_are_on_two_different_objects() -> None:
    """The defect this module was rewritten on: a file ceiling applied to the canonical JSON."""
    assert tiers.ceiling_of("A") == ("the canonical JSON of the record", tiers.HARD_A)
    assert tiers.ceiling_of("B") == ("the whole HTML file", tiers.CEILING_B)
    assert tiers.ceiling_of("C") is None


def test_the_renderer_does_not_edit_what_it_embeds(record: dict) -> None:
    sealed = plan_and_seal(record)
    before = copy.deepcopy(sealed)
    render(sealed)
    assert sealed == before


# ---------------------------------------------------------------------------------------------
# The records the bands are read off
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tier_b_record(reference: dict) -> dict:
    """Past the 5 MB hard line on the canonical JSON, so the tables would travel as Parquet.

    Nothing supplies their payloads, so the record this fixture seals is the honest tier B a
    caller gets with the bundle nowhere in reach: the tables are named under `omitted_tables`
    and the page says they are in the bundle, rather than a block of the twelve sample rows.
    """
    record = copy.deepcopy(reference)
    declare_tables(record, ["samples", "groups"])
    inflate(record, entries=30, filler=200_000)
    sealed = plan_and_seal(record)
    assert len(canonical_bytes(sealed)) > tiers.HARD_A
    assert sealed["embedding"]["tier"] == "B"
    assert sealed["embedding"]["omitted_tables"] == ["samples", "groups"]
    return sealed


def claiming_b(record: dict) -> tuple[dict, dict]:
    """The same record declaring tier B with nothing omitted, and the payloads to let it try.

    The renderer embeds a table only from that table's complete payload, so a record claiming
    every table is in the file is refused for the missing payload long before anything is weighed.
    Handing it payloads that match what it declares is what puts the ceiling back in the dock.
    """
    lying = copy.deepcopy(record)
    lying["embedding"] = {"tier": "B", "omitted_tables": []}
    for table in lying["tables"]:
        table["rows"] = 12
    return lying, {table["id"]: sample_rows(table["id"], 12) for table in lying["tables"]}


# ---------------------------------------------------------------------------------------------
# (a) the embedded record is the sealed record, at every tier
# ---------------------------------------------------------------------------------------------


def test_tier_a_embeds_the_sealed_record_byte_for_byte(record: dict) -> None:
    sealed = plan_and_seal(record)
    assert sealed["embedding"]["tier"] == "A"
    embedded = embedded_record(render(sealed))
    assert embedded == sealed
    assert digest(embedded) == sealed["assay_id"]


def test_tier_b_embeds_the_sealed_record_byte_for_byte(tier_b_record: dict) -> None:
    embedded = embedded_record(render(tier_b_record))
    assert embedded == tier_b_record
    assert digest(embedded) == tier_b_record["assay_id"]


def test_tier_c_embeds_the_sealed_record_byte_for_byte(over_the_ceiling: dict) -> None:
    assert over_the_ceiling["embedding"]["tier"] == "C"
    embedded = embedded_record(render(over_the_ceiling))
    assert embedded == over_the_ceiling
    assert digest(embedded) == over_the_ceiling["assay_id"]


def test_the_store_would_accept_every_record_the_page_carries(
    record: dict, tier_b_record: dict, over_the_ceiling: dict
) -> None:
    """The store's own check is `digest(data) == data["assay_id"]`; the page keeps that true."""
    for sealed in (plan_and_seal(record), tier_b_record, over_the_ceiling):
        assert digest(seal(embedded_record(render(sealed)))) == sealed["assay_id"]


# ---------------------------------------------------------------------------------------------
# (b) the sizes of the files that were written
# ---------------------------------------------------------------------------------------------


def test_the_file_just_under_the_ceiling_is_under_it_when_it_is_weighed(
    under_the_ceiling: dict, tmp_path_factory: pytest.TempPathFactory
) -> None:
    assert under_the_ceiling["embedding"]["tier"] == "B"
    out = render_to(under_the_ceiling, tmp_path_factory.mktemp("under") / "report.html")
    written = out.stat().st_size
    assert written <= tiers.CEILING_B, f"{written} bytes written against {tiers.CEILING_B}"
    assert written == under_the_ceiling["embedding"]["html_bytes"]
    assert tiers.CEILING_B - written < 4096, "the boundary record is meant to sit on the line"


def test_the_record_just_over_the_ceiling_renders_as_c_and_says_so(
    over_the_ceiling: dict, tmp_path_factory: pytest.TempPathFactory
) -> None:
    assert over_the_ceiling["embedding"]["tier"] == "C"
    assert over_the_ceiling["embedding"]["omitted_tables"] == ["samples", "groups"]
    out = render_to(over_the_ceiling, tmp_path_factory.mktemp("over") / "report.html")
    text = out.read_bytes().decode("utf-8")
    assert "2 of 2 tables are in the bundle" in text
    assert 'id="table:samples"' not in text
    assert out.stat().st_size == over_the_ceiling["embedding"]["html_bytes"]


def test_the_canonical_json_is_not_what_the_ceiling_is_on(over_the_ceiling: dict) -> None:
    """The old rule called this record tier B: its canonical JSON is under 25 MB, its file is not.

    The record the plan sent to tier C still has a canonical JSON under the ceiling, so a rule
    measured on the record would have inlined it and written a file past the line. Declaring B and
    asking the renderer to honour it is what weighs the file, and the file is what is over.
    """
    assert len(canonical_bytes(over_the_ceiling)) < tiers.CEILING_B
    lying, payloads = claiming_b(over_the_ceiling)
    with pytest.raises(RecordInvalid) as raised:
        render(lying, tables=payloads)
    assert raised.value.context["measured_bytes"] > tiers.CEILING_B


# ---------------------------------------------------------------------------------------------
# (c) the measurement is a fixed point
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["record", "tier_b_record", "over_the_ceiling"])
def test_planning_a_planned_record_returns_the_same_plan(fixture: str, request) -> None:
    subject = request.getfixturevalue(fixture)
    first = plan_embedding(subject)
    replanned = dict(subject, embedding=first.to_dict())
    assert plan_embedding(replanned) == first


def test_the_plan_settles_and_says_in_how_many_passes(record: dict) -> None:
    plan = plan_embedding(record)
    assert 1 <= plan.passes <= 5
    assert plan.html_bytes == len(render(seal(dict(record, embedding=plan.to_dict()))))


# ---------------------------------------------------------------------------------------------
# (d) the refusal
# ---------------------------------------------------------------------------------------------


def test_a_record_declaring_b_whose_file_is_over_the_ceiling_is_refused(
    over_the_ceiling: dict,
) -> None:
    lying, payloads = claiming_b(over_the_ceiling)
    with pytest.raises(RecordInvalid) as raised:
        render(lying, tables=payloads)
    assert raised.value.code == "RL0604"
    assert raised.value.context["declared_tier"] == "B"
    assert raised.value.context["ceiling_bytes"] == tiers.CEILING_B
    assert raised.value.context["measured_bytes"] > tiers.CEILING_B
    assert "the whole HTML file" in str(raised.value)


def test_a_record_declaring_a_whose_json_is_over_the_hard_line_is_refused(
    tier_b_record: dict,
) -> None:
    lying = copy.deepcopy(tier_b_record)
    lying["embedding"] = {"tier": "A", "omitted_tables": []}
    with pytest.raises(RecordInvalid) as raised:
        render(lying)
    assert raised.value.code == "RL0604"
    assert raised.value.context["ceiling_bytes"] == tiers.HARD_A
    assert "the canonical JSON" in str(raised.value)


def test_a_record_declaring_tier_a_on_a_record_bytes_it_does_not_have_is_refused(
    tier_b_record: dict,
) -> None:
    """The defect: `record_bytes` was read off the record and believed.

    A sealed record could declare tier A beside a small number it chose, and the hard line was
    then measured against the number rather than against the record. Forty megabytes rendered as
    an A. The canonical JSON is weighed every time now, and the two numbers are put side by side.
    """
    forged = copy.deepcopy(tier_b_record)
    forged["embedding"] = {
        "tier": "A",
        "omitted_tables": [],
        "record_bytes": 1,
        "html_bytes": 0,
    }
    with pytest.raises(RecordInvalid) as raised:
        render(forged)
    assert raised.value.code == "RL0604"
    assert raised.value.context["declared_field"] == "record_bytes"
    assert raised.value.context["declared_bytes"] == 1
    assert raised.value.context["measured_bytes"] > tiers.HARD_A
    assert "1 bytes" in str(raised.value)
    assert f"{raised.value.context['measured_bytes']:,} bytes" in str(raised.value)


def test_a_record_whose_html_bytes_is_off_by_one_is_refused(record: dict) -> None:
    """One byte, because a size that is nearly right is a size nobody checked."""
    sealed = plan_and_seal(record)
    honest = sealed["embedding"]["html_bytes"]
    forged = copy.deepcopy(sealed)
    forged["embedding"]["html_bytes"] = (
        honest + 1 if len(str(honest + 1)) == len(str(honest)) else honest - 1
    )
    with pytest.raises(RecordInvalid) as raised:
        render(forged)
    assert raised.value.code == "RL0604"
    assert raised.value.context["declared_field"] == "html_bytes"
    assert raised.value.context["measured_on"] == "the whole HTML file"
    assert abs(raised.value.context["declared_bytes"] - honest) == 1
    assert raised.value.context["measured_bytes"] == honest


def test_weighing_every_number_still_renders_the_records_that_were_always_honest(
    record: dict, under_the_ceiling: dict, over_the_ceiling: dict
) -> None:
    """The check earns nothing if it also refuses the reference fixture and the two boundaries."""
    for sealed, tier in (
        (plan_and_seal(record), "A"),
        (under_the_ceiling, "B"),
        (over_the_ceiling, "C"),
    ):
        assert sealed["embedding"]["tier"] == tier
        html = render(sealed)
        assert len(html) == sealed["embedding"]["html_bytes"]
        assert len(canonical_bytes(sealed)) == sealed["embedding"]["record_bytes"]


def test_nothing_is_written_when_a_record_is_refused(
    over_the_ceiling: dict, tmp_path_factory: pytest.TempPathFactory
) -> None:
    lying = copy.deepcopy(over_the_ceiling)
    lying["embedding"] = {"tier": "B", "omitted_tables": []}
    out = tmp_path_factory.mktemp("refused") / "report.html"
    with pytest.raises(RecordInvalid):
        render_to(lying, out)
    assert not out.exists()


# ---------------------------------------------------------------------------------------------
# (f) honest omissions, and (g) the tier B that cannot be written
# ---------------------------------------------------------------------------------------------


def test_tier_c_names_every_table_it_left_out(over_the_ceiling: dict) -> None:
    text = render(over_the_ceiling).decode("utf-8")
    for table in ("samples", "groups"):
        assert f"omitted-table:{table}" in text
    assert 'data-truncated="C"' in text
    assert "past the 26,214,400-byte ceiling" in text


def test_tier_b_with_every_payload_carries_every_table_and_omits_none(
    reference: dict,
) -> None:
    record = copy.deepcopy(reference)
    declare_tables(record, ["samples", "groups"], rows=120, inline=0)
    inflate(record, entries=30, filler=200_000)
    payloads = {name: sample_rows(name, 120) for name in ("samples", "groups")}
    sealed = plan_and_seal(record, tables=payloads)
    assert sealed["embedding"]["tier"] == "B"
    assert sealed["embedding"]["omitted_tables"] == []
    html = render(sealed, tables=payloads)
    for table in ("samples", "groups"):
        assert has_block(html, table)
        assert block_rows(html, table) == 120
    assert "Nothing is missing and nothing is fetched." in html.decode("utf-8")


def test_a_record_past_the_soft_line_but_under_the_hard_one_says_so(reference: dict) -> None:
    record = copy.deepcopy(reference)
    inflate(record, entries=16, filler=200_000)
    sealed = plan_and_seal(record)
    assert sealed["embedding"]["tier"] == "A"
    assert tiers.SOFT_A < sealed["embedding"]["record_bytes"] <= tiers.HARD_A
    assert "past the size a page carries comfortably" in render(sealed).decode("utf-8")


def test_without_a_parquet_writer_the_plan_chooses_c_and_the_page_says_why(
    reference: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pyarrow` lives in the `trace` extra, so its absence is a fact about the installation."""
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    record = copy.deepcopy(reference)
    declare_tables(record, ["samples", "groups"])
    inflate(record, entries=30, filler=200_000)
    sealed = plan_and_seal(record)
    assert sealed["embedding"]["tier"] == "C"
    assert sealed["embedding"]["omitted_tables"] == ["samples", "groups"]
    text = render(sealed).decode("utf-8")
    assert "2 of 2 tables are in the bundle" in text
    assert "needs pyarrow (the `trace` extra)" in text


def test_a_record_declaring_b_with_no_writer_is_refused_rather_than_re_tiered(
    tier_b_record: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    with pytest.raises(RecordInvalid) as raised:
        render(tier_b_record)
    assert raised.value.code == "RL0604"
    assert "no Parquet writer is installed" in str(raised.value)


# ---------------------------------------------------------------------------------------------
# (e) a table is in the file because its payload was supplied, or it is disclosed as absent
# ---------------------------------------------------------------------------------------------


def tier_b_table(reference: dict, *, rows: int, inline: int) -> dict:
    """A record over the hard line declaring one table of `rows` rows and `inline` sample rows."""
    record = copy.deepcopy(reference)
    declare_tables(record, ["samples"], rows=rows, inline=inline)
    inflate(record, entries=30, filler=200_000)
    return record


def test_a_declared_table_with_no_payload_is_disclosed_and_never_a_zero_row_block(
    reference: dict,
) -> None:
    """The defect: 120 declared rows, no preview, and a Parquet block holding nothing at all.

    The page said every table was carried. What it carried was an empty block, which reads as a
    table with no rows rather than as a table that is somewhere else.
    """
    sealed = plan_and_seal(tier_b_table(reference, rows=120, inline=0))
    assert sealed["embedding"]["tier"] == "B"
    assert sealed["embedding"]["omitted_tables"] == ["samples"]
    html = render(sealed)
    text = html.decode("utf-8")
    assert not has_block(html, "samples"), "an unsupplied table is disclosed, not approximated"
    assert "1 of 1 tables are in the bundle" in text
    assert "payload not supplied" in text
    assert 'data-omitted="omitted-table:samples"' in text
    assert 'data-truncated="B"' in text
    assert "Nothing is missing and nothing is fetched." not in text


def test_a_preview_alone_never_becomes_a_block(reference: dict) -> None:
    """Twelve sample rows are twelve sample rows, whatever the table says it has."""
    sealed = plan_and_seal(tier_b_table(reference, rows=120, inline=12))
    assert sealed["embedding"]["omitted_tables"] == ["samples"]
    html = render(sealed)
    assert not has_block(html, "samples")
    assert "preview only: 12 of 120 rows" in html.decode("utf-8")


def test_a_source_holding_the_whole_table_embeds_it_with_its_own_row_count(
    reference: dict,
) -> None:
    payload = {"samples": sample_rows("samples", 120)}
    sealed = plan_and_seal(tier_b_table(reference, rows=120, inline=12), tables=payload)
    assert sealed["embedding"]["tier"] == "B"
    assert sealed["embedding"]["omitted_tables"] == []
    html = render(sealed, tables=payload)
    assert block_rows(html, "samples") == 120, "the block is the payload, not the preview"
    assert "Nothing is missing and nothing is fetched." in html.decode("utf-8")


def test_a_source_short_of_the_declaration_is_refused(reference: dict) -> None:
    record = tier_b_table(reference, rows=120, inline=12)
    with pytest.raises(RecordInvalid) as raised:
        plan_and_seal(record, tables={"samples": sample_rows("samples", 100)})
    assert raised.value.code == "RL0604"
    assert raised.value.context == {
        "table": "samples",
        "declared_rows": 120,
        "found_rows": 100,
    }
    assert "samples" in str(raised.value)
    assert "120" in str(raised.value) and "100" in str(raised.value)


def test_a_record_claiming_a_table_the_render_has_no_payload_for_is_refused(
    reference: dict,
) -> None:
    """Planned with the payload and rendered without it: the page would have to make it up."""
    payload = {"samples": sample_rows("samples", 120)}
    sealed = plan_and_seal(tier_b_table(reference, rows=120, inline=12), tables=payload)
    with pytest.raises(RecordInvalid) as raised:
        render(sealed)
    assert raised.value.code == "RL0604"
    assert raised.value.context["tables"] == ["samples"]
    assert "preview only: 12 of 120 rows" in str(raised.value)


def test_a_payload_that_digests_to_something_else_is_refused(reference: dict) -> None:
    """Bytes are the one payload a digest can be taken of, so bytes are where it is checked."""
    import io

    import pyarrow.parquet as pq

    record = tier_b_table(reference, rows=120, inline=0)
    buffer = io.BytesIO()
    pq.write_table(_arrow(sample_rows("samples", 120)), buffer)
    payload = buffer.getvalue()
    with pytest.raises(RecordInvalid) as raised:
        plan_and_seal(record, tables={"samples": payload})
    assert raised.value.code == "RL0604"
    assert raised.value.context["declared_digest"] == "sha256:" + "1" * 64
    assert raised.value.context["found_digest"].startswith("sha256:")


def test_the_bundles_own_bytes_are_embedded_when_the_digest_is_the_declared_one(
    reference: dict,
) -> None:
    import hashlib
    import io

    import pyarrow.parquet as pq

    buffer = io.BytesIO()
    pq.write_table(_arrow(sample_rows("samples", 120)), buffer)
    payload = buffer.getvalue()
    record = tier_b_table(reference, rows=120, inline=0)
    record["tables"][0]["digest"] = "sha256:" + hashlib.sha256(payload).hexdigest()
    sealed = plan_and_seal(record, tables={"samples": payload})
    assert sealed["embedding"]["omitted_tables"] == []
    assert block_rows(render(sealed, tables={"samples": payload}), "samples") == 120


def _arrow(rows: list[dict]):
    import pyarrow as pa

    return pa.Table.from_pylist(rows)
