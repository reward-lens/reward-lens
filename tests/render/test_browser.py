"""The stranger test: open the file from file:// with no network and read the first screen.

Chrome runs with `MAP * ~NOTFOUND`, so nothing resolves; the page has to be everything it needs.
Every assertion here is on the DOM after the script ran, because the rendering rules of section 6.4
are about what a reader sees, not about what the shell contains.
"""

from __future__ import annotations

import copy
import json
import re
import sys

import pytest
from conftest import (
    ROOT,
    declare_tables,
    dump_dom,
    embedded_record,
    inflate,
    parse,
    plan_and_seal,
    sample_rows,
)

from reward_lens.contracts import canonical_bytes
from reward_lens.render.report import parquet, render_to, tiers
from reward_lens.store.project import digest

SECTION_ORDER = ["answer", "subject", "findings", "evidence", "holes", "provenance"]


@pytest.fixture(scope="module")
def rendered(reference, tmp_path_factory):
    out = tmp_path_factory.mktemp("report") / "reference.assay.html"
    render_to(reference, out, bundle_manifest_digest="sha256:" + "ef" * 32)
    return parse(dump_dom(out, out.parent / "profile"))


@pytest.fixture(scope="module")
def no_finding(reference, tmp_path_factory):
    record = copy.deepcopy(reference)
    record["findings"] = []
    record["rules"] = []
    out = tmp_path_factory.mktemp("clean") / "clean.assay.html"
    render_to(record, out)
    return parse(dump_dom(out, out.parent / "profile"))


def test_the_first_screen_renders_with_the_network_off(rendered) -> None:
    answer = rendered.find_all(attr="data-section")
    assert answer, "nothing rendered: the page is a shell"
    first = [n for n in answer if n.attrs["data-section"] == "answer"]
    assert len(first) == 1
    assert len(first[0].text) > 40
    assert "code-reward" in rendered.text


def test_the_six_sections_are_in_the_order_6_4_fixes(rendered) -> None:
    seen = [n.attrs["data-section"] for n in rendered.find_all(attr="data-section")]
    assert seen == SECTION_ORDER


def test_the_scope_is_on_the_first_screen_and_not_in_a_footer(rendered) -> None:
    answer = [n for n in rendered.find_all(attr="data-section") if n.attrs["data-section"] == "answer"][0]
    scope = answer.find_all(attr="data-scope")
    assert scope, "a certificate without its scope is a lie"
    assert "evaluator comparison" in scope[0].text.lower()


def test_every_status_is_a_word_and_a_glyph_and_a_colour(rendered) -> None:
    statuses = rendered.find_all(attr="data-status")
    assert len(statuses) >= 4
    for node in statuses:
        glyphs = node.find_all(cls="glyph")
        words = node.find_all(cls="word")
        assert glyphs and words, f"status {node.attrs['data-status']!r} is missing a glyph or a word"
        assert glyphs[0].attrs.get("aria-hidden") == "true"
        assert words[0].text.strip()
        assert node.attrs.get("data-tone") in {"pass", "fail", "warn", "absent", "info"}


def test_not_measured_is_never_zero_and_never_a_pass(rendered) -> None:
    absences = [n for n in rendered.find_all(attr="data-entry") if n.attrs.get("data-state") == "absent"]
    assert absences
    for node in absences:
        value = node.find_all(attr="data-value")
        assert not value, "an absence has no value to draw"
        assert "NOT MEASURED" in node.text
        assert node.attrs.get("data-tone") == "absent"
    assert "NOT MEASURED" in rendered.text


def test_an_interval_is_drawn_wherever_a_point_estimate_is(rendered) -> None:
    estimates = rendered.find_all(attr="data-estimate")
    assert estimates
    for node in estimates:
        interval = node.find_all(cls="interval")
        assert interval, f"{node.attrs['data-estimate']} draws a point with no interval"
        assert interval[0].text.strip()


def test_the_holes_carry_their_missing_access_and_remedy(rendered) -> None:
    holes = [n for n in rendered.find_all(attr="data-section") if n.attrs["data-section"] == "holes"][0]
    assert "sampled responses grouped by prompt" in holes.text
    assert "reward-lens audit ./demo --responses r.jsonl" in holes.text


def test_provenance_carries_the_bundle_digest_offline_state_and_sandbox_tier(rendered) -> None:
    assert "sha256:" + "ef" * 32 in rendered.text
    prov = [n for n in rendered.find_all(attr="data-section") if n.attrs["data-section"] == "provenance"][0]
    assert "L2" in prov.text
    assert "offline" in prov.text.lower()


def test_a_panel_that_could_not_detect_states_its_limitation(rendered) -> None:
    assert "fresh state only" in rendered.text


def test_a_record_with_no_finding_renders_the_6_2_sentence(no_finding) -> None:
    assert "No blocking finding under this protocol" in no_finding.text
    findings = [n for n in no_finding.find_all(attr="data-section") if n.attrs["data-section"] == "findings"][0]
    named = findings.find_all(attr="data-holes")
    assert named, "the sentence stands beside the holes the record names"
    assert "signal.contrast_fraction" in named[0].attrs["data-holes"]
    assert "NOT MEASURED" in no_finding.text, "and beside its holes"
    assert not no_finding.find_all(cls="banner-green")


def test_every_chart_has_a_table_alternative(rendered) -> None:
    for chart in rendered.find_all(attr="data-chart"):
        assert chart.find_all(tag="table"), f"chart {chart.attrs['data-chart']} has no table alternative"
        assert chart.find_all(tag="svg"), "the report's marks are SVG, not canvas"


def test_a_truncated_report_says_it_is_truncated(
    reference, tmp_path_factory, monkeypatch
) -> None:
    """Tier C in a browser, with no constant patched: the writer tier B needs is not installed.

    A record that reaches tier C by size alone is a 26 MB page, and what this test is for is the
    sentence on the page rather than Chrome's parse time. Taking `pyarrow` away is the other real
    road to tier C, it is the one a reader without the `trace` extra is on, and the record that
    comes out of the plan declares C for a reason the page prints.
    """
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    record = declare_tables(
        inflate(copy.deepcopy(reference), entries=30, filler=200_000), ["samples", "groups"]
    )
    stamp = "sha256:" + "ab" * 32
    sealed = plan_and_seal(record, bundle_manifest_digest=stamp)
    assert sealed["embedding"]["tier"] == "C"
    assert len(canonical_bytes(sealed)) > tiers.HARD_A
    out = tmp_path_factory.mktemp("tierc") / "big.assay.html"
    render_to(sealed, out, bundle_manifest_digest=stamp)
    embedded = embedded_record(out.read_bytes())
    assert embedded == sealed, "the renderer never edits what it embeds"
    dom = parse(dump_dom(out, out.parent / "profile"))
    assert "2 of 2 tables are in the bundle" in dom.text
    assert "needs pyarrow" in dom.text
    assert dom.find_all(attr="data-truncated")
    assert "code-reward" in dom.text, "the page still opens and reads its record back"
    assert dom.find_all(attr="data-section"), "a summary page is still a page"
    assert not dom.find_all(attr="data-table-rows"), "tier C carries no block to read"


TIER_B_PAYLOADS = {name: sample_rows(name, 120) for name in ("samples", "groups")}


@pytest.fixture(scope="module")
def tier_b_dom(reference, tmp_path_factory):
    """A tier B page whose blocks are the tables themselves, supplied through the table source.

    This fixture used to hand the renderer nothing and let it write the record's twelve sample
    rows into each block, so the page read `12 of 25000 rows` under a notice that said nothing
    was missing. A preview is not the table: the blocks here are the complete payloads, and the
    row count the page prints is the row count the record declares.
    """
    record = declare_tables(
        inflate(copy.deepcopy(reference), entries=30, filler=200_000),
        ["samples", "groups"],
        rows=120,
        inline=12,
    )
    stamp = "sha256:" + "ba" * 32
    sealed = plan_and_seal(record, bundle_manifest_digest=stamp, tables=TIER_B_PAYLOADS)
    assert sealed["embedding"]["tier"] == "B"
    assert sealed["embedding"]["omitted_tables"] == []
    out = tmp_path_factory.mktemp("tierb") / "tables.assay.html"
    render_to(sealed, out, bundle_manifest_digest=stamp, tables=TIER_B_PAYLOADS)
    return parse(dump_dom(out, out.parent / "profile"))


def test_tier_b_reads_its_parquet_blocks_in_the_page(tier_b_dom) -> None:
    """The differentiator: rows out of a Parquet block, from file://, with nothing resolving."""
    shown = {n.attrs["data-table-rows"]: n for n in tier_b_dom.find_all(attr="data-table-rows")}
    assert set(shown) == {"samples", "groups"}
    assert not tier_b_dom.find_all(attr="data-table-error")
    assert not tier_b_dom.find_all(attr="data-table-pending")
    for name, node in shown.items():
        assert node.attrs["data-rows"] == "120", "the block is the table, not the preview"
        assert f"{name}: 120 of 120 rows, read from the Parquet block in this file" in node.text
        assert f"{name}-0000" in node.text, "the row values themselves reached the page"
        assert f"{name}-0119" in node.text, "and the rows the preview never held"
    assert "in this file, as a Parquet block" in tier_b_dom.text
    assert "Nothing is missing and nothing is fetched." in tier_b_dom.text


# --- what the record does not carry, and what it carries outside [0, 1] (review answer 4) ---


def _absence(template: dict, entry_id: str, section: str, state: str, missing: str) -> dict:
    entry = copy.deepcopy(template)
    entry["entry_id"] = entry_id
    entry["section"] = section
    entry["measurand"] = "whether the grader keeps its word here"
    entry["method"] = dict(entry["method"], id=f"rl.{section}.probe", procedure="not run")
    entry["limitations"] = []
    entry["absence"] = {
        "state": state,
        "missing_access": missing,
        "affected_claims": ["whether this reward can be trusted for that claim"],
        "remedy": "reward-lens audit with the access named above",
    }
    return entry


def _seconds(template: dict, entry_id: str, value: float, interval: list[float]) -> dict:
    entry = copy.deepcopy(template)
    entry["entry_id"] = entry_id
    entry["section"] = "reward_statistics"
    entry["measurand"] = "wall time the grader takes on one response"
    entry["value"] = value
    entry["unit"] = "seconds"
    entry["uncertainty"] = {
        "interval": interval,
        "method": "task_level_bootstrap",
        "level": 0.95,
    }
    return entry


@pytest.fixture(scope="module")
def three_kinds_record(reference) -> dict:
    """One record carrying all three of the renderings review answer 4 asked for.

    Three absences, one per hole state; a D-75 estimate whose uncertainty states that no interval
    is available; and two values well outside [0, 1] in a unit that is not a fraction.
    """
    record = copy.deepcopy(reference)
    estimate = record["measurement"]["soundness"][0]
    absence = record["measurement"]["signal"][0]

    thin = copy.deepcopy(estimate)
    thin["entry_id"] = "soundness.cluster_thin"
    thin["value"] = 0.42
    thin["n"] = 8
    thin["sampling_unit"] = "cluster"
    thin["uncertainty"] = {
        "method": "no_interval_below_15_clusters",
        "level": 0.95,
        "reason": "fewer clusters than the method needs",
        "clusters": 8,
    }
    record["measurement"]["soundness"].append(thin)

    record["measurement"]["reward_statistics"] = [
        _seconds(estimate, "reward_statistics.grader_wall_median", 47.5, [40.0, 55.0]),
        _seconds(estimate, "reward_statistics.grader_wall_worst", 95.0, [90.0, 99.0]),
    ]

    holes = [
        ("reach.holdout_sweep", "reach", "NOT_MEASURED", "held-out prompts the grader never saw"),
        ("exploits.seed_probe", "exploits", "COULD_NOT_CHECK", "a sandbox that may execute code"),
        ("framing.vendor_prompt", "framing", "REFUSED", "the vendor prompt, which the vendor withheld"),
    ]
    for entry_id, section, state, missing in holes:
        record["measurement"][section] = [_absence(absence, entry_id, section, state, missing)]
        record["holes"].append(
            {
                "section": section,
                "entry_id": entry_id,
                "state": state,
                "missing_access": missing,
                "affected_claims": ["whether this reward can be trusted for that claim"],
                "remedy": "reward-lens audit with the access named above",
            }
        )
    return record


@pytest.fixture(scope="module")
def three_kinds(three_kinds_record, tmp_path_factory):
    out = tmp_path_factory.mktemp("kinds") / "kinds.assay.html"
    render_to(three_kinds_record, out)
    return parse(dump_dom(out, out.parent / "profile"))


def _row(dom, entry_id: str):
    hits = [n for n in dom.find_all(attr="data-entry") if n.attrs["data-entry"] == entry_id]
    assert hits, f"{entry_id} is not on the page at all"
    return hits[0]


def _mark(dom, label: str):
    return [n for n in dom.find_all(attr="data-mark") if n.attrs["data-mark"] == label]


def test_an_absence_names_the_missing_access_and_is_never_a_zero(three_kinds) -> None:
    for entry_id, word in (
        ("reach.holdout_sweep", "NOT MEASURED"),
        ("exploits.seed_probe", "COULD NOT CHECK"),
        ("framing.vendor_prompt", "REFUSED"),
    ):
        row = _row(three_kinds, entry_id)
        assert word in row.text
        assert not row.find_all(attr="data-value"), "an absence has no value to render"
        assert "0" not in row.text, "Number(value ?? 0) is what this test exists to refuse"
        absent = row.find_all(attr="data-absent")
        assert absent, f"{entry_id} does not say what state it is in"
        assert "missing access" in absent[0].text
        assert not _mark(three_kinds, entry_id), "an unmeasured entry is drawn nowhere"


def test_an_unbounded_estimate_keeps_its_value_and_draws_no_bar(three_kinds) -> None:
    row = _row(three_kinds, "soundness.cluster_thin")
    assert "0.42" in row.text
    assert "no interval" in row.text
    assert "8 clusters" in row.text
    marks = _mark(three_kinds, "soundness.cluster_thin")
    assert marks, "a value the record carries is drawn"
    assert marks[0].find_all(tag="circle"), "the point stands"
    assert not marks[0].find_all(cls="whisker"), "no bar where the record states there is none"


def test_a_value_outside_zero_to_one_is_drawn_where_it_falls(three_kinds) -> None:
    row = _row(three_kinds, "reward_statistics.grader_wall_worst")
    assert "95" in row.text and "seconds" in row.text
    dots = []
    for label in ("reward_statistics.grader_wall_median", "reward_statistics.grader_wall_worst"):
        marks = _mark(three_kinds, label)
        assert marks, f"{label} carries a value and is drawn"
        circles = marks[0].find_all(tag="circle")
        assert circles, f"{label} has no point"
        dots.append(float(circles[0].attrs["cx"]))
    assert dots[0] != dots[1], "clamping into [0, 1] drew every value above 1 in one place"
    charts = [n for n in three_kinds.find_all(attr="data-chart")]
    seconds = [c for c in charts if c.attrs["data-chart"] == "estimates-seconds"]
    assert seconds, "estimates are grouped by unit; a share and a second share no axis"
    domain = seconds[0].find_all(attr="data-domain")
    assert domain and domain[0].attrs["data-domain"] == "40:99"
    assert "40" in domain[0].text and "99" in domain[0].text


def test_no_number_on_the_findings_panel_was_computed_in_the_browser(rendered, reference) -> None:
    """Review answer 3: the removed rate is gone from the source and from the built bundle."""
    src = ROOT / "frontend" / "report" / "src"
    app = (src / "App.tsx").read_text(encoding="utf-8")
    assert "executed / rows.length" not in app, "the per-section rate is the removed computation"
    assert "Executed share" not in app and "data-coverage" not in app
    ui = (src / "ui.tsx").read_text(encoding="utf-8").replace(" ", "")
    assert "Math.min(1" not in ui, "the mark clamp is gone"
    bundle = (ROOT / "src" / "reward_lens" / "render" / "report" / "assets" / "report.js")
    built = bundle.read_text(encoding="utf-8")
    assert "Executed share of each section" not in built
    assert "data-coverage" not in built
    panel = [
        n for n in rendered.find_all(attr="data-section") if n.attrs["data-section"] == "findings"
    ][0]
    carried = json.dumps(reference, ensure_ascii=False)
    for run in re.findall(r"\d+", panel.text):
        assert run in carried, f"{run!r} is on the panel and not in the record"


def test_a_tier_c_reached_by_the_ceiling_states_its_omission_in_the_browser(
    over_the_ceiling, tmp_path_factory
) -> None:
    """The other road to tier C, and the one a reader with the `trace` extra installed is on.

    The test above takes the Parquet writer away, which is a fact about an installation. This
    record reaches tier C with the writer in place, by being a page that will not fit: twenty-six
    megabytes through Chrome, the tables left in the bundle, and the reader told so on the page
    rather than in a footnote. It is the expensive one, which is why it is one test.
    """
    assert parquet.available(), "this record has to reach tier C by size, not by a missing writer"
    assert over_the_ceiling["embedding"]["tier"] == "C"
    out = tmp_path_factory.mktemp("ceiling") / "over.assay.html"
    render_to(over_the_ceiling, out)
    assert out.stat().st_size > tiers.CEILING_B, "the point of this record is that it is too big"
    embedded = embedded_record(out.read_bytes())
    assert embedded == over_the_ceiling, "the renderer never edits what it embeds"
    assert digest(embedded) == over_the_ceiling["assay_id"]
    dom = parse(dump_dom(out, out.parent / "profile"))
    assert "2 of 2 tables are in the bundle, not here" in dom.text
    assert "past the 26,214,400-byte ceiling" in dom.text
    listed = [node.attrs["data-omitted"] for node in dom.find_all(attr="data-omitted")]
    assert listed == ["omitted-table:samples", "omitted-table:groups"]
    assert dom.find_all(attr="data-truncated")
    assert not dom.find_all(attr="data-table-rows"), "tier C carries no block to read"
    assert dom.find_all(attr="data-section"), "a summary page is still a page"
