"""One file, no network, and a refusal before any byte of it exists."""

from __future__ import annotations

import json

import pytest
from conftest import JSON_BLOCK, embedded_record

from reward_lens.contracts import RecordInvalid, canonical_bytes
from reward_lens.render.report import render, render_to

FORBIDDEN = (
    'href="http',
    'src="http',
    "fetch(",
    "import(",
    "new Worker",
    'type="module"',
    "importScripts",
    "XMLHttpRequest",
    "WebAssembly",
    "@import url",
    "cdn.jsdelivr",
    "fonts.googleapis",
)


@pytest.fixture(scope="module")
def html(reference: dict) -> bytes:
    return render(reference)


def test_it_is_one_html_document(html: bytes) -> None:
    text = html.decode("utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert text.rstrip().endswith("</html>")
    assert '<script type="application/json" id="assay">' in text


def test_nothing_in_the_page_reaches_off_the_page(html: bytes) -> None:
    text = html.decode("utf-8")
    for needle in FORBIDDEN:
        assert needle not in text, f"the single file must not contain {needle!r}"


def test_the_script_is_classic_and_there_is_exactly_one(html: bytes) -> None:
    """Two script elements: the record, which is data, and the bundle, which is a classic script.

    Counting the substring will not do it, because React's own minified source carries the text of
    a script tag. The shell writes each element at the start of its own line, so the test reads
    the lines rather than the text.
    """
    lines = html.decode("utf-8").splitlines()
    opens = [line for line in lines if line.startswith("<script")]
    assert len(opens) == 2
    assert lines.count("<script>") == 1
    assert sum(line.startswith('<script type="application/json"') for line in opens) == 1
    assert lines.count("</script>") == 1  # the bundle; the JSON block closes on its own line


def test_sizes_are_inside_tier_a(reference: dict, html: bytes) -> None:
    assert len(canonical_bytes(reference)) < 2 * 1024 * 1024
    assert len(html) < 8 * 1024 * 1024


def test_the_font_stack_is_the_system_one_and_no_font_is_embedded(html: bytes) -> None:
    text = html.decode("utf-8")
    assert "@font-face" not in text
    assert "data:font/" not in text
    assert "ui-sans-serif" in text
    assert "ui-monospace" in text


def test_print_and_dark_mode_are_css_not_javascript(html: bytes) -> None:
    text = html.decode("utf-8")
    assert "@media print" in text
    assert "print-color-adjust: exact" in text
    assert "-webkit-print-color-adjust: exact" in text
    assert "@page" in text
    assert "prefers-color-scheme: dark" in text
    assert "color-scheme:" in text


def test_the_bundle_manifest_digest_appears_when_given(reference: dict) -> None:
    digest = "sha256:" + "ab" * 32
    text = render(reference, bundle_manifest_digest=digest).decode("utf-8")
    assert digest in text
    assert digest not in JSON_BLOCK.search(text).group(1), (
        "D-16: the record never references the bundle; the digest is shown, not stored"
    )


def test_an_invalid_record_is_refused_with_rl0604_before_any_html(record: dict, tmp_path) -> None:
    record["measurement"]["validity"][0]["kind"] = "not-a-kind"
    out = tmp_path / "x.assay.html"
    with pytest.raises(RecordInvalid) as caught:
        render_to(record, out)
    assert caught.value.code == "RL0604"
    assert caught.value.exit_code == 4
    assert not out.exists()


def test_render_to_writes_the_bytes_render_returns(reference: dict, tmp_path) -> None:
    out = render_to(reference, tmp_path / "r.assay.html")
    assert out.read_bytes() == render(reference)


def test_the_record_is_not_mutated_by_rendering(reference: dict) -> None:
    before = json.dumps(reference, sort_keys=True)
    render(reference)
    assert json.dumps(reference, sort_keys=True) == before


def test_embedding_tier_is_written_into_the_embedded_record(reference: dict, html: bytes) -> None:
    assert embedded_record(html)["embedding"]["tier"] == "A"
    assert embedded_record(html)["embedding"]["omitted_tables"] == []
