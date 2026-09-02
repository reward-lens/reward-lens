"""The record survives the trip into the page (D-15).

A JSON data block is raw text to the HTML parser except for the `</script` terminator, so the
escaping is what keeps the record readable and the page unbroken. Never HTML-escape the block: a
JSON block does not decode entities.
"""

from __future__ import annotations

import json

from conftest import JSON_BLOCK, embedded_record

from reward_lens.render.report import embed, render

HOSTILE = 'a </script><img src=x> b   c   d <!-- e --> f & g'


def test_escaper_leaves_no_raw_angle_bracket_or_separator() -> None:
    escaped = embed.escape_json_text(json.dumps({"k": HOSTILE}))
    assert "<" not in escaped
    assert ">" not in escaped
    assert " " not in escaped
    assert " " not in escaped
    assert json.loads(escaped)["k"] == HOSTILE


def test_hostile_string_survives_the_round_trip(record: dict) -> None:
    record["intent"]["success"] = HOSTILE
    html = render(record)
    text = html.decode("utf-8")
    block = JSON_BLOCK.search(text)
    assert block is not None
    assert "</script>" not in block.group(1)
    assert embedded_record(html)["intent"]["success"] == HOSTILE


def test_the_block_is_the_only_place_the_record_lives(record: dict) -> None:
    html = render(record)
    assert embedded_record(html)["subject"]["reward_system"]["id"] == "code-reward"
    assert html.decode("utf-8").count('id="assay"') == 1


def test_the_json_block_is_not_html_escaped(record: dict) -> None:
    record["intent"]["success"] = "a & b"
    text = render(record).decode("utf-8")
    block = JSON_BLOCK.search(text).group(1)
    assert "&amp;" not in block
    assert "\\u0026" in block
