"""Carrying the record into the page without breaking either of them (D-15).

A `<script type="application/json">` block is raw text to the HTML parser: character references
are not decoded there, so HTML-escaping the payload would corrupt it. The one terminator that
matters is `</script`, and the two Unicode line separators break a JavaScript string context, so
the escape set is `<`, `>`, `&`, U+2028 and U+2029, each written as its own JSON `\\uXXXX`, which
leaves the text valid JSON and identical after `JSON.parse`.
"""

from __future__ import annotations

ESCAPES = {
    "<": "\\u003c",
    ">": "\\u003e",
    "&": "\\u0026",
    " ": "\\u2028",
    " ": "\\u2029",
}


def escape_json_text(text: str) -> str:
    """Escape a JSON document so it can sit inside a script block unchanged."""
    for raw, escaped in ESCAPES.items():
        text = text.replace(raw, escaped)
    return text


def escape_attribute(value: str) -> str:
    """The ordinary HTML attribute escape, for the few values the shell writes itself."""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def escape_text(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
