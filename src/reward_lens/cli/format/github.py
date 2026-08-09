"""`--format github`: GitHub workflow annotations, written by the export engine.

Not in this build, and it refuses rather than falling back (D-23). P-EXPORT replaces the body with
the `::error file=...,line=...,title=...::` lines of section 5.7.
"""

from __future__ import annotations


def render(document: dict) -> str:
    from ..output import _export_unavailable

    raise _export_unavailable("github")
