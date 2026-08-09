"""`--format sarif`: written by the export engine, which is not in this build.

A placeholder that refuses is the honest shape here. D-23 forbids a format flag that is silently
ignored, which is appendix G defect 7, so this raises RL0701 and exits 5 rather than emitting JSON
under a name the caller did not ask for. P-EXPORT replaces the body.
"""

from __future__ import annotations


def render(document: dict) -> str:
    from ..output import _export_unavailable

    raise _export_unavailable("sarif")
