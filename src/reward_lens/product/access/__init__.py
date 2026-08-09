"""The doctor and the access ladder (D-20, D-37, section 5.6).

`reward_lens.api.doctor` reaches `doctor` here through the dispatch seam; P-CLI's `doctor` verb and
`audit --dry-run` reach `render_text`, `render_json` and `dry_run`.
"""

from __future__ import annotations

from .doctor import (
    INTERFACE_PANELS,
    doctor,
    dry_run,
    render_json,
    render_text,
)
from .endpoints import Endpoint, scan
from .ladder import Resolution, resolve, rung_ids, sandbox_probe
from .matrix import Conformance, conformance
from .matrix import build as build_matrix

__all__ = [
    "Conformance",
    "Endpoint",
    "INTERFACE_PANELS",
    "Resolution",
    "build_matrix",
    "conformance",
    "doctor",
    "dry_run",
    "render_json",
    "render_text",
    "resolve",
    "rung_ids",
    "sandbox_probe",
    "scan",
]
