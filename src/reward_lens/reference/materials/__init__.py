"""Compatibility names for adopted reference material.

``diagnostic_v3.py`` in this directory is an integrity-anchored byte-for-byte adopted payload.
It must remain executable by :mod:`reward_lens.reference.load` in its isolated verifier namespace,
but importing that payload as a normal library module would register its dataset loaders a second
time.  Bind the historical import path to the canonical builtin module instead.  The registry keeps
its duplicate-key refusal; normal consumers simply reach the one builder that owns those keys.
"""

from __future__ import annotations

import sys

from reward_lens.data.builtin import diagnostic_v3 as _canonical_diagnostic_v3

sys.modules[f"{__name__}.diagnostic_v3"] = _canonical_diagnostic_v3
