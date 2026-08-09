"""`--format json`: one envelope on stdout and nothing else (D-21, D-23).

The serialisation itself lives in `output.Output.deliver`, because stdout, `--fields` and the exit
code are one decision and splitting them is how a warning ends up inside a payload.
"""

from __future__ import annotations

import json as _json

SCHEMA = "assay-result/1.0"


def render(document: dict) -> str:
    return _json.dumps(document, indent=2, sort_keys=False)
