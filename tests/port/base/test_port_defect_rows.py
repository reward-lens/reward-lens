"""Every defect row names a site that is still in this tree.

This is the one thing about the rows worth asserting from here. A row's disposition is a judgement
and a test cannot check a judgement; what a test can check is that the evidence it cites has not
moved or been deleted underneath it, so that a later packet reading `assigned:P-PORT-4` off a row
finds the file the row is about.

Rows 5 and 6 used to carry a test each asserting that the ledger recorded an obligation and that
the handoff quoted the diff. Those tests asserted the ledger's own text and passed whether or not
the defect was ever fixed. They are gone: the rows are `assigned:P-PORT-4`, which the wave-1 exit
refuses until the packet that owns those paths applies the fix and writes the behavioural test.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEDGER = json.loads(
    (ROOT / "tests" / "fixtures" / "port" / "base-defects.json").read_text(encoding="utf-8")
)


def test_the_six_defect_rows_are_present_and_each_cites_a_file_that_exists() -> None:
    assert [d["n"] for d in LEDGER["defects"]] == [1, 2, 3, 4, 5, 6]
    for d in LEDGER["defects"]:
        path, _, line = d["evidence"].rpartition(":")
        assert (ROOT / path).is_file(), f"defect {d['n']}: {d['evidence']} does not resolve"
        assert line.isdigit()


def test_an_assigned_row_names_the_packet_and_what_it_hands_over() -> None:
    """An assignment that does not say what is being handed over is not an assignment."""
    for d in LEDGER["defects"]:
        disposition = str(d["disposition"])
        if not disposition.startswith("assigned:"):
            continue
        assert disposition.split(":", 1)[1].strip(), f"defect {d['n']}: assigned to nobody"
        assert d.get("hunks"), f"defect {d['n']}: assigned without the hunks it hands over"
        assert "test" not in d, (
            f"defect {d['n']}: an assigned row cannot also name a test that carries it"
        )
