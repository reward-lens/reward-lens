"""The held-out guard, tested on the failure it was rewritten for.

The guard this replaces asserted that no `*.py` file under five directories contained the substring
`SEALED`. It passed for the life of the project while three files in the tree published the held-out
answer, two of them at extensions the scan could not open. The seal was never breached; the guard was
testing the wrong thing.

So the cases below are about reconstruction, not reference. The load-bearing one is
`test_a_file_that_reproduces_a_protected_value_without_naming_it_is_caught`: a file containing the
number and not the word, which is exactly the shape that survived for a year.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reward_lens.artifacts.sealed import (
    fingerprint,
    is_fingerprintable,
    load_fingerprint,
    scan,
)

REPO = Path(__file__).resolve().parents[1]
FINGERPRINT = REPO / "e1_preflight" / "SEALED" / "leak_fingerprint.json"

#: Files permitted to carry the protected quantities: the sealed directory itself, the package that
#: writes it, and the guards. Adding a path here is a decision, not a formality.
ALLOWED = {
    "e1_preflight/SEALED/leak_fingerprint.json",
    "e1_preflight/SEALED/null_counts.json",
    "e1_preflight/SEALED/README.md",
    "experiments/e1_preflight/report.py",
    "tests/test_e1_preflight.py",
    "tests/test_sealed_guard.py",
}

#: Decisive reproductions found by this guard on the day it was written, each recorded with why it
#: is tolerated rather than silently allowed. A new entry here is a leak somebody accepted, and it
#: should be argued for in the commit that adds it.
KNOWN = {
    (
        "experiments/x3_release/evidence/evidence.jsonl",
        "rh_always_equal.adoption_share_of_hacks",
    ),
    ("experiments/x3_release/evidence/evidence.jsonl", "rh_exit.adoption_share_of_hacks"),
    ("experiments/x3_release/evidence/evidence.jsonl", "rh_conftest.adoption_share_of_hacks"),
}

SALT = "test-salt"


def _fp(scalars, arrays=None):
    return fingerprint(scalars, arrays, salt=SALT)


def test_a_file_that_reproduces_a_protected_value_without_naming_it_is_caught(tmp_path: Path):
    """The failure the old guard could not see: the number without the word."""
    payload = _fp({"secret_rate": 0.021819546453653568})
    (tmp_path / "innocent.md").write_text(
        "The adoption rate we measured was 0.021819546453653568 across the run.\n",
        encoding="utf-8",
    )
    leaks = scan(tmp_path, payload)
    assert len(leaks) == 1
    assert leaks[0].quantity == "secret_rate"
    assert leaks[0].decisive
    # And it never mentions the sealed directory, which is why the string guard passed on it.
    assert "SEALED" not in (tmp_path / "innocent.md").read_text(encoding="utf-8")


def test_every_extension_is_scanned_not_only_python(tmp_path: Path):
    """Two of the three real leaks were a `.jsonl` and an `.npz`."""
    payload = _fp({"rate": 0.0006969388302149788})
    for name in ("a.py", "b.jsonl", "c.md", "d.txt", "e.yaml", "f.json"):
        (tmp_path / name).write_text("value: 0.0006969388302149788\n", encoding="utf-8")
    leaks = scan(tmp_path, payload)
    assert {leak.path for leak in leaks} == {"a.py", "b.jsonl", "c.md", "d.txt", "e.yaml", "f.json"}


def test_a_stored_array_reproducing_a_protected_trajectory_is_caught(tmp_path: Path):
    numpy = pytest.importorskip("numpy")
    trajectory = [0.0, 0.125, 0.375, 0.625, 0.875, 1.0]
    payload = _fp({}, {"traj": trajectory})
    numpy.savez(tmp_path / "finds.npz", something_else=numpy.array(trajectory))
    leaks = scan(tmp_path, payload)
    assert [leak.quantity for leak in leaks] == ["traj"]
    assert leaks[0].channel == "array"
    assert leaks[0].decisive


def test_a_protected_trajectory_written_as_a_literal_list_is_caught(tmp_path: Path):
    trajectory = [0.0, 0.125, 0.375, 0.625, 0.875, 1.0]
    payload = _fp({}, {"traj": trajectory})
    (tmp_path / "series.json").write_text(
        json.dumps({"label": "something harmless", "values": trajectory}), encoding="utf-8"
    )
    leaks = [leak for leak in scan(tmp_path, payload) if leak.channel == "array"]
    assert [leak.quantity for leak in leaks] == ["traj"]


def test_an_interleaved_two_column_table_is_a_known_gap_in_the_array_channel(tmp_path: Path):
    """Stated as a test so the limit is checked rather than believed.

    The array channel looks for the protected values as a *contiguous* run. A two-column table
    interleaves the step index between them, so the run is broken and the channel does not fire.
    A leak in that shape is caught only if one of its individual values is distinctive enough for
    the scalar channel, which a trajectory of round fractions is not.

    Recording it as a passing test rather than a comment means that if somebody later teaches the
    scanner to stride, this test fails and tells them to update the claim.
    """
    trajectory = [0.0, 0.125, 0.375, 0.625, 0.875, 1.0]
    payload = _fp({}, {"traj": trajectory})
    (tmp_path / "table.md").write_text(
        "step | rate\n" + "\n".join(f"{i} | {v:.10g}" for i, v in enumerate(trajectory)),
        encoding="utf-8",
    )
    assert [leak for leak in scan(tmp_path, payload) if leak.channel == "array"] == []


def test_a_round_literal_does_not_match_a_protected_value_at_a_precision_it_does_not_carry(
    tmp_path: Path,
):
    """The 3,098-hit failure. `0` must not match a protected value rounded to two decimals."""
    payload = _fp({"rate": 0.00012345678})
    (tmp_path / "ordinary.py").write_text("x = 0\ny = 1\nz = 0.0\n", encoding="utf-8")
    assert scan(tmp_path, payload) == []


def test_an_integer_protected_value_is_reported_as_unguardable_rather_than_guarded():
    """An onset step cannot be fingerprinted, and the gap is named instead of hidden."""
    payload = _fp({"onset_step": 90.0, "rate": 0.0218195})
    assert payload["unguardable_scalars"] == ["onset_step"]
    assert "rate" in payload["scalars"]
    assert not is_fingerprintable(90.0)
    assert not is_fingerprintable(0.0)
    assert not is_fingerprintable(1.0)
    assert is_fingerprintable(0.0218195)


def test_the_fingerprint_file_does_not_contain_the_values_it_protects():
    """A guard that lists the protected values is the leak it exists to prevent."""
    if not FINGERPRINT.exists():
        pytest.skip("the preflight has not been run in this tree")
    payload = load_fingerprint(FINGERPRINT)
    text = FINGERPRINT.read_text(encoding="utf-8")
    # Every digest is 32 hex characters; no decimal probability should appear anywhere.
    import re

    decimals = [t for t in re.findall(r"0\.\d{3,}", text)]
    assert decimals == [], f"the fingerprint file carries literal values: {decimals[:5]}"
    assert payload["scalars"]


def test_the_tree_carries_no_unacknowledged_reconstruction_of_a_held_out_quantity():
    """The guard, run over the whole repository at every extension.

    `KNOWN` holds the reproductions this guard found when it was written. They are recorded rather
    than removed because they predate it and because deleting a published evidence record is a
    worse remedy than declaring it: what they cost is that the affected quantities are
    `DESCRIPTIVE` permanently, which `EXPOSURE.md` states. A path appearing here that is not in
    `KNOWN` is a new leak.
    """
    if not FINGERPRINT.exists():
        pytest.skip("the preflight has not been run in this tree")
    payload = load_fingerprint(FINGERPRINT)
    decisive = [leak for leak in scan(REPO, payload, allow=ALLOWED) if leak.decisive]
    unacknowledged = [leak for leak in decisive if (leak.path, leak.quantity) not in KNOWN]
    assert unacknowledged == [], "new reconstructions of a held-out quantity:\n" + "\n".join(
        f"  {leak}" for leak in unacknowledged
    )
