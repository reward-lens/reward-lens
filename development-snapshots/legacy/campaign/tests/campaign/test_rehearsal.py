"""End-to-end rehearsal drills: the tiny campaign produces the complete artifact set.

These tests run the real driver (``campaign.tinyrun.main``) against private roots and read
the artifacts it writes: ``close_summary.json`` for the verdicts, ``RESULTS.md`` for the
claim tags, ``specs/frozen`` for the rehearsal freeze, and the figures directory for the
hero panel. Three passes cover the operational story end to end: a fresh run adjudicates
every registry card through the real resolvers, a resume pass recomputes nothing and
reproduces the same verdict table, and a mid-fleet kill drill resumes into the identical
artifact set. Everything is seeded, CPU-only, and offline; the whole module costs a few
rehearsal runs of under a minute each.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest

from campaign import arcs, config, tinyrun
from campaign import evidence_keys as ek
from campaign.registry import CARDS

_NON_META = sorted(c.card for c in CARDS if c.card != "META-LEDGER")
_SCORED = ("confirmed", "refuted", "mixed")


def _summary(root: Path) -> dict:
    return json.loads(
        (root / "runs" / "campaign" / "close_summary.json").read_text(encoding="utf-8")
    )


def _verdict_rows(root: Path) -> list[str]:
    """The RESULTS.md verdict table rows with the run-specific evidence ids stripped."""
    rows = []
    for line in (root / "RESULTS.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("| ") and "| ev:" in line:
            rows.append(line.rsplit("| ev:", 1)[0])
    return rows


@pytest.fixture(scope="module")
def fresh_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("rehearsal")
    assert tinyrun.main(["--fresh", "--root", str(root)]) == 0
    return root


def test_fresh_rehearsal_adjudicates_every_card(fresh_root):
    summary = _summary(fresh_root)
    assert sorted(summary["cards"]) == _NON_META
    stuck = {k: v for k, v in summary["cards"].items() if v not in _SCORED}
    assert not stuck, f"cards not adjudicated: {stuck}"
    assert summary["meta"] in _SCORED
    assert not summary["staged_violations"]


def test_fresh_rehearsal_writes_the_full_frozen_record_set(fresh_root):
    frozen = fresh_root / "specs" / "frozen"
    spec_ids = {c.spec_id for c in CARDS}
    written = {p.stem for p in frozen.glob("*.json")}
    assert spec_ids <= written
    for staged in ("probabilities.json", "manifest.json",
                   "ladder_intervals.json", "hump_prediction.json"):
        assert (frozen / staged).exists(), f"missing {staged}"
    manifest = json.loads((frozen / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rehearsal"] is True
    assert manifest["code_tree_hash"].startswith("code:")
    assert manifest["power_verdicts"], "power verdicts must be recorded, not refused"


def test_fresh_rehearsal_closes_into_checked_artifacts(fresh_root):
    summary = _summary(fresh_root)
    assert summary["claims"]["checked"] > 0
    assert summary["claims"]["ok"] and summary["claims"]["failures"] == 0
    assert summary["works_ledger"]["pass"] > 0
    # The rehearsal world is deliberately refuted and fires the instrument-class kills
    # (K-verif, K-welch), whose registered text promises a works-ledger finding; the only
    # FAIL rows allowed are exactly those, each naming its kill.
    ledger_text = (fresh_root / "runs" / "campaign" / "WORKS_LEDGER.md").read_text(
        encoding="utf-8"
    )
    fail_lines = [
        ln for ln in ledger_text.splitlines()
        if ln.startswith("|") and "| FAIL |" in ln
    ]
    assert len(fail_lines) == summary["works_ledger"]["fail"]
    assert all("instrument-class kill" in ln for ln in fail_lines), fail_lines
    assert any(name.startswith("hero1") for name in summary["figures"])
    hero = list((fresh_root / "figures").glob("hero1*"))
    assert hero, "the hero panel must exist on disk"
    assert (fresh_root / "RESULTS.md").exists()


def test_second_run_skips_completed_work_and_reproduces_verdicts(fresh_root, capsys):
    before = _summary(fresh_root)
    rows_before = _verdict_rows(fresh_root)
    assert tinyrun.main(["--resume", "--root", str(fresh_root)]) == 0
    out = capsys.readouterr().out
    match = re.search(r"arcs done in [\d.]+s: computed=(\d+) skipped=(\d+) appended=(\d+)", out)
    assert match, out
    computed, skipped, appended = (int(g) for g in match.groups())
    assert appended == 0, "a resume over a complete root must append nothing"
    assert computed == 0
    assert skipped > 0
    after = _summary(fresh_root)
    assert after["cards"] == before["cards"]
    assert after["meta"] == before["meta"]
    assert _verdict_rows(fresh_root) == rows_before


def test_kill_drill_resumes_into_the_uninterrupted_artifact_set(fresh_root, tmp_path):
    root = tmp_path / "killed"
    label = "model:skywork-v2-llama31-8b"
    with pytest.raises(SystemExit) as exc:
        tinyrun.main(["--fresh", "--root", str(root), "--kill-at", label])
    assert exc.value.code == 3

    # The kill landed right after a durable append (the hook fires only on those; capture
    # chunks are never counted), so the shard holds at least one intermediate the resume
    # can trust, while the arc's later work is visibly missing: the first checkpoint is
    # the rb2-full capture manifest, so the rmbench bank cannot exist yet.
    store = arcs.shard_store(root / "store", label)
    rows = [
        ev for ev in store
        if (ev.subject.extra or {}).get("roster_key") == "skywork-v2-llama31-8b"
    ]
    assert rows, "no durable append survived the kill"
    done = {
        ev.subject.extra.get("slice") for ev in rows if ev.observable == ek.OBS_SCORES
    }
    assert "rmbench-full" not in done, "the killed arc must not have finished its menu"

    assert tinyrun.main(["--resume", "--root", str(root)]) == 0
    resumed = _summary(root)
    reference = _summary(fresh_root)
    assert resumed["cards"] == reference["cards"]
    assert resumed["meta"] == reference["meta"]
    assert resumed["figures"] == reference["figures"]
    assert resumed["claims"]["ok"] and resumed["claims"]["checked"] > 0
    assert _verdict_rows(root) == _verdict_rows(fresh_root)


def test_config_paths_are_restored_after_reporting(fresh_root):
    # The driver redirects the frozen-record paths at the config module for the merge and
    # must put them back; a leaked override would poison later real-path readers.
    assert config.FROZEN_DIR == config.SPECS_DIR / "frozen"
    assert config.LADDER_INTERVALS_PATH == config.FROZEN_DIR / "ladder_intervals.json"


def test_modal_surface_imports_clean():
    pytest.importorskip("modal")
    import importlib

    app_module = importlib.import_module("campaign.app")
    for name in ("data_arc", "patch_arc", "surgery_arc", "evalaware_arc",
                 "atlas_arc", "generation_arc"):
        assert hasattr(app_module, name)
