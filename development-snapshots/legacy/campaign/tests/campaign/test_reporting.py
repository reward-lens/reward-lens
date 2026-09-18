"""End-to-end tests for the reporting layer on a synthetic mini-campaign.

The mini-campaign lives in a temp directory with the real on-disk layout: two evidence shards
holding intermediates for two real cards (TOPO-HODGE and CAL-TRANSFER, both with confirming
tiny-shaped plants), frozen spec records built through campaign.specs plus the library freeze,
and the frozen probabilities manifest. ``merge_and_report`` must close it into RESULTS.md that
passes the claims gate; a fabricated claim must fail that gate; a back-dated staged-freeze
record must void the affected card's outcomes; a tampered spec hash must stop the world.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from reward_lens.core.store import EvidenceStore
from reward_lens.studies.freeze import freeze
from reward_lens.studies.spec import StudyResult

from campaign import caltransfer, topo
from campaign import evidence_keys as ek
from campaign.config import task_seed
from campaign.payloads import MatrixPayload, ScoreBank
from campaign.registry import CARDS_BY_ID
from campaign.reporting import (
    ClaimsGateError,
    FigureDataError,
    SpecHashDriftError,
    merge_and_report,
    merge_shards,
    run_claims_gate,
    run_rundown,
)
from campaign.reporting.figures import _fig_generic
from campaign.reporting.rundown import STAGED_VIOLATION, CardVerdict
from campaign.specs import build_spec

_FROZEN_AT = "2026-07-16T00:00:00+00:00"


def _write_frozen_record(frozen_dir: Path, card_id: str) -> None:
    card = CARDS_BY_ID[card_id]
    spec = build_spec(card)
    fz = freeze(spec, frozen_at=_FROZEN_AT)
    record = {
        "study_id": str(fz.study_id),
        "git_sha": fz.git_sha,
        "frozen_at": fz.frozen_at,
        "spec_hash": fz.spec_hash,
        "spec": spec.__canonical__(),
    }
    frozen_dir.mkdir(parents=True, exist_ok=True)
    (frozen_dir / f"{card.spec_id}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )


def _mini_campaign(root: Path) -> Path:
    """Two shards of real-card intermediates plus frozen records for three cards."""
    rng = np.random.default_rng(task_seed("test-reporting", "mini"))

    # Shard A: the TOPO-HODGE tournament intermediates, shaped as its resolver reads them.
    shard_a = EvidenceStore(root / "store" / "shards" / "shard-a")
    tiny = topo.tiny_subjects(rng)
    for key, slice_name in (
        ("nectar", "nectar-tournaments"),
        ("ultrafeedback", "ultrafeedback-tournaments"),
    ):
        shard_a.append(
            ek.record_intermediate(
                observable=ek.OBS_TOURNAMENTS,
                roster_key=ek.DATA_KEY,
                slice_name=slice_name,
                value=MatrixPayload(
                    name=f"{key} tournaments",
                    matrix=np.asarray(tiny[key]["edges"], dtype=np.float64),
                    meta={"n_items": tiny[key]["n_items"]},
                ),
                arc="test-ingest",
            )
        )

    # Shard B: the CAL-TRANSFER scorecard, a confirming plant by construction.
    shard_b = EvidenceStore(root / "store" / "shards" / "shard-b")
    scorecard = caltransfer.tiny_subjects(rng)["scorecard"]
    shard_b.append(
        ek.record_intermediate(
            observable=ek.OBS_SCORECARD,
            roster_key="skywork-v2-qwen3-0.6b",
            slice_name="caltransfer-organisms",
            value=scorecard,
            gpu="H100",
            arc="test-organisms",
            gpu_seconds=12.0,
        )
    )

    frozen_dir = root / "specs" / "frozen"
    for card_id in ("TOPO-HODGE", "CAL-TRANSFER", "META-LEDGER"):
        _write_frozen_record(frozen_dir, card_id)
    # The manifest in the exact schema campaign.freeze writes, when the meta module that
    # owns it has landed; the flat fallback keeps this fixture independent of build order.
    try:
        from campaign.metaledger import probabilities_payload

        payload = probabilities_payload()
        payload["frozen_at"] = _FROZEN_AT
    except ImportError:
        payload = {
            "frozen_at": _FROZEN_AT,
            "probabilities": {
                "TOPO-HODGE/H-intransitive": 0.70,
                "CAL-TRANSFER/H-transfer": 0.60,
            },
        }
    (frozen_dir / "probabilities.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def closed_campaign(tmp_path_factory):
    """One full merge_and_report over the mini-campaign, shared by the read-only tests."""
    root = _mini_campaign(tmp_path_factory.mktemp("mini-campaign"))
    summary = merge_and_report(root=root)
    return root, summary


def test_close_out_produces_every_artifact(closed_campaign):
    root, summary = closed_campaign
    assert (root / "RESULTS.md").is_file()
    assert (root / "runs" / "campaign" / "evidence.jsonl").is_file()
    assert (root / "runs" / "campaign" / "SCOREBOARD.md").is_file()
    assert (root / "runs" / "campaign" / "WORKS_LEDGER.md").is_file()
    assert (root / "runs" / "campaign" / "checkpoints" / "checkpoint-tier3.json").is_file()
    assert summary["claims"]["ok"] and summary["claims"]["checked"] > 0


def test_confirming_plants_adjudicate_confirmed(closed_campaign):
    _, summary = closed_campaign
    assert summary["cards"]["TOPO-HODGE"] == "confirmed"
    assert summary["cards"]["CAL-TRANSFER"] == "confirmed"
    # META-LEDGER adjudicates when its analysis module exists; a parallel build may not have
    # landed it yet, and the rundown must then degrade rather than crash.
    assert summary["meta"] in ("confirmed", "mixed", "refuted", "inconclusive")
    assert summary["staged_violations"] == []


def test_results_md_claims_are_store_backed(closed_campaign):
    root, _ = closed_campaign
    text = (root / "RESULTS.md").read_text(encoding="utf-8")
    assert "[[claim value=" in text
    assert "field=metrics.intransitive_mass" in text
    assert "field=metrics.max_abs_auc_difference" in text
    # The affirmative-null wording for the equivalence-bound card.
    assert "affirmative null" in text
    store = EvidenceStore(root / "runs" / "campaign")
    report = run_claims_gate([root / "RESULTS.md"], store)
    assert report.ok


def test_figures_exist_in_both_themes_with_captions(closed_campaign):
    root, summary = closed_campaign
    figures_dir = root / "figures"
    captions = json.loads((figures_dir / "captions.json").read_text(encoding="utf-8"))
    assert set(summary["figures"]) == set(captions)
    assert "hero1-ledger" in captions
    assert "card-topo-hodge" in captions
    assert "cal-transfer" in captions  # the specialty figure, from the per-instrument detail
    for name in captions:
        assert (figures_dir / f"{name}.light.png").is_file(), name
        assert (figures_dir / f"{name}.dark.png").is_file(), name
        assert captions[name].strip()


def test_works_ledger_carries_excused_rows_and_gates(closed_campaign):
    root, _ = closed_campaign
    text = (root / "runs" / "campaign" / "WORKS_LEDGER.md").read_text(encoding="utf-8")
    assert "EXCUSED" in text and "PathEffect" in text
    assert "Gate firings" in text
    assert "v1 failure-class regression" in text
    # The claims-checker feature is exercised live by the ledger itself.
    assert "live probe" in text
    # CAL-TRANSFER adjudicated, so the organism foundry row passes with its evidence id.
    assert "organism foundry (real trunk)" in text


def test_rerun_is_idempotent(closed_campaign):
    root, first = closed_campaign
    again = merge_and_report(root=root)
    assert again["claims"]["ok"]
    assert again["cards"] == first["cards"]
    # Nothing new to merge: content-derived ids make the merge a no-op on a re-run.
    assert again["merge_report"]["merged"] == 0


def test_fabricated_claim_fails_the_gate(tmp_path):
    root = _mini_campaign(tmp_path / "camp")
    summary = merge_and_report(root=root)
    assert summary["claims"]["ok"]
    results = root / "RESULTS.md"
    store = EvidenceStore(root / "runs" / "campaign")
    fabricated = (
        results.read_text(encoding="utf-8")
        + "\nThe effect is [[claim value=0.99 ev=ev:00000000000000000000000000000000 "
        + "field=metrics.intransitive_mass tol=1e-6]].\n"
    )
    results.write_text(fabricated, encoding="utf-8")
    with pytest.raises(ClaimsGateError):
        run_claims_gate([results], store)


def test_wrong_value_against_real_evidence_fails_the_gate(tmp_path):
    root = _mini_campaign(tmp_path / "camp")
    merge_and_report(root=root)
    store = EvidenceStore(root / "runs" / "campaign")
    adj = next(ev for ev in store if ev.observable == "campaign.adjudication.TOPO-HODGE")
    text = (
        f"The mass is [[claim value=123.456 ev={adj.id} "
        f"field=metrics.intransitive_mass tol=1e-6]]."
    )
    doc = tmp_path / "wrong.md"
    doc.write_text(text, encoding="utf-8")
    with pytest.raises(ClaimsGateError):
        run_claims_gate([doc], store)


def test_corrupt_shard_is_held_out_not_fatal(tmp_path):
    """A shard with a dangling parent is held out with a reason; healthy shards merge."""
    root = tmp_path / "camp"
    healthy = EvidenceStore(root / "store" / "shards" / "shard-good")
    ok = ek.record_intermediate(
        observable=ek.OBS_TOURNAMENTS,
        roster_key=ek.DATA_KEY,
        slice_name="nectar-tournaments",
        value=MatrixPayload(name="t", matrix=np.zeros((1, 5)), meta={"n_items": [3]}),
    )
    healthy.append(ok)
    # A shard whose one row declares a parent no shard holds: written by hand because the
    # store's own append correctly refuses to create this state.
    bad_dir = root / "store" / "shards" / "shard-bad"
    bad_dir.mkdir(parents=True)
    orphan = ek.record_intermediate(
        observable=ek.OBS_SCORES,
        roster_key="skywork-v2-qwen3-0.6b",
        slice_name="rb2-full",
        value={"placeholder": 1.0},
        parents=(),
    )
    env = orphan.envelope(sidecar_dir=bad_dir / "payloads")
    env["provenance"]["parents"] = ["ev:ffffffffffffffffffffffffffffffff"]
    (bad_dir / "evidence.jsonl").write_text(json.dumps(env) + "\n", encoding="utf-8")

    store, report = merge_shards(root / "store", root / "runs" / "campaign")
    assert ok.id in store
    assert report["merged"] == 1
    assert len(report["failed_shards"]) == 1
    assert "shard-bad" in report["failed_shards"][0]


def test_staged_freeze_violation_voids_hump(tmp_path):
    """Gold-sweep evidence created before the frozen tail record is exploratory by rule."""
    root = _mini_campaign(tmp_path / "camp")
    shard = EvidenceStore(root / "store" / "shards" / "shard-hump")
    shard.append(
        ek.record_intermediate(
            observable=ek.OBS_SCORES,
            roster_key="skywork-v2-llama31-8b",
            slice_name="hump-prompts",
            value=ScoreBank(item_ids=["p0", "p1"], scores=np.zeros((2, 4)), layout="bank"),
            gpu="H100",
            arc="test-generation",
        )
    )
    frozen_dir = root / "specs" / "frozen"
    _write_frozen_record(frozen_dir, "HUMP")
    # The prediction record postdates the sweep evidence, which is exactly the leak the
    # staged check exists to catch.
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    (frozen_dir / "hump_prediction.json").write_text(
        json.dumps({"frozen_at": future, "hump_kl_predicted": 0.6}), encoding="utf-8"
    )

    store, _ = merge_shards(root / "store", root / "runs" / "campaign")
    rundown = run_rundown(store, frozen_dir)
    hump = rundown.verdicts["HUMP"]
    assert hump.staged_violation
    assert hump.status == "inconclusive"
    assert hump.reason.startswith(STAGED_VIOLATION)
    assert set(hump.outcomes.values()) == {"inconclusive"}
    assert rundown.staged_violations


def test_spec_hash_drift_stops_the_world(tmp_path):
    root = _mini_campaign(tmp_path / "camp")
    record_path = root / "specs" / "frozen" / "campaign-topo-hodge.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["spec_hash"] = "spec:0000000000000000000000000000dead"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SpecHashDriftError):
        merge_and_report(root=root)


def test_missing_intermediates_degrade_to_inconclusive(tmp_path):
    """A frozen card whose arc never ran must come back inconclusive with the reason."""
    root = tmp_path / "camp"
    frozen_dir = root / "specs" / "frozen"
    _write_frozen_record(frozen_dir, "CAL-TRANSFER")
    (root / "store" / "shards").mkdir(parents=True)
    store, _ = merge_shards(root / "store", root / "runs" / "campaign")
    rundown = run_rundown(store, frozen_dir)
    verdict = rundown.verdicts["CAL-TRANSFER"]
    assert verdict.status == "inconclusive"
    assert "missing intermediate" in verdict.reason
    assert set(verdict.outcomes.values()) == {"inconclusive"}


def test_figures_refuse_unbacked_numbers(tmp_path):
    """A plotted number whose evidence id is not in the store is a hard figure error."""
    store = EvidenceStore(tmp_path / "store")
    card = CARDS_BY_ID["TOPO-HODGE"]
    verdict = CardVerdict(card=card, record={})
    verdict.status = "adjudicated"
    verdict.result = StudyResult(
        outcomes={"H-intransitive": "confirmed"}, metrics={"intransitive_mass": 0.4}
    )
    verdict.summary_evidence_id = "ev:00000000000000000000000000000000"
    verdict.adjudication_evidence_id = "ev:00000000000000000000000000000000"
    with pytest.raises(FigureDataError):
        _fig_generic(verdict, store, tmp_path / "figures")


def test_reproduce_guards_refuse_before_the_staged_freeze(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("reproduce", repo_root / "reproduce.py")
    reproduce = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reproduce)

    monkeypatch.setattr(reproduce, "FROZEN", tmp_path / "frozen")
    # Listing never executes and always succeeds.
    assert reproduce.main(["5c"]) == 0
    # Running the guarded steps refuses while the staged records are absent. The gold
    # sweep's guard sits on 8c: 8a (base banks) and 8b (the staged hump freeze) must be
    # free to run first, because 8b is what writes the record 8c requires.
    assert reproduce.main(["5c", "--run"]) == 2
    assert reproduce.main(["8c", "--run"]) == 2
    # Once the records exist the guards stand down (the commands themselves would then run;
    # they are not executed here).
    (tmp_path / "frozen").mkdir()
    (tmp_path / "frozen" / "ladder_intervals.json").write_text("{}", encoding="utf-8")
    assert reproduce._guard_ladder() is None
