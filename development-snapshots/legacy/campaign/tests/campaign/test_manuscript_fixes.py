"""Tests for the manuscript-layer adjudication and disclosure rules.

These cover the reporting behaviors the scored ledger depends on beyond the happy path: the
hero panel's oriented margins, the outcome-conditional gauge caption, the low-power
disclosures read from the freeze manifest, the SURGERY certificate gate, the pre-freeze
evidence check, the rehearsal banner, admit-versus-cut rendering, count consistency, the
works-ledger's hardware and substrate honesty, the tier1-at-close rule, the freeze-manifest
presence check, the card gallery, and the mandatory FDR overlay wiring.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from reward_lens.core.store import EvidenceStore
from reward_lens.studies.spec import StudyResult

from campaign import evidence_keys as ek
from campaign.adjudicate import AdjudicatedResult, AdjudicationNote
from campaign.payloads import FreezeManifest, Heartbeat
from campaign.registry import CARDS_BY_ID
from campaign.reporting import merge_and_report, merge_shards, run_rundown
from campaign.reporting.close import _spend_checkpoints
from campaign.reporting.figures import gauge_caption, margin_to_threshold
from campaign.reporting.results_md import build_results_md
from campaign.reporting.rundown import (
    PRE_FREEZE_VIOLATION,
    UNCERTIFIED_ERASURE,
    CardVerdict,
    Rundown,
    _apply_certificate_gate,
)
from campaign.reporting.works_ledger import build_works_ledger

from tests.campaign.test_reporting import _mini_campaign

_PAST = "2020-01-01T00:00:00+00:00"


def _write_manifest(root: Path, **extra) -> None:
    record = {"created_at": _PAST, "tag": "test-freeze", **extra}
    (root / "specs" / "frozen" / "manifest.json").write_text(
        json.dumps(record), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Finding 1: the hero panel's oriented, standardized margins
# ---------------------------------------------------------------------------


def test_margin_orientation_right_of_zero_always_means_held():
    # Greater-than: a value above its threshold held, so the margin is positive.
    m, kind = margin_to_threshold("chi_bon_spearman", ">", 0.3, 0.5)
    assert m > 0 and kind == "threshold"
    # Less-than: a value below its threshold held, so the sign flips to positive.
    m, _ = margin_to_threshold("forecast_mae", "<", 0.06, 0.04)
    assert m > 0
    m, _ = margin_to_threshold("forecast_mae", "<", 0.06, 0.09)
    assert m < 0
    # abs<: the magnitude rule applies to |value|.
    m, _ = margin_to_threshold("raw_cos_v01_v02", "abs<", 0.02, -0.005)
    assert m > 0
    # Equality scores the negated distance, zero when exactly met.
    m, _ = margin_to_threshold("bon_lowerquantile_beats_mean", "==", 1.0, 1.0)
    assert m == 0.0
    m, _ = margin_to_threshold("bon_lowerquantile_beats_mean", "==", 1.0, 0.0)
    assert m < 0


def test_margin_uses_log_scale_for_p_values_and_ci_halfwidth_when_registered():
    m, kind = margin_to_threshold("xfam_perm_p", "<", 0.05, 0.005)
    assert kind == "log10"
    assert m == pytest.approx(1.0)
    m, _ = margin_to_threshold("xfam_perm_p", "<", 0.05, 0.5)
    assert m == pytest.approx(-1.0)
    # A registered CI standardizes the margin in CI half-widths.
    m, kind = margin_to_threshold("recovery_gap", ">", 0.0, 0.2, ci=(0.1, 0.3))
    assert kind == "ci"
    assert m == pytest.approx(2.0)


def test_hero_panel_renders_margin_form_with_orientation_rule(tmp_path):
    root = _mini_campaign(tmp_path / "camp")
    _write_manifest(root)
    summary = merge_and_report(root=root)
    captions = json.loads(
        (root / "figures" / "captions.json").read_text(encoding="utf-8")
    )
    assert "hero1-ledger" in captions
    assert "zero line" in captions["hero1-ledger"]
    assert "hero1-ledger" in summary["figures"]
    assert (root / "figures" / "hero1-ledger.light.png").is_file()
    assert (root / "figures" / "hero1-ledger.dark.png").is_file()


# ---------------------------------------------------------------------------
# Finding 2: the gauge caption conditions on both hypotheses
# ---------------------------------------------------------------------------


def test_gauge_caption_covers_all_four_outcome_combinations():
    both = gauge_caption(0.005, 0.7, "confirmed", "confirmed")
    assert "largely the same function" in both
    raw_only = gauge_caption(0.005, 0.1, "confirmed", "refuted")
    assert "did not recover co-rotation" in raw_only
    assert "multiplicity" in raw_only
    canon_only = gauge_caption(0.4, 0.7, "refuted", "confirmed")
    assert "did not reproduce the near-zero" in canon_only
    assert "no orthogonality to reconcile" in canon_only
    neither = gauge_caption(0.4, 0.1, "refuted", "refuted")
    assert neither.startswith("Neither frozen call held")
    # A caption keyed on H-canon alone would have asserted orthogonality here.
    assert "orthogonal in raw coordinates" not in canon_only
    assert "orthogonal in raw coordinates" not in neither


# ---------------------------------------------------------------------------
# Finding 3: low-power acceptances are read, disclosed, and flagged inline
# ---------------------------------------------------------------------------


def test_low_power_acceptance_discloses_and_flags_the_band(tmp_path):
    root = _mini_campaign(tmp_path / "camp")
    _write_manifest(
        root,
        power_acceptances=[
            {"card": "CAL-TRANSFER", "power": 0.55, "mde": 0.001,
             "note": "kept for the portfolio"},
        ],
    )
    summary = merge_and_report(root=root)
    assert summary["power_acceptances"]["CAL-TRANSFER"]["power"] == pytest.approx(0.55)
    text = (root / "RESULTS.md").read_text(encoding="utf-8")
    assert "Low-power acceptances recorded at the freeze" in text
    assert "power 0.55" in text and "MDE 0.001" in text
    assert "low-power acceptance" in text  # the standing per-card disclosure
    # The tiny CAL-TRANSFER plant confirms between 0.001 and the 0.15 threshold, which is
    # exactly the band the accepted power cannot resolve, so the inline flag must fire.
    assert "LOW-POWER-BAND DETECTION" in text
    assert summary["claims"]["ok"]


def test_no_acceptances_means_no_front_matter_subsection(tmp_path):
    root = _mini_campaign(tmp_path / "camp")
    _write_manifest(root)
    merge_and_report(root=root)
    text = (root / "RESULTS.md").read_text(encoding="utf-8")
    assert "Low-power acceptances recorded at the freeze" not in text


# ---------------------------------------------------------------------------
# Finding 4: a failed erasure certificate blocks an H-erase confirmation
# ---------------------------------------------------------------------------


def test_failed_certificate_downgrades_a_confirmed_h_erase():
    verdict = CardVerdict(card=CARDS_BY_ID["SURGERY"], record={})
    verdict.result = StudyResult(
        outcomes={"H-erase": "confirmed", "H-retain": "confirmed"},
        metrics={"erasure_certificate_pass": 0.0, "exploit_drift_reduction": 0.9},
    )
    verdict.overlay = AdjudicatedResult(result=verdict.result)
    _apply_certificate_gate(verdict)
    assert verdict.result.outcomes["H-erase"] == "inconclusive"
    assert verdict.result.outcomes["H-retain"] == "confirmed"
    assert any(
        n.action == "downgraded-uncertified" and n.detail == UNCERTIFIED_ERASURE
        for n in verdict.overlay.notes
    )


def test_certificate_gate_never_upgrades_or_touches_other_cards():
    refuted = CardVerdict(card=CARDS_BY_ID["SURGERY"], record={})
    refuted.result = StudyResult(
        outcomes={"H-erase": "refuted"}, metrics={"erasure_certificate_pass": 0.0}
    )
    refuted.overlay = AdjudicatedResult(result=refuted.result)
    _apply_certificate_gate(refuted)
    assert refuted.result.outcomes["H-erase"] == "refuted"

    other = CardVerdict(card=CARDS_BY_ID["TOPO-HODGE"], record={})
    other.result = StudyResult(
        outcomes={"H-intransitive": "confirmed"},
        metrics={"erasure_certificate_pass": 0.0},
    )
    other.overlay = AdjudicatedResult(result=other.result)
    _apply_certificate_gate(other)
    assert other.result.outcomes["H-intransitive"] == "confirmed"


# ---------------------------------------------------------------------------
# Finding 5: every card's consumed evidence must postdate the main freeze
# ---------------------------------------------------------------------------


def test_pre_freeze_evidence_voids_every_affected_card(tmp_path):
    root = _mini_campaign(tmp_path / "camp")
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    (root / "specs" / "frozen" / "manifest.json").write_text(
        json.dumps({"created_at": future}), encoding="utf-8"
    )
    store, _ = merge_shards(root / "store", root / "runs" / "campaign")
    rundown = run_rundown(store, root / "specs" / "frozen")
    for name in ("TOPO-HODGE", "CAL-TRANSFER"):
        verdict = rundown.verdicts[name]
        assert verdict.pre_freeze_violation, name
        assert verdict.status == "inconclusive"
        assert PRE_FREEZE_VIOLATION in verdict.reason
        assert set(verdict.outcomes.values()) == {"inconclusive"}
    assert rundown.pre_freeze_violations


def test_post_freeze_evidence_passes_the_check(tmp_path):
    root = _mini_campaign(tmp_path / "camp")
    _write_manifest(root)
    store, _ = merge_shards(root / "store", root / "runs" / "campaign")
    rundown = run_rundown(store, root / "specs" / "frozen")
    assert not rundown.pre_freeze_violations
    assert rundown.verdicts["TOPO-HODGE"].status == "adjudicated"
    assert rundown.verdicts["CAL-TRANSFER"].status == "adjudicated"


def test_missing_manifest_is_a_recorded_caveat_not_a_silent_pass(tmp_path):
    root = _mini_campaign(tmp_path / "camp")
    store, _ = merge_shards(root / "store", root / "runs" / "campaign")
    rundown = run_rundown(store, root / "specs" / "frozen")
    assert any("pre-freeze evidence check" in n for n in rundown.notes)


# ---------------------------------------------------------------------------
# Finding 6: a rehearsal manifest brands the ledger end to end
# ---------------------------------------------------------------------------


def test_rehearsal_manifest_brands_results_and_close_summary(tmp_path):
    root = _mini_campaign(tmp_path / "camp")
    _write_manifest(root, rehearsal=True)
    summary = merge_and_report(root=root)
    assert summary["rehearsal"] is True
    assert "REHEARSAL" in summary["banner"]
    text = (root / "RESULTS.md").read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("# REHEARSAL")
    assert "REHEARSAL RUN." in text


def test_real_manifest_leaves_the_title_unbranded(tmp_path):
    root = _mini_campaign(tmp_path / "camp")
    _write_manifest(root)
    summary = merge_and_report(root=root)
    assert summary["rehearsal"] is False
    assert "banner" not in summary
    first = (root / "RESULTS.md").read_text(encoding="utf-8").splitlines()[0]
    assert "REHEARSAL" not in first


# ---------------------------------------------------------------------------
# Finding 7: the mandated caveats hold in every world
# ---------------------------------------------------------------------------


def test_standing_caveats_present_even_in_an_empty_world(tmp_path):
    text = build_results_md(Rundown(), EvidenceStore(tmp_path / "store"))
    assert "Standing caveats, present in every outcome" in text
    assert "It would not establish causation" in text
    assert "WildChat" in text and "not an independence control" in text
    assert "ArmoRM" in text and "native gating-head" in text
    assert "Beirami bound" in text


def test_hump_section_carries_the_kl_bound_statement(tmp_path):
    rundown = Rundown()
    rundown.verdicts["HUMP"] = CardVerdict(card=CARDS_BY_ID["HUMP"], record={})
    text = build_results_md(rundown, EvidenceStore(tmp_path / "store"))
    hump_section = text.split("### HUMP", 1)[1]
    assert "Beirami best-of-n bound" in hump_section.split("###")[0]


# ---------------------------------------------------------------------------
# Finding 8: admissions render as admissions, cuts as cuts
# ---------------------------------------------------------------------------


def test_admit_and_cut_render_under_their_own_labels(tmp_path):
    checkpoints = [
        {"checkpoint": "tier3", "evaluated": True, "proceed": True,
         "cuts": ["ADMIT T3-27B", "ADMIT T3-DECOMP"], "reasons": ["fits"],
         "total_usd": 12.5},
        {"checkpoint": "fleet", "evaluated": True, "proceed": False,
         "cuts": ["HUMP-27b-gold-arm"], "reasons": ["projection breached the cap"],
         "total_usd": 24.0},
        {"checkpoint": "tier1", "evaluated": False, "proceed": True, "cuts": [],
         "reasons": ["not evaluated at close-out: no pilot heartbeat"],
         "total_usd": 24.0},
    ]
    text = build_results_md(
        Rundown(), EvidenceStore(tmp_path / "store"), checkpoints=checkpoints
    )
    assert "Admitted at the Tier 3 gate" in text
    assert "T3-27B, T3-DECOMP" in text
    assert "dropped: HUMP-27b-gold-arm" in text
    assert "not evaluated at close-out" in text
    assert "CUT ADMIT" not in text
    # Admitted cards never appear on a dropped line.
    for line in text.splitlines():
        if "dropped:" in line:
            assert "ADMIT" not in line


# ---------------------------------------------------------------------------
# Finding 9: one source for card counts and hypothesis counts
# ---------------------------------------------------------------------------


def test_lead_states_both_units_and_matches_the_summary(tmp_path):
    root = _mini_campaign(tmp_path / "camp")
    _write_manifest(root)
    summary = merge_and_report(root=root)
    text = (root / "RESULTS.md").read_text(encoding="utf-8")
    match = re.search(
        r"(\d+) of (\d+) frozen cards adjudicated.*?(\d+) of (\d+) frozen hypotheses "
        r"confirmed, (\d+) refuted, (\d+) inconclusive; (\d+) kill",
        text,
        re.DOTALL,
    )
    assert match, "the lead must state cards and hypotheses in one sentence"
    n_adj, n_frozen = int(match.group(1)), int(match.group(2))
    scored_cards = len(summary["cards"]) + 1  # the meta-ledger is counted, and says so
    assert "counting the meta-ledger" in text
    assert n_frozen == scored_cards
    assert n_adj <= n_frozen


# ---------------------------------------------------------------------------
# Finding 10: works-ledger hardware and substrate honesty
# ---------------------------------------------------------------------------


@pytest.fixture()
def mini_rundown(tmp_path):
    root = _mini_campaign(tmp_path / "camp")
    _write_manifest(root)
    store, _ = merge_shards(root / "store", root / "runs" / "campaign")
    rundown = run_rundown(store, root / "specs" / "frozen")
    return root, store, rundown


def test_hardware_column_reads_the_gpu_stamps(mini_rundown):
    _, store, rundown = mini_rundown
    rows = {r.feature: r for r in build_works_ledger(rundown, store)}
    # CAL-TRANSFER's scorecard intermediate is stamped H100, so the foundry row says so.
    foundry = rows["organism foundry (real trunk)"]
    assert foundry.status == "PASS"
    assert foundry.hardware == "H100"
    # TOPO-HODGE consumed cpu-stamped tournament tables; the row must not claim a GPU.
    studies = rows["StudySpec / freeze / run_study / scoreboard"]
    assert studies.status == "PASS"
    assert studies.hardware.startswith("cpu")
    assert "no GPU stamp" in studies.hardware


def test_substrate_rows_refuse_pass_when_the_substrate_never_ran(mini_rundown):
    _, store, rundown = mini_rundown
    rows = {r.feature: r for r in build_works_ledger(rundown, store)}
    for feature in ("GenerativeJudge", "ProcessRM", "ImplicitRM", "DistributionalSignal",
                    "ClassifierRM (InternLM2 registration path)"):
        row = rows[feature]
        assert row.status == "NOT-RUN", feature
        assert "substrate never ran" in row.note or "not frozen" in row.note, feature


def test_substrate_evidence_lifts_the_registration_row(mini_rundown):
    _, store, rundown = mini_rundown
    store.append(
        ek.record_intermediate(
            observable=ek.OBS_SCORES,
            roster_key="internlm2-1.8b",
            slice_name="rb2-full",
            value={"probe": 1.0},
            gpu="H100",
            arc="test-substrate",
        )
    )
    rows = {r.feature: r for r in build_works_ledger(rundown, store)}
    row = rows["ClassifierRM (InternLM2 registration path)"]
    assert row.status == "PASS"
    assert row.evidence_id
    assert row.hardware == "H100"


# ---------------------------------------------------------------------------
# Finding 12: tier1 is a pilot-phase decision, not a close-out artifact
# ---------------------------------------------------------------------------


def test_tier1_not_evaluated_without_pilot_heartbeats(tmp_path):
    root = _mini_campaign(tmp_path / "camp")
    _write_manifest(root)
    summary = merge_and_report(root=root)
    tier1 = next(c for c in summary["checkpoints"] if c["checkpoint"] == "tier1")
    assert tier1["evaluated"] is False
    assert tier1["proceed"] is True and tier1["cuts"] == []
    assert any("no pilot heartbeat" in r for r in tier1["reasons"])
    text = (root / "RESULTS.md").read_text(encoding="utf-8")
    assert "not evaluated at close-out" in text


def _heartbeat_evidence(seq_per_s: float, floor: float):
    return ek.record_intermediate(
        observable=ek.OBS_HEARTBEAT,
        roster_key="skywork-v2-qwen3-0.6b",
        slice_name="_heartbeat",
        value=Heartbeat(
            arc="model:pilot", gpu="H100", roster_key="skywork-v2-qwen3-0.6b",
            seq_per_s=seq_per_s, floor_seq_per_s=floor, n_scored=32,
            tokens_mean=512.0, container_start_s=1.0,
        ),
        gpu="H100",
        arc="model:pilot",
    )


def test_tier1_evaluates_only_while_spend_looks_like_a_pilot(tmp_path):
    pilot_store = EvidenceStore(tmp_path / "pilot-store")
    pilot_store.append(_heartbeat_evidence(seq_per_s=240.0, floor=120.0))
    records, _ = _spend_checkpoints(pilot_store, tmp_path / "cp-a")
    tier1 = next(r for r in records if r["checkpoint"] == "tier1")
    assert tier1["evaluated"] is True and tier1["proceed"] is True

    late_store = EvidenceStore(tmp_path / "late-store")
    late_store.append(_heartbeat_evidence(seq_per_s=240.0, floor=120.0))
    late_store.append(
        ek.record_intermediate(
            observable=ek.OBS_SCORES,
            roster_key="skywork-v2-llama31-8b",
            slice_name="rb2-full",
            value={"bank": 1.0},
            gpu="H100",
            arc="model:fleet",
            gpu_seconds=4000.0,
        )
    )
    records, _ = _spend_checkpoints(late_store, tmp_path / "cp-b")
    tier1 = next(r for r in records if r["checkpoint"] == "tier1")
    assert tier1["evaluated"] is False
    assert any("past the pilot phase" in r for r in tier1["reasons"])


# ---------------------------------------------------------------------------
# Finding 11: freeze-manifest presence and the card gallery
# ---------------------------------------------------------------------------


def test_missing_freeze_manifest_row_is_a_caveat(tmp_path):
    root = _mini_campaign(tmp_path / "camp")
    _write_manifest(root)
    summary = merge_and_report(root=root)
    assert summary["freeze_manifest_in_store"] is False
    text = (root / "RESULTS.md").read_text(encoding="utf-8")
    assert "no FreezeManifest evidence row" in text


def test_freeze_manifest_row_in_a_shard_clears_the_caveat(tmp_path):
    root = _mini_campaign(tmp_path / "camp")
    _write_manifest(root)
    shard = EvidenceStore(root / "store" / "shards" / "shard-freeze")
    shard.append(
        ek.record_intermediate(
            observable=ek.OBS_FREEZE_MANIFEST,
            roster_key=ek.DATA_KEY,
            slice_name="_freeze",
            value=FreezeManifest(
                study_ids=["study:test"], git_sha="deadbeef", tag="test-freeze",
                frozen_at=_PAST, spec_hashes=["spec:test"],
                external_registration="manual:test",
            ),
        )
    )
    summary = merge_and_report(root=root)
    assert summary["freeze_manifest_in_store"] is True
    text = (root / "RESULTS.md").read_text(encoding="utf-8")
    assert "no FreezeManifest evidence row" not in text


def test_card_gallery_renders_signal_stamped_fleet_models(tmp_path):
    root = _mini_campaign(tmp_path / "camp")
    _write_manifest(root)
    shard = EvidenceStore(root / "store" / "shards" / "shard-atlas")
    shard.append(
        ek.record_intermediate(
            observable=ek.OBS_INDEX_TABLE,
            roster_key="skywork-v2-qwen3-8b",
            slice_name="rb2-full",
            value={"snr": 1.5},
            signals=("skywork-v2-qwen3-8b",),
            gpu="H100",
            arc="test-atlas",
        )
    )
    summary = merge_and_report(root=root)
    assert "skywork-v2-qwen3-8b" in summary["card_gallery"]
    card_path = Path(summary["card_gallery"]["skywork-v2-qwen3-8b"])
    assert card_path.is_file() and card_path.suffix == ".html"
    assert card_path.parent == root / "cards"
    ledger = (root / "runs" / "campaign" / "WORKS_LEDGER.md").read_text(encoding="utf-8")
    assert "per-model cards rendered" in ledger


def test_gallery_without_signal_stamps_is_a_note_not_a_pass(tmp_path):
    root = _mini_campaign(tmp_path / "camp")
    _write_manifest(root)
    summary = merge_and_report(root=root)
    assert summary["card_gallery"] == {}
    ledger = (root / "runs" / "campaign" / "WORKS_LEDGER.md").read_text(encoding="utf-8")
    assert "the per-model gallery rendered no card" in ledger


# ---------------------------------------------------------------------------
# The FDR overlay wiring: applied per card, mandatory at close-out
# ---------------------------------------------------------------------------


def test_fdr_overlay_notes_fold_into_the_card_overlay(tmp_path, monkeypatch):
    import campaign.adjudicate as adjudicate_mod

    def fake_fdr(spec, result, alpha=0.05):
        note = AdjudicationNote(
            hypothesis_id="H-test", metric="m", action="downgraded-fdr",
            detail=f"stand-in at alpha {alpha}",
        )
        return AdjudicatedResult(result=result, notes=[note])

    monkeypatch.setattr(adjudicate_mod, "apply_fdr_overlay", fake_fdr)
    root = _mini_campaign(tmp_path / "camp")
    _write_manifest(root)
    store, _ = merge_shards(root / "store", root / "runs" / "campaign")
    rundown = run_rundown(store, root / "specs" / "frozen")
    topo = rundown.verdicts["TOPO-HODGE"]
    assert any(n.action == "downgraded-fdr" for n in topo.overlay.notes)


def test_close_out_refuses_without_the_fdr_overlay(tmp_path, monkeypatch):
    import campaign.adjudicate as adjudicate_mod

    monkeypatch.delattr(adjudicate_mod, "apply_fdr_overlay")
    root = _mini_campaign(tmp_path / "camp")
    _write_manifest(root)
    with pytest.raises(RuntimeError, match="apply_fdr_overlay"):
        merge_and_report(root=root)


# ---------------------------------------------------------------------------
# Finding 13: the claims scope rule is stated and thresholds are tagged
# ---------------------------------------------------------------------------


def test_scope_note_and_threshold_tags_are_present(tmp_path):
    root = _mini_campaign(tmp_path / "camp")
    _write_manifest(root)
    summary = merge_and_report(root=root)
    text = (root / "RESULTS.md").read_text(encoding="utf-8")
    assert "What the claims gate covers" in text
    assert "field=thresholds.H-intransitive" in text
    assert "field=thresholds.H-transfer" in text
    assert "metered sum" in text
    assert summary["claims"]["ok"]
