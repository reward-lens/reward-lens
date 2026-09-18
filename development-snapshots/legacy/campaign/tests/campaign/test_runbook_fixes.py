"""Runbook, budget, and resilience behavior: menus, pilot quarantine, admission
arithmetic, checkpoint refusals, freeze retry-safety, the shard writer lock, the
capture-derived banks, and the EMB checkpoint validation. CPU-only, offline, seconds."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pytest

from campaign import arcs, captures, config, freeze, spend
from campaign import evidence_keys as ek
from campaign.reporting import checkpoint as checkpoint_cli
from reward_lens.signals.loaders import from_tiny

KEY = "skywork-v2-qwen3-0.6b"


# ---------------------------------------------------------------------------
# Menus and the CLI parse contract
# ---------------------------------------------------------------------------


def test_parse_menu_resolves_the_three_spellings():
    assert arcs.parse_menu(KEY, "") is None
    assert arcs.parse_menu(KEY, "ppe") == tuple(
        f"ppe-best-of-k::{s}" for s in arcs.PPE_SETS
    )
    assert arcs.parse_menu(KEY, "rb2-full, hh-ood") == ("rb2-full", "hh-ood")


def test_parse_menu_refuses_typos_before_any_billing():
    with pytest.raises(KeyError):
        arcs.parse_menu(KEY, "rb2-fulll")
    with pytest.raises(ValueError):
        arcs.parse_menu(KEY, "pilot")


def test_ppe_moves_off_the_8b_default_menus_but_stays_on_the_small_seat():
    small = arcs.default_menu("skywork-v2-qwen3-0.6b")
    assert any(s.startswith("ppe-best-of-k::") for s in small)
    for eight_b in ("skywork-v2-qwen3-8b", "skywork-v2-llama31-8b"):
        menu = arcs.default_menu(eight_b)
        assert not any(s.startswith("ppe-best-of-k::") for s in menu)
    # ppe-human is scored only through the runbook's explicit command for the anchor.
    for key in config.ROSTER:
        if config.ROSTER[key].adapter == "policy":
            continue
        assert "ppe-human" not in arcs.default_menu(key)


# ---------------------------------------------------------------------------
# Pilot quarantine
# ---------------------------------------------------------------------------


def test_pilot_arc_writes_outside_the_shards_glob(tmp_path):
    store_root = tmp_path / "store"
    report = arcs.pilot_arc(store_root, tmp_path / "weights", True, roster_key=KEY)
    assert report["computed"] > 0
    pilot_dir = arcs.pilot_store_dir(store_root, KEY)
    assert (pilot_dir / "evidence.jsonl").is_file()
    shards = store_root / "shards"
    hits = list(shards.rglob("evidence.jsonl")) if shards.is_dir() else []
    assert not hits, "pilot evidence must never land under shards/"
    # The pilot slice bank exists in quarantine and nowhere else.
    from reward_lens.core.store import EvidenceStore

    bank = ek.find_one(
        EvidenceStore(pilot_dir), ek.OBS_SCORES, roster_key=KEY,
        slice_name=arcs.PILOT_SLICE,
    )
    assert bank.value.layout == "pairs"
    # Idempotent: a second run loads nothing and appends nothing.
    again = arcs.pilot_arc(store_root, tmp_path / "weights", True, roster_key=KEY)
    assert again["appended"] == 0


def test_pilot_slice_is_never_a_confirmatory_dataset():
    from campaign.registry import CARDS

    for card in CARDS:
        assert arcs.PILOT_SLICE not in card.datasets


def test_real_pilot_slice_converter_derives_disjoint_pairs():
    # Uses the committed cache; both derived slices load without network.
    from campaign.data import load_slice

    pilot = load_slice("rb2-pilot-256")
    calibration = load_slice("rb2-calibration-200")
    pilot_ids = {item.lineage.seed_id for item in pilot.view}
    cal_ids = {item.lineage.seed_id for item in calibration.view}
    assert len(pilot.view) == 256
    assert not pilot_ids & cal_ids


# ---------------------------------------------------------------------------
# Spend: admission order, no skip-ahead, fleet cuts, pilot metering
# ---------------------------------------------------------------------------


def _report(total_usd: float) -> spend.SpendReport:
    return spend.SpendReport(h100_seconds=total_usd / config.H100_USD_PER_S)


def test_admission_follows_reverse_kill_order_and_stops_at_first_non_fit():
    assert spend.STRETCH_ADMISSION_ORDER == ("T3-DECOMP", "T3-27B", "T3-FIELD")
    # Strict-floor constants after the round-two cost audit: 27B at floor plus load,
    # FIELD's Lanczos at floor, DECOMP as the CPU-only line it actually is.
    assert spend.STRETCH_COST_USD == {"T3-27B": 4.2, "T3-FIELD": 1.9, "T3-DECOMP": 0.05}
    # Headroom fits DECOMP (0.05) but not 27B (4.2); FIELD (1.9) would fit but must not
    # be skipped ahead to.
    headroom = 0.15 + 0.05 + 2.0
    verdict = spend.tier3_admission(_report(config.SPEND_CAP_USD - headroom))
    assert verdict.cuts == ["ADMIT T3-DECOMP"]
    assert any("no skip-ahead" in r for r in verdict.reasons)
    assert any("T3-FIELD not considered" in r for r in verdict.reasons)


def test_fleet_checkpoint_automates_only_remaining_cuts_and_holds_for_operator():
    verdict = spend.checkpoint_fleet(_report(config.SPEND_CAP_USD + 5.0))
    # The automated cut names the gold arm that exists (the 8B; no 27B gold arm was ever
    # implemented) so acting on the cut actually saves its dollars.
    assert verdict.cuts == ["HUMP-gold-arm"]
    assert not verdict.proceed
    assert any("human decision" in r for r in verdict.reasons)
    assert not any("GAUGE-XFAM" in c or "PPE" in c for c in verdict.cuts)


def test_measure_spend_reads_the_pilot_root_too(tmp_path):
    def _write(dir_path: Path, seconds: float) -> None:
        from reward_lens.core.store import EvidenceStore

        store = EvidenceStore(dir_path)
        store.append(
            ek.record_intermediate(
                observable=ek.OBS_SCORES,
                roster_key=KEY,
                slice_name="x",
                value={"n": 1},
                gpu="H100",
                arc="test",
                gpu_seconds=seconds,
            )
        )

    _write(tmp_path / "shards" / "model--a", 100.0)
    _write(tmp_path / "pilot" / KEY, 50.0)
    without = spend.measure_spend([tmp_path / "shards" / "model--a"])
    with_pilot = spend.measure_spend(
        [tmp_path / "shards" / "model--a"], pilot_root=tmp_path / "pilot"
    )
    assert with_pilot.h100_seconds == pytest.approx(without.h100_seconds + 50.0)


# ---------------------------------------------------------------------------
# The step 9 checkpoint refusals and the admitted-card commands
# ---------------------------------------------------------------------------


def test_checkpoint_refuses_a_missing_store_root(tmp_path, capsys):
    code = checkpoint_cli.run_checkpoint(tmp_path / "nope", tmp_path / "dest")
    assert code == 2
    assert "does not exist" in capsys.readouterr().out


def test_checkpoint_refuses_zero_shards(tmp_path, capsys):
    (tmp_path / "store" / "shards").mkdir(parents=True)
    code = checkpoint_cli.run_checkpoint(tmp_path / "store", tmp_path / "dest")
    assert code == 2
    assert "no evidence shards" in capsys.readouterr().out


def test_checkpoint_prints_the_exact_admitted_commands(tmp_path, capsys):
    from reward_lens.core.store import EvidenceStore

    shard = tmp_path / "store" / "shards" / "model--a"
    EvidenceStore(shard).append(
        ek.record_intermediate(
            observable=ek.OBS_SCORES, roster_key=KEY, slice_name="x",
            value={"n": 1}, gpu="H100", arc="test", gpu_seconds=1.0,
        )
    )
    code = checkpoint_cli.run_checkpoint(tmp_path / "store", tmp_path / "dest")
    out = capsys.readouterr().out
    assert code == 0
    assert "--roster-key skywork-gemma2-27b" in out
    assert "--roster-key skywork-v2-llama31-8b --hessian" in out
    assert "closes from existing banks" in out
    assert set(checkpoint_cli.TIER3_COMMANDS) == set(spend.STRETCH_COST_USD)


# ---------------------------------------------------------------------------
# Freeze retry-safety
# ---------------------------------------------------------------------------


def _git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.check_call(["git", "init", "-q"], cwd=repo)
    subprocess.check_call(["git", "config", "user.email", "t@t"], cwd=repo)
    subprocess.check_call(["git", "config", "user.name", "t"], cwd=repo)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "."], cwd=repo)
    subprocess.check_call(["git", "commit", "-q", "-m", "base"], cwd=repo)
    return repo


def test_clean_tree_exempts_the_freezes_own_outputs(tmp_path):
    repo = _git_repo(tmp_path)
    (repo / "specs" / "frozen").mkdir(parents=True)
    (repo / "specs" / "frozen" / "x.json").write_text("{}\n", encoding="utf-8")
    (repo / "specs" / "nulls.json").write_text("{}\n", encoding="utf-8")
    (repo / "runs").mkdir()
    (repo / "runs" / "log.txt").write_text("run\n", encoding="utf-8")
    (repo / "campaign" / "data").mkdir(parents=True)
    (repo / "campaign" / "data" / "slices.lock.json").write_text("{}\n", encoding="utf-8")
    sha = freeze.assert_clean_tree(repo)
    assert len(sha) == 40
    # A non-exempt dirty path still refuses.
    (repo / "b.py").write_text("y = 2\n", encoding="utf-8")
    with pytest.raises(freeze.FreezeRefused):
        freeze.assert_clean_tree(repo)


def test_staged_freezes_refuse_overwrite_without_force(tmp_path):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "ladder_intervals.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(freeze.FreezeRefused, match="--force"):
        freeze.stage_ladder(tmp_path / "store", frozen_dir=frozen)
    (frozen / "hump_prediction.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(freeze.FreezeRefused, match="--force"):
        freeze.stage_hump(tmp_path / "store", frozen_dir=frozen)


# ---------------------------------------------------------------------------
# The shard writer lock
# ---------------------------------------------------------------------------


def test_fresh_writer_lock_refuses_and_stale_lock_is_stolen(tmp_path):
    store_root = tmp_path / "store"
    shard_dir = store_root / "shards" / arcs.shard_id(arcs.arc_label("model", KEY))
    shard_dir.mkdir(parents=True)
    lock = shard_dir / "writer.lock"
    lock.write_text('{"pid": 1, "host": "other"}\n', encoding="utf-8")
    with pytest.raises(arcs.ShardLocked):
        arcs.model_arc(
            store_root, tmp_path / "weights", True, roster_key=KEY, menu=("rb2-full",)
        )
    # Age the lock past the freshness window: the rerun steals it and completes.
    stale = time.time() - (arcs.SHARD_LOCK_FRESH_S + 60)
    os.utime(lock, (stale, stale))
    report = arcs.model_arc(
        store_root, tmp_path / "weights", True, roster_key=KEY, menu=("rb2-full",)
    )
    assert report["appended"] > 0
    assert not lock.exists(), "a clean exit must release the lock"


def test_kill_drill_releases_the_lock_so_resume_needs_no_steal(tmp_path):
    def kill_hook(label: str, n: int) -> None:
        raise SystemExit(3)

    arcs.checkpoint_hook = kill_hook
    try:
        with pytest.raises(SystemExit):
            arcs.model_arc(
                tmp_path / "store", tmp_path / "weights", True,
                roster_key=KEY, menu=("rb2-full",),
            )
    finally:
        arcs.checkpoint_hook = None
    shard_dir = tmp_path / "store" / "shards" / arcs.shard_id(arcs.arc_label("model", KEY))
    assert not (shard_dir / "writer.lock").exists()
    report = arcs.model_arc(
        tmp_path / "store", tmp_path / "weights", True, roster_key=KEY, menu=("rb2-full",)
    )
    assert report["appended"] > 0


# ---------------------------------------------------------------------------
# Single pass: capture-derived banks equal the scoring pass, derived slices index
# ---------------------------------------------------------------------------


def test_capture_pass_scores_equal_the_scoring_path(tmp_path):
    signal = from_tiny(seed=0)
    acts = captures.activation_store(tmp_path / "store")
    view = [(f"q{i}", f"answer {i} with [fact] content") for i in range(10)]
    plan = captures.union_plans(
        [("rb2-full", captures.default_observational_sites(signal.meta.n_layers))]
    )[0]
    warm = captures.warm_capture(acts, signal, view, plan, chunk_size=4,
                                 score_readout="reward")
    assert warm.scores is not None and warm.scores.shape == (10,)
    direct = np.asarray(signal.score(view).value.values, dtype=np.float64).ravel()
    np.testing.assert_allclose(warm.scores, direct, rtol=0, atol=1e-6)
    # A found-warm entry runs no forward and returns no scores.
    again = captures.warm_capture(acts, signal, view, plan, chunk_size=4,
                                  score_readout="reward")
    assert not again.computed and again.scores is None


def test_model_arc_single_passes_rb2_and_indexes_derived_slices(tmp_path):
    store_root = tmp_path / "store"
    report = arcs.model_arc(
        store_root, tmp_path / "weights", True,
        roster_key=KEY, menu=("rb2-full", "rb2-calibration-200"),
    )
    assert report["computed"] > 0
    store = arcs.shard_store(store_root, arcs.arc_label("model", KEY))
    bank = ek.find_one(store, ek.OBS_SCORES, roster_key=KEY, slice_name="rb2-full")
    assert bank.value.meta.get("derived_from") == "capture-pass"
    hits = ek.find_intermediates(store, ek.OBS_SCORES, roster_key=KEY)
    part_slices = {e.subject.extra["slice"] for e in hits}
    assert not any(s.startswith("rb2-full::part") for s in part_slices), (
        "rb2-full must not be forwarded a second time by the chunk scorer"
    )
    derived = ek.find_one(
        store, ek.OBS_SCORES, roster_key=KEY, slice_name="rb2-calibration-200"
    )
    assert derived.value.meta.get("derived_from") == "rb2-full"
    assert derived.value.layout == "pairs"


def test_derived_pair_bank_indexes_matching_rows_exactly(tmp_path):
    from campaign.payloads import ScoreBank

    signal = from_tiny(seed=1)
    ctx = arcs.ArcContext(
        label="model:test", roster_key=KEY, gpu="cpu", tiny=True,
        store=arcs.shard_store(tmp_path, "model:test"),
        store_root=tmp_path, weights_root=tmp_path,
    )
    parent_ids = ["item-a", "item-b", "item-c"]
    parent_scores = np.arange(12, dtype=np.float32).reshape(3, 4)
    ctx.record(
        ek.OBS_SCORES, "rb2-full",
        ScoreBank(item_ids=parent_ids, scores=parent_scores, layout="best-of-4", meta={}),
    )
    view = arcs.SliceView(
        slice_name="rb2-calibration-200", kind="pairs",
        item_ids=["item-b", "item-missing"],
        rows=[("p1", "chosen one"), ("p1", "rejected one"),
              ("p2", "chosen two"), ("p2", "rejected two")],
        meta={"columns": ["chosen", "rejected"]},
    )
    assert arcs._derived_pair_bank(ctx, signal, view)
    bank = ek.find_one(
        ctx.store, ek.OBS_SCORES, roster_key=KEY, slice_name="rb2-calibration-200"
    )
    got = np.asarray(bank.value.scores, dtype=np.float64)
    np.testing.assert_allclose(got[0], parent_scores[1, :2])
    assert bank.value.meta["indexed_rows"] == 1
    assert bank.value.meta["scored_rows"] == 1
    direct = np.asarray(
        signal.score(view.rows[2:]).value.values, dtype=np.float64
    ).ravel()
    np.testing.assert_allclose(got[1], direct, atol=1e-6)


# ---------------------------------------------------------------------------
# The instrument-independent ADJ-AVP key and the FIELD Lanczos basis
# ---------------------------------------------------------------------------


def test_labels_from_deltas_uses_the_half_max_rule_with_both_classes():
    labels = arcs._labels_from_deltas(np.array([1.0, 0.6, 0.1, 0.05]))
    assert labels.tolist() == [1, 1, 0, 0]
    # Degenerate all-planted input falls back to the top half so both classes exist.
    labels = arcs._labels_from_deltas(np.array([1.0, 1.0, 1.0, 1.0]))
    assert 0 < labels.sum() < 4


def test_field_lanczos_basis_meets_the_resolver_contract(tmp_path):
    report = arcs.big_arc(
        tmp_path / "store", tmp_path / "weights", True,
        roster_key="skywork-v2-llama31-8b", hessian=True,
    )
    assert report["computed"] == 1
    store = arcs.shard_store(
        tmp_path / "store", arcs.arc_label("big", "skywork-v2-llama31-8b")
    )
    ev = ek.find_one(
        store, "campaign.subspace.flat",
        roster_key="skywork-v2-llama31-8b", slice_name="rb2-full",
    )
    basis = np.asarray(ev.value.matrix, dtype=np.float64)
    assert basis.ndim == 2 and 1 <= basis.shape[1] < basis.shape[0]
    assert ev.value.meta["method"] == "lanczos"
    assert len(ev.value.meta["curvatures"]) == basis.shape[1]
    band = ev.value.meta["eigenvalue_band"]
    assert band[0] == -band[1]


# ---------------------------------------------------------------------------
# EMB checkpoint validation
# ---------------------------------------------------------------------------


def test_expected_steps_mirror_the_trainer_rule():
    # 48 pairs, batch 8, 2 epochs: 12 optimizer steps; save_every=4 saves 0,4,8,12.
    assert arcs._expected_steps(48, 2, 8, 4) == [0, 4, 8, 12]
    # A final step off the interval is saved too.
    assert arcs._expected_steps(10, 1, 4, 2) == [0, 2, 3]


def test_validated_steps_rejects_torn_and_half_trajectories(tmp_path):
    ckpt = tmp_path / "emb"
    for step in (0, 4, 8):
        d = ckpt / f"step_{step}"
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}\n", encoding="utf-8")
    # Fewer steps than the recipe expects is a half-trajectory: retrain.
    assert arcs._validated_steps(ckpt, [0, 4, 8, 12]) == []
    # The full set validates.
    d = ckpt / "step_12"
    d.mkdir()
    (d / "config.json").write_text("{}\n", encoding="utf-8")
    assert arcs._validated_steps(ckpt, [0, 4, 8, 12]) == [0, 4, 8, 12]
    # A torn directory (no config.json) is deleted and forces retraining.
    (d / "config.json").unlink()
    assert arcs._validated_steps(ckpt, [0, 4, 8, 12]) == []
    assert not d.exists()


# ---------------------------------------------------------------------------
# Runbook shape
# ---------------------------------------------------------------------------


def test_runbook_covers_the_whole_fleet_with_the_real_flag():
    import reproduce

    assert {"qrm", "tulu-dpo", "internlm2-1.8b"} <= set(reproduce._FLEET_5A)
    all_commands = [c for s in reproduce.STEPS for c in s["commands"]]
    assert not any("--model " in c for c in all_commands), "the CLI flag is --roster-key"
    joined = "\n".join(all_commands)
    assert "--roster-key" in joined
    assert "--accept-underpowered" in joined and "EMB-LORA" in joined
    assert "campaign.app::checkpoint" in joined
    assert "campaign.app::checkpoint --stage fleet" in joined
    assert "campaign.app::stage_ladder" in joined and "campaign.app::stage_hump" in joined
    assert joined.count("--menu ppe") >= 2
    assert "--menu ppe-human" in joined
    # Step 10 retrieves the volume artifacts BEFORE the local claims re-check.
    step10 = reproduce._BY_ID["10"]["commands"]
    gate_idx = next(i for i, c in enumerate(step10) if "claims_gate" in c)
    assert any("volume get rl-store RESULTS.md" in c for c in step10[:gate_idx])
    assert any("volume get rl-store runs/campaign" in c for c in step10[:gate_idx])
    # The held-out seat's PPE work sits at 5c, behind the interval-freeze guard.
    step5a = "\n".join(reproduce._BY_ID["5a"]["commands"])
    assert "skywork-v2-qwen3-8b" not in step5a


def test_documented_acceptance_list_matches_the_real_power_gate():
    """The step-4 alternate command must name exactly the cards the gate fails.

    Round two proved a substring check cannot catch list drift: the gate's failing set
    moved when the power arithmetic was repriced and both runbooks kept a stale list that
    the freeze then refused. This asserts set equality against the real gate at the
    committed lock and nulls, so any future repricing fails here first.
    """
    import json
    import re

    import reproduce
    from campaign import power
    from campaign.registry import TIER2_CARDS

    lock = json.loads(
        (reproduce.REPO_ROOT / "campaign" / "data" / "slices.lock.json").read_text()
    )
    failing = {v.card for v in power.power_gate(TIER2_CARDS, lock) if not v.passes}
    step4 = "\n".join(reproduce._BY_ID["4"]["commands"])
    match = re.search(r"--accept-underpowered\s+([A-Z0-9,\-]+)", step4)
    assert match, "step 4 documents no acceptance command"
    documented = set(match.group(1).split(","))
    assert documented == failing, (
        f"the documented acceptance list drifted from the gate: documented "
        f"{sorted(documented)}, gate fails {sorted(failing)}"
    )


def test_category_accuracy_excludes_calibration_rows(tmp_path):
    store_root = tmp_path / "store"
    arcs.model_arc(store_root, tmp_path / "weights", True, roster_key=KEY, menu=("rb2-full",))
    store = arcs.shard_store(store_root, arcs.arc_label("model", KEY))
    acc = ek.find_one(store, ek.OBS_CATEGORY_ACC, roster_key=KEY, slice_name="rb2-full")
    assert acc.value.meta["exclusion"] == "rb2-calibration-200 source ids"


def test_freeze_manifest_evidence_lands_in_a_merged_shard(tmp_path):
    # The rehearsal path already targets <store>/shards/freeze; assert the shape holds.
    root = tmp_path / "root"
    store = root / "store"
    (store / "shards").mkdir(parents=True)
    from campaign import tinyrun

    tinyrun.rehearse_main_freeze(root, store)
    assert (store / "shards" / "freeze" / "evidence.jsonl").is_file()
