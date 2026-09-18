"""Data-plane tests: tiny views, the serde cache, tournament matrices, and the lock.

Everything here is CPU-only and network-free. Tests over the real ingested slices are
guarded on the cache and lock files existing (they are produced by
``python -m campaign.data.ingest`` and are gitignored), so a fresh checkout skips them
rather than downloading anything.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from reward_lens.core.types import content_hash
from reward_lens.data.schema import DataView, EdgeObs, Response, Tournament, content_of

from campaign.config import SLICES, SLICES_LOCK_PATH
from campaign.data import loaders
from campaign.data.ingest import lock_entry
from campaign.data.loaders import CONVERTERS, SliceResult, build_receipt_triple, load_slice
from campaign.data.serde import item_from_dict, item_to_dict
from campaign.data.tiny import tiny_view
from campaign.data.tournaments import (
    edge_matrix,
    edges_from_ranks,
    edges_from_scores,
    matrix_payload,
    subjects_entry,
)

_LOCK_KEYS = {
    "content_hash",
    "n",
    "n_effective",
    "duplicate_fraction",
    "source",
    "revision",
    "config",
    "split",
    "seed",
    "notes",
}


# ---------------------------------------------------------------------------
# Tiny views
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CONVERTERS))
def test_tiny_view_loads_with_honest_lineage(name):
    view = tiny_view(name)
    assert 8 <= len(view) <= 30
    assert view.effective_n() > 0
    for item in view:
        assert item.lineage.content_hash == content_hash(content_of(item), "ch")
    # Determinism: a second build is byte-identical in content.
    assert str(view.checksum()) == str(tiny_view(name).checksum())


def test_tiny_view_rejects_organism_slices_with_a_pointer():
    with pytest.raises(KeyError, match="tiny_subjects"):
        tiny_view("caltransfer-organisms")
    with pytest.raises(KeyError, match="unknown slice"):
        tiny_view("not-a-slice")


def test_every_text_slice_has_a_converter():
    organism = {"caltransfer-organisms", "adjavp-organism"}
    assert set(CONVERTERS) == set(SLICES) - organism


def test_tiny_receipt_triples_follow_the_arm_semantics():
    view = tiny_view("receipt-triples-1600")
    for t in view:
        arms = [r.meta["arm"] for r in t.responses]
        assert arms == ["valid", "absent", "failing"]
        valid, absent, failing = (r.text for r in t.responses)
        # The absent arm deletes the receipt sentence, the failing arm flips its verdict.
        assert valid.startswith(absent)
        assert len(valid) > len(absent)
        assert failing != valid
        assert failing.startswith(absent)


def test_tiny_best_of_k_containers_keep_the_chosen_block_first():
    for t in tiny_view("rb2-full"):
        n_chosen = t.meta["n_chosen"]
        assert 1 <= n_chosen < len(t.responses)
        for e in t.edges:
            assert e.i < n_chosen <= e.j


# ---------------------------------------------------------------------------
# Receipt triple construction (the real builder on a fixed claim)
# ---------------------------------------------------------------------------


def test_receipt_triple_arms_are_the_exact_edit_semantics():
    claim = "The measurement report concludes that the reading stayed inside the band."
    t = build_receipt_triple("Summarize the report.", claim, seed_id="fix:0", index=0)
    valid, absent, failing = (r.text for r in t.responses)
    assert absent == claim
    assert valid == f"{claim} {loaders.RECEIPT_TEMPLATES[0]}"
    assert failing == valid.replace("pass", "fail")
    assert t.lineage.seed_id == "fix:0"
    assert "receipt-triple:valid-absent-failing" in t.lineage.ops


def test_receipt_templates_flip_one_keyword_each():
    claim = "The archived summary restates the committee's finding in plain language today."
    for idx in range(len(loaders.RECEIPT_TEMPLATES)):
        t = build_receipt_triple("p", claim, seed_id=f"fix:{idx}", index=idx)
        valid, absent, failing = (r.text for r in t.responses)
        assert absent == claim
        assert valid != failing
        # The edit is a single keyword substitution, never a rewrite of the claim.
        assert failing.startswith(claim)


# ---------------------------------------------------------------------------
# Serde round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["rb2-full", "processbench-full", "receipt-triples-1600",
                                  "evalaware-paired", "nectar-tournaments"])
def test_serde_roundtrip_preserves_content_and_lineage(name):
    for item in tiny_view(name):
        clone = item_from_dict(json.loads(json.dumps(item_to_dict(item))))
        assert content_of(clone) == content_of(item)
        assert clone.lineage == item.lineage
        assert clone.meta == item.meta


def test_serde_rejects_a_tampered_item():
    d = item_to_dict(tiny_view("hh-ood")[0])
    d["chosen"]["text"] += " tampered"
    with pytest.raises(Exception, match="content hash mismatch"):
        item_from_dict(d)


def test_cache_roundtrip_reproduces_the_checksum(tmp_path, monkeypatch):
    # cache_dir() honors CAMPAIGN_DATA_CACHE over the module attribute, so isolate via the
    # env var: patching only CACHE_DIR is a no-op under a volume-backed container (where the
    # var is set) and would write this tiny fixture over the real slice on the shared cache.
    monkeypatch.setenv("CAMPAIGN_DATA_CACHE", str(tmp_path))
    monkeypatch.setattr(loaders, "CACHE_DIR", tmp_path)
    view = tiny_view("ultrafeedback-tournaments")
    result = SliceResult(
        name="ultrafeedback-tournaments", view=view, source="tiny", revision=None,
        config=None, split="tiny", seed=0, notes="roundtrip fixture",
    )
    loaders._write_cache(result)
    again = load_slice("ultrafeedback-tournaments")
    assert str(again.view.checksum()) == str(view.checksum())
    assert again.notes == "roundtrip fixture"
    # Run twice on the same cached data: same hash both times.
    assert str(load_slice("ultrafeedback-tournaments").view.checksum()) == str(view.checksum())


# ---------------------------------------------------------------------------
# Tournament matrices (hand-built fixture)
# ---------------------------------------------------------------------------


def _fixture_tournaments() -> list[Tournament]:
    # Tournament 0: a pure 3-cycle. Tournament 1: a clean order over 4 items from ranks.
    cycle = loaders._tournament(
        "cycle",
        [Response(text=f"c{k}") for k in range(3)],
        [
            EdgeObs(i=0, j=1, wins_i=1, wins_j=0, annotator_id="fix"),
            EdgeObs(i=1, j=2, wins_i=1, wins_j=0, annotator_id="fix"),
            EdgeObs(i=0, j=2, wins_i=0, wins_j=1, annotator_id="fix"),
        ],
        seed_id="fix:cycle",
        ops=("fixture",),
    )
    ordered = loaders._tournament(
        "ordered",
        [Response(text=f"o{k}") for k in range(4)],
        edges_from_ranks([1.0, 2.0, 3.0, 4.0], judge_id="fix"),
        seed_id="fix:ordered",
        ops=("fixture",),
    )
    return [cycle, ordered]


def test_edge_matrix_layout_matches_the_topo_contract():
    matrix, n_items = edge_matrix(_fixture_tournaments())
    assert n_items == [3, 4]
    assert matrix.shape == (3 + 6, 5)
    # Tournament 0 rows, exactly as constructed.
    np.testing.assert_array_equal(
        matrix[:3],
        np.asarray([[0, 0, 1, 1, 0], [0, 1, 2, 1, 0], [0, 0, 2, 0, 1]], dtype=np.float64),
    )
    # Tournament 1: every pair decisive, lower rank wins.
    ordered = matrix[matrix[:, 0] == 1]
    assert ordered.shape == (6, 5)
    for _, i, j, wi, wj in ordered:
        assert i < j and wi == 1.0 and wj == 0.0


def test_edge_builders_drop_ties_and_orient_wins():
    assert len(edges_from_scores([5.0, 5.0, 5.0])) == 0
    edges = edges_from_scores([9.0, 2.0, 9.0], judge_id="j")
    assert [(e.i, e.j, e.wins_i, e.wins_j) for e in edges] == [(0, 1, 1, 0), (1, 2, 0, 1)]
    edges = edges_from_ranks([2.0, 1.0], annotator_id="a")
    assert [(e.i, e.j, e.wins_i, e.wins_j) for e in edges] == [(0, 1, 0, 1)]


def test_fixture_cycle_carries_pure_intransitive_mass():
    # The shipped Hodge decomposition is the consumer; a three-cycle must be all curl.
    from studies.s06_topology.hodge import decompose_corpus

    cycle = _fixture_tournaments()[0]
    d = decompose_corpus([cycle])
    assert d.intransitive_mass == pytest.approx(1.0)
    # A clean order is gradient-dominated; the saturated unit margins on the complete
    # graph leave a small curl residual, so this bound is deliberately not exact.
    ordered = _fixture_tournaments()[1]
    d2 = decompose_corpus([ordered])
    assert d2.gradient_mass > 0.8
    assert d2.intransitive_mass < 0.2


def test_subjects_entry_and_payload_agree_with_edge_matrix():
    ts = _fixture_tournaments()
    entry = subjects_entry(ts)
    payload = matrix_payload("nectar-tournaments", ts)
    matrix, n_items = edge_matrix(ts)
    np.testing.assert_array_equal(entry["edges"], matrix)
    assert entry["n_items"] == n_items
    np.testing.assert_array_equal(payload.matrix, matrix)
    assert payload.meta["n_items"] == n_items
    assert payload.col_labels == ["t_idx", "i", "j", "wins_i", "wins_j"]


def test_record_tournaments_feeds_the_topo_resolver(tmp_path, monkeypatch):
    # The whole seam: cached slices in, OBS_TOURNAMENTS intermediates out, and the topo
    # card's resolver reads back the same matrices under the model-free join key.
    from reward_lens.core.store import EvidenceStore

    from campaign import topo
    from campaign.data.tournaments import record_tournaments

    monkeypatch.setenv("CAMPAIGN_DATA_CACHE", str(tmp_path))
    monkeypatch.setattr(loaders, "CACHE_DIR", tmp_path)
    for name in ("nectar-tournaments", "ultrafeedback-tournaments"):
        loaders._write_cache(
            SliceResult(
                name=name, view=tiny_view(name), source="tiny", revision=None,
                config=None, split="tiny", seed=0, notes="",
            )
        )
    store = EvidenceStore(tmp_path / "store")
    evs = record_tournaments(store)
    assert len(evs) == 2
    subjects = topo.resolve_subjects(store)
    for key, name in (("nectar", "nectar-tournaments"), ("ultrafeedback", "ultrafeedback-tournaments")):
        matrix, n_items = edge_matrix(load_slice(name).view)
        np.testing.assert_array_equal(np.asarray(subjects[key]["edges"]), matrix)
        assert subjects[key]["n_items"] == n_items
        assert subjects[f"{key}__ev"] in {e.id for e in evs}


def test_topo_rebuild_roundtrips_the_matrix():
    # The card's _rebuild must reproduce the same decomposition from the flat matrix.
    from studies.s06_topology.hodge import decompose_corpus

    from campaign import topo

    ts = _fixture_tournaments()
    matrix, n_items = edge_matrix(ts)
    rebuilt = topo._rebuild("fix", matrix, n_items)
    assert len(rebuilt) == 2
    direct = decompose_corpus(ts)
    replay = decompose_corpus(rebuilt)
    assert replay.intransitive_mass == pytest.approx(direct.intransitive_mass)


# ---------------------------------------------------------------------------
# Lock schema
# ---------------------------------------------------------------------------


def test_lock_entry_schema_on_a_tiny_result():
    view = tiny_view("hh-ood")
    entry = lock_entry(
        SliceResult(
            name="hh-ood", view=view, source="Anthropic/hh-rlhf", revision="deadbeef",
            config=None, split="train", seed=7, notes="fixture",
        )
    )
    assert set(entry) == _LOCK_KEYS
    assert entry["n"] == len(view)
    assert entry["n_effective"] == pytest.approx(view.effective_n())
    assert entry["duplicate_fraction"] == 0.0
    assert entry["content_hash"].startswith("ds:")


def test_lock_entry_counts_clones_honestly():
    item = tiny_view("hh-ood")[0]
    view = DataView([item, item, item], name="clones")
    entry = lock_entry(
        SliceResult(
            name="clones", view=view, source="x", revision=None, config=None,
            split="train", seed=0, notes="",
        )
    )
    assert entry["n"] == 3
    assert entry["n_effective"] == pytest.approx(1.0)
    assert entry["duplicate_fraction"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# The real ingested slices (guarded: skipped when the ingest has not run here)
# ---------------------------------------------------------------------------

_HAS_LOCK = SLICES_LOCK_PATH.exists()


@pytest.mark.skipif(not _HAS_LOCK, reason="slices.lock.json not built on this machine")
def test_real_lock_matches_the_exact_schema():
    lock = json.loads(SLICES_LOCK_PATH.read_text(encoding="utf-8"))
    assert set(lock) <= {"created_at", "slices", "failed"}
    assert lock["slices"]
    for name, entry in lock["slices"].items():
        assert name in SLICES
        assert set(entry) == _LOCK_KEYS, name
        assert entry["n"] > 0
        assert 0 < entry["n_effective"] <= entry["n"]
        assert 0.0 <= entry["duplicate_fraction"] < 1.0


@pytest.mark.skipif(not _HAS_LOCK, reason="slices.lock.json not built on this machine")
def test_cached_slices_reproduce_their_locked_hashes():
    lock = json.loads(SLICES_LOCK_PATH.read_text(encoding="utf-8"))
    checked = 0
    for name, entry in lock["slices"].items():
        if not loaders.cache_path(name).exists():
            continue
        result = load_slice(name)
        assert str(result.view.checksum()) == entry["content_hash"], name
        assert len(result.view) == entry["n"], name
        checked += 1
    assert checked > 0


@pytest.mark.skipif(
    not loaders.cache_path("rb2-full").exists(), reason="rb2-full cache not built here"
)
def test_derived_converter_is_deterministic_from_cache():
    # The derived RB2 pair slice rebuilds from the cached rb2-full without any network;
    # two builds must agree exactly (same seed, same hash).
    a = CONVERTERS["rb2-patch-40"]()
    b = CONVERTERS["rb2-patch-40"]()
    assert str(a.view.checksum()) == str(b.view.checksum())
    assert [p.lineage.seed_id for p in a.view] == [p.lineage.seed_id for p in b.view]


@pytest.mark.skipif(
    not (loaders.cache_path("hump-prompts").exists()
         and loaders.cache_path("ultrafeedback-bank").exists()),
    reason="ultrafeedback caches not built here",
)
def test_hump_prompts_are_disjoint_from_the_bank():
    bank = load_slice("ultrafeedback-bank")
    hump = load_slice("hump-prompts")
    bank_rows = {int(t.meta["source_row"]) for t in bank.view}
    bank_texts = {t.prompt_text for t in bank.view}
    hump_rows = {int(t.meta["source_row"]) for t in hump.view}
    hump_texts = {t.prompt_text for t in hump.view}
    assert not (bank_rows & hump_rows)
    assert not (bank_texts & hump_texts)
