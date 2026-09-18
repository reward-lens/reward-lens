"""Arc-layer tests: shard idempotency, the kill/resume drill, the R1 assertion, and the
chunked capture store keys. Everything runs the tiny mode on CPU in seconds with no network;
these are the same code paths the Modal functions drive, minus real weights."""

from __future__ import annotations

import os
import sys
import types

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pytest

from campaign import arcs, captures, config, throughput
from campaign import evidence_keys as ek
from reward_lens.signals.loaders import from_tiny

MENU = ("rb2-full",)
KEY = "skywork-v2-qwen3-0.6b"


def _run_model_arc(tmp_path, **kwargs):
    return arcs.model_arc(
        tmp_path / "store", tmp_path / "weights", True, roster_key=KEY, menu=MENU, **kwargs
    )


def test_model_arc_second_run_appends_nothing(tmp_path):
    first = _run_model_arc(tmp_path)
    assert first["appended"] > 0
    assert first["checkpoints"] > 0
    second = _run_model_arc(tmp_path)
    assert second["appended"] == 0
    assert second["computed"] == 0


def test_kill_after_checkpoint_bounds_loss_and_resume_completes(tmp_path):
    """The dead-container drill on a chunk-scored slice.

    rb2-full is single-pass now (its bank rides the capture forward), so the scored-parts
    drill runs on rmbench-full, which still goes through the checkpointed chunk scorer:
    the kill lands after durable part appends and before the assembled bank, and the
    resume reuses every pre-kill part.
    """
    calls: list[int] = []

    def kill_hook(label: str, n_checkpoints: int) -> None:
        calls.append(n_checkpoints)
        if n_checkpoints >= 2:
            raise SystemExit(3)

    arcs.checkpoint_hook = kill_hook
    try:
        with pytest.raises(SystemExit):
            arcs.model_arc(
                tmp_path / "store", tmp_path / "weights", True,
                roster_key=KEY, menu=("rmbench-full",),
            )
    finally:
        arcs.checkpoint_hook = None

    # The kill landed after durable appends: the shard holds the checkpointed parts but not
    # the assembled bank.
    store = arcs.shard_store(tmp_path / "store", arcs.arc_label("model", KEY))
    parts = ek.find_intermediates(store, ek.OBS_SCORES, roster_key=KEY)
    part_slices = {e.subject.extra["slice"] for e in parts}
    assert any(s.startswith("rmbench-full::part") for s in part_slices)
    assert "rmbench-full" not in part_slices

    resumed = arcs.model_arc(
        tmp_path / "store", tmp_path / "weights", True,
        roster_key=KEY, menu=("rmbench-full",),
    )
    assert resumed["skipped"] >= 1  # the pre-kill parts were reused, not recomputed
    store = arcs.shard_store(tmp_path / "store", arcs.arc_label("model", KEY))
    final = ek.find_one(store, ek.OBS_SCORES, roster_key=KEY, slice_name="rmbench-full")
    assert final.value.layout == "pairs"
    assert np.asarray(final.value.scores).shape[1] == 2


def test_assert_throughput_fires_below_half_floor():
    with pytest.raises(throughput.ThroughputError):
        throughput.assert_throughput(5.9, 12.0, "model:test", KEY)
    # At or above half the floor the arc proceeds; a zero floor disables the check entirely.
    throughput.assert_throughput(6.0, 12.0, "model:test", KEY)
    throughput.assert_throughput(0.001, 0.0, "model:test", KEY)


def test_heartbeat_records_evidence_even_when_it_aborts(tmp_path):
    from reward_lens.core.store import EvidenceStore

    store = EvidenceStore(tmp_path / "shard")
    with pytest.raises(throughput.ThroughputError):
        throughput.first_chunk_heartbeat(
            store,
            lambda: 1,  # one sequence in however long the call takes: far below any floor
            arc="model:test",
            gpu="cpu",
            roster_key=KEY,
            floor=1e9,
        )
    beats = ek.find_intermediates(store, ek.OBS_HEARTBEAT, roster_key=KEY)
    assert len(beats) == 1
    assert beats[0].value.floor_seq_per_s == 1e9


def test_capture_chunking_writes_the_expected_store_keys(tmp_path):
    signal = from_tiny(seed=0)
    acts = captures.activation_store(tmp_path / "store")
    view = [(f"q{i}", f"answer {i} with [fact] content") for i in range(10)]
    n_layers = signal.meta.n_layers
    plans = captures.union_plans(
        [("rb2-full", captures.default_observational_sites(n_layers))]
    ) + captures.union_plans(
        [(arcs.FRAME_SLICE, captures.default_observational_sites(n_layers)[-1:])],
        dtype=config.FRAME_DTYPE,
    )
    chunks_seen: list[tuple[int, int]] = []
    results = [
        captures.warm_capture(
            acts, signal, view, plan, chunk_size=4,
            on_chunk=lambda done, total: chunks_seen.append((done, total)),
        )
        for plan in plans
    ]

    # The dataset id hands the store the dtype and position the shipped key ignores, so the
    # fp16 and fp32 entries land under distinct keys and both exist.
    fp = signal.meta.fingerprint
    for plan, warm in zip(plans, results):
        expected = acts.key(
            fp,
            captures.store_dataset_id(plan.slice_name, plan.dtype, plan.position),
            plan.sites,
            plan.position,
            plan.dtype,
            "none",
        )
        assert warm.store_key == expected
        assert acts.has(fp, expected)
        assert warm.computed
        assert warm.n_items == len(view)
    assert results[0].store_key != results[1].store_key
    assert chunks_seen[:3] == [(1, 3), (2, 3), (3, 3)]  # 10 items in chunks of 4

    # The chunks concatenated into one full-view entry of the requested dtype.
    handle = acts.get(fp, results[0].store_key)
    cap = next(iter(handle))
    site = captures.default_observational_sites(n_layers)[0]
    tensor = cap.get(site) if hasattr(cap, "get") else cap.tensors[site]
    assert tuple(tensor.shape) == (len(view), signal.meta.d_model)
    assert "float16" in str(tensor.dtype)

    # Re-warming is a no-op: the entry exists, so no forward runs and no time is metered.
    again = captures.warm_capture(acts, signal, view, plans[0], chunk_size=4)
    assert not again.computed
    assert again.gpu_seconds == 0.0
    assert again.content_hash == results[0].content_hash


def test_feature_hook_prefers_campaign_features_when_present(tmp_path, monkeypatch):
    """The lazy hook delegates to campaign.features when it exposes internal_features.

    A stub stands in for that entry point, which is optional on the module. Both sys.modules
    and the parent package attribute are patched because ``import a.b as x`` binds through
    the parent attribute when it exists.
    """
    import campaign as campaign_pkg
    from campaign.payloads import FeatureMatrix

    calls: dict[str, object] = {}

    def fake_internal_features(
        capture, *, sites, item_ids, lens_scores, roster_key, slice_name
    ):
        calls["slice"] = slice_name
        return FeatureMatrix(
            item_ids=list(item_ids), names=["stub"], matrix=np.zeros((len(item_ids), 1))
        )

    stub = types.ModuleType("campaign.features")
    stub.internal_features = fake_internal_features
    monkeypatch.setitem(sys.modules, "campaign.features", stub)
    monkeypatch.setattr(campaign_pkg, "features", stub, raising=False)

    signal = from_tiny(seed=0)
    acts = captures.activation_store(tmp_path / "store")
    plan = captures.union_plans(
        [("rb2-full", captures.default_observational_sites(signal.meta.n_layers))]
    )[0]
    view = [("q", "a short answer")]
    warm = captures.warm_capture(acts, signal, view, plan, chunk_size=4)
    cap = next(iter(acts.get(signal.meta.fingerprint, warm.store_key)))
    fm = captures.internal_feature_matrix(
        cap,
        sites=list(cap.tensors.keys()),
        item_ids=["rb2-full::row0"],
        lens_scores=np.zeros(1),
        roster_key=KEY,
        slice_name="rb2-full",
        tiny=True,
    )
    assert calls["slice"] == "rb2-full"
    assert fm.names == ["stub"]


def test_data_and_model_banks_share_prompt_ids(tmp_path):
    """The seam CHI-DRIFT asserts at merge time: the data arc's banked features and a model
    arc's bank are built from the same deterministic view, so their prompt ids align."""
    store_root, weights = tmp_path / "store", tmp_path / "weights"
    arcs.data_arc(store_root, weights, True)
    arcs.model_arc(
        store_root, weights, True, roster_key=KEY, menu=("ultrafeedback-bank",)
    )
    data_store = arcs.shard_store(store_root, "data")
    feats = ek.find_one(
        data_store, ek.OBS_BANKED_FEATURES,
        roster_key=ek.DATA_KEY, slice_name="ultrafeedback-bank",
    )
    model_store = arcs.shard_store(store_root, arcs.arc_label("model", KEY))
    bank = ek.find_one(
        model_store, ek.OBS_SCORES, roster_key=KEY, slice_name="ultrafeedback-bank"
    )
    assert bank.value.layout == "bank"
    assert list(feats.value.prompt_ids) == list(bank.value.item_ids)
    assert np.asarray(feats.value.tensor).shape[:2] == np.asarray(bank.value.scores).shape
