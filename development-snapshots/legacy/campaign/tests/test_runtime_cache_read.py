"""M1 acceptance: the v1 ``.pt`` activation cache reads through the adapter (section 2.2.3, M1).

The v1 campaign left 2.5 GB of cached activations for four models under
``outputs/.../_shared_cache/<model>/floor-population-<hash>.pt``. Reading them back for free is what
makes the E-parity runs cheap on hardware that cannot hold the 8B models. This test loads exactly one
real ``.pt`` shard and asserts it returns per-layer activation tensors with sane shapes and the fp16
dtype the v1 writer used. It loads a single file (not the whole 2.5 GB) and skips with a clear
message if the campaign outputs are not present on this machine.

The second half covers the v3 ``ActivationStore``'s crash tolerance: writes land atomically (no
truncated shard can ever occupy the content-addressed name), and the existence checks the
skip-if-cached logic relies on validate the safetensors header rather than trusting the filename,
so a shard left corrupt by a killed writer is dropped and recomputed instead of poisoning re-runs.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from reward_lens.core.types import ModelFP, Site
from reward_lens.runtime.backend import Capture, CaptureSpec
from reward_lens.runtime.store import ActivationStore, InMemoryCaptureHandle, read_v1_cache

# Supply the omitted historical cache explicitly; the public snapshot has no local default.
_V1_CACHE_ROOT = (
    Path(os.environ["REWARD_LENS_V1_CACHE"])
    if os.environ.get("REWARD_LENS_V1_CACHE")
    else None
)


def _first_shard() -> Path | None:
    if _V1_CACHE_ROOT is None or not _V1_CACHE_ROOT.exists():
        return None
    shards = sorted(_V1_CACHE_ROOT.glob("*/floor-population-*.pt"))
    return shards[0] if shards else None


def test_read_one_v1_cache_shard():
    shard = _first_shard()
    if shard is None:
        pytest.skip("Set REWARD_LENS_V1_CACHE to an available v1 shared cache; E-parity fixture absent")

    cache = read_v1_cache(shard, device="cpu")

    # The residual-stream table must be non-empty and layer-keyed.
    layers = cache.layers()
    assert layers, "no residual-stream layers in the v1 cache"
    assert all(isinstance(layer, int) for layer in layers)

    # Final-token residual tensors are (B, d_model) in fp16 (the v1 writer downcast on save).
    sample = cache.residual_streams[layers[0]]
    assert sample.ndim == 2, f"expected (B, d_model), got shape {tuple(sample.shape)}"
    assert str(sample.dtype) == "torch.float16", f"expected fp16, got {sample.dtype}"

    batch_size = cache.batch_size()
    assert batch_size > 0
    # Every residual layer shares the same batch dimension.
    for layer in layers:
        assert cache.residual_streams[layer].shape[0] == batch_size

    # The shape summary is well-formed and reports the fp16 dtype.
    shapes = cache.shapes()
    assert shapes["residual_streams"]["layers"] == len(layers)
    assert shapes["residual_streams"]["dtype"] == "torch.float16"


def test_missing_path_raises_cleanly(tmp_path):
    """A missing shard raises ``FileNotFoundError`` so a fixture test can skip on it."""
    with pytest.raises(FileNotFoundError):
        read_v1_cache(tmp_path / "does-not-exist.pt")


# ---------------------------------------------------------------------------
# ActivationStore crash tolerance
# ---------------------------------------------------------------------------

_FP = ModelFP("mfp:cache-demo")
_SITE = Site(0, "resid_post")


def _capture(seed: int = 0) -> Capture:
    torch.manual_seed(seed)
    return Capture(
        tensors={_SITE: torch.randn(4, 8, dtype=torch.float32)},
        positions=[[7], [7], [7], [7]],
        dtype="float32",
    )


class _CountingSignal:
    """A stand-in signal for ``get_or_compute``: counts how often a capture is computed."""

    def __init__(self, capture: Capture):
        self._capture = capture
        self.calls = 0
        self.meta = SimpleNamespace(fingerprint=_FP)

    def capture(self, view, spec):
        self.calls += 1
        return InMemoryCaptureHandle(self._capture)


def test_put_leaves_no_temp_files_and_round_trips(tmp_path):
    store = ActivationStore(tmp_path)
    cap = _capture()
    key = store.key(_FP, "ds", (_SITE,), "final", "float32")
    shard = store.put(_FP, key, cap)

    assert store.has(_FP, key)
    assert not list(shard.parent.glob("*.tmp*")), "put left a temp file behind"
    got = store.get(_FP, key).get(_SITE)
    assert torch.equal(got, cap.tensors[_SITE])


def test_truncated_shard_is_detected_dropped_and_recomputed(tmp_path):
    # A writer killed mid-save used to leave a truncated safetensors file that every re-run
    # trusted by name. The atomic rename prevents that for new writes; this simulates a shard
    # corrupted anyway (an old writer, a bad disk) and proves the skip logic drops it and
    # get_or_compute recomputes rather than returning garbage.
    store = ActivationStore(tmp_path)
    cap = _capture()
    signal = _CountingSignal(cap)
    spec = CaptureSpec(sites=(_SITE,), dtype="float32")

    handle = store.get_or_compute(signal, view=["item"], spec=spec, dataset_id="ds")
    assert signal.calls == 1
    assert torch.equal(handle.get(_SITE), cap.tensors[_SITE])

    key = store.key(_FP, "ds", (_SITE,), "final", "float32")
    shard = store._shard_path(_FP, key)
    raw = shard.read_bytes()
    shard.write_bytes(raw[: len(raw) // 2])

    with pytest.warns(RuntimeWarning, match="corrupt activation shard"):
        assert not store.has(_FP, key)
    assert not shard.exists(), "the corrupt shard was left on disk to be trusted again"

    handle = store.get_or_compute(signal, view=["item"], spec=spec, dataset_id="ds")
    assert signal.calls == 2, "the corrupt shard was not recomputed"
    assert torch.equal(handle.get(_SITE), cap.tensors[_SITE])
    assert store.has(_FP, key)


def test_shard_without_index_reports_missing_and_put_heals_it(tmp_path):
    # A kill between the payload rename and the index write leaves an intact payload with no
    # index. That used to read back as a silently empty capture; now it counts as a miss (the
    # payload is kept, put simply rewrites both files).
    store = ActivationStore(tmp_path)
    cap = _capture()
    key = store.key(_FP, "ds", (_SITE,), "final", "float32")
    shard = store.put(_FP, key, cap)
    shard.with_suffix(".json").unlink()

    assert not store.has(_FP, key)
    assert shard.exists(), "an intact payload should survive a missing-index miss"

    store.put(_FP, key, cap)
    assert store.has(_FP, key)
    assert torch.equal(store.get(_FP, key).get(_SITE), cap.tensors[_SITE])
