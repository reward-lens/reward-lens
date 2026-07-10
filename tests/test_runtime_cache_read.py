"""M1 acceptance: the v1 ``.pt`` activation cache reads through the adapter (section 2.2.3, M1).

The v1 campaign left 2.5 GB of cached activations for four models under
``outputs/.../_shared_cache/<model>/floor-population-<hash>.pt``. Reading them back for free is what
makes the E-parity runs cheap on hardware that cannot hold the 8B models. This test loads exactly one
real ``.pt`` shard and asserts it returns per-layer activation tensors with sane shapes and the fp16
dtype the v1 writer used. It loads a single file (not the whole 2.5 GB) and skips with a clear
message if the campaign outputs are not present on this machine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reward_lens.runtime.store import read_v1_cache

# The v1 shared cache lives beside the original (untouched) repo; one shard per (model, pairset).
_V1_CACHE_ROOT = Path(
    "/home/suhail-nadaf/final-reward/reward-lens/outputs/v2_20260506_222648_unknown/_shared_cache"
)


def _first_shard() -> Path | None:
    if not _V1_CACHE_ROOT.exists():
        return None
    shards = sorted(_V1_CACHE_ROOT.glob("*/floor-population-*.pt"))
    return shards[0] if shards else None


def test_read_one_v1_cache_shard():
    shard = _first_shard()
    if shard is None:
        pytest.skip(f"v1 shared cache not present under {_V1_CACHE_ROOT}; E-parity fixture absent")

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
