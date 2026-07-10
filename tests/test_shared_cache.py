"""
Tests for the cross-experiment activation cache and the H200 batch knobs.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from experiments.utils.shared_cache import ActivationFloor, cached_forward


@pytest.fixture(scope="module")
def tiny_rm():
    from experiments.utils.tiny_model import make_tiny_reward_model

    return make_tiny_reward_model()


def test_setup_torch_perf_idempotent():
    from reward_lens.model import setup_torch_perf

    s1 = setup_torch_perf()
    s2 = setup_torch_perf()
    assert s1.get("matmul_precision") == s2.get("matmul_precision")


def test_auto_batch_size_cpu_default():
    """On a CPU-only environment ``auto_batch_size`` returns the safe
    default of 32 — never zero, never a number that would OOM."""
    from reward_lens.model import auto_batch_size

    bs = auto_batch_size(d_model=4096, n_layers=32, weight_gb=16.0)
    assert bs == 32  # CPU branch


def test_floor_disk_roundtrip_equivalent(tiny_rm):
    """The cached path must produce activations bit-equivalent (within
    fp16 precision) to the uncached path."""
    pairs = [("Hello", "World"), ("What is 2+2?", "Four"), ("Why?", "Because")]
    with tempfile.TemporaryDirectory() as td:
        floor_a = ActivationFloor(root=td, model_short="tiny", tag="t")
        c1 = floor_a.get_or_compute(tiny_rm, pairs, side="preferred", batch_size=2, max_length=64)
        # Fresh floor — forces a disk hit, not memory hit.
        floor_b = ActivationFloor(root=td, model_short="tiny", tag="t")
        c2 = floor_b.get_or_compute(tiny_rm, pairs, side="preferred", batch_size=2, max_length=64)

        assert sorted(c1.residual_streams.keys()) == sorted(c2.residual_streams.keys())
        for k in c1.residual_streams:
            a = c1.residual_streams[k].float()
            b = c2.residual_streams[k].float()
            assert torch.allclose(a, b, atol=1e-2), (
                f"layer {k} cache roundtrip drift: max diff {(a - b).abs().max()}"
            )


def test_floor_side_does_not_alias(tiny_rm):
    """Same prompts, different responses, must not share a cache file."""
    pairs_w = [("Q?", "A_chosen")]
    pairs_l = [("Q?", "A_rejected")]
    with tempfile.TemporaryDirectory() as td:
        floor = ActivationFloor(root=td, model_short="tiny", tag="t")
        # Computed for the side effect of writing distinct cache files.
        floor.get_or_compute(tiny_rm, pairs_w, side="preferred", batch_size=1, max_length=32)
        floor.get_or_compute(tiny_rm, pairs_l, side="dispreferred", batch_size=1, max_length=32)
        # Different responses → different rewards (probabilistically).
        # At minimum the disk paths are distinct.
        files = sorted(Path(td).rglob("*.pt"))
        assert len(files) == 2, f"expected 2 cache files, got {[f.name for f in files]}"


def test_cached_forward_no_root_fallback(tiny_rm):
    """When ``cfg.extra["shared_cache_root"]`` is unset, ``cached_forward``
    behaves as a plain forward (no caching). This is the path every
    runner takes when ``--shared-activation-cache`` is off."""

    class _Cfg:
        batch_size = 2
        max_length = 32
        extra: dict = {}

    cache = cached_forward(
        tiny_rm, [("hi", "hello")], side="preferred", cfg=_Cfg(), model_short="tiny"
    )
    assert cache.batch_size == 1
    assert -1 in cache.residual_streams


def test_length_bucket_preserves_order(tiny_rm):
    """``length_bucket=True`` reorders pairs internally for batching but
    must return activations in input order so call sites' indexing stays
    correct."""
    pairs = [
        ("longer prompt that takes more tokens", "even longer response with many words" * 3),
        ("short", "ok"),
        ("medium length here", "medium response text"),
    ]
    cache_a = tiny_rm.forward_with_cache_batch(
        pairs, batch_size=2, max_length=64, length_bucket=False
    )
    cache_b = tiny_rm.forward_with_cache_batch(
        pairs, batch_size=2, max_length=64, length_bucket=True
    )
    # Rewards must align by input position regardless of internal ordering.
    assert torch.allclose(cache_a.rewards, cache_b.rewards, atol=1e-3)
