"""
Tests for the RewardModel performance knobs and length-bucketed batching.

These exercise the CPU-safe defaults of ``setup_torch_perf`` and
``auto_batch_size`` and prove that ``forward_with_cache_batch`` returns
activations in input order even when it reorders pairs internally for length
bucketing. They run on the package's own tiny CPU trunk
(``reward_lens.organisms._tiny.make_micro_trunk``), so no experiment tree and
no GPU are needed.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="module")
def tiny_rm():
    from reward_lens.organisms._tiny import make_micro_trunk

    return make_micro_trunk()


def test_setup_torch_perf_idempotent():
    from reward_lens.model import setup_torch_perf

    s1 = setup_torch_perf()
    s2 = setup_torch_perf()
    assert s1.get("matmul_precision") == s2.get("matmul_precision")


def test_auto_batch_size_cpu_default():
    """On a CPU-only environment ``auto_batch_size`` returns the safe default
    of 32 — never zero, never a number that would OOM."""
    from reward_lens.model import auto_batch_size

    bs = auto_batch_size(d_model=4096, n_layers=32, weight_gb=16.0)
    assert bs == 32  # CPU branch


def test_length_bucket_preserves_order(tiny_rm):
    """``length_bucket=True`` reorders pairs internally for batching but must
    return activations in input order so call sites' indexing stays correct."""
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
