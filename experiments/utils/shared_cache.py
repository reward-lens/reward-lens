"""
Shared per-(model, pair-set) activation cache.

The deep_analysisv1 GPU audit found that e02 (lens), e03 (attribution),
e04 (faithfulness), e07 (cascade), and e09 (conflict) all call
``rm.forward_with_cache_batch`` on the *same* set of preferred /
dispreferred pairs, then differ only in how they read out the cached
tensors. With three working models × five experiments × two forwards
each, we run 30 redundant full population forwards per campaign — which
on H200 is ~30 × 30 s ≈ 15 minutes of pure waste, or ~1 hour on the
27B Gemma.

This module gives the runners an opt-in shared cache:

  - The first runner that needs activations computes them via
    :class:`ActivationFloor` (one batched forward per (model, side)),
    persists final-token residual / attn / mlp tensors to disk, and
    returns them in memory.
  - Every later runner in the same campaign loads from disk in <1 s
    instead of paying for another forward.

The on-disk format is a single ``.pt`` file per (model_short, pair_set,
side) holding a dict of ``int_layer -> torch.HalfTensor`` (final-token
only, so size scales as ``B * d_model * n_layers * 2 bytes`` ≈ 100 MB
per model for a 360-pair population on a 32-layer 8B Llama).

Usage
-----
    from experiments.utils.shared_cache import ActivationFloor

    floor = ActivationFloor(cfg.extra.get("shared_cache_root"),
                            model_short=mc.short_name(),
                            tag="population")
    cache_w = floor.get_or_compute(
        rm, [(p.prompt, p.preferred) for p in pairs], side="preferred",
        batch_size=cfg.batch_size, max_length=cfg.max_length,
        capture_heads=False,
    )
    # cache_w is a BatchedActivationCache (in-memory representation
    # identical to what forward_with_cache_batch returns directly).

If ``shared_cache_root`` is None or the cache root doesn't exist yet,
the floor degrades gracefully to a simple in-memory cache scoped to a
single ``ActivationFloor`` instance.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import torch

from reward_lens.model import BatchedActivationCache


def _hash_pair_set(pairs: list[tuple[str, str]], side: str, capture_heads: bool) -> str:
    """Stable hash of a pair-set so reusing across runs is safe.

    Includes ``side`` and ``capture_heads`` so a "preferred" cache and a
    "dispreferred" cache for the same prompts don't collide, and so a
    head-capture cache is never silently aliased to a non-head one.
    """
    h = hashlib.sha256()
    h.update(side.encode())
    h.update(b"|heads=")
    h.update(b"1" if capture_heads else b"0")
    for prompt, response in pairs:
        h.update(b"|")
        h.update(prompt.encode("utf-8")[:512])
        h.update(b"||")
        h.update(response.encode("utf-8")[:512])
    return h.hexdigest()[:16]


def _save_cache(cache: BatchedActivationCache, path: Path) -> None:
    """Persist a BatchedActivationCache to a single .pt file.

    We deliberately downcast to half-precision before saving — every
    downstream consumer (lens projection, attribution dot product, conflict
    cosine) re-promotes to float for math. Saves ~2x disk and IO.
    """
    payload = {
        "residual_streams": {k: v.detach().cpu().half() for k, v in cache.residual_streams.items()},
        "attn_outputs":     {k: v.detach().cpu().half() for k, v in cache.attn_outputs.items()},
        "mlp_outputs":      {k: v.detach().cpu().half() for k, v in cache.mlp_outputs.items()},
        "attn_head_outputs": {k: v.detach().cpu().half() for k, v in cache.attn_head_outputs.items()},
        "rewards": cache.rewards.detach().cpu().float() if cache.rewards is not None else None,
        "final_token_positions":
            cache.final_token_positions.detach().cpu().to(torch.int32)
            if cache.final_token_positions is not None else None,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_cache(path: Path, device: torch.device) -> BatchedActivationCache:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    cache = BatchedActivationCache()
    for layer, t in payload["residual_streams"].items():
        cache.residual_streams[int(layer)] = t.to(device)
    for layer, t in payload["attn_outputs"].items():
        cache.attn_outputs[int(layer)] = t.to(device)
    for layer, t in payload["mlp_outputs"].items():
        cache.mlp_outputs[int(layer)] = t.to(device)
    for layer, t in payload["attn_head_outputs"].items():
        cache.attn_head_outputs[int(layer)] = t.to(device)
    if payload.get("rewards") is not None:
        cache.rewards = payload["rewards"].to(device)
    if payload.get("final_token_positions") is not None:
        cache.final_token_positions = payload["final_token_positions"].to(device)
    return cache


class ActivationFloor:
    """Per-campaign shared cache for ``forward_with_cache_batch`` outputs.

    Args:
        root: Disk root (typically ``run_root/_shared_cache``). When
            ``None``, all caching is in-memory and scoped to this
            instance — useful for one-shot scripts where there's no
            run-root to attach to.
        model_short: Short name of the model (e.g. "Skywork-Reward-Llama-3.1-8B-v0.2");
            used as a directory name to keep models from colliding.
        tag: Optional human-friendly tag for the pair-set (e.g.
            "population", "patching_subset"). Cached together with the
            content hash so distinct pair-sets don't share cache files.
    """

    def __init__(self, root: Optional[str], model_short: str, tag: str = ""):
        self.root = Path(root) / model_short if root is not None else None
        self.tag = tag
        self._mem: dict[str, BatchedActivationCache] = {}

    # -- public API --------------------------------------------------------

    def cache_path(self, pair_hash: str) -> Optional[Path]:
        if self.root is None:
            return None
        suffix = f"-{self.tag}" if self.tag else ""
        return self.root / f"floor{suffix}-{pair_hash}.pt"

    def get_or_compute(
        self,
        rm,
        pairs: list[tuple[str, str]],
        *,
        side: str,
        batch_size: int,
        max_length: int = 2048,
        capture_heads: bool = False,
        length_bucket: bool = False,
    ) -> BatchedActivationCache:
        """Return a BatchedActivationCache for ``pairs`` on ``rm``.

        First checks the in-memory cache, then disk, then runs the
        forward pass (and writes back to disk if a root was set).
        """
        h = _hash_pair_set(pairs, side, capture_heads)
        if h in self._mem:
            return self._mem[h]
        path = self.cache_path(h)
        if path is not None and path.exists():
            try:
                cache = _load_cache(path, rm.device)
                self._mem[h] = cache
                return cache
            except Exception:
                # Corrupt cache — recompute and overwrite.
                pass
        cache = rm.forward_with_cache_batch(
            pairs,
            batch_size=batch_size,
            max_length=max_length,
            capture_heads=capture_heads,
            length_bucket=length_bucket,
            progress=False,
        )
        if path is not None:
            try:
                _save_cache(cache, path)
            except Exception:
                # Best-effort cache write; never block the experiment.
                pass
        self._mem[h] = cache
        return cache

    @staticmethod
    def for_cfg(cfg, model_short: str, tag: str = "") -> "ActivationFloor":
        """Build a floor from an :class:`ExperimentConfig`. Reads
        ``cfg.extra["shared_cache_root"]`` — when unset returns an
        in-memory-only floor, so call sites can opt in transparently
        without branching."""
        root = cfg.extra.get("shared_cache_root") if hasattr(cfg, "extra") else None
        return ActivationFloor(root=root, model_short=model_short, tag=tag)

    def evict_disk(self) -> int:
        """Delete the on-disk cache for this floor's model. Returns the
        number of files removed. Used by the orchestrator after a
        successful run to reclaim space."""
        if self.root is None or not self.root.exists():
            return 0
        n = 0
        for p in self.root.glob("floor*.pt"):
            try:
                p.unlink()
                n += 1
            except Exception:
                pass
        return n


def cached_forward(
    rm,
    pairs: list[tuple[str, str]],
    *,
    side: str,
    cfg,
    model_short: str,
    tag: str = "population",
    capture_heads: bool = False,
) -> BatchedActivationCache:
    """Drop-in replacement for ``rm.forward_with_cache_batch`` that uses
    the shared activation floor when ``cfg.extra["shared_cache_root"]``
    is set, and behaves identically (no caching, plain forward) otherwise.

    Args:
        rm: A :class:`reward_lens.model.RewardModel`.
        pairs: List of (prompt, response) tuples.
        side: ``"preferred"`` or ``"dispreferred"`` — folded into the
            cache key so the two sides don't alias.
        cfg: The experiment config; we read ``cfg.batch_size``,
            ``cfg.max_length``, and ``cfg.extra``.
        model_short: Stable per-model directory name.
        tag: Pair-set label; share between experiments that use the
            same population (default ``"population"``).
        capture_heads: Whether to also capture per-head attn outputs.

    Returns:
        :class:`BatchedActivationCache` in the same shape as
        ``rm.forward_with_cache_batch`` would return.
    """
    floor = ActivationFloor.for_cfg(cfg, model_short=model_short, tag=tag)
    length_bucket = bool(cfg.extra.get("length_bucket", False)) if hasattr(cfg, "extra") else False
    return floor.get_or_compute(
        rm, pairs, side=side,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        capture_heads=capture_heads,
        length_bucket=length_bucket,
    )
