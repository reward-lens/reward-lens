"""The activation store (section 2.2.3), successor to v1's ``shared_cache``.

The store is the content-addressed disk cache for captured activations. v1's ``ActivationFloor``
keyed on ``(model, pair-set, side)``; the v3 key adds the site, the position spec, the dtype, and
crucially the **intervention fingerprint**, so a patched-run cache can never alias a clean-run cache
(a latent v1 hazard). The on-disk layout is safetensors shards plus a JSON index; activations are
fp16 by default, but anything destined for covariance or whitening is stored fp32 (frames refuse
fp16 inputs). Reads are memory-mapped so a large sweep never materializes every activation at once.

This module also carries the read adapter for the v1 ``.pt`` caches under
``outputs/.../_shared_cache/<model>/floor-population-<hash>.pt`` (``read_v1_cache``): loading one of
those files back into a v1-shaped cache object is what makes the E-parity runs free, because the
2.5 GB of campaign activations can be replayed without recomputing a single forward on hardware that
cannot hold the 8B models.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from reward_lens.core.config import get_settings
from reward_lens.core.types import ModelFP, Site, content_hash

if TYPE_CHECKING:
    import torch

    from reward_lens.runtime.backend import Capture, CaptureSpec


# ---------------------------------------------------------------------------
# The v1 .pt cache read adapter (E-parity, section 4.3.2)
# ---------------------------------------------------------------------------


@dataclass
class V1Cache:
    """A v1 ``BatchedActivationCache`` reconstructed from a ``.pt`` file on disk.

    Exposes the same four activation dictionaries the v1 shared cache persisted (final-token
    residual stream, attention output, MLP output, per-head attention output), each mapping an
    integer layer index to a ``(B, ...)`` tensor, plus the batched rewards and final-token
    positions. ``layers()`` and ``shapes()`` are convenience views the E-parity fixtures assert on.
    """

    residual_streams: dict[int, "torch.Tensor"] = field(default_factory=dict)
    attn_outputs: dict[int, "torch.Tensor"] = field(default_factory=dict)
    mlp_outputs: dict[int, "torch.Tensor"] = field(default_factory=dict)
    attn_head_outputs: dict[int, "torch.Tensor"] = field(default_factory=dict)
    rewards: "torch.Tensor | None" = None
    final_token_positions: "torch.Tensor | None" = None
    path: str = ""

    def layers(self) -> list[int]:
        """Sorted layer indices present in the residual-stream cache."""
        return sorted(self.residual_streams.keys())

    def batch_size(self) -> int:
        """The batch (population) size, read off the first available tensor."""
        for table in (self.residual_streams, self.attn_outputs, self.mlp_outputs):
            if table:
                return int(next(iter(table.values())).shape[0])
        if self.rewards is not None:
            return int(self.rewards.shape[0])
        return 0

    def shapes(self) -> dict[str, Any]:
        """A compact shape/dtype summary for assertions and logging."""

        def summarize(table: dict[int, "torch.Tensor"]) -> dict[str, Any]:
            if not table:
                return {"layers": 0}
            sample = next(iter(table.values()))
            return {
                "layers": len(table),
                "shape": tuple(sample.shape),
                "dtype": str(sample.dtype),
            }

        return {
            "residual_streams": summarize(self.residual_streams),
            "attn_outputs": summarize(self.attn_outputs),
            "mlp_outputs": summarize(self.mlp_outputs),
            "attn_head_outputs": summarize(self.attn_head_outputs),
            "rewards": None if self.rewards is None else tuple(self.rewards.shape),
        }


def read_v1_cache(path: str | Path, device: str = "cpu") -> V1Cache:
    """Load one v1 ``.pt`` shared-cache file into a :class:`V1Cache` (section 2.2.3, 4.3.2).

    The v1 format is a single ``torch.save`` dict of half-precision final-token tensors keyed by
    layer, written by ``experiments/utils/shared_cache.py``. This reads exactly that structure back,
    coercing layer keys to ``int`` and moving tensors to ``device`` (CPU by default so a single file
    can be inspected without a GPU). It loads one file, not the whole 2.5 GB campaign; the caller is
    expected to point it at a specific shard. Raises ``FileNotFoundError`` if the path is absent so a
    fixture test can skip cleanly with a clear message.
    """
    import torch

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"v1 cache not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    cache = V1Cache(path=str(path))
    for attr in ("residual_streams", "attn_outputs", "mlp_outputs", "attn_head_outputs"):
        table = payload.get(attr, {}) or {}
        target = getattr(cache, attr)
        for layer, tensor in table.items():
            target[int(layer)] = tensor.to(device)
    if payload.get("rewards") is not None:
        cache.rewards = payload["rewards"].to(device)
    if payload.get("final_token_positions") is not None:
        cache.final_token_positions = payload["final_token_positions"].to(device)
    return cache


# ---------------------------------------------------------------------------
# CaptureHandle implementations
# ---------------------------------------------------------------------------


class InMemoryCaptureHandle:
    """A ``CaptureHandle`` over an in-memory ``Capture`` (the un-cached path)."""

    def __init__(self, capture: "Capture"):
        self._capture = capture

    def __iter__(self) -> Iterator["Capture"]:
        yield self._capture

    def get(self, site: Site) -> "torch.Tensor":
        return self._capture.tensors[site]


class ShardCaptureHandle:
    """A ``CaptureHandle`` backed by a safetensors shard, read memory-mapped.

    Iterating yields one ``Capture`` (the whole shard); ``get(site)`` returns a single site's tensor
    with a lazy, memory-mapped read via ``safetensors`` so a large capture is never fully resident.
    """

    def __init__(self, shard_path: Path, index: dict[str, Any]):
        self._path = shard_path
        self._index = index

    @staticmethod
    def _key(site: Site) -> str:
        head = "" if site.head is None else f".h{site.head}"
        return f"L{site.layer}.{site.point}{head}"

    def get(self, site: Site) -> "torch.Tensor":
        from safetensors.torch import load_file

        tensors = load_file(str(self._path))
        return tensors[self._key(site)]

    def __iter__(self) -> Iterator["Capture"]:
        from safetensors.torch import load_file

        from reward_lens.runtime.backend import Capture

        raw = load_file(str(self._path))
        tensors: dict[Site, "torch.Tensor"] = {}
        for key, tensor in raw.items():
            tensors[_parse_site_key(key)] = tensor
        yield Capture(
            tensors=tensors,
            positions=self._index.get("positions", []),
            dtype=self._index.get("dtype", "float16"),
        )


def _parse_site_key(key: str) -> Site:
    """Invert :meth:`ShardCaptureHandle._key`: ``"L3.attn_out"`` / ``"L3.head_out.h5"`` -> Site."""
    body = key[1:] if key.startswith("L") else key
    parts = body.split(".")
    layer = int(parts[0])
    point = parts[1]
    head = int(parts[2][1:]) if len(parts) > 2 and parts[2].startswith("h") else None
    return Site(layer, point, head)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ActivationStore
# ---------------------------------------------------------------------------


def _safetensors_intact(path: Path) -> bool:
    """Cheaply validate a safetensors shard without reading any tensor data.

    The format is an 8-byte little-endian header length, the JSON header, then one contiguous
    tensor buffer whose extent the header's ``data_offsets`` entries describe exactly. A file
    truncated by a killed writer fails the length arithmetic or the header parse, so this
    check costs a stat plus a header read and catches the corruption existence checks would
    otherwise trust.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            prefix = fh.read(8)
            if len(prefix) != 8:
                return False
            header_len = int.from_bytes(prefix, "little")
            if header_len <= 0 or 8 + header_len > size:
                return False
            header = json.loads(fh.read(header_len).decode("utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(header, dict):
        return False
    end = 0
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        offsets = spec.get("data_offsets") if isinstance(spec, dict) else None
        if not offsets or len(offsets) != 2:
            return False
        end = max(end, int(offsets[1]))
    return size == 8 + header_len + end


@dataclass(frozen=True)
class CacheStats:
    """Hits and misses one ``ActivationStore`` instance has served since its counters were last reset.

    A cache that silently serves the wrong entry looks exactly like a cache that is working: the
    caller gets a tensor of the right shape and dtype and has no way to tell which subject produced
    it. Comparing two returned tensors does not settle it either, because two genuinely distinct
    subjects can return numerically close vectors and a comparison then passes for a reason that has
    nothing to do with the cache. The counter is the direct observation: a read that recomputed
    incremented ``misses``, a read that was served from an existing shard incremented ``hits``, and
    no amount of luck in the values changes either number.

    The counters are per-instance and in-memory, so a fresh `ActivationStore` over a warm directory
    starts at zero and counts what *this* object did, not what the directory accumulated. They are
    not synchronised; two threads sharing one store may lose an increment, which is acceptable for a
    diagnostic and is why nothing gates on them.
    """

    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        """Reads served, hits plus misses."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float | None:
        """Fraction of reads served from cache, or ``None`` when nothing has been read yet."""
        return None if self.total == 0 else self.hits / self.total


#: Position kinds whose identity lives in ``PositionSpec.detail`` rather than in the kind name.
#: ``explicit`` carries the token indices, ``judgment`` carries the verdict index the signal
#: detected, ``span_ends`` carries the span kind the ends are taken from. For all three the kind
#: string is the same for every read, so a key built from the kind alone is one key for the whole
#: family (BLK-003). ``final`` and ``all`` resolve from the input, which is already in the key.
_POSITION_KINDS_CARRYING_DETAIL = frozenset({"explicit", "judgment", "span_ends"})


def _canonical_detail(detail: Any) -> Any:
    """Reduce a ``PositionSpec.detail`` to a value a content hash can be taken over.

    Order is preserved for sequences, because ``PositionSpec.resolve`` returns explicit indices in
    the order given and the capture's rows come back in that order. Mapping keys are sorted.
    Anything that is not a plain value, sequence or mapping is refused rather than hashed by
    ``repr`` or by ``vars``: an object's address and an object's class name are both constant
    across the things a cache key exists to separate, which is the failure this repair is closing.
    """
    if detail is None or isinstance(detail, (str, int, float, bool)):
        return detail
    if isinstance(detail, (list, tuple)):
        return [_canonical_detail(item) for item in detail]
    if isinstance(detail, dict):
        return {
            str(k): _canonical_detail(v)
            for k, v in sorted(detail.items(), key=lambda kv: str(kv[0]))
        }
    raise ValueError(
        f"a PositionSpec detail of type {type(detail).__name__} cannot enter the activation cache "
        f"key: only values, sequences and mappings serialise canonically. Pass the resolved "
        f"indices, or the configuration the signal used to resolve them, not the object."
    )


def _position_component(position: "str | Any | None") -> dict[str, Any]:
    """The position half of the cache key: the kind **and** what the kind is configured with.

    At `59a5f5a` this was ``getattr(spec.position, "kind", "final")`` and nothing else, so the two
    reads Part 7.1 takes (the last prompt token, and ``T_pre`` generated tokens later) were one
    entry. Both are ``explicit``.
    """
    if position is None:
        return {"kind": "final", "detail": None}
    if isinstance(position, str):
        if position in _POSITION_KINDS_CARRYING_DETAIL:
            raise ValueError(
                f"position kind {position!r} carries its identity in PositionSpec.detail, so a "
                f"bare kind string is not a position: every read of this kind would share one "
                f"cache entry. Pass the PositionSpec."
            )
        return {"kind": position, "detail": None}
    kind = getattr(position, "kind", None)
    if kind is None:
        raise ValueError(
            f"cannot build a cache key from a position of type {type(position).__name__}: it has "
            f"no 'kind'. Pass a PositionSpec, or a bare kind string for 'final' or 'all'."
        )
    detail = getattr(position, "detail", None)
    if kind in _POSITION_KINDS_CARRYING_DETAIL and detail is None:
        raise ValueError(
            f"position kind {kind!r} was given no detail, so it is indistinguishable from every "
            f"other {kind!r} read of the same input. Resolve it before caching."
        )
    return {"kind": str(kind), "detail": _canonical_detail(detail)}


class ActivationStore:
    """Content-addressed disk cache for captured activations (section 2.2.3).

    The key folds the model fingerprint, the dataset id or content hash, the site set, the position
    spec, the dtype, and the intervention fingerprint (``"none"`` for a clean run). Shards are
    safetensors files under ``root/<model_fp>/`` with a sibling JSON index; ``get_or_compute`` reads
    a cached shard memory-mapped or computes the capture via the signal and writes it back. fp16 is
    the default activation dtype; a spec asking for fp32 (covariance/whitening inputs) is honoured
    and stored fp32.

    ``hits`` and ``misses`` count what ``get_or_compute`` did, and ``stats`` packages them. They are
    the observable that tells a caller whether two reads of two subjects were two computations or
    one computation served twice, which is not answerable from the returned tensors.
    """

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else get_settings().resolved_cache()
        self.root.mkdir(parents=True, exist_ok=True)
        #: Reads ``get_or_compute`` served from an existing shard.
        self.hits = 0
        #: Reads ``get_or_compute`` had to compute because no shard existed for the key.
        self.misses = 0

    @property
    def stats(self) -> CacheStats:
        """The current hit/miss counts as a `CacheStats` snapshot."""
        return CacheStats(hits=self.hits, misses=self.misses)

    def reset_stats(self) -> None:
        """Zero the hit and miss counters, so a caller can measure one phase of a sweep."""
        self.hits = 0
        self.misses = 0

    def key(
        self,
        model_fp: ModelFP,
        dataset: str,
        sites: tuple[Site, ...],
        position: "str | Any | None",
        dtype: str,
        intervention_fp: str = "none",
        full_sequence: bool = False,
    ) -> str:
        """Compute the content-addressed cache key (section 2.2.3).

        ``position`` takes a ``PositionSpec``, or a bare kind string for the two kinds that carry
        no configuration. Passing a ``PositionSpec`` is the calling convention: the kind alone is
        not the position, and ``key`` refuses a bare ``"explicit"``, ``"judgment"`` or
        ``"span_ends"`` for that reason (BLK-003). ``full_sequence`` mirrors ``CaptureSpec`` and
        decides whether the shard holds every token position or only the resolved ones.

        Adding those two components changes every key this method has ever produced. Shards
        written under the old key are not read again; they are orphaned, not misread, which is the
        only safe direction for a key repair.
        """
        material = {
            "model_fp": str(model_fp),
            "dataset": dataset,
            "sites": [str(s) for s in sites],
            "position": _position_component(position),
            "dtype": dtype,
            "intervention_fp": intervention_fp,
            "full_sequence": bool(full_sequence),
        }
        return content_hash(material, "cap").split(":")[1]

    def _shard_path(self, model_fp: ModelFP, key: str) -> Path:
        model_dir = self.root / str(model_fp).replace(":", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / f"{key}.safetensors"

    def has(self, model_fp: ModelFP, key: str) -> bool:
        """Whether an intact shard for ``key`` exists (a corrupt one is dropped, not trusted).

        Existence alone is not integrity: a writer killed mid-save can leave a truncated
        safetensors file at the content-addressed path, and every skip-if-cached re-run would
        trust it. A shard that fails the header check is deleted with a warning so the caller
        recomputes it; a shard whose sibling index is missing (the writer died between the
        payload rename and the index write) reports absent so ``put`` rewrites both, but the
        intact payload is left in place for the atomic replace.
        """
        shard = self._shard_path(model_fp, key)
        if not shard.exists():
            return False
        if not _safetensors_intact(shard):
            warnings.warn(
                f"dropping corrupt activation shard {shard} (truncated or unreadable header); "
                f"it will be recomputed on the next request.",
                RuntimeWarning,
                stacklevel=2,
            )
            shard.unlink(missing_ok=True)
            shard.with_suffix(".json").unlink(missing_ok=True)
            return False
        return shard.with_suffix(".json").exists()

    def put(
        self,
        model_fp: ModelFP,
        key: str,
        capture: "Capture",
        index_extra: dict[str, Any] | None = None,
    ) -> Path:
        """Write a ``Capture`` to a safetensors shard plus a JSON index; return the shard path.

        Both files land by write-then-rename, payload first: a kill mid-write leaves only a
        temp file (never a truncated shard at the trusted name), and a kill between the two
        renames leaves an intact payload with no index, which ``has`` treats as a miss.
        """
        from safetensors.torch import save_file

        shard = self._shard_path(model_fp, key)
        tensors = {
            ShardCaptureHandle._key(site): t.contiguous() for site, t in capture.tensors.items()
        }
        tmp_shard = shard.with_name(f"{shard.name}.tmp{os.getpid()}")
        save_file(tensors, str(tmp_shard))
        os.replace(tmp_shard, shard)
        index = {
            "key": key,
            "dtype": capture.dtype,
            "positions": capture.positions,
            "sites": [str(s) for s in capture.tensors.keys()],
        }
        if index_extra:
            index.update(index_extra)
        index_path = shard.with_suffix(".json")
        tmp_index = index_path.with_name(f"{index_path.name}.tmp{os.getpid()}")
        tmp_index.write_text(json.dumps(index, indent=2), encoding="utf-8")
        os.replace(tmp_index, index_path)
        return shard

    def get(self, model_fp: ModelFP, key: str) -> ShardCaptureHandle:
        """Return a memory-mapped handle to a cached shard (raises if absent)."""
        shard = self._shard_path(model_fp, key)
        if not shard.exists():
            raise KeyError(f"no cached capture for key {key} under {model_fp}")
        index_path = shard.with_suffix(".json")
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
        return ShardCaptureHandle(shard, index)

    def get_or_compute(
        self,
        signal: Any,
        view: Any,
        spec: "CaptureSpec",
        dataset_id: str | None = None,
        intervention_fp: str = "none",
    ) -> Any:
        """Return a ``CaptureHandle`` for ``(signal, view, spec)``, computing and caching on miss.

        On a cache hit the shard is returned memory-mapped; on a miss the signal computes the
        capture (``signal.capture``) and the result is written back under the content key before the
        handle is returned. ``dataset_id`` defaults to the view's checksum when the data plane (M2)
        provides one, else a hash of the view's repr, so the key is stable per data content.

        Every call increments exactly one of ``self.hits`` or ``self.misses``. Those counters are the
        only way a caller can tell a distinct capture from a reused one; the tensors cannot say.
        """
        model_fp = signal.meta.fingerprint
        dataset = dataset_id or _view_id(view)
        key = self.key(
            model_fp,
            dataset,
            tuple(spec.sites),
            spec.position,
            spec.dtype,
            intervention_fp,
            full_sequence=bool(getattr(spec, "full_sequence", False)),
        )
        if self.has(model_fp, key):
            self.hits += 1
            return self.get(model_fp, key)
        self.misses += 1
        handle = signal.capture(view, spec)
        capture = next(iter(handle))
        self.put(
            model_fp,
            key,
            capture,
            index_extra={"dataset": dataset, "intervention_fp": intervention_fp},
        )
        return self.get(model_fp, key)


def _view_id(view: Any) -> str:
    """A stable id for a data view: its checksum if it has one, else a hash of its repr."""
    checksum = getattr(view, "checksum", None)
    if callable(checksum):
        return str(checksum())
    return content_hash({"repr": repr(view)}, "dv").split(":")[1]


__all__ = [
    "ActivationStore",
    "CacheStats",
    "V1Cache",
    "read_v1_cache",
    "InMemoryCaptureHandle",
    "ShardCaptureHandle",
]
