"""Model fingerprints and lineage (section 2.2.5).

``fingerprint`` produces the ``ModelFP`` that identifies a signal everywhere in the store: it is
the ``mfp:`` prefix on every ``SubjectRef``, the key that stops a patched-run cache from aliasing a
clean-run cache, and the axis the Atlas, kinship, and monoculture measurements are computed over.
It costs almost nothing to collect at load and is impossible to reconstruct afterwards, which is
why it is mandatory rather than optional (RK9).

The digest hashes seven things: the base weight content, the parameter manifest, the injected
adapter weights, the adapter's activation state, the normalized config JSON, the tokenizer identity,
and the caller's adapter id. Weight content is streamed, never materialized whole: for a model whose
safetensors live on disk (the 8B campaign case) the files are hashed block by block off disk; for a
tiny in-memory model with no files, the ``state_dict`` is serialized one tensor at a time. Both paths
use ``reward_lens.core.hash_bytes``/``content_hash`` so the id format matches the rest of the kernel.

The four components after the base weight content exist because the disk hash answers a narrower
question than it appears to. It hashes the *directory the config names*, and under a parameter
efficient fine-tune that directory holds the **base** model: the LoRA tensors live in a separate
adapter checkpoint, or only in memory, and hashing the base directory returns the same digest for
every adapter ever loaded on top of it. A run whose subjects are forty adapters over one base then
has one fingerprint for forty distinct models, and the activation store, which is keyed on the
fingerprint, serves the first capture to all of them. That is the failure this module was carrying.

So the identity is decomposed rather than widened by guesswork. Enumerated, the dimensions that
change what a forward pass produces and that a fingerprint must therefore separate:

    base weight bytes         the checkpoint on disk, or the state dict when there are no files
    parameter manifest        every parameter's name, dtype and shape; catches a wrapper being
                              present at all, a changed LoRA rank, a resized head, and a load at a
                              different dtype, none of which move the bytes in the base directory
    adapter weight bytes      the injected tensors themselves, hashed exactly, which is the only
                              thing separating two adapters of identical shape over one base
    adapter activation state  which adapters are active, whether they are disabled, and whether they
                              have been merged. A reference pass taken with the adapter disabled is
                              a different model from the same object with it enabled, and nothing in
                              the weights or the manifest records the difference
    config, tokenizer         as before
    caller's adapter id       the checkpoint path or spec source the caller resolved

Two things are deliberately **not** in the identity. Device is a property of the machine, not of the
model, and including it would defeat the point of a content-derived id. And a base parameter mutated
in memory after a load from disk is not detected on the disk fast path: the files still hash to what
they held. Closing that would mean hashing every live parameter on every fingerprint call, which is
the cost the disk path exists to avoid. Callers that mutate base weights (interventions, planted
heads) work on models with no resolvable directory, so they take the state-dict path and are hashed
correctly; this is recorded so the limit is known rather than assumed away.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from reward_lens.core.types import ModelFP, content_hash, hash_bytes

if TYPE_CHECKING:
    import torch

# Config keys that vary run to run or machine to machine and must not enter the identity. Dropping
# them is what lets two loads of the same checkpoint on different boxes fingerprint identically.
_VOLATILE_CONFIG_KEYS = frozenset(
    {
        "_name_or_path",
        "transformers_version",
        "torch_dtype",
        "use_cache",
        "_attn_implementation",
        "device_map",
        "name_or_path",
    }
)

# Adapter-config keys that name a path or a machine and must not enter the identity, on top of the
# model-config ones. ``base_model_name_or_path`` is where the base was found on the box that trained
# the adapter, which says nothing about the adapter; the base itself is already hashed as weights.
_VOLATILE_ADAPTER_CONFIG_KEYS = _VOLATILE_CONFIG_KEYS | frozenset(
    {"base_model_name_or_path", "runtime_config", "revision"}
)

# Read weight files in 8 MiB blocks so an 8B checkpoint never lands in RAM to be hashed.
_STREAM_BLOCK = 8 * 1024 * 1024

# Substrings that mark a state-dict key as belonging to an injected adapter rather than to the base
# checkpoint. This is the fallback rule; the primary rule reads PEFT's own ``adapter_layer_names``
# contract off the module tree, which is exact. The fallback exists because a tuner PEFT adds later,
# or an adapter somebody hand-rolled, would otherwise be classified as base and hashed only through
# a directory that does not contain it. A false positive here costs a few extra bytes of hashing; a
# false negative costs a collision between two checkpoints, so the rule errs wide on purpose.
_ADAPTER_KEY_MARKERS = (
    "lora_a",
    "lora_b",
    "lora_embedding",
    "lora_magnitude",
    "modules_to_save",
    "ia3_l",
    "loha_",
    "lokr_",
    "hada_",
    "oft_",
    "boft_",
    "poly_",
    "vera_",
    "ln_tuning",
    "prompt_encoder",
    "prefix_encoder",
    "adaption_prompt",
    "adapter",
)


def _hash_weights_from_disk(local_dir: Path) -> str | None:
    """Stream-hash the safetensors shards under ``local_dir`` in sorted order.

    Returns a hex digest, or ``None`` if the directory holds no safetensors (the caller then falls
    back to the state-dict path). Shards are read in filename order and in fixed-size blocks, so the
    peak memory is one block regardless of checkpoint size. This is the 8B path; it reads the full
    weight bytes off disk (gigabytes) but holds only a block at a time.
    """
    shards = sorted(local_dir.glob("*.safetensors"))
    if not shards:
        return None
    h = hashlib.blake2b(digest_size=16)
    for shard in shards:
        h.update(shard.name.encode("utf-8"))
        with shard.open("rb") as fh:
            while True:
                block = fh.read(_STREAM_BLOCK)
                if not block:
                    break
                h.update(block)
    return h.hexdigest()


def _hash_weights_from_state_dict(
    model: "torch.nn.Module", names: "frozenset[str] | None" = None
) -> str:
    """Serialize and hash a model's ``state_dict`` one tensor at a time (the in-memory path).

    Used for the tiny synthetic models that have no files on disk, and for the adapter tensors of a
    model whose base weights were hashed off disk. Each parameter is moved to CPU, made contiguous,
    and serialized with safetensors (which handles bf16/fp16 losslessly, unlike ``numpy().tobytes()``),
    then folded into a running digest along with its name, dtype, and shape. Serializing per tensor
    rather than the whole dict keeps the transient allocation to one tensor.

    ``names`` restricts the hash to a subset of state-dict keys. That is what makes hashing an
    adapter cheap: a rank-32 LoRA over an 8B base is tens of megabytes, so hashing exactly those
    tensors and leaving the base to the streamed disk read costs almost nothing and is what
    separates two checkpoints that share a base directory.
    """
    import safetensors.torch as st
    import torch

    h = hashlib.blake2b(digest_size=16)
    sd = model.state_dict()
    keys = sorted(sd.keys()) if names is None else sorted(k for k in sd.keys() if k in names)
    for name in keys:
        tensor = sd[name]
        if not hasattr(tensor, "detach"):
            continue
        cpu = tensor.detach().to("cpu").contiguous()
        header = f"{name}|{cpu.dtype}|{tuple(cpu.shape)}".encode("utf-8")
        h.update(header)
        # safetensors requires at least one tensor; save this one to a byte buffer and fold it in.
        try:
            h.update(st.save({"t": cpu}))
        except (ValueError, RuntimeError):
            # A rare dtype safetensors will not serialize (e.g. an integer buffer view); fall back
            # to the raw storage bytes, which are still deterministic for identity purposes.
            h.update(bytes(cpu.flatten().to("cpu").view(torch.uint8).numpy().tobytes()))
    return h.hexdigest()


def _adapter_param_names(model: "torch.nn.Module") -> "frozenset[str]":
    """The state-dict keys that belong to an injected adapter rather than to the base checkpoint.

    Two rules, unioned. The first reads PEFT's own contract off the module tree: every tuner layer
    class declares ``adapter_layer_names``, the attributes holding its per-adapter weights, so a
    module at path ``P`` declaring ``("lora_A", "lora_B")`` contributes every state-dict key under
    ``P.lora_A.`` and ``P.lora_B.``. That is exact and needs no import of ``peft``, which is not a
    dependency of this library. ``ModulesToSaveWrapper`` declares ``("modules_to_save",)`` by the same
    contract, which is how a reward head retrained alongside the adapter is caught.

    The second rule matches ``_ADAPTER_KEY_MARKERS`` against the key text, and exists so an adapter
    this library has never seen still lands in the adapter bucket instead of silently being treated
    as base weights that the directory hash already covers. It does not.

    Returns an empty set for a plain model, which is the signal that there is no adapter component.
    """
    names: set[str] = set()
    try:
        state_keys = list(model.state_dict().keys())
    except Exception:  # pragma: no cover - a model with no usable state dict has no adapter to find
        return frozenset()

    prefixes: list[str] = []
    try:
        for module_name, module in model.named_modules():
            layer_names = getattr(module, "adapter_layer_names", None)
            if not layer_names or isinstance(layer_names, str):
                continue
            head = f"{module_name}." if module_name else ""
            prefixes.extend(f"{head}{attr}." for attr in layer_names)
    except Exception:  # pragma: no cover - fall through to the marker rule
        prefixes = []

    for key in state_keys:
        if any(key.startswith(prefix) for prefix in prefixes):
            names.add(key)
            continue
        lowered = key.lower()
        if any(marker in lowered for marker in _ADAPTER_KEY_MARKERS):
            names.add(key)
    return frozenset(names)


def _hash_manifest(model: "torch.nn.Module") -> str:
    """Hash every parameter's name, dtype and shape, without touching a single tensor's bytes.

    The manifest is what makes the disk fast path honest about the model in hand rather than about
    the directory it was loaded from. A LoRA wrapper renames ``q_proj.weight`` to
    ``q_proj.base_layer.weight`` and adds two tensors; a different rank changes their shapes; a load
    at bf16 instead of fp32 changes every dtype. None of that moves a byte in the base directory, so
    without this the disk path reports all of them as the same model. It also removes an internal
    disagreement: the state-dict path has always folded name, dtype and shape into its digest, so
    before this the two paths did not agree on what identity meant.

    Reading names, dtypes and shapes off a state dict is metadata only. It does not copy, move or
    materialize a tensor, so this is free even for an 8B model.
    """
    try:
        sd = model.state_dict()
    except Exception:  # pragma: no cover - nothing to describe
        return "no-manifest"
    rows = []
    for name in sorted(sd.keys()):
        tensor = sd[name]
        dtype = getattr(tensor, "dtype", None)
        shape = getattr(tensor, "shape", None)
        rows.append(f"{name}|{dtype}|{tuple(shape) if shape is not None else ()}")
    blob = "\n".join(rows).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


def _hash_adapter_state(model: "torch.nn.Module") -> str:
    """Hash which adapters are active, disabled or merged, plus their configs.

    Weights and manifest between them cannot express the difference between a model with its adapter
    enabled and the same object with it disabled: the tensors are identical and only a boolean on
    each tuner layer differs. That boolean is a real read in a GRPO run, where the KL reference is
    the base model with the adapter disabled, and it is a different model from the policy for every
    purpose an activation capture serves. The same holds for which of several loaded adapters is
    active and for whether an adapter has been merged into the base.

    Everything here is read by duck typing against PEFT's public attribute names, so no dependency is
    introduced and a model that has no adapter machinery returns the same constant it always would.
    Path-valued and machine-valued config keys are dropped for the same reason the model config drops
    them: they differ between boxes that hold the same checkpoint.
    """
    parts: dict[str, Any] = {}

    try:
        peft_config = getattr(model, "peft_config", None)
    except Exception:  # pragma: no cover - a refusing property is not state we have
        peft_config = None
    if isinstance(peft_config, dict) and peft_config:
        configs: dict[str, Any] = {}
        for adapter_name in sorted(str(k) for k in peft_config):
            raw_config = peft_config[adapter_name]
            try:
                raw = raw_config.to_dict()
            except (AttributeError, TypeError):
                raw = {k: v for k, v in vars(raw_config).items() if not k.startswith("_")}
            configs[adapter_name] = {
                k: v for k, v in sorted(raw.items()) if k not in _VOLATILE_ADAPTER_CONFIG_KEYS
            }
        parts["configs"] = configs

    disabled: set[bool] = set()
    merged: set[str] = set()
    active: set[str] = set()
    try:
        modules = list(model.named_modules())
    except Exception:  # pragma: no cover - nothing to walk
        modules = []
    for _, module in modules:
        # Every read here is a property on someone else's class, so every read is guarded. A
        # fingerprint that raises on an exotic model is worse than one that records less about it.
        try:
            if not getattr(module, "adapter_layer_names", None):
                continue
            flag = getattr(module, "disable_adapters", None)
            if flag is not None:
                disabled.add(bool(flag))
            for attr, sink in (("merged_adapters", merged), ("active_adapters", active)):
                value = getattr(module, attr, None)
                if value:
                    sink.update(str(v) for v in value)
        except Exception:  # pragma: no cover - a property that refuses is not state we have
            continue
    if disabled or merged or active:
        parts["disabled"] = sorted(disabled)
        parts["merged"] = sorted(merged)
        parts["active"] = sorted(active)

    if not parts:
        return "no-adapter"
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()


def _resolve_local_dir(model: "torch.nn.Module") -> Path | None:
    """Best-effort resolution of a local weights directory from the model config.

    Note what this resolves to under a parameter efficient fine-tune: ``config`` on a wrapped model
    is the **base** model's config, so ``_name_or_path`` is the base directory and the adapter is
    nowhere in it. That is correct for the base weight component and wrong for anything that treats
    the returned digest as the whole identity, which is why `fingerprint` hashes the adapter tensors
    and the adapter state separately rather than trusting this alone.
    """
    cfg = getattr(model, "config", None)
    for attr in ("_name_or_path", "name_or_path"):
        candidate = getattr(cfg, attr, None) if cfg is not None else None
        if candidate:
            path = Path(str(candidate))
            if path.is_dir():
                return path
    return None


def _hash_config(model: "torch.nn.Module") -> str:
    """Hash the model config as normalized JSON, dropping volatile keys."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return "no-config"
    try:
        raw = cfg.to_dict()
    except (AttributeError, TypeError):
        raw = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    normalized = {k: v for k, v in raw.items() if k not in _VOLATILE_CONFIG_KEYS}
    # json with a string fallback for anything not natively serializable (e.g. nested config
    # objects) so the hash is stable and never raises on an exotic config value.
    blob = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()


def _hash_tokenizer(tokenizer: Any) -> str:
    """Hash the tokenizer identity: class, vocab size, specials, and chat template.

    Full tokenizer-file hashing is available when the files are on disk, but the identity that
    actually matters for a reward signal is the vocabulary and the template that turns a pair into
    tokens; those are what change a score. Hashing them (plus the class name) is stable and cheap.
    """
    if tokenizer is None:
        return "no-tokenizer"
    parts: dict[str, Any] = {"class": type(tokenizer).__name__}
    for attr in ("vocab_size", "name_or_path", "padding_side"):
        parts[attr] = getattr(tokenizer, attr, None)
    specials = getattr(tokenizer, "all_special_tokens", None)
    if specials is not None:
        parts["special_tokens"] = sorted(str(t) for t in specials)
    template = getattr(tokenizer, "chat_template", None)
    parts["chat_template"] = template if isinstance(template, str) else None
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()


def fingerprint(model: "torch.nn.Module", tokenizer: Any = None, adapter_id: str = "") -> ModelFP:
    """Compute the content-derived ``ModelFP`` for a loaded model (section 2.2.5).

    The digest folds the base weight-content hash (streamed from disk when the safetensors are
    present, else serialized from the ``state_dict``), the parameter manifest, the injected adapter
    weights, the adapter's activation state, the normalized config hash, the tokenizer-identity hash,
    and the caller's adapter id into one ``mfp:`` id via ``content_hash``. Two loads of the same
    checkpoint at the same dtype, with the same tokenizer and the same adapter id, produce the same
    id on any machine; a changed weight, adapter, config field, dtype or tokenizer changes it. Never
    raises on a well-formed model.

    ``adapter_id`` is the caller's provenance string: the checkpoint path or the spec source it
    resolved. It is one component among seven and not the load-bearing one, which is deliberate. The
    defect this replaced had four call sites all passing a class name, and a fingerprint that depends
    on every caller getting one argument right will eventually meet a caller that does not. The
    adapter tensors are read off the model here, so two checkpoints separate even when the caller
    passes nothing at all.

    One consequence a caller should know about: an ``adapter_id`` that is an absolute filesystem
    path makes the id machine-specific, because that path is. A hub id or a run-relative path keeps
    it portable. Nothing about the separation depends on the choice; the content components do that
    work, and a machine-specific id costs recomputation rather than correctness.
    """
    local_dir = _resolve_local_dir(model)
    weights = _hash_weights_from_disk(local_dir) if local_dir is not None else None
    if weights is None:
        weights = _hash_weights_from_state_dict(model)
    adapter_names = _adapter_param_names(model)
    material = {
        "weights": weights,
        "manifest": _hash_manifest(model),
        "adapter_weights": (
            _hash_weights_from_state_dict(model, adapter_names) if adapter_names else "none"
        ),
        "adapter_state": _hash_adapter_state(model),
        "config": _hash_config(model),
        "tokenizer": _hash_tokenizer(tokenizer),
        "adapter": adapter_id or "",
    }
    return ModelFP(content_hash(material, "mfp"))


def fingerprint_bytes(data: bytes, prefix: str = "mfp") -> ModelFP:
    """Fingerprint raw bytes (used by tests and by callers that already have a serialized blob)."""
    return ModelFP(hash_bytes(data, prefix))


__all__ = ["fingerprint", "fingerprint_bytes"]
