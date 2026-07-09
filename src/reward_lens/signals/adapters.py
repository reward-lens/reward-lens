"""Signal adapters: capabilities and site addressing over the v1 architecture adapters (R3).

v1 discovered model structure by duck typing (``hasattr(adapter, "get_attn_o_proj")``), which is
exactly the pattern R3 outlaws: a missing method surfaced as a deep ``AttributeError`` or, worse, a
silently skipped code path (the InternLM2/QRM class of silent exclusion). This module keeps the v1
``ModelAdapter`` navigation logic intact (it is correct and hard-won, and duplicating it would be a
maintenance liability) and layers the two things v3 needs on top of it:

  - a **declared** ``Capability`` set per family, resolved before any GPU work, and
  - a ``SiteMap`` that resolves a logical ``Site`` to a concrete module path, so no Observable ever
    hardcodes a module name.

The v1 adapters (Llama/Mistral/Gemma2/ArmoRM/InternLM2/Generic) are imported and reused verbatim;
this is the "keep as model_adapters bridged" option the design allows (section 2.3.3). The
``SiteMap`` is built by walking the adapter's own navigation methods and reading each module's
qualified name out of ``named_modules``, which is robust to the exact nesting a family uses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from reward_lens.core.types import Capability, Site
from reward_lens.model_adapters import (
    ArmoRMAdapter,
    Gemma2Adapter,
    GenericAdapter,
    InternLM2Adapter,
    LlamaAdapter,
    MistralAdapter,
    ModelAdapter,
)
from reward_lens.model_adapters import get_adapter as _v1_get_adapter
from reward_lens.runtime.backend import SiteMap

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

# The capability every classifier-family adapter has: scalar scores, prefix scores, activation
# capture, autograd and its second order, a linear readout. GENERATIVE/PAIRED_MODELS/SPAN_TYPES are
# the judge/implicit/trajectory adapters' business (section 2.3.3) and are not claimed here.
_CLASSIFIER_CAPS = (
    Capability.SCORES
    | Capability.PREFIX_SCORES
    | Capability.ACTIVATIONS
    | Capability.GRADIENTS
    | Capability.HVP
    | Capability.LINEAR_READOUT
)

# ArmoRM adds MULTI_READOUT (its 19 objective rows are first-class, not row-meaned; section 2.3.3).
_ARMORM_CAPS = _CLASSIFIER_CAPS | Capability.MULTI_READOUT


def capabilities_for(adapter: ModelAdapter) -> Capability:
    """Return the declared ``Capability`` set for a v1 adapter instance (R3).

    ArmoRM declares ``MULTI_READOUT`` because its head has nineteen objective rows the design keeps
    separate; every other classifier family declares the common set. This is a table, not a probe:
    the runner checks an Observable's ``requires`` against it before any forward, and fails with a
    precise message instead of a late ``AttributeError``.
    """
    if isinstance(adapter, ArmoRMAdapter):
        return _ARMORM_CAPS
    return _CLASSIFIER_CAPS


def is_multi_readout(adapter: ModelAdapter, model: "nn.Module") -> bool:
    """Whether this signal exposes multiple readout rows (a non-scalar reward head).

    True for ArmoRM-class heads and for any head whose weight matrix has more than one row (QRM's
    nineteen quantile/objective rows land here too). The row count is read off the checkpoint, so a
    multi-objective model is never silently collapsed to a row mean (liability 6).
    """
    weight = _reward_head_weight(adapter, model)
    return weight is not None and weight.ndim == 2 and weight.shape[0] > 1


def reward_head_module(adapter: ModelAdapter, model: "nn.Module") -> "nn.Module | None":
    """Return the ``nn.Module`` that maps the final hidden state to the reward (the head).

    Mirrors the search in the v1 ``get_reward_head_params`` (``score`` / ``regression_layer`` /
    ``v_head`` / ``reward_head``) but returns the module itself, because the runtime pre-hooks it to
    capture the exact head-input hidden state (the tensor the head consumes) for fp32 scoring, grad,
    and hvp. Returns ``None`` if no linear-like head is found; the caller then falls back to reading
    the model's native logits.
    """
    import torch.nn as nn

    for name in ("score", "regression_layer", "v_head", "reward_head", "classifier"):
        head = getattr(model, name, None)
        if isinstance(head, nn.Linear):
            return head
        if isinstance(head, nn.Sequential):
            for sub in reversed(list(head.modules())):
                if isinstance(sub, nn.Linear):
                    return sub
    return None


def _reward_head_weight(adapter: ModelAdapter, model: "nn.Module") -> "torch.Tensor | None":
    head = reward_head_module(adapter, model)
    if head is None:
        return None
    return head.weight.data


def build_site_map(adapter: ModelAdapter, model: "nn.Module") -> SiteMap:
    """Resolve every logical ``Site`` this architecture exposes to a module path (section 2.2.1).

    Walks the adapter's own navigation (``get_layers``/``get_attn_module``/``get_mlp_module``/
    ``get_attn_o_proj``/``get_embedding``) and records the qualified name of each module by identity
    lookup against ``named_modules``. The result maps:

      - ``Site(L, "resid_post")`` and ``Site(L, "resid_pre")`` to the decoder block at layer L (a
        forward hook reads the block output as resid_post; a pre-hook reads its input as resid_pre),
      - ``Site(L, "attn_out")`` to the attention sublayer, ``Site(L, "mlp_out")`` to the MLP,
      - ``Site(L, "head_out")`` (head-agnostic key) to the attention output projection, whose input
        is the concatenated per-head outputs the runtime slices by head, and
      - ``Site(-1, "embed")`` to the token embedding (the pre-first-layer residual).

    Storing paths rather than module objects is what the frozen ``SiteMap`` wants, and the identity
    lookup makes it correct regardless of how deeply a family nests its backbone.
    """
    name_by_id = {id(module): name for name, module in model.named_modules()}

    def path_of(module: "nn.Module | None") -> str | None:
        if module is None:
            return None
        return name_by_id.get(id(module))

    paths: dict[Site, str] = {}
    layers = adapter.get_layers(model)
    n_layers = len(layers)
    for layer_idx, layer in enumerate(layers):
        layer_path = path_of(layer)
        if layer_path is not None:
            paths[Site(layer_idx, "resid_post")] = layer_path
            paths[Site(layer_idx, "resid_pre")] = layer_path
        attn_path = path_of(adapter.get_attn_module(layer))
        if attn_path is not None:
            paths[Site(layer_idx, "attn_out")] = attn_path
        mlp_path = path_of(adapter.get_mlp_module(layer))
        if mlp_path is not None:
            paths[Site(layer_idx, "mlp_out")] = mlp_path
        o_proj_path = path_of(adapter.get_attn_o_proj(layer))
        if o_proj_path is not None:
            paths[Site(layer_idx, "head_out", None)] = o_proj_path

    embed_path = path_of(adapter.get_embedding(model))
    if embed_path is not None:
        paths[Site(-1, "embed")] = embed_path

    d_model = int(_reward_head_weight(adapter, model).shape[-1])
    return SiteMap(
        module_paths=paths,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=int(adapter.n_heads(model)),
    )


def resolve_adapter(model: "nn.Module", model_name: str = "") -> ModelAdapter:
    """Select the v1 adapter for a model (delegates to the proven v1 dispatch)."""
    return _v1_get_adapter(model, model_name)


__all__ = [
    "ModelAdapter",
    "LlamaAdapter",
    "MistralAdapter",
    "Gemma2Adapter",
    "ArmoRMAdapter",
    "InternLM2Adapter",
    "GenericAdapter",
    "resolve_adapter",
    "capabilities_for",
    "is_multi_readout",
    "reward_head_module",
    "build_site_map",
]
