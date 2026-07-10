"""Model loading + adapter health-check helper."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from ..config import ModelConfig


def _torch_dtype(name: str):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_reward_model(cfg: ModelConfig):
    """Load a RewardModel from a ModelConfig."""
    from reward_lens import RewardModel
    rm = RewardModel.from_pretrained(
        cfg.name,
        device=cfg.device,
        torch_dtype=_torch_dtype(cfg.torch_dtype),
        trust_remote_code=cfg.trust_remote_code,
        attn_implementation=cfg.attn_implementation,
    )
    return rm


@contextmanager
def safe_per_model(
    experiment_name: str,
    mc: ModelConfig,
    out_dir,
    cfg_dict: dict,
    *,
    seed: int = 0,
    skip_on_error: bool = True,
) -> Iterator[Optional[object]]:
    """Per-model loader + manifest writer that isolates failures.

    Yields the loaded ``RewardModel`` (or ``None`` if the load failed), and
    swallows any exception raised inside the with-block when
    ``skip_on_error=True``. The manifest for this (experiment, model) cell
    is written with status ``complete`` on success, ``failed`` (with the
    exception class + message) on any error.

    Always frees GPU memory on exit, even if the body raised, so the next
    model in the loop has clean memory to load into.

    Use case (replaces the old try/except-around-load pattern that left
    forward-pass failures uncaught — see deep_analysis_v2 §2.7)::

        for mc in cfg.models:
            with safe_per_model("e04_...", mc, out_dir, cfg.__dict__,
                                 seed=cfg.seed,
                                 skip_on_error=cfg.skip_models_on_error) as rm:
                if rm is None:
                    continue
                # ... use rm.score(...) etc; any exception lands as a
                # cell-level failed manifest, the next model still runs.
    """
    import time
    from pathlib import Path

    from .io import write_manifest
    from .parallel import tprint, clear_gpu

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    write_manifest(out_path, experiment_name, cfg_dict, status="running",
                   model=mc.name, seed=seed)

    rm = None
    error: Optional[BaseException] = None
    t0 = time.time()
    try:
        try:
            rm = load_reward_model(mc)
        except Exception as e:
            error = e
            tprint(f"[{experiment_name}] failed to load {mc.name}: "
                   f"{type(e).__name__}: {e}")
            if not skip_on_error:
                raise
            # rm stays None; we still yield so the user's `with` block
            # can detect it and `continue`.
        try:
            yield rm
        except Exception as e:
            error = e
            tprint(f"[{experiment_name}] {mc.name} failed: "
                   f"{type(e).__name__}: {e}")
            if not skip_on_error:
                raise
    finally:
        runtime = time.time() - t0
        if error is None and rm is not None:
            write_manifest(out_path, experiment_name, cfg_dict, status="complete",
                           model=mc.name, seed=seed, runtime_seconds=runtime)
        else:
            note = (f"{type(error).__name__}: {error}" if error is not None
                    else "load returned None")
            write_manifest(out_path, experiment_name, cfg_dict, status="failed",
                           model=mc.name, seed=seed, runtime_seconds=runtime,
                           notes=note)
        if rm is not None:
            try:
                del rm
            except Exception:
                pass
        clear_gpu()


def adapter_health_check(rm, n_pairs: int = 5) -> dict:
    """Five-pair sanity test: forward_with_cache, lens, attribution, batched
    forward all return finite numbers. Used by E01.

    Returns a dict with per-check pass/fail flags + diagnostic info.
    Failures don't raise — caller decides whether to drop the model."""
    import math
    import numpy as np
    from reward_lens.lens import RewardLens
    from reward_lens.attribution import ComponentAttribution
    from reward_lens.diagnostic_data_v2 import get_pairs_v2

    results = {"checks": {}, "overall": True}
    pairs = get_pairs_v2()[:n_pairs]
    if not pairs:
        results["overall"] = False
        results["checks"]["no_pairs"] = "diagnostic_v2 is empty"
        return results

    # 1. score_pair
    try:
        sw, sl = rm.score_pair(pairs[0].prompt, pairs[0].preferred, pairs[0].dispreferred)
        ok = math.isfinite(sw) and math.isfinite(sl)
        results["checks"]["score_pair"] = {"ok": ok, "preferred": sw, "dispreferred": sl}
        results["overall"] &= ok
    except Exception as e:
        results["checks"]["score_pair"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        results["overall"] = False

    # 2. forward_with_cache
    try:
        r, cache = rm.forward_with_cache(pairs[0].prompt, pairs[0].preferred)
        ok = math.isfinite(r) and len(cache.residual_streams) > 0
        results["checks"]["forward_with_cache"] = {"ok": ok, "n_layers_cached": len(cache.residual_streams)}
        results["overall"] &= ok
    except Exception as e:
        results["checks"]["forward_with_cache"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        results["overall"] = False

    # 3. RewardLens
    try:
        lens = RewardLens(rm)
        lr = lens.trace(pairs[0].prompt, pairs[0].preferred, pairs[0].dispreferred)
        ok = np.all(np.isfinite(lr.differential))
        results["checks"]["lens"] = {"ok": bool(ok), "crystal_layer": int(lr.crystallization_layer)}
        results["overall"] &= bool(ok)
    except Exception as e:
        results["checks"]["lens"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        results["overall"] = False

    # 4. ComponentAttribution
    try:
        attr = ComponentAttribution(rm)
        cr = attr.attribute(pairs[0].prompt, pairs[0].preferred, pairs[0].dispreferred)
        ok = np.all(np.isfinite(cr.differential_contributions))
        results["checks"]["attribution"] = {"ok": bool(ok), "n_components": len(cr.component_names)}
        results["overall"] &= bool(ok)
    except Exception as e:
        results["checks"]["attribution"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        results["overall"] = False

    # 5. forward_with_cache_batch
    try:
        sample = [(p.prompt, p.preferred) for p in pairs]
        cache = rm.forward_with_cache_batch(sample, batch_size=min(len(sample), 4))
        ok = (cache.batch_size == len(sample) and cache.rewards is not None
              and bool(cache.rewards.isfinite().all().item()))
        results["checks"]["forward_with_cache_batch"] = {
            "ok": ok, "batch_size": cache.batch_size,
            "n_layers": len(cache.residual_streams),
        }
        results["overall"] &= ok
    except Exception as e:
        results["checks"]["forward_with_cache_batch"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        results["overall"] = False

    return results
