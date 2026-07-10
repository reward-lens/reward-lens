"""
E14 — Cross-architecture circuit comparison.

For all loaded models, on a shared set of pairs (RewardBench helpfulness),
compute per-pair lens trajectories and report:
  - distribution of per-pair Pearson correlation between every pair of models
  - family-internal vs family-cross correlations
  - clustered heatmap

The "family" tag is derived from the model name (Llama / Mistral / Gemma /
DeBERTa); override via cfg.extra["families"] = {"short_name": "family"}.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import ExperimentConfig
from ..utils.io import manifest_run, save_json, write_csv, JsonlWriter
from ..utils.figures import setup_matplotlib, savefig
from ..utils.parallel import tprint, clear_gpu
from ..utils.batching import batch_lens_curves
from ..utils.datasets import load_rewardbench
from ..utils.diagnostics import load_diagnostic_v2
from ..utils.models import load_reward_model


def _family(model_name: str) -> str:
    n = model_name.lower()
    for fam in ("llama", "mistral", "gemma", "deberta", "qwen", "internlm"):
        if fam in n:
            return fam
    return "unknown"


def run(cfg: ExperimentConfig) -> dict:
    out = cfg.out_path
    (out / "figures").mkdir(parents=True, exist_ok=True)

    pairs = load_rewardbench(subset="chat", limit=cfg.n_pairs_per_dim)
    if not pairs:
        # Fallback: helpfulness from diagnostic_v2
        pairs = load_diagnostic_v2(["helpfulness"], limit_per_dim=cfg.n_pairs_per_dim)

    model_curves: dict[str, np.ndarray] = {}  # short -> (n_pairs, L+1) — interpolated to common grid
    model_family: dict[str, str] = {}
    families_override = cfg.extra.get("families", {})

    common_grid = np.linspace(0.0, 1.0, 33)
    for mc in cfg.models:
        short = mc.short_name()
        model_out = out / short
        model_out.mkdir(parents=True, exist_ok=True)
        with manifest_run(model_out, "e14_cross_architecture", cfg.__dict__,
                          model=mc.name, seed=cfg.seed,
                          swallow_exceptions=cfg.skip_models_on_error):
            try:
                rm = load_reward_model(mc)
            except Exception as e:
                tprint(f"[e14] load failed: {e}")
                raise
            cache_w = rm.forward_with_cache_batch(
                [(p.prompt, p.preferred) for p in pairs],
                batch_size=cfg.batch_size, max_length=cfg.max_length, progress=cfg.progress,
            )
            cache_l = rm.forward_with_cache_batch(
                [(p.prompt, p.dispreferred) for p in pairs],
                batch_size=cfg.batch_size, max_length=cfg.max_length, progress=cfg.progress,
            )
            curves = batch_lens_curves(rm, cache_w) - batch_lens_curves(rm, cache_l)
            n_layers = rm.n_layers
            xs = np.array([(-1 + (i)) / max(1, n_layers) for i in range(curves.shape[1])])
            xs = (xs + 1.0) / 2  # in [0, ~1]
            # Interpolate each pair's curve onto the common grid
            interp = np.zeros((curves.shape[0], common_grid.size))
            for i in range(curves.shape[0]):
                ci = curves[i]
                if not np.all(np.isfinite(ci)):
                    interp[i] = np.nan
                    continue
                interp[i] = np.interp(common_grid, np.linspace(0, 1, ci.size), ci)
            model_curves[short] = interp
            model_family[short] = families_override.get(short, _family(mc.name))
            del rm
            clear_gpu()

    if len(model_curves) < 2:
        tprint(f"[e14] need >=2 models; got {len(model_curves)}")
        return {"rows": []}

    # Per-pair correlation between every model pair
    shorts = list(model_curves.keys())
    n_pairs = next(iter(model_curves.values())).shape[0]
    rows = []
    intra = []
    inter = []
    for i, mi in enumerate(shorts):
        for j, mj in enumerate(shorts):
            if j <= i:
                continue
            ci = model_curves[mi]; cj = model_curves[mj]
            corrs = []
            for p in range(n_pairs):
                a, b = ci[p], cj[p]
                if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                    continue
                if np.std(a) == 0 or np.std(b) == 0:
                    continue
                corrs.append(float(np.corrcoef(a, b)[0, 1]))
            if not corrs:
                continue
            arr = np.asarray(corrs)
            r_med = float(np.median(arr)); r_mean = float(arr.mean())
            same = model_family[mi] == model_family[mj]
            (intra if same else inter).extend(corrs)
            rows.append({
                "model_i": mi, "model_j": mj,
                "family_i": model_family[mi], "family_j": model_family[mj],
                "same_family": same,
                "n_pairs": len(corrs),
                "median_correlation": r_med, "mean_correlation": r_mean,
            })

    write_csv(rows, out / "e14_cross_arch.csv")
    save_json({
        "intra_family_n": len(intra),
        "intra_family_mean": float(np.mean(intra)) if intra else float("nan"),
        "inter_family_n": len(inter),
        "inter_family_mean": float(np.mean(inter)) if inter else float("nan"),
    }, out / "e14_summary.json")
    return {"rows": rows}
