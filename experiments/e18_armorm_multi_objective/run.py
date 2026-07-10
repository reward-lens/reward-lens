"""
E18 — ArmoRM multi-objective deep dive (19 objectives).

Per-objective lens, attribution, hacking profile, and conflict matrix
on the 19 directions. ArmoRM-specific.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..config import ExperimentConfig
from ..utils.io import manifest_run, save_json, write_csv
from ..utils.figures import setup_matplotlib, savefig
from ..utils.parallel import tprint, clear_gpu
from ..utils.diagnostics import load_diagnostic_v2
from ..utils.models import load_reward_model


def run(cfg: ExperimentConfig) -> dict:
    out = cfg.out_path
    (out / "figures").mkdir(parents=True, exist_ok=True)

    pairs = load_diagnostic_v2(dimensions=cfg.dimensions, limit_per_dim=cfg.n_pairs_per_dim)
    rows_obj_lens = []
    rows_obj_conflict = []

    for mc in cfg.models:
        short = mc.short_name()
        model_out = out / short
        model_out.mkdir(parents=True, exist_ok=True)
        with manifest_run(model_out, "e18_armorm_multi_objective", cfg.__dict__,
                          model=mc.name, seed=cfg.seed,
                          swallow_exceptions=cfg.skip_models_on_error):
            try:
                rm = load_reward_model(mc)
            except Exception as e:
                tprint(f"[e18] load failed: {e}")
                raise
            adapter = rm.adapter
            if not hasattr(adapter, "per_objective_directions"):
                tprint(f"[e18] {short} adapter has no per_objective_directions; skipping")
                continue
            try:
                D = adapter.per_objective_directions(rm.model)  # (n_obj, d_model)
            except Exception as e:
                tprint(f"[e18] failed to extract per-obj directions: {e}")
                continue
            D = D.detach().to(torch.float32)
            n_obj = D.shape[0]

            # Conflict matrix: pairwise cosines of objective directions
            D_norm = D / (D.norm(dim=1, keepdim=True) + 1e-12)
            cos_mat = (D_norm @ D_norm.T).cpu().numpy()
            save_json({"cosine_matrix": cos_mat.tolist(), "n_objectives": n_obj},
                      model_out / "armo_obj_cosine.json")
            _heatmap(cos_mat, [f"obj{i}" for i in range(n_obj)],
                     out / "figures" / f"e18_objective_cosine_{short}")

            for i in range(n_obj):
                for j in range(n_obj):
                    rows_obj_conflict.append({
                        "model": short, "obj_i": i, "obj_j": j,
                        "cosine": float(cos_mat[i, j]),
                    })

            # Per-objective lens — for each objective, project residual streams
            # at each layer onto that objective's direction (no chat-template
            # gating; this is "what would the objective head say if read here?").
            cache = rm.forward_with_cache_batch(
                [(p.prompt, p.preferred) for p in pairs[:cfg.n_pairs_per_dim]],
                batch_size=cfg.batch_size, max_length=cfg.max_length, progress=cfg.progress,
            )
            cache_l = rm.forward_with_cache_batch(
                [(p.prompt, p.dispreferred) for p in pairs[:cfg.n_pairs_per_dim]],
                batch_size=cfg.batch_size, max_length=cfg.max_length, progress=cfg.progress,
            )
            n_layers = rm.n_layers
            for L in range(n_layers):
                hw = cache.residual_streams.get(L)
                hl = cache_l.residual_streams.get(L)
                if hw is None or hl is None:
                    continue
                # (B, d_model) @ (d_model, n_obj) -> (B, n_obj)
                pw = (hw.float() @ D.T.to(hw.device))
                pl = (hl.float() @ D.T.to(hl.device))
                diff = (pw - pl).detach().cpu().numpy()  # (B, n_obj)
                for o in range(n_obj):
                    rows_obj_lens.append({
                        "model": short, "objective": o, "layer": L,
                        "mean_diff": float(diff[:, o].mean()),
                    })
            del rm
            clear_gpu()

    write_csv(rows_obj_conflict, out / "e18_objective_conflict.csv")
    write_csv(rows_obj_lens, out / "e18_objective_lens.csv")
    return {"conflict": rows_obj_conflict, "lens": rows_obj_lens}


def _heatmap(matrix: np.ndarray, labels: list[str], path: Path) -> None:
    setup_matplotlib()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, label="cosine")
    ax.set_title("E18 ArmoRM objective-direction cosines")
    savefig(fig, path)
