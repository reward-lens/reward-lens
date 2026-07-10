"""
E09 — Reward conflict analysis at population scale.

Fit term directions per dimension from >=30 pairs and report cosine
similarities with bootstrap CIs over pair-resamples. Test whether
"alignments" are distinguishable from orthogonal (CI excludes 0).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import ExperimentConfig
from ..utils.io import manifest_run, save_json, write_csv
from ..utils.figures import setup_matplotlib, savefig
from ..utils.parallel import tprint, clear_gpu
from ..utils.diagnostics import load_diagnostic_v2
from ..utils.models import load_reward_model
from ..utils.shared_cache import cached_forward


def run(cfg: ExperimentConfig) -> dict:
    out = cfg.out_path
    (out / "figures").mkdir(parents=True, exist_ok=True)
    from reward_lens.diagnostic_data_v2 import ALL_DIMENSIONS_V2

    pairs_by_dim = {}
    all_unique_pairs = []
    seen_pairs = set()
    for d in ALL_DIMENSIONS_V2:
        ps = list(load_diagnostic_v2([d], limit_per_dim=cfg.n_pairs_per_dim))
        pairs_by_dim[d] = ps
        for p in ps:
            pid = f"{p.source}:{p.pair_id}"
            if pid not in seen_pairs:
                all_unique_pairs.append(p)
                seen_pairs.add(pid)

    master_rows = []
    for mc in cfg.models:
        short = mc.short_name()
        model_out = out / short
        model_out.mkdir(parents=True, exist_ok=True)
        with manifest_run(model_out, "e09_conflict_population", cfg.__dict__,
                          model=mc.name, seed=cfg.seed,
                          swallow_exceptions=cfg.skip_models_on_error):
            try:
                rm = load_reward_model(mc)
            except Exception as e:
                tprint(f"[e09] load failed: {e}")
                raise

            # 1. Pre-calculate deltas for all unique pairs using batching
            tprint(f"[e09] {short}: caching deltas for {len(all_unique_pairs)} pairs...")
            import time
            import torch
            t0 = time.time()
            cache_w = cached_forward(
                rm, [(p.prompt, p.preferred) for p in all_unique_pairs],
                side="preferred", cfg=cfg, model_short=short,
            )
            cache_l = cached_forward(
                rm, [(p.prompt, p.dispreferred) for p in all_unique_pairs],
                side="dispreferred", cfg=cfg, model_short=short,
            )
            
            # Get final layer activations
            final_layer = rm.n_layers - 1
            h_pref = cache_w.residual_streams.get(final_layer)
            h_disp = cache_l.residual_streams.get(final_layer)
            
            if h_pref is None or h_disp is None:
                tprint(f"[e09] {short}: failed to get activations")
                del rm
                clear_gpu()
                continue
                
            # Move to CPU for memory-efficient bootstrapping
            all_deltas = (h_pref - h_disp).squeeze().cpu().float()
            pair_to_delta = {f"{p.source}:{p.pair_id}": all_deltas[i] for i, p in enumerate(all_unique_pairs)}
            tprint(f"[e09] {short}: cached deltas in {time.time()-t0:.1f}s")

            # Organize deltas by dimension
            deltas_by_dim_tensor = {}
            dims = sorted(pairs_by_dim.keys())
            for d in dims:
                ps = pairs_by_dim[d]
                ds = [pair_to_delta[f"{p.source}:{p.pair_id}"] for p in ps]
                if ds:
                    deltas_by_dim_tensor[d] = torch.stack(ds)

            # 2. Bootstrap over cached deltas
            n = len(dims)
            n_boot = min(cfg.n_resamples, 1000) # Faster bootstrap with 1k resamples
            rng = np.random.default_rng(cfg.seed)
            cosines_acc = np.zeros((n_boot, n, n))

            tprint(f"[e09] {short}: bootstrapping {n_boot} resamples...")
            t0 = time.time()
            
            # Pre-compute all directions for each dimension across all resamples
            dim_directions = {} # dim -> (n_boot, D)
            for d in dims:
                if d not in deltas_by_dim_tensor: continue
                deltas = deltas_by_dim_tensor[d] # (N, D)
                N = len(deltas)
                idx = rng.integers(0, N, size=(n_boot, N))
                
                # Batch compute means: (n_boot, N, D) -> (n_boot, D)
                # Using chunks to avoid large memory allocations if N*n_boot*D is huge
                chunk_size = 100
                dir_samples_list = []
                for i in range(0, n_boot, chunk_size):
                    chunk_idx = idx[i:i+chunk_size]
                    dir_samples_list.append(deltas[chunk_idx].mean(dim=1))
                dir_samples = torch.cat(dir_samples_list, dim=0)
                
                # Normalize
                dir_samples = dir_samples / (dir_samples.norm(dim=1, keepdim=True) + 1e-12)
                dim_directions[d] = dir_samples

            # Compute all pairwise cosines for each bootstrap iteration
            for i, di in enumerate(dims):
                if di not in dim_directions: continue
                vi = dim_directions[di] # (n_boot, D)
                for j, dj in enumerate(dims):
                    if dj not in dim_directions: continue
                    vj = dim_directions[dj] # (n_boot, D)
                    # Batch dot product over bootstrap iterations
                    cosines_acc[:, i, j] = (vi * vj).sum(dim=1).numpy()

            tprint(f"[e09] {short}: bootstrap done in {time.time()-t0:.1f}s")

            mean_mat = cosines_acc.mean(axis=0)
            alpha = (1 - cfg.ci) / 2
            ci_low_mat = np.quantile(cosines_acc, alpha, axis=0)
            ci_high_mat = np.quantile(cosines_acc, 1 - alpha, axis=0)

            for i, di in enumerate(dims):
                for j, dj in enumerate(dims):
                    master_rows.append({
                        "model": short, "dim_i": di, "dim_j": dj,
                        "mean_cosine": float(mean_mat[i, j]),
                        "ci_low": float(ci_low_mat[i, j]),
                        "ci_high": float(ci_high_mat[i, j]),
                        "distinguishable_from_zero": bool(
                            (ci_low_mat[i, j] > 0) or (ci_high_mat[i, j] < 0)
                        ),
                    })
            save_json({"dimensions": dims,
                       "mean_cosine": mean_mat.tolist(),
                       "ci_low": ci_low_mat.tolist(),
                       "ci_high": ci_high_mat.tolist()},
                      model_out / "conflict_summary.json")
            _heatmap(mean_mat, dims, out / "figures" / f"e09_conflict_{short}")
            del rm
            clear_gpu()

    write_csv(master_rows, out / "e09_conflict.csv")
    return {"rows": master_rows}


def _heatmap(matrix: np.ndarray, dims: list[str], path: Path) -> None:
    setup_matplotlib()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(dims))); ax.set_xticklabels(dims, rotation=30, ha="right")
    ax.set_yticks(range(len(dims))); ax.set_yticklabels(dims)
    for i in range(len(dims)):
        for j in range(len(dims)):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if abs(matrix[i,j]) < 0.6 else "white")
    fig.colorbar(im, ax=ax)
    ax.set_title("E09 reward-conflict cosines")
    savefig(fig, path)
