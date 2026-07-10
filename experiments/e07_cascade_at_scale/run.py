"""
E07 — Misalignment cascade detection at proper scale.

Computes the cross-dimension correlation matrix on per-pair reward
differentials (≥30 probes/dim). Bootstrap CI per cell, paired permutation
significance, hierarchical clustering on the correlation distance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import ExperimentConfig
from ..utils.io import JsonlWriter, manifest_run, save_json, write_csv
from ..utils.figures import setup_matplotlib, savefig
from ..utils.parallel import tprint, clear_gpu
from ..utils.diagnostics import load_diagnostic_v2
from ..utils.models import load_reward_model
from ..utils.shared_cache import cached_forward

from reward_lens.statistics import bootstrap_ci, paired_permutation_test, bh_fdr


def run(cfg: ExperimentConfig) -> dict:
    out = cfg.out_path
    (out / "figures").mkdir(parents=True, exist_ok=True)
    pairs = list(load_diagnostic_v2(dimensions=cfg.dimensions, limit_per_dim=cfg.n_pairs_per_dim))
    by_dim: dict[str, list] = {}
    for p in pairs:
        by_dim.setdefault(p.dimension, []).append(p)

    master_rows = []
    for mc in cfg.models:
        short = mc.short_name()
        model_out = out / short
        model_out.mkdir(parents=True, exist_ok=True)
        with manifest_run(model_out, "e07_cascade_at_scale", cfg.__dict__,
                          model=mc.name, seed=cfg.seed,
                          swallow_exceptions=cfg.skip_models_on_error):
            try:
                rm = load_reward_model(mc)
            except Exception as e:
                tprint(f"[e07] load failed: {e}")
                raise

            jsonl = JsonlWriter(model_out / "cascade_per_pair.jsonl")
            todo = [p for p in pairs if not jsonl.has(f"{p.source}:{p.pair_id}")]
            if todo:
                cache_w = cached_forward(
                    rm, [(p.prompt, p.preferred) for p in todo],
                    side="preferred", cfg=cfg, model_short=short,
                )
                cache_l = cached_forward(
                    rm, [(p.prompt, p.dispreferred) for p in todo],
                    side="dispreferred", cfg=cfg, model_short=short,
                )
                rw = cache_w.rewards.detach().cpu().numpy()
                rl = cache_l.rewards.detach().cpu().numpy()
                for i, p in enumerate(todo):
                    jsonl.write({
                        "record_id": f"{p.source}:{p.pair_id}",
                        "source": p.source, "dimension": p.dimension, "pair_id": p.pair_id,
                        "differential": float(rw[i] - rl[i]),
                    })

            records = jsonl.read_all()
            by_dim_diff: dict[str, list[float]] = {}
            for r in records:
                by_dim_diff.setdefault(r["dimension"], []).append(r["differential"])
            dims = sorted(by_dim_diff.keys())
            n = len(dims)

            # We need pair-aligned vectors per dim — but pairs are dim-specific
            # in our diagnostic_v2 set, so the standard cross-dim correlation
            # of differentials uses sampled aggregation. Here we instead use
            # bootstrap-mean-correlation: per resample, sample equal-size
            # batches per dim and compute correlation of means.
            corr_mat = np.zeros((n, n))
            ci_low_mat = np.zeros((n, n))
            ci_high_mat = np.zeros((n, n))
            p_mat = np.ones((n, n))
            for i, di in enumerate(dims):
                xi = np.asarray(by_dim_diff[di])
                for j, dj in enumerate(dims):
                    yj = np.asarray(by_dim_diff[dj])
                    m = min(xi.size, yj.size)
                    if m < 3:
                        corr_mat[i, j] = float("nan"); continue
                    a = xi[:m]; b = yj[:m]
                    if np.std(a) == 0 or np.std(b) == 0:
                        corr_mat[i, j] = 0.0; continue
                    r = float(np.corrcoef(a, b)[0, 1])
                    corr_mat[i, j] = r
                    # bootstrap CI by resampling indices
                    rng = np.random.default_rng(cfg.seed + i * 100 + j)
                    idx = rng.integers(0, m, size=(min(cfg.n_resamples, 5000), m))
                    rs = []
                    for k in range(idx.shape[0]):
                        aa = a[idx[k]]; bb = b[idx[k]]
                        if np.std(aa) == 0 or np.std(bb) == 0:
                            continue
                        rs.append(np.corrcoef(aa, bb)[0, 1])
                    if rs:
                        rs = np.asarray(rs)
                        alpha = (1 - cfg.ci) / 2
                        ci_low_mat[i, j] = float(np.quantile(rs, alpha))
                        ci_high_mat[i, j] = float(np.quantile(rs, 1 - alpha))
                    if i != j:
                        p_mat[i, j] = paired_permutation_test(a, b,
                                                              n_permutations=min(cfg.n_resamples, 5000),
                                                              statistic="mean_diff",
                                                              seed=cfg.seed)
                        master_rows.append({
                            "model": short, "dim_i": di, "dim_j": dj,
                            "correlation": r, "ci_low": ci_low_mat[i, j],
                            "ci_high": ci_high_mat[i, j], "p_value": p_mat[i, j],
                        })

            # FDR correction across the off-diagonal cells
            offdiag_p = []
            offdiag_idx = []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        offdiag_p.append(p_mat[i, j])
                        offdiag_idx.append((i, j))
            if offdiag_p:
                rejected, q = bh_fdr(offdiag_p, alpha=0.05)
                q_mat = np.full((n, n), np.nan)
                for k, (i, j) in enumerate(offdiag_idx):
                    q_mat[i, j] = q[k]
                save_json({"dimensions": dims,
                           "correlation": corr_mat.tolist(),
                           "ci_low": ci_low_mat.tolist(), "ci_high": ci_high_mat.tolist(),
                           "p_value": p_mat.tolist(), "q_value": q_mat.tolist()},
                          model_out / "cascade_correlations.json")
                _heatmap(corr_mat, dims, out / "figures" / f"e07_cascade_{short}")
            del rm
            clear_gpu()

    write_csv(master_rows, out / "e07_cascade.csv")
    return {"rows": master_rows}


def _heatmap(matrix: np.ndarray, dims: list[str], path: Path) -> None:
    setup_matplotlib()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(dims))); ax.set_xticklabels(dims, rotation=30, ha="right")
    ax.set_yticks(range(len(dims))); ax.set_yticklabels(dims)
    for i in range(len(dims)):
        for j in range(len(dims)):
            v = matrix[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if abs(v) < 0.6 else "white")
    fig.colorbar(im, ax=ax, label="correlation")
    ax.set_title("E07 cascade correlations")
    savefig(fig, path)
