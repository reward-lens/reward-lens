"""
E05 — Cross-dimension circuit overlap.

Reads E03's per-pair attribution intermediates (or recomputes them if
absent), computes per-pair top-K Jaccard between dimensions, and reports
the distribution per dimension-pair with bootstrap CI. Hierarchical
clustering on the Jaccard distance matrix yields a dendrogram per model.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import ExperimentConfig
from ..utils.io import JsonlWriter, manifest_run, save_json, write_csv, load_json
from ..utils.figures import setup_matplotlib, savefig
from ..utils.parallel import tprint, clear_gpu


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _cluster_dendrogram(matrix: np.ndarray, dims: list[str], path: Path) -> None:
    setup_matplotlib()
    import matplotlib.pyplot as plt
    try:
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
    except ImportError:
        return
    dist = 1.0 - matrix
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2  # symmetrize
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    fig, ax = plt.subplots(figsize=(8, 4 + 0.2 * len(dims)))
    dendrogram(Z, labels=dims, ax=ax, leaf_rotation=30)
    ax.set_title("E05 dimension dendrogram (Jaccard distance)")
    savefig(fig, path)


def _heatmap(matrix: np.ndarray, dims: list[str], path: Path, title: str) -> None:
    setup_matplotlib()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlOrRd")
    ax.set_xticks(range(len(dims))); ax.set_xticklabels(dims, rotation=30, ha="right")
    ax.set_yticks(range(len(dims))); ax.set_yticklabels(dims)
    for i in range(len(dims)):
        for j in range(len(dims)):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    color="black" if matrix[i, j] < 0.6 else "white", fontsize=8)
    fig.colorbar(im, ax=ax, label="Jaccard")
    ax.set_title(title)
    savefig(fig, path)


def run(cfg: ExperimentConfig) -> dict:
    out = cfg.out_path
    (out / "figures").mkdir(parents=True, exist_ok=True)
    top_k = int(cfg.extra.get("top_k", 10))
    e03_root = cfg.extra.get("e03_root")
    if e03_root:
        e03_root = Path(e03_root)

    master_rows = []
    for mc in cfg.models:
        short = mc.short_name()
        model_out = out / short
        model_out.mkdir(parents=True, exist_ok=True)
        with manifest_run(model_out, "e05_circuit_overlap", cfg.__dict__,
                          model=mc.name, seed=cfg.seed,
                          swallow_exceptions=cfg.skip_models_on_error):
            # Try E03's intermediates first.
            attr_path = None
            if e03_root is not None:
                cand = e03_root / short / "attribution_per_pair.jsonl"
                if cand.exists():
                    attr_path = cand
            if attr_path is None:
                cand = out / short / "attribution_per_pair.jsonl"
                if cand.exists():
                    attr_path = cand
            if attr_path is None or not attr_path.exists():
                tprint(f"[e05] no attribution intermediates for {short}; skipping")
                continue

            jw = JsonlWriter(attr_path)
            records = jw.read_all()
            by_dim: dict[str, list[set]] = {}
            for r in records:
                d = r["dimension"]
                names = r["component_names"]
                contribs = np.asarray(r["differential_contributions"])
                top = np.argsort(np.abs(contribs))[::-1][:top_k]
                by_dim.setdefault(d, []).append(set(names[i] for i in top))

            dims = sorted(by_dim.keys())
            n = len(dims)
            mean_mat = np.zeros((n, n))
            ci_low_mat = np.zeros((n, n))
            ci_high_mat = np.zeros((n, n))

            rng = np.random.default_rng(cfg.seed)
            for i, di in enumerate(dims):
                for j, dj in enumerate(dims):
                    if i == j:
                        mean_mat[i, j] = 1.0
                        ci_low_mat[i, j] = 1.0
                        ci_high_mat[i, j] = 1.0
                        continue
                    Si, Sj = by_dim[di], by_dim[dj]
                    if not Si or not Sj:
                        continue
                    pairs = [(a, b) for a in Si for b in Sj]
                    js = np.array([_jaccard(a, b) for a, b in pairs])
                    mean_mat[i, j] = float(js.mean())
                    # bootstrap CI
                    idx = rng.integers(0, js.size, size=(cfg.n_resamples, js.size))
                    means = js[idx].mean(axis=1)
                    alpha = (1 - cfg.ci) / 2
                    ci_low_mat[i, j] = float(np.quantile(means, alpha))
                    ci_high_mat[i, j] = float(np.quantile(means, 1 - alpha))
                    master_rows.append({
                        "model": short, "dim_i": di, "dim_j": dj,
                        "mean_jaccard": mean_mat[i, j],
                        "ci_low": ci_low_mat[i, j], "ci_high": ci_high_mat[i, j],
                    })

            save_json({"dimensions": dims,
                       "mean_jaccard": mean_mat.tolist(),
                       "ci_low": ci_low_mat.tolist(),
                       "ci_high": ci_high_mat.tolist()},
                      model_out / "circuit_overlap.json")
            _heatmap(mean_mat, dims,
                     out / "figures" / f"e05_overlap_{short}",
                     f"E05 circuit overlap — {short}")
            _cluster_dendrogram(mean_mat, dims,
                                out / "figures" / f"e05_dendrogram_{short}")
    write_csv(master_rows, out / "e05_overlap.csv")
    return {"rows": master_rows}
