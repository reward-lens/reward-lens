"""
E03 — Component attribution at population scale.

For every (pair, model) we compute per-component attribution and emit
per-pair top-15 + the full vector. Aggregate stats: top-10-by-frequency
with bootstrap CI on each frequency, plus rank-stability heatmaps.
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..config import ExperimentConfig
from ..utils.io import JsonlWriter, manifest_run, save_json, write_csv
from ..utils.figures import setup_matplotlib, savefig
from ..utils.parallel import tprint, clear_gpu
from ..utils.batching import batch_attribution
from ..utils.diagnostics import load_diagnostic_v2
from ..utils.datasets import load_rewardbench
from ..utils.models import load_reward_model
from ..utils.shared_cache import cached_forward


def _bootstrap_freq_ci(values: list[bool], n_resamples: int, seed: int, ci: float
                       ) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    return float(arr.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def run(cfg: ExperimentConfig) -> dict:
    out = cfg.out_path
    (out / "figures").mkdir(parents=True, exist_ok=True)

    pairs = list(load_diagnostic_v2(dimensions=cfg.dimensions, limit_per_dim=cfg.n_pairs_per_dim))
    for sub in ("chat", "safety", "reasoning"):
        pairs += load_rewardbench(subset=sub, limit=cfg.n_pairs_per_dim)
    tprint(f"[e03] total pairs: {len(pairs)}")

    master_rows: list[dict] = []
    for mc in cfg.models:
        short = mc.short_name()
        model_out = out / short
        model_out.mkdir(parents=True, exist_ok=True)
        with manifest_run(model_out, "e03_attribution_population", cfg.__dict__,
                          model=mc.name, seed=cfg.seed,
                          swallow_exceptions=cfg.skip_models_on_error):
            try:
                rm = load_reward_model(mc)
            except Exception as e:
                tprint(f"[e03] failed to load {mc.name}: {e}")
                raise

            jsonl = JsonlWriter(model_out / "attribution_per_pair.jsonl")
            todo = [p for p in pairs if not jsonl.has(f"{p.source}:{p.pair_id}")]

            if todo:
                t0 = time.time()
                cache_w = cached_forward(
                    rm, [(p.prompt, p.preferred) for p in todo],
                    side="preferred", cfg=cfg, model_short=short,
                )
                cache_l = cached_forward(
                    rm, [(p.prompt, p.dispreferred) for p in todo],
                    side="dispreferred", cfg=cfg, model_short=short,
                )
                tprint(f"[e03] {short}: forwards in {time.time()-t0:.1f}s")
                names_w, types_w, layer_idxs, contribs_w = batch_attribution(rm, cache_w)
                names_l, _, _, contribs_l = batch_attribution(rm, cache_l)
                diff = contribs_w - contribs_l  # (B, C)

                for i, p in enumerate(todo):
                    d = diff[i]
                    order = np.argsort(np.abs(d))[::-1]
                    top15 = [(names_w[k], float(d[k])) for k in order[:15]]
                    rec = {
                        "record_id": f"{p.source}:{p.pair_id}",
                        "source": p.source, "dimension": p.dimension, "pair_id": p.pair_id,
                        "component_names": names_w,
                        "differential_contributions": d.tolist(),
                        "top_15": top15,
                    }
                    jsonl.write(rec)

            records = jsonl.read_all()

            # Aggregate per-dimension top-K with bootstrap CI on frequency.
            by_dim: dict[str, list[dict]] = {}
            for r in records:
                by_dim.setdefault(r["dimension"], []).append(r)

            top_components_by_dim: dict[str, list[str]] = {}
            for dim, rs in by_dim.items():
                # For each component, frequency of appearing in top-10 (per pair).
                freq: dict[str, list[bool]] = {}
                for r in rs:
                    in_top = set(name for name, _ in r["top_15"][:10])
                    for name in in_top:
                        freq.setdefault(name, []).extend([True])
                    # Mark components NOT in this pair's top-10 as False
                    # only if they ever appeared — to keep n consistent we
                    # accept that components never seen don't enter freq.
                # Build full-Bernoulli vectors against pair count
                n_pairs = len(rs)
                rows = []
                for name, hits in freq.items():
                    binary = [True] * len(hits) + [False] * (n_pairs - len(hits))
                    f, lo, hi = _bootstrap_freq_ci(binary, cfg.n_resamples, cfg.seed, cfg.ci)
                    rows.append({"component": name, "frequency": f, "ci": [lo, hi],
                                 "n_pairs": n_pairs})
                rows.sort(key=lambda r: r["frequency"], reverse=True)
                top_components_by_dim[dim] = [r["component"] for r in rows[:10]]
                save_json(rows, model_out / f"top_components_{dim}.json")
                for r in rows[:10]:
                    master_rows.append({"model": short, "dimension": dim, **r})
            save_json({"top_components_by_dim": top_components_by_dim},
                      model_out / "attribution_summary.json")
            del rm
            clear_gpu()

    write_csv(master_rows, out / "e03_top_components.csv")
    return {"rows": master_rows}
