"""
E02 — Reward-Lens at population scale.

For every (pair, model), run the reward lens and record:
  - the per-layer differential trajectory
  - the crystallization layer (first layer at >=50% of final differential)

Population stats per dimension: mean/median crystallization fraction with
bootstrap CIs. Per-pair JSONL intermediate so re-aggregation is free.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..config import ExperimentConfig
from ..utils.io import JsonlWriter, manifest_run, save_json, write_csv
from ..utils.figures import setup_matplotlib, savefig, PALETTE
from ..utils.parallel import tprint, clear_gpu
from ..utils.batching import batch_lens_curves
from ..utils.diagnostics import load_diagnostic_v2
from ..utils.datasets import load_rewardbench
from ..utils.models import load_reward_model
from ..utils.shared_cache import cached_forward


def _bootstrap_mean_ci(values: list[float], n_resamples: int, seed: int,
                       ci: float = 0.95) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    return float(arr.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def _crystal_frac(diff_curve: np.ndarray, layer_keys: list[int], n_layers: int) -> float:
    """Fraction-of-depth at which differential first reaches 50% of final.

    Defense-in-depth: when ``final`` is zero or non-finite (Gemma-2's
    logit soft-cap can collapse the late-layer differential to numerical
    zero — see deep_analysis_v2 §2.5), we fall back to ``max |diff|``
    as the reference magnitude. If even that is below 1e-8 the pair is
    genuinely degenerate and we return 1.0 (crystallises at the top).
    """
    arr = np.asarray(diff_curve, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    final = arr[-1]
    ref = final
    if not np.isfinite(ref) or abs(ref) < 1e-8:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return float("nan")
        # Use the largest-magnitude finite differential as the reference.
        idx_max = int(np.argmax(np.abs(finite)))
        ref = float(finite[idx_max])
        if abs(ref) < 1e-8:
            return 1.0
    threshold = 0.5 * ref
    for i, d in enumerate(arr):
        if not np.isfinite(d):
            continue
        if (ref > 0 and d >= threshold) or (ref < 0 and d <= threshold):
            # layer_keys[i] is in [-1, n_layers-1]; map to [0, 1]
            return max(0.0, layer_keys[i] / max(1, n_layers))
    return 1.0


def run(cfg: ExperimentConfig) -> dict:
    out = cfg.out_path
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # Lens uses diagnostic_v2 + RewardBench helpfulness/safety/reasoning if
    # available. Population sizes per dimension >= 200 in practice.
    pairs = list(load_diagnostic_v2(dimensions=cfg.dimensions, limit_per_dim=cfg.n_pairs_per_dim))
    for sub in ("chat", "safety", "reasoning"):
        pairs += load_rewardbench(subset=sub, limit=cfg.n_pairs_per_dim)
    tprint(f"[e02] total pairs: {len(pairs)}")

    master_rows: list[dict] = []
    for mc in cfg.models:
        short = mc.short_name()
        model_out = out / short
        model_out.mkdir(parents=True, exist_ok=True)
        with manifest_run(model_out, "e02_lens_population", cfg.__dict__,
                          model=mc.name, seed=cfg.seed,
                          swallow_exceptions=cfg.skip_models_on_error):
            try:
                rm = load_reward_model(mc)
            except Exception as e:
                tprint(f"[e02] {mc.name} failed to load: {e}")
                raise
            n_layers = rm.n_layers
            layer_keys = [-1] + list(range(n_layers))
            jsonl = JsonlWriter(model_out / "lens_per_pair.jsonl")

            todo = [(p, p) for p in pairs if not jsonl.has(f"{p.source}:{p.pair_id}")]
            todo_pairs = [t[0] for t in todo]

            if todo_pairs:
                # Run two batched forwards (preferred + dispreferred), then
                # compute per-pair lens curves from cached residual streams.
                t0 = time.time()
                cache_w = cached_forward(
                    rm, [(p.prompt, p.preferred) for p in todo_pairs],
                    side="preferred", cfg=cfg, model_short=short,
                )
                cache_l = cached_forward(
                    rm, [(p.prompt, p.dispreferred) for p in todo_pairs],
                    side="dispreferred", cfg=cfg, model_short=short,
                )
                tprint(f"[e02] {short}: forwards done in {time.time()-t0:.1f}s")

                lens_w = batch_lens_curves(rm, cache_w)  # (B, L+1)
                lens_l = batch_lens_curves(rm, cache_l)
                diffs = lens_w - lens_l
                rewards_w = cache_w.rewards.detach().cpu().numpy()
                rewards_l = cache_l.rewards.detach().cpu().numpy()

                for i, p in enumerate(todo_pairs):
                    diff = diffs[i]
                    rec = {
                        "record_id": f"{p.source}:{p.pair_id}",
                        "source": p.source, "dimension": p.dimension, "pair_id": p.pair_id,
                        "n_layers": n_layers,
                        "differential_curve": diff.tolist(),
                        "lens_preferred": lens_w[i].tolist(),
                        "lens_dispreferred": lens_l[i].tolist(),
                        "reward_preferred": float(rewards_w[i]),
                        "reward_dispreferred": float(rewards_l[i]),
                        "crystallization_frac": _crystal_frac(diff, layer_keys, n_layers),
                    }
                    jsonl.write(rec)

            records = jsonl.read_all()
            # Aggregate per-dimension
            by_dim: dict[str, list[dict]] = {}
            for r in records:
                by_dim.setdefault(r["dimension"], []).append(r)
            summary = {"per_dimension": {}}
            for dim, rs in by_dim.items():
                fracs = [r["crystallization_frac"] for r in rs]
                m, lo, hi = _bootstrap_mean_ci(fracs, cfg.n_resamples, cfg.seed, cfg.ci)
                med = float(np.nanmedian(np.asarray(fracs))) if fracs else float("nan")
                summary["per_dimension"][dim] = {
                    "n": len(rs), "mean_crystal_frac": m,
                    "mean_ci": [lo, hi], "median_crystal_frac": med,
                }
                master_rows.append({
                    "model": short, "dimension": dim, "n": len(rs),
                    "mean_crystal_frac": m, "ci_low": lo, "ci_high": hi,
                    "median_crystal_frac": med,
                })
            save_json(summary, model_out / "lens_summary.json")
            _plot_ridge(short, by_dim, model_out / "figures" / f"e02_crystal_{short}")

            del rm
            clear_gpu()

    write_csv(master_rows, out / "e02_crystallization.csv")
    return {"rows": master_rows}


def _plot_ridge(model_short: str, by_dim: dict[str, list[dict]], path: Path) -> None:
    setup_matplotlib()
    import matplotlib.pyplot as plt
    dims = sorted(by_dim.keys())
    if not dims:
        return
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(dims) + 1.5))
    for i, d in enumerate(dims):
        fracs = np.asarray([r["crystallization_frac"] for r in by_dim[d]])
        fracs = fracs[np.isfinite(fracs)]
        if fracs.size == 0:
            continue
        # Lay each dimension out as a violin
        ax.violinplot([fracs], positions=[i], vert=False, showmeans=True, showmedians=True)
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels(dims)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Crystallization depth (fraction of model)")
    ax.set_title(f"E02 reward-lens crystallization — {model_short}")
    savefig(fig, path)
