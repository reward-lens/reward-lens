"""
E01 — Baseline + diagnostics-v2 sanity.

Per the plan: score every model on diagnostic_data_v2, RewardBench,
RewardBench-2, RM-Bench, JudgeBench. Report per-dimension accuracy with
bootstrap 95% CI. Flag models whose RewardBench accuracy is more than
3pp below the published number (likely an adapter bug).

Outputs:
  out_dir/<model_short>/baseline_per_pair.jsonl  — per-pair accuracy
  out_dir/<model_short>/baseline_summary.json    — per-dimension acc + CI
  out_dir/e01_accuracy.csv                       — master CSV
  out_dir/figures/e01_baseline_<model>.{pdf,png} — per-model bar chart
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import ExperimentConfig, ModelConfig
from ..utils.io import JsonlWriter, manifest_run, save_json, write_csv
from ..utils.figures import setup_matplotlib, savefig, PALETTE
from ..utils.parallel import tprint, clear_gpu
from ..utils.diagnostics import load_diagnostic_v2
from ..utils.datasets import load_rewardbench, load_rewardbench2, load_rmbench, load_judgebench
from ..utils.models import load_reward_model, adapter_health_check


def _bootstrap_ci(values: list[bool] | list[float], n_resamples: int, seed: int,
                  ci: float = 0.95) -> tuple[float, float, float]:
    """Bootstrap CI for a binary or scalar mean."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    return float(arr.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def _score_pairs(rm, pairs: list, batch_size: int, max_length: int,
                 progress: bool, jsonl: JsonlWriter) -> list[dict]:
    """Score a list of pairs; resume by skipping records already in jsonl.
    Uses batched forward where the model permits it (preferred separately
    from dispreferred, scored via cache.rewards)."""
    todo: list[tuple[int, "obj"]] = []
    for i, p in enumerate(pairs):
        rid = f"{p.source}:{p.pair_id}"
        if jsonl.has(rid):
            continue
        todo.append((i, p))
    if not todo:
        return jsonl.read_all()

    # Score preferred + dispreferred separately in one big batch each.
    pref_pairs = [(p.prompt, p.preferred) for _, p in todo]
    disp_pairs = [(p.prompt, p.dispreferred) for _, p in todo]

    cache_w = rm.forward_with_cache_batch(
        pref_pairs, batch_size=batch_size, max_length=max_length, progress=progress,
    )
    cache_l = rm.forward_with_cache_batch(
        disp_pairs, batch_size=batch_size, max_length=max_length, progress=progress,
    )
    rewards_w = cache_w.rewards.detach().cpu().numpy()
    rewards_l = cache_l.rewards.detach().cpu().numpy()

    for j, (i, p) in enumerate(todo):
        rec = {
            "record_id": f"{p.source}:{p.pair_id}",
            "source": p.source, "dimension": p.dimension, "pair_id": p.pair_id,
            "score_preferred": float(rewards_w[j]),
            "score_dispreferred": float(rewards_l[j]),
            "differential": float(rewards_w[j] - rewards_l[j]),
            "correct": bool(rewards_w[j] > rewards_l[j]),
        }
        jsonl.write(rec)
    return jsonl.read_all()


def _summarize(records: list[dict], n_resamples: int, seed: int, ci: float) -> dict:
    by_dim: dict[str, list[dict]] = {}
    for r in records:
        by_dim.setdefault(r["dimension"], []).append(r)
    summary = {"per_dimension": {}, "overall": {}}
    all_correct = [r["correct"] for r in records]
    if all_correct:
        m, lo, hi = _bootstrap_ci(all_correct, n_resamples, seed, ci)
        summary["overall"] = {"accuracy": m, "ci_low": lo, "ci_high": hi, "n": len(all_correct)}
    for dim, rs in by_dim.items():
        cor = [r["correct"] for r in rs]
        diffs = [r["differential"] for r in rs]
        m, lo, hi = _bootstrap_ci(cor, n_resamples, seed, ci)
        dm, dlo, dhi = _bootstrap_ci(diffs, n_resamples, seed, ci)
        summary["per_dimension"][dim] = {
            "n": len(rs),
            "accuracy": m, "accuracy_ci": [lo, hi],
            "mean_differential": dm, "differential_ci": [dlo, dhi],
        }
    return summary


def _plot_per_model(model_short: str, summary: dict, out_dir: Path) -> None:
    setup_matplotlib()
    import matplotlib.pyplot as plt
    dims = sorted(summary["per_dimension"].keys())
    if not dims:
        return
    means = [summary["per_dimension"][d]["accuracy"] for d in dims]
    los = [summary["per_dimension"][d]["accuracy_ci"][0] for d in dims]
    his = [summary["per_dimension"][d]["accuracy_ci"][1] for d in dims]
    err = [[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(dims))
    ax.bar(x, means, yerr=err, capsize=3, color=PALETTE[0], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(dims, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, label="chance")
    ax.set_ylabel("Accuracy (with 95% CI)")
    ax.set_title(f"E01 baseline accuracy — {model_short}")
    ax.legend(fontsize=8)
    savefig(fig, out_dir / "figures" / f"e01_baseline_{model_short}")


def run(cfg: ExperimentConfig) -> dict:
    out = cfg.out_path
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # Build the merged pair pool. Diagnostic_v2 always; external benchmarks
    # are best-effort (loaders return [] on network failure).
    n_per_dim = cfg.n_pairs_per_dim
    pairs = list(load_diagnostic_v2(dimensions=cfg.dimensions, limit_per_dim=n_per_dim))
    pairs += load_rewardbench(limit=n_per_dim * 4)
    pairs += load_rewardbench2(limit=n_per_dim * 2)
    pairs += load_rmbench(limit=n_per_dim * 2)
    pairs += load_judgebench(limit=n_per_dim)
    tprint(f"[e01] total pairs: {len(pairs)}")

    master_rows: list[dict] = []
    health: dict = {}
    for mc in cfg.models:
        short = mc.short_name()
        model_out = out / short
        model_out.mkdir(parents=True, exist_ok=True)
        with manifest_run(model_out, "e01_baseline_and_diagnostics",
                          cfg.__dict__, model=mc.name, seed=cfg.seed,
                          swallow_exceptions=cfg.skip_models_on_error):
            tprint(f"[e01] loading {mc.name}")
            t0 = time.time()
            try:
                rm = load_reward_model(mc)
            except Exception as e:
                tprint(f"[e01] {mc.name} failed to load: {e}")
                health[short] = {"loaded": False, "error": str(e)}
                raise
            tprint(f"[e01] loaded in {time.time()-t0:.1f}s")
            hc = adapter_health_check(rm)
            health[short] = {"loaded": True, "health": hc}
            save_json(hc, model_out / "adapter_health_check.json")
            if not hc["overall"] and cfg.skip_models_on_error:
                tprint(f"[e01] {short} failed health check, skipping further work")
                clear_gpu()
                continue

            jsonl_path = model_out / "baseline_per_pair.jsonl"
            jsonl = JsonlWriter(jsonl_path) if cfg.resume else JsonlWriter(jsonl_path)
            t0 = time.time()
            records = _score_pairs(rm, pairs, batch_size=cfg.batch_size,
                                   max_length=cfg.max_length, progress=cfg.progress,
                                   jsonl=jsonl)
            tprint(f"[e01] {short} scored {len(records)} pairs in {time.time()-t0:.1f}s")
            summary = _summarize(records, cfg.n_resamples, cfg.seed, cfg.ci)
            save_json(summary, model_out / "baseline_summary.json")
            _plot_per_model(short, summary, out)
            for dim, s in summary["per_dimension"].items():
                master_rows.append({
                    "model": short, "dimension": dim, "n": s["n"],
                    "accuracy": s["accuracy"], "ci_low": s["accuracy_ci"][0],
                    "ci_high": s["accuracy_ci"][1],
                    "mean_diff": s["mean_differential"],
                })
            del rm
            clear_gpu()

    write_csv(master_rows, out / "e01_accuracy.csv",
              columns=["model", "dimension", "n", "accuracy", "ci_low", "ci_high", "mean_diff"])
    save_json(health, out / "adapter_health_summary.json")
    return {"per_model_rows": master_rows, "health": health}
