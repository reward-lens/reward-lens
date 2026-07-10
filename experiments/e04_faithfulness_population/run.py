"""
E04 — Attribution Faithfulness at population scale (THE spine).

For each (pair, model):
  1. Compute attribution on all sublayer components (attn_L*, mlp_L*).
  2. Compute patching effects under noising mode for the SAME components.
  3. Compute Spearman rho between |attribution| and |patch effect| ON THIS PAIR.

Population stats: distribution of per-pair rho over n>=150 pairs/dim/model.
Stratify by dimension and component subset (attn vs mlp, early vs late).

This is the experiment that supersedes Table 3 of the v1 paper.

Patching is expensive — we do not run it on the full population by default.
Instead we sample ``cfg.extra.get("patching_pairs_per_dim", 30)`` pairs per
dimension per model for full patching, while attribution runs on the full
population. The faithfulness rho is computed ONLY on the patched subset.
The remaining attribution data feeds rank-stability and aggregate
distributions.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..config import ExperimentConfig
from ..utils.io import JsonlWriter, manifest_run, save_json, write_csv
from ..utils.figures import setup_matplotlib, savefig, PALETTE
from ..utils.parallel import tprint, clear_gpu
from ..utils.batching import batch_attribution
from ..utils.diagnostics import load_diagnostic_v2
from ..utils.datasets import load_rewardbench
from ..utils.models import load_reward_model
from ..utils.shared_cache import cached_forward

from reward_lens.statistics import spearman_with_ci, bootstrap_ci, bh_fdr


def _patch_one_pair(rm, p) -> dict:
    """Run noising patching on one pair; return per-component effects.
    Wraps the existing ActivationPatcher; expensive."""
    from reward_lens.patching import ActivationPatcher
    patcher = ActivationPatcher(rm)
    result = patcher.patch_all_components(
        p.prompt, p.preferred, p.dispreferred, mode="noising", show_progress=False,
    )
    return {
        "component_names": result.component_names,
        "patch_effects": result.patch_effects.tolist(),
        "original_differential": float(result.original_differential),
    }


def run(cfg: ExperimentConfig) -> dict:
    out = cfg.out_path
    (out / "figures").mkdir(parents=True, exist_ok=True)
    patching_per_dim = int(cfg.extra.get("patching_pairs_per_dim", 30))

    pairs = list(load_diagnostic_v2(dimensions=cfg.dimensions, limit_per_dim=cfg.n_pairs_per_dim))
    for sub in ("chat", "safety", "reasoning"):
        pairs += load_rewardbench(subset=sub, limit=cfg.n_pairs_per_dim)
    by_dim: dict[str, list] = {}
    for p in pairs:
        by_dim.setdefault(p.dimension, []).append(p)
    tprint(f"[e04] dims: {list(by_dim.keys())}")

    master_rows: list[dict] = []
    for mc in cfg.models:
        short = mc.short_name()
        model_out = out / short
        model_out.mkdir(parents=True, exist_ok=True)
        with manifest_run(model_out, "e04_faithfulness_population", cfg.__dict__,
                          model=mc.name, seed=cfg.seed,
                          swallow_exceptions=cfg.skip_models_on_error):
            try:
                rm = load_reward_model(mc)
            except Exception as e:
                tprint(f"[e04] failed to load {mc.name}: {e}")
                raise

            attr_jsonl = JsonlWriter(model_out / "attribution_per_pair.jsonl")
            patch_jsonl = JsonlWriter(model_out / "patching_per_pair.jsonl")
            faith_jsonl = JsonlWriter(model_out / "faithfulness_per_pair.jsonl")

            # ---- attribution at population scale (batched) ----
            todo_attr = [p for p in pairs if not attr_jsonl.has(f"{p.source}:{p.pair_id}")]
            if todo_attr:
                t0 = time.time()
                cache_w = cached_forward(
                    rm, [(p.prompt, p.preferred) for p in todo_attr],
                    side="preferred", cfg=cfg, model_short=short,
                )
                cache_l = cached_forward(
                    rm, [(p.prompt, p.dispreferred) for p in todo_attr],
                    side="dispreferred", cfg=cfg, model_short=short,
                )
                tprint(f"[e04] {short}: attribution forwards in {time.time()-t0:.1f}s")
                names, types, layer_idxs, contribs_w = batch_attribution(rm, cache_w)
                _, _, _, contribs_l = batch_attribution(rm, cache_l)
                diff = contribs_w - contribs_l
                for i, p in enumerate(todo_attr):
                    attr_jsonl.write({
                        "record_id": f"{p.source}:{p.pair_id}",
                        "source": p.source, "dimension": p.dimension, "pair_id": p.pair_id,
                        "component_names": names,
                        "component_types": types,
                        "differential_contributions": diff[i].tolist(),
                    })

            # ---- patching on a sampled subset per dimension ----
            rng = np.random.default_rng(cfg.seed)
            patched_pairs_by_dim: dict[str, list] = {}
            for dim, ps in by_dim.items():
                idx = np.arange(len(ps))
                rng.shuffle(idx)
                patched_pairs_by_dim[dim] = [ps[i] for i in idx[:patching_per_dim]]

            patch_records_by_pair: dict[str, dict] = {
                r["record_id"]: r for r in patch_jsonl.read_all()
            }
            for dim, ps in patched_pairs_by_dim.items():
                for p in ps:
                    rid = f"{p.source}:{p.pair_id}"
                    if rid in patch_records_by_pair:
                        continue
                    try:
                        result = _patch_one_pair(rm, p)
                    except Exception as e:
                        tprint(f"[e04] patching failed on {rid}: {e}")
                        continue
                    rec = {"record_id": rid, "source": p.source, "dimension": p.dimension,
                           "pair_id": p.pair_id, **result}
                    patch_jsonl.write(rec)
                    patch_records_by_pair[rid] = rec

            # ---- faithfulness rho per (patched) pair ----
            attr_records = {r["record_id"]: r for r in attr_jsonl.read_all()}
            faith_records = {r["record_id"]: r for r in faith_jsonl.read_all()}

            for rid, p_rec in patch_records_by_pair.items():
                if rid in faith_records:
                    continue
                a_rec = attr_records.get(rid)
                if a_rec is None:
                    continue
                attr_by_name = dict(zip(a_rec["component_names"], a_rec["differential_contributions"]))
                xs, ys = [], []
                for cn, eff in zip(p_rec["component_names"], p_rec["patch_effects"]):
                    av = attr_by_name.get(cn)
                    if av is None:
                        continue
                    xs.append(abs(av))
                    ys.append(abs(eff))
                if len(xs) < 5:
                    continue
                spearman = spearman_with_ci(xs, ys, n_resamples=cfg.n_resamples,
                                            seed=cfg.seed, ci=cfg.ci)
                rec = {
                    "record_id": rid,
                    "source": p_rec["source"],
                    "dimension": p_rec["dimension"],
                    "pair_id": p_rec["pair_id"],
                    "spearman_rho": spearman.point,
                    "spearman_ci_low": spearman.ci_low,
                    "spearman_ci_high": spearman.ci_high,
                    "n_components": len(xs),
                }
                faith_jsonl.write(rec)
                faith_records[rid] = rec

            # ---- aggregate per-pair rho distribution per dimension ----
            by_dim_rho: dict[str, list[float]] = {}
            for r in faith_records.values():
                if np.isfinite(r["spearman_rho"]):
                    by_dim_rho.setdefault(r["dimension"], []).append(r["spearman_rho"])
            summary = {"per_dimension": {}}
            p_values = []
            dims_for_fdr = []
            for dim, rhos in by_dim_rho.items():
                arr = np.asarray(rhos)
                m_boot = bootstrap_ci(arr, statistic=np.mean, n_resamples=cfg.n_resamples,
                                       seed=cfg.seed, ci=cfg.ci)
                # one-sample sign-flip vs zero for whether mean rho is non-zero
                rng2 = np.random.default_rng(cfg.seed)
                signs = rng2.choice([-1.0, 1.0], size=(min(cfg.n_resamples, 5000), arr.size))
                replicates = (signs * arr[None, :]).mean(axis=1)
                p_val = float((np.sum(np.abs(replicates) >= abs(arr.mean())) + 1)
                               / (replicates.size + 1))
                p_values.append(p_val)
                dims_for_fdr.append(dim)
                summary["per_dimension"][dim] = {
                    "n": int(arr.size),
                    "mean_rho": float(arr.mean()),
                    "median_rho": float(np.median(arr)),
                    "mean_ci": [m_boot.ci_low, m_boot.ci_high],
                    "p5": float(np.quantile(arr, 0.05)),
                    "p95": float(np.quantile(arr, 0.95)),
                    "frac_negative": float((arr < 0).mean()),
                    "p_value": p_val,
                }
                master_rows.append({
                    "model": short, "dimension": dim, "n": int(arr.size),
                    "mean_rho": float(arr.mean()), "ci_low": m_boot.ci_low,
                    "ci_high": m_boot.ci_high, "frac_negative": float((arr < 0).mean()),
                    "p_value": p_val,
                })
            if p_values:
                rejected, q = bh_fdr(p_values, alpha=0.05)
                for i, dim in enumerate(dims_for_fdr):
                    summary["per_dimension"][dim]["q_value"] = float(q[i])
                    summary["per_dimension"][dim]["significant_fdr05"] = bool(rejected[i])
                    for row in master_rows:
                        if row["model"] == short and row["dimension"] == dim:
                            row["q_value"] = float(q[i])
                            row["significant_fdr05"] = bool(rejected[i])
            save_json(summary, model_out / "faithfulness_summary.json")
            _plot_violin(short, by_dim_rho, model_out / "figures" / f"e04_rho_violin_{short}")
            del rm
            clear_gpu()

    write_csv(master_rows, out / "e04_faithfulness.csv")
    return {"rows": master_rows}


def _plot_violin(model_short: str, by_dim: dict[str, list[float]], path: Path) -> None:
    setup_matplotlib()
    import matplotlib.pyplot as plt
    dims = sorted(by_dim.keys())
    if not dims:
        return
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(dims)), 5))
    data = [np.asarray(by_dim[d]) for d in dims]
    parts = ax.violinplot(data, positions=range(len(dims)), showmeans=True, showmedians=True)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(dims, rotation=30, ha="right")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("per-pair Spearman ρ (|attr| vs |patch|)")
    ax.set_title(f"E04 faithfulness (per-pair rho) — {model_short}")
    savefig(fig, path)
