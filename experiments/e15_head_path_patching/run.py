"""
E15 (merged with E16) — Head-level path patching.

Two stages:
  1. Head-level direct patching: identify the top-K heads per dimension on
     each model. If a single head dominates a dimension's causal effect,
     that's a reportable finding ("a sycophancy head").
  2. Path patching at head granularity: for a small set of (sender_head ->
     receiver) hypotheses derived from stage 1 and the population
     attribution data, measure path effects. Compare against direct
     patching to find paths whose path effect exceeds direct effect (i.e.
     two-hop circuits not visible at sublayer resolution).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import ExperimentConfig
from ..utils.io import JsonlWriter, manifest_run, save_json, write_csv
from ..utils.figures import setup_matplotlib, savefig, PALETTE
from ..utils.parallel import tprint, clear_gpu
from ..utils.diagnostics import load_diagnostic_v2
from ..utils.models import load_reward_model


def run(cfg: ExperimentConfig) -> dict:
    out = cfg.out_path
    (out / "figures").mkdir(parents=True, exist_ok=True)
    pairs_per_dim = int(cfg.extra.get("pairs_per_dim", 5))
    top_k_heads = int(cfg.extra.get("top_k_heads", 10))
    n_path_hypotheses = int(cfg.extra.get("n_path_hypotheses", 10))

    pairs_by_dim = {}
    for d in (cfg.dimensions or ["helpfulness", "safety", "sycophancy"]):
        ps = load_diagnostic_v2([d], limit_per_dim=pairs_per_dim)
        pairs_by_dim[d] = ps[:pairs_per_dim]

    master_rows = []
    path_rows = []
    for mc in cfg.models:
        short = mc.short_name()
        model_out = out / short
        model_out.mkdir(parents=True, exist_ok=True)
        with manifest_run(model_out, "e15_head_path_patching", cfg.__dict__,
                          model=mc.name, seed=cfg.seed,
                          swallow_exceptions=cfg.skip_models_on_error):
            try:
                rm = load_reward_model(mc)
            except Exception as e:
                tprint(f"[e15] load failed: {e}")
                raise

            from reward_lens.patching import ActivationPatcher
            from reward_lens.path_patching import PathPatcher

            patcher = ActivationPatcher(rm)
            path_patcher = PathPatcher(rm)

            # Stage 1 — head-level direct patching
            head_jsonl = JsonlWriter(model_out / "head_patching_per_pair.jsonl")
            for dim, ps in pairs_by_dim.items():
                for p in ps:
                    rid = f"{p.source}:{p.pair_id}:head"
                    if head_jsonl.has(rid):
                        continue
                    try:
                        result = patcher.patch_all_heads(
                            p.prompt, p.preferred, p.dispreferred,
                            mode="noising", show_progress=False,
                        )
                        rec = {
                            "record_id": rid, "dimension": dim,
                            "component_names": result.component_names,
                            "patch_effects": result.patch_effects.tolist(),
                            "original_differential": float(result.original_differential),
                        }
                    except Exception as e:
                        rec = {"record_id": rid, "error": str(e)}
                    head_jsonl.write(rec)

            # Aggregate top heads per dimension
            head_records = head_jsonl.read_all()
            top_heads_by_dim: dict[str, list[tuple[str, float]]] = {}
            for dim in pairs_by_dim:
                effects: dict[str, list[float]] = {}
                for r in head_records:
                    if r.get("dimension") != dim or "patch_effects" not in r:
                        continue
                    for n, e in zip(r["component_names"], r["patch_effects"]):
                        effects.setdefault(n, []).append(abs(e))
                ranked = sorted(effects.items(), key=lambda kv: np.mean(kv[1]) if kv[1] else 0.0,
                                 reverse=True)
                top_heads_by_dim[dim] = [(n, float(np.mean(es))) for n, es in ranked[:top_k_heads]]
                for name, mean_eff in top_heads_by_dim[dim]:
                    master_rows.append({
                        "model": short, "dimension": dim, "head": name,
                        "mean_abs_effect": mean_eff,
                    })

            save_json(top_heads_by_dim, model_out / "top_heads_by_dim.json")

            # Stage 2 — head-level path patching
            # Build (sender, receiver) hypotheses: take top heads as senders,
            # pair with downstream MLPs that appear in the per-pair top-15 of
            # E03/E04 attributions (if available). Otherwise just pair with
            # MLPs roughly 1/3 deeper than the sender.
            n_layers = rm.n_layers
            for dim, top in top_heads_by_dim.items():
                p = pairs_by_dim[dim][0] if pairs_by_dim[dim] else None
                if p is None:
                    continue
                ct = 0
                for head_name, _ in top:
                    if not head_name.startswith("head_L"):
                        continue
                    parts = head_name.split("_")
                    sender_layer = int(parts[1][1:])
                    sender_head = int(parts[2][1:])
                    receiver_layer = min(n_layers - 1,
                                          sender_layer + max(1, (n_layers - sender_layer) // 3))
                    if receiver_layer <= sender_layer:
                        continue
                    try:
                        result = path_patcher.patch(
                            p.prompt, p.preferred, p.dispreferred,
                            sender=("head", sender_layer, sender_head),
                            receiver=("mlp", receiver_layer, None),
                            mode="noising",
                        )
                        path_rows.append({
                            "model": short, "dimension": dim,
                            "sender": head_name,
                            "receiver": f"mlp_L{receiver_layer}",
                            "path_effect": float(result.path_effect),
                            "original_diff": float(result.original_differential),
                        })
                    except Exception as e:
                        path_rows.append({
                            "model": short, "dimension": dim,
                            "sender": head_name,
                            "receiver": f"mlp_L{receiver_layer}",
                            "error": str(e),
                        })
                    ct += 1
                    if ct >= n_path_hypotheses:
                        break

            del rm
            clear_gpu()

    write_csv(master_rows, out / "e15_top_heads.csv")
    write_csv(path_rows, out / "e15_path_effects.csv")
    return {"top_heads": master_rows, "paths": path_rows}
