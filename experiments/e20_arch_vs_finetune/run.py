"""
E20 — Architecture vs fine-tune decomposition (3-way Skywork-family compare).

The deep_analysisv1 §5.4 finding: the original paper's "single-scalar
vs multi-objective" attribution claims were derived from a 2-model design
(Skywork single-scalar vs ArmoRM multi-objective). The 3-Skywork-family
run (Llama-orig + Llama-v0.2 + Gemma-27B) showed that within the
single-scalar family, two architectures (Llama vs Gemma) differ on
attribution mix as much as the paper attributes to objective structure
(Llama 85% MLP routing, Gemma 87% attention routing). The paper's
architectural attribution is therefore **under-determined** — it cannot
distinguish "objective structure" effects from "Llama vs Gemma" effects.

This experiment makes that decomposition explicit. With 3 models that
differ on two axes (architecture × fine-tune):

    model        architecture   fine-tune
    -----------  -------------- ---------
    Llama-orig   Llama          orig-RL
    Llama-v0.2   Llama          v0.2-RL
    Gemma-27B    Gemma          v0.2-RL

we can compute, for any per-model metric m:
  - architecture_effect:  m(Llama-v0.2) - m(Gemma-v0.2)   (controls fine-tune)
  - fine_tune_effect:     m(Llama-v0.2) - m(Llama-orig)   (controls arch)
  - interaction:          architecture_effect - corresponding fine-tune effect

The output JSON reports both effect magnitudes per dimension/concept/
component, plus a "majority attribution" verdict for each metric:

  - "architecture-dominated" if |arch_effect| > 2 × |finetune_effect|
  - "finetune-dominated"     if |finetune_effect| > 2 × |arch_effect|
  - "mixed"                  otherwise

This is the experiment that lets a v2 paper say "the attribution-mix
divergence is *architectural*, not objective-structure" with evidence
rather than hand-waving.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import ExperimentConfig, ModelConfig
from ..utils.io import manifest_run, save_json, write_csv
from ..utils.figures import setup_matplotlib, savefig, PALETTE
from ..utils.parallel import tprint, clear_gpu
from ..utils.diagnostics import load_diagnostic_v2
from ..utils.models import load_reward_model
from ..utils.batching import batch_attribution
from ..utils.shared_cache import cached_forward


def _per_model_metrics(rm, pairs: list, *, model_short: str,
                        cfg: ExperimentConfig) -> dict:
    """Compute the per-model summary used as the input to E20's
    decomposition. We deliberately keep this small — three numbers per
    component is enough to cleanly separate "architecture moved this"
    from "fine-tune moved this"."""
    cache_w = cached_forward(rm, [(p.prompt, p.preferred) for p in pairs],
                             side="preferred", cfg=cfg, model_short=model_short)
    cache_l = cached_forward(rm, [(p.prompt, p.dispreferred) for p in pairs],
                             side="dispreferred", cfg=cfg, model_short=model_short)
    names, types, layer_idxs, contribs_w = batch_attribution(rm, cache_w)
    _,     _,     _,         contribs_l = batch_attribution(rm, cache_l)
    contrib_diff = contribs_w - contribs_l  # (B, C)

    # Per-component magnitude (mean |contribution| across pairs).
    mag = np.abs(contrib_diff).mean(axis=0)

    # Per-component-type fraction in top-15 (this is the §4.3 "attn vs
    # mlp" headline number that flipped between Llama and Gemma).
    order = np.argsort(mag)[::-1][:15]
    top_types = [types[i] for i in order]
    type_fractions = {t: top_types.count(t) / max(1, len(top_types))
                      for t in set(types)}

    # Per-dim mean differential at the final layer.
    rewards_w = cache_w.rewards.detach().cpu().numpy()
    rewards_l = cache_l.rewards.detach().cpu().numpy()
    diffs_by_dim: dict[str, list[float]] = {}
    for i, p in enumerate(pairs):
        diffs_by_dim.setdefault(p.dimension, []).append(float(rewards_w[i] - rewards_l[i]))
    mean_diff = {d: float(np.mean(v)) for d, v in diffs_by_dim.items()}

    return {
        "component_names": names,
        "component_mag": mag,
        "type_fractions_in_top15": type_fractions,
        "mean_diff_per_dim": mean_diff,
        "n_layers": rm.n_layers,
        "d_model": rm.d_model,
    }


def run(cfg: ExperimentConfig) -> dict:
    out = cfg.out_path
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # Pull the three Skywork-family models. Allow override via cfg.extra
    # for users who have a different 2x2 design they want to decompose.
    arch_finetune_map: dict[tuple[str, str], str] = cfg.extra.get(
        "arch_finetune_map",
        {
            ("Llama", "orig"): "Skywork/Skywork-Reward-Llama-3.1-8B",
            ("Llama", "v02"):  "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
            ("Gemma", "v02"):  "Skywork/Skywork-Reward-Gemma-2-27B-v0.2",
        },
    )
    if cfg.models:
        # If the orchestrator passed cfg.models, they take precedence;
        # we attempt to identify each by name.
        def _classify(name: str) -> tuple[str, str]:
            n = name.lower()
            arch = "Gemma" if "gemma" in n else "Llama"
            ft = "v02" if "v0.2" in n or "v02" in n else "orig"
            return (arch, ft)
        arch_finetune_map = {_classify(mc.name): mc.name for mc in cfg.models}

    pairs = list(load_diagnostic_v2(dimensions=cfg.dimensions,
                                     limit_per_dim=cfg.n_pairs_per_dim))
    if not pairs:
        tprint("[e20] no diagnostic_v2 pairs available; aborting")
        return {"summary": {}}

    metrics_by_cell: dict[tuple[str, str], dict] = {}
    for (arch, ft), model_id in arch_finetune_map.items():
        mc = ModelConfig(name=model_id)
        short = mc.short_name()
        model_out = out / short
        model_out.mkdir(parents=True, exist_ok=True)
        with manifest_run(model_out, "e20_arch_vs_finetune", cfg.__dict__,
                          model=model_id, seed=cfg.seed,
                          swallow_exceptions=cfg.skip_models_on_error):
            try:
                rm = load_reward_model(mc)
            except Exception as e:
                tprint(f"[e20] {short} ({arch},{ft}) load failed: {e}")
                raise
            metrics_by_cell[(arch, ft)] = _per_model_metrics(
                rm, pairs, model_short=short, cfg=cfg)
            del rm
            clear_gpu()

    cells = list(metrics_by_cell.keys())
    if len(cells) < 3:
        tprint(f"[e20] need 3 cells, got {len(cells)}: {cells} — cannot decompose")
        save_json({"cells_loaded": cells, "decomposition": None},
                  out / "e20_summary.json")
        return {"summary": {"cells": cells}}

    # Decomposition. The "anchor" cell is (Llama, v02) — the cell that
    # all three of (orig, v02, Gemma) can reach by changing exactly one
    # axis.
    anchor = ("Llama", "v02")
    if anchor not in metrics_by_cell:
        # Fall back to the first cell that shares an arch with another
        # cell on the orthogonal axis.
        anchor = cells[0]

    arch_partner = next(((a, f) for (a, f) in cells
                         if f == anchor[1] and a != anchor[0]), None)
    ft_partner = next(((a, f) for (a, f) in cells
                       if a == anchor[0] and f != anchor[1]), None)
    if arch_partner is None or ft_partner is None:
        tprint(f"[e20] partners missing — anchor={anchor}, arch_partner={arch_partner}, "
               f"ft_partner={ft_partner}")
        save_json({"cells_loaded": cells, "decomposition": None,
                   "anchor": anchor, "arch_partner": arch_partner,
                   "ft_partner": ft_partner}, out / "e20_summary.json")
        return {"summary": {"cells": cells}}

    anchor_m = metrics_by_cell[anchor]
    arch_m = metrics_by_cell[arch_partner]
    ft_m = metrics_by_cell[ft_partner]

    # --- attribution-type fractions (§4.3 headline) ---
    types = sorted(set(anchor_m["type_fractions_in_top15"].keys())
                   | set(arch_m["type_fractions_in_top15"].keys())
                   | set(ft_m["type_fractions_in_top15"].keys()))
    type_rows = []
    for t in types:
        anc = anchor_m["type_fractions_in_top15"].get(t, 0.0)
        arc = arch_m["type_fractions_in_top15"].get(t, 0.0)
        fnt = ft_m["type_fractions_in_top15"].get(t, 0.0)
        arch_eff = anc - arc        # anchor minus arch-partner = arch effect
        ft_eff   = anc - fnt        # anchor minus ft-partner   = ft effect
        verdict = _verdict(arch_eff, ft_eff)
        type_rows.append({
            "component_type": t,
            "fraction_anchor": anc, "fraction_arch_partner": arc,
            "fraction_ft_partner": fnt,
            "arch_effect": float(arch_eff), "ft_effect": float(ft_eff),
            "verdict": verdict,
        })

    # --- mean per-dim differential ---
    dims = sorted(set(anchor_m["mean_diff_per_dim"].keys())
                  | set(arch_m["mean_diff_per_dim"].keys())
                  | set(ft_m["mean_diff_per_dim"].keys()))
    dim_rows = []
    for dim in dims:
        anc = anchor_m["mean_diff_per_dim"].get(dim, float("nan"))
        arc = arch_m["mean_diff_per_dim"].get(dim, float("nan"))
        fnt = ft_m["mean_diff_per_dim"].get(dim, float("nan"))
        arch_eff = anc - arc
        ft_eff   = anc - fnt
        dim_rows.append({
            "dimension": dim,
            "diff_anchor": anc, "diff_arch_partner": arc, "diff_ft_partner": fnt,
            "arch_effect": float(arch_eff), "ft_effect": float(ft_eff),
            "verdict": _verdict(arch_eff, ft_eff),
        })

    # --- per-component magnitude (only when arch-partners share d_model
    # AND component schema; otherwise skip with a note) ---
    component_rows = []
    if (anchor_m["component_names"] == ft_m["component_names"]):
        # ft comparison is dimensionally safe (same arch).
        for i, name in enumerate(anchor_m["component_names"]):
            anc = float(anchor_m["component_mag"][i])
            fnt = float(ft_m["component_mag"][i])
            component_rows.append({
                "component": name,
                "mag_anchor": anc, "mag_ft_partner": fnt,
                "ft_effect": anc - fnt,
                "verdict": "finetune-eligible (cross-arch comparison skipped)",
            })

    summary = {
        "cells_loaded": [list(c) for c in cells],
        "anchor": list(anchor),
        "arch_partner": list(arch_partner),
        "ft_partner": list(ft_partner),
        "anchor_id": arch_finetune_map[anchor],
        "arch_partner_id": arch_finetune_map[arch_partner],
        "ft_partner_id": arch_finetune_map[ft_partner],
        "n_pairs": len(pairs),
        "type_fraction_decomposition": type_rows,
        "per_dim_diff_decomposition": dim_rows,
    }
    save_json(summary, out / "e20_summary.json")
    write_csv(type_rows, out / "e20_attribution_type_decomposition.csv")
    write_csv(dim_rows, out / "e20_per_dim_diff_decomposition.csv")
    write_csv(component_rows, out / "e20_component_finetune_delta.csv")

    _plot_decomposition(type_rows, dim_rows, out / "figures" / "e20_decomp")
    return {"summary": summary}


def _verdict(arch_eff: float, ft_eff: float) -> str:
    a, f = abs(arch_eff), abs(ft_eff)
    if a > 2 * f and a > 0.05:
        return "architecture-dominated"
    if f > 2 * a and f > 0.05:
        return "finetune-dominated"
    if a < 0.05 and f < 0.05:
        return "neither (both effects below noise)"
    return "mixed"


def _plot_decomposition(type_rows: list, dim_rows: list, path) -> None:
    setup_matplotlib()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if type_rows:
        ax = axes[0]
        labels = [r["component_type"] for r in type_rows]
        arch = [r["arch_effect"] for r in type_rows]
        ft = [r["ft_effect"] for r in type_rows]
        x = np.arange(len(labels))
        w = 0.4
        ax.bar(x - w / 2, arch, w, label="architecture effect", color=PALETTE[0])
        ax.bar(x + w / 2, ft,   w, label="fine-tune effect",    color=PALETTE[1])
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel("Δ fraction in top-15")
        ax.set_title("Attribution-mix decomposition")
        ax.legend()

    if dim_rows:
        ax = axes[1]
        labels = [r["dimension"] for r in dim_rows]
        arch = [r["arch_effect"] for r in dim_rows]
        ft = [r["ft_effect"] for r in dim_rows]
        x = np.arange(len(labels))
        w = 0.4
        ax.bar(x - w / 2, arch, w, label="architecture effect", color=PALETTE[0])
        ax.bar(x + w / 2, ft,   w, label="fine-tune effect",    color=PALETTE[1])
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel("Δ mean per-pair differential")
        ax.set_title("Per-dimension differential decomposition")
        ax.legend()

    fig.suptitle("E20 architecture vs fine-tune decomposition")
    fig.tight_layout()
    savefig(fig, path)
