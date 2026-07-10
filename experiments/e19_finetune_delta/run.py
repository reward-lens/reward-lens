"""
E19 — Fine-tune delta between two same-architecture reward models.

The deep_analysisv1 campaign surfaced a finding the original paper
cannot make because it only trained on the v0.2 fine-tune: ``Cohen's d
on the repetition probe flipped from +0.49 (Skywork-Reward-Llama-3.1-8B,
the orig-RL fine-tune) to −3.97 (the v0.2 fine-tune)``. That's the
v0.2 second-stage RL *introducing* a new bias, not propagating one.

This experiment makes the cross-fine-tune comparison a first-class
result. Given two model configs (orig + v0.2 by default), it computes:

  1. Per-hacking-dim Cohen's *d* delta (was the bias introduced or
     amplified by the second-stage fine-tune?)
  2. Per-concept dose-response slope delta (e08 numbers, with shared
     prompts so the deltas are paired).
  3. Per-component attribution-rank delta (which components moved up or
     down in the top-15 between checkpoints?).
  4. Per-dim crystallisation-depth delta (does the v0.2 fine-tune push
     decisions earlier or later in the residual stream?).
  5. Reward-direction angle (cosine between ``w_r`` of the two models;
     same architecture so this is dimensionally well-defined unlike the
     §3.7 cross-architecture concept comparison).

Output schema (per JSON):
  - ``per_hacking_dim``: list of {dim, d_orig, d_v02, delta}
  - ``per_concept``:    list of {concept, slope_orig, slope_v02, delta}
  - ``per_dim_crystal``: list of {dim, crystal_orig, crystal_v02, delta}
  - ``component_rank``:  list of {component, rank_orig, rank_v02, rank_delta}
  - ``reward_direction_cosine``: scalar
  - ``summary``: top movers by absolute delta in each category

Defaults compare ``Skywork/Skywork-Reward-Llama-3.1-8B`` (orig) and
``Skywork/Skywork-Reward-Llama-3.1-8B-v0.2``. Override via:
  cfg.extra["model_a"] = "<HF id>"     # baseline
  cfg.extra["model_b"] = "<HF id>"     # comparison
The two models must share architecture (same d_model). ArmoRM cannot be
compared with Skywork via ``reward_direction_cosine`` because their
heads live in incompatible spaces; ``model_a`` × ``model_b`` is checked
for dimensional compatibility before the cosine is computed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..config import ExperimentConfig, ModelConfig
from ..utils.io import manifest_run, save_json, write_csv
from ..utils.figures import setup_matplotlib, savefig, PALETTE
from ..utils.parallel import tprint, clear_gpu
from ..utils.diagnostics import load_diagnostic_v2
from ..utils.models import load_reward_model
from ..utils.batching import batch_attribution, batch_lens_curves
from ..utils.shared_cache import cached_forward


def _crystal_frac(diff_curve: np.ndarray) -> float:
    """Fraction-of-depth at which |diff| first reaches 50% of |final|."""
    final = diff_curve[-1]
    if not np.isfinite(final) or abs(final) < 1e-8:
        return float("nan")
    threshold = 0.5 * final
    for i, d in enumerate(diff_curve):
        if (final > 0 and d >= threshold) or (final < 0 and d <= threshold):
            return i / max(1, len(diff_curve) - 1)
    return 1.0


def _model_payload(rm, pairs: list, *, model_short: str, cfg: ExperimentConfig) -> dict:
    """Compute everything we need from one model in two batched forwards."""
    cache_w = cached_forward(rm, [(p.prompt, p.preferred) for p in pairs],
                             side="preferred", cfg=cfg, model_short=model_short)
    cache_l = cached_forward(rm, [(p.prompt, p.dispreferred) for p in pairs],
                             side="dispreferred", cfg=cfg, model_short=model_short)

    # Lens curves (B, L+1) → per-pair crystallization
    lens_w = batch_lens_curves(rm, cache_w)
    lens_l = batch_lens_curves(rm, cache_l)
    diffs = lens_w - lens_l
    crystals = np.array([_crystal_frac(d) for d in diffs])

    # Attribution (B, C) → component contribution magnitudes
    names, types, layer_idxs, contribs_w = batch_attribution(rm, cache_w)
    _,     _,     _,         contribs_l = batch_attribution(rm, cache_l)
    contrib_diff = contribs_w - contribs_l

    return {
        "rewards_w": cache_w.rewards.detach().cpu().numpy(),
        "rewards_l": cache_l.rewards.detach().cpu().numpy(),
        "diffs": diffs,
        "crystals": crystals,
        "component_names": names,
        "component_diff": contrib_diff,
        "w_r": rm.reward_direction.detach().cpu().float().numpy(),
        "n_layers": rm.n_layers,
        "d_model": rm.d_model,
    }


def _hacking_d(rm, dim_to_probes: dict, *, max_length: int) -> dict:
    """Run the e06 hacking detector inline. Returns dict[dim] = Cohen's d."""
    from reward_lens.hacking import HackingDetector
    detector = HackingDetector(rm)
    out: dict[str, float] = {}
    for dim, probes in dim_to_probes.items():
        try:
            r = detector._run_bias_test(
                dim, probes, max_length=max_length,
                n_resamples=200, ci=0.95, seed=0,
            )
            out[dim] = float(r.effect_size)
        except Exception:
            out[dim] = float("nan")
    return out


def _concept_slopes(rm, concept_pairs: dict, held_out: list,
                    *, max_length: int) -> dict:
    """Compute per-concept dose-response slopes (a stripped-down e08)."""
    from reward_lens.concepts import ConceptExtractor
    extractor = ConceptExtractor(rm)
    vecs = extractor.extract_concepts(concept_pairs)
    strengths = [-2, -1, 0, 1, 2]
    slopes: dict[str, float] = {}
    layers_list = rm.adapter.get_layers(rm.model)
    inject_layer_idx = max(0, rm.n_layers - 3)
    layer_module = layers_list[inject_layer_idx]
    for cname, vec in vecs.items():
        per_prompt: list[float] = []
        v_gpu = vec.to(rm.device).float()
        for prompt, resp in held_out:
            ys = []
            for s in strengths:
                if s == 0:
                    ys.append(0.0)
                    continue
                def make_hook(strength=s):
                    def _hook(module, _inp, out):
                        h = rm.adapter.extract_layer_output(out)
                        delta = (strength * v_gpu).to(h.dtype)
                        h[:, -1, :] = h[:, -1, :] + delta
                        if isinstance(out, tuple):
                            return (h,) + out[1:]
                        return h
                    return _hook
                handle = layer_module.register_forward_hook(make_hook())
                try:
                    base = float(rm.score(prompt, resp, max_length=max_length))
                except Exception:
                    base = float("nan")
                handle.remove()
                ys.append(base)
            xs = np.asarray(strengths, dtype=np.float64)
            ys_a = np.asarray(ys, dtype=np.float64)
            if np.all(np.isfinite(ys_a)):
                per_prompt.append(float(np.polyfit(xs, ys_a, 1)[0]))
        slopes[cname] = float(np.mean(per_prompt)) if per_prompt else float("nan")
    return slopes


def run(cfg: ExperimentConfig) -> dict:
    out = cfg.out_path
    (out / "figures").mkdir(parents=True, exist_ok=True)

    model_a_id = cfg.extra.get(
        "model_a", "Skywork/Skywork-Reward-Llama-3.1-8B")
    model_b_id = cfg.extra.get(
        "model_b", "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
    # Honour user-supplied two-model config if it overrides the defaults.
    if len(cfg.models) >= 2:
        model_a = cfg.models[0]
        model_b = cfg.models[1]
    else:
        model_a = ModelConfig(name=model_a_id)
        model_b = ModelConfig(name=model_b_id)
    short_a = model_a.short_name()
    short_b = model_b.short_name()

    pairs = list(load_diagnostic_v2(dimensions=cfg.dimensions,
                                     limit_per_dim=cfg.n_pairs_per_dim))
    if not pairs:
        tprint("[e19] no diagnostic_v2 pairs available; aborting")
        return {"rows": []}

    # Hacking probes — load the same set the e06 runner uses.
    from reward_lens.hacking import ALL_TESTS
    dim_to_probes = {dim: tests[:cfg.extra.get("hacking_probes_per_dim", 30)]
                     for dim, tests in ALL_TESTS.items()}

    # Concept setup — small but diverse held-out for slope robustness.
    from reward_lens.concepts import CONCEPT_PAIRS
    held_out = [
        ("What is X?", "X is defined as Y."),
        ("How do I do Z?", "Step 1: do A. Step 2: do B."),
        ("Explain T briefly.", "T is a process that involves U and V."),
        ("Define W.", "W means having property P."),
    ][:cfg.extra.get("concept_held_out", 4)]

    payload_by_short: dict[str, dict] = {}
    hacking_by_short: dict[str, dict] = {}
    concepts_by_short: dict[str, dict] = {}

    for mc in (model_a, model_b):
        short = mc.short_name()
        model_out = out / short
        model_out.mkdir(parents=True, exist_ok=True)
        with manifest_run(model_out, "e19_finetune_delta", cfg.__dict__,
                          model=mc.name, seed=cfg.seed,
                          swallow_exceptions=cfg.skip_models_on_error):
            try:
                rm = load_reward_model(mc)
            except Exception as e:
                tprint(f"[e19] {short} load failed: {e}")
                raise
            payload_by_short[short] = _model_payload(
                rm, pairs, model_short=short, cfg=cfg)
            hacking_by_short[short] = _hacking_d(
                rm, dim_to_probes, max_length=cfg.max_length)
            concepts_by_short[short] = _concept_slopes(
                rm, CONCEPT_PAIRS, held_out, max_length=cfg.max_length)
            del rm
            clear_gpu()

    if short_a not in payload_by_short or short_b not in payload_by_short:
        tprint("[e19] one or both models failed to load; cannot compute delta")
        return {"rows": []}

    pa = payload_by_short[short_a]
    pb = payload_by_short[short_b]

    # ---- 1. Per-hacking-dim Cohen's d delta ----
    hacking_rows = []
    for dim in sorted(set(hacking_by_short[short_a]) | set(hacking_by_short[short_b])):
        da = hacking_by_short[short_a].get(dim, float("nan"))
        db = hacking_by_short[short_b].get(dim, float("nan"))
        hacking_rows.append({
            "dimension": dim,
            f"d_{short_a}": float(da),
            f"d_{short_b}": float(db),
            "delta": float(db - da),
        })

    # ---- 2. Per-concept slope delta ----
    concept_rows = []
    for cname in sorted(set(concepts_by_short[short_a]) | set(concepts_by_short[short_b])):
        sa = concepts_by_short[short_a].get(cname, float("nan"))
        sb = concepts_by_short[short_b].get(cname, float("nan"))
        concept_rows.append({
            "concept": cname,
            f"slope_{short_a}": float(sa),
            f"slope_{short_b}": float(sb),
            "delta": float(sb - sa),
        })

    # ---- 3. Per-dim crystallisation depth delta ----
    dim_rows = []
    by_dim_a: dict[str, list] = {}
    by_dim_b: dict[str, list] = {}
    for i, p in enumerate(pairs):
        by_dim_a.setdefault(p.dimension, []).append(pa["crystals"][i])
        by_dim_b.setdefault(p.dimension, []).append(pb["crystals"][i])
    for dim in sorted(by_dim_a):
        ca = float(np.nanmean(by_dim_a[dim]))
        cb = float(np.nanmean(by_dim_b[dim]))
        dim_rows.append({
            "dimension": dim,
            f"crystal_{short_a}": ca,
            f"crystal_{short_b}": cb,
            "delta": float(cb - ca),
        })

    # ---- 4. Per-component attribution rank delta ----
    # Average |contribution| across pairs to get a per-component magnitude,
    # then rank within each model and report the rank delta.
    comp_rows = []
    if pa["component_names"] == pb["component_names"]:
        names = pa["component_names"]
        mag_a = np.abs(pa["component_diff"]).mean(axis=0)
        mag_b = np.abs(pb["component_diff"]).mean(axis=0)
        # Higher magnitude = lower rank (rank 1 = strongest).
        rank_a = np.argsort(np.argsort(-mag_a)) + 1
        rank_b = np.argsort(np.argsort(-mag_b)) + 1
        for i, name in enumerate(names):
            comp_rows.append({
                "component": name,
                f"mag_{short_a}": float(mag_a[i]),
                f"mag_{short_b}": float(mag_b[i]),
                f"rank_{short_a}": int(rank_a[i]),
                f"rank_{short_b}": int(rank_b[i]),
                "rank_delta": int(rank_b[i] - rank_a[i]),
            })

    # ---- 5. Reward-direction cosine ----
    cos = float("nan")
    if pa["d_model"] == pb["d_model"]:
        wa = pa["w_r"]; wb = pb["w_r"]
        cos = float(wa @ wb / (np.linalg.norm(wa) * np.linalg.norm(wb) + 1e-12))

    # ---- Top movers per category ----
    def _top_n(rows: list[dict], key: str, n: int = 5) -> list[dict]:
        sorted_rows = sorted(rows, key=lambda r: -abs(r.get(key, 0.0)))
        return sorted_rows[:n]

    summary = {
        "model_a": model_a.name, "model_b": model_b.name,
        "reward_direction_cosine": cos,
        "n_pairs": len(pairs),
        "n_hacking_dims": len(hacking_rows),
        "top_hacking_movers":   _top_n(hacking_rows, "delta"),
        "top_concept_movers":   _top_n(concept_rows, "delta"),
        "top_crystal_movers":   _top_n(dim_rows, "delta"),
        "top_component_movers": _top_n(comp_rows, "rank_delta"),
    }

    save_json(summary, out / "e19_summary.json")
    write_csv(hacking_rows, out / "e19_hacking_delta.csv")
    write_csv(concept_rows, out / "e19_concept_delta.csv")
    write_csv(dim_rows, out / "e19_crystal_delta.csv")
    write_csv(comp_rows, out / "e19_component_delta.csv")

    _plot_top_movers(summary, out / "figures" / "e19_top_movers")
    return {"summary": summary,
            "hacking": hacking_rows, "concept": concept_rows,
            "crystal": dim_rows, "components": comp_rows}


def _plot_top_movers(summary: dict, path) -> None:
    setup_matplotlib()
    import matplotlib.pyplot as plt
    sections = [
        ("hacking d Δ",     summary["top_hacking_movers"], "dimension", "delta"),
        ("concept slope Δ", summary["top_concept_movers"], "concept",   "delta"),
        ("crystal depth Δ", summary["top_crystal_movers"], "dimension", "delta"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (title, rows, label_key, val_key) in zip(axes, sections):
        if not rows:
            ax.set_title(title + " (no data)")
            continue
        labels = [r[label_key] for r in rows]
        vals = [r[val_key] for r in rows]
        colors = [PALETTE[0] if v >= 0 else PALETTE[2] for v in vals]
        ax.barh(range(len(labels)), vals, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_title(title)
    fig.suptitle(f"E19 fine-tune delta: {summary['model_a']} → {summary['model_b']}",
                 fontsize=10)
    fig.tight_layout()
    savefig(fig, path)
