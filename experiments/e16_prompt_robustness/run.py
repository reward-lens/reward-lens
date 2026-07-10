"""
E16 — Prompt robustness probe.

A reward model is "robust" if minor, semantically-equivalent paraphrases
of the prompt do not move the reward by more than the within-pair
preference signal. This is a missing-from-the-paper question with two
practical motivations:

  1. The hacking detector (e06) looks for *deliberate* surface bias
     (length, formatting, sycophancy). Prompt robustness measures
     *accidental* surface sensitivity — does the model give the same
     score to "What is X?" and "Could you tell me X?" for the same
     response? If not, RewardBench-style accuracy numbers are more
     fragile than reported.
  2. It surfaces a class of OOD failures that crystal/attribution can't:
     a model can have textbook-quality lens curves but be 0.3-σ unstable
     under paraphrase, which would make any downstream PPO pipeline
     unreliable.

For each diagnostic_v2 pair, we generate K paraphrases of the prompt
via a fixed mutation set (no extra LLM dependency — these are
template-based and conservative). We re-score the unchanged response
on each paraphrase and report:

  - mean / std of the reward across paraphrases (per pair)
  - paraphrase-vs-baseline correlation across pairs (per dimension)
  - "robust fraction": fraction of pairs where the paraphrase-induced
    reward σ is smaller than the pair's preferred-vs-dispreferred gap

Output schema:
  - ``robust_per_pair.jsonl``: per-pair {pair_id, dim, baseline_reward,
    paraphrase_rewards, paraphrase_std, signal_to_noise}
  - ``e16_robustness.csv``: per-dimension robust-fraction + mean σ
  - ``e16_summary.json``: per-model rollup
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


# Conservative paraphrase mutations. Each is a (prefix, suffix) pair
# applied to the prompt only; the response is unchanged. Keeps the
# *semantic content* identical while changing surface form.
_MUTATIONS = [
    ("",                        ""),  # baseline (no mutation)
    ("Could you tell me — ",    ""),
    ("Hi! ",                    ""),
    ("Quick question: ",        ""),
    ("Hey, ",                   " Thanks!"),
    ("",                        " Please be brief."),
    ("Please answer: ",         ""),
    ("",                        " (One sentence is fine.)"),
]


def _paraphrase(prompt: str, mut: tuple[str, str]) -> str:
    pre, suf = mut
    if pre and not pre.endswith(" "):
        pre = pre + " "
    return f"{pre}{prompt}{suf}"


def run(cfg: ExperimentConfig) -> dict:
    out = cfg.out_path
    (out / "figures").mkdir(parents=True, exist_ok=True)

    pairs = list(load_diagnostic_v2(dimensions=cfg.dimensions,
                                     limit_per_dim=cfg.n_pairs_per_dim))
    if not pairs:
        tprint("[e16] no diagnostic_v2 pairs available; aborting")
        return {"rows": []}
    n_mut = int(cfg.extra.get("paraphrases_per_prompt", len(_MUTATIONS)))
    mutations = _MUTATIONS[:n_mut]

    master_rows = []
    for mc in cfg.models:
        short = mc.short_name()
        model_out = out / short
        model_out.mkdir(parents=True, exist_ok=True)
        with manifest_run(model_out, "e16_prompt_robustness", cfg.__dict__,
                          model=mc.name, seed=cfg.seed,
                          swallow_exceptions=cfg.skip_models_on_error):
            try:
                rm = load_reward_model(mc)
            except Exception as e:
                tprint(f"[e16] {short} load failed: {e}")
                raise

            jsonl = JsonlWriter(model_out / "robust_per_pair.jsonl")
            todo = [p for p in pairs if not jsonl.has(f"{p.source}:{p.pair_id}")]
            if not todo:
                pass

            # Build a flat list of (pair_idx, mutation_idx, prompt_text,
            # response_text) so we can do one big batched forward.
            flat_pairs: list[tuple[str, str]] = []
            ptrs: list[tuple[int, int, str]] = []  # (pair_idx, mut_idx, "preferred"/"dispreferred")
            for pi, p in enumerate(todo):
                for mi, mut in enumerate(mutations):
                    paraphrased_prompt = _paraphrase(p.prompt, mut)
                    flat_pairs.append((paraphrased_prompt, p.preferred))
                    ptrs.append((pi, mi, "preferred"))
                    flat_pairs.append((paraphrased_prompt, p.dispreferred))
                    ptrs.append((pi, mi, "dispreferred"))

            if flat_pairs:
                cache = rm.forward_with_cache_batch(
                    flat_pairs, batch_size=cfg.batch_size,
                    max_length=cfg.max_length, progress=cfg.progress,
                    length_bucket=bool(cfg.extra.get("length_bucket", False)),
                )
                rewards = cache.rewards.detach().cpu().numpy()
            else:
                rewards = np.array([])

            # Group rewards back per pair.
            per_pair: dict[int, dict] = {}
            for k, (pi, mi, side) in enumerate(ptrs):
                d = per_pair.setdefault(pi, {"preferred": {}, "dispreferred": {}})
                d[side][mi] = float(rewards[k])

            for pi, p in enumerate(todo):
                if pi not in per_pair:
                    continue
                w_by_mut = per_pair[pi]["preferred"]
                l_by_mut = per_pair[pi]["dispreferred"]
                w_arr = np.array([w_by_mut[mi] for mi in sorted(w_by_mut)])
                l_arr = np.array([l_by_mut[mi] for mi in sorted(l_by_mut)])
                baseline_w = w_by_mut[0]
                baseline_l = l_by_mut[0]
                preference_gap = abs(baseline_w - baseline_l)
                # σ across paraphrases for this pair (preferred side only).
                pref_std = float(w_arr.std(ddof=0))
                # Signal-to-noise: how big is the within-pair gap relative
                # to paraphrase-induced wobble?
                snr = float(preference_gap / (pref_std + 1e-9))
                rec = {
                    "record_id": f"{p.source}:{p.pair_id}",
                    "source": p.source, "dimension": p.dimension, "pair_id": p.pair_id,
                    "n_paraphrases": int(len(w_arr)),
                    "baseline_preferred": baseline_w,
                    "baseline_dispreferred": baseline_l,
                    "paraphrase_preferred":   w_arr.tolist(),
                    "paraphrase_dispreferred": l_arr.tolist(),
                    "preferred_std": pref_std,
                    "dispreferred_std": float(l_arr.std(ddof=0)),
                    "preference_gap": float(preference_gap),
                    "signal_to_noise": snr,
                    "robust": bool(snr > 1.0),
                }
                jsonl.write(rec)

            records = jsonl.read_all()
            by_dim: dict[str, list[dict]] = {}
            for r in records:
                if "robust" in r:
                    by_dim.setdefault(r["dimension"], []).append(r)
            for dim, rs in by_dim.items():
                if not rs:
                    continue
                robust_frac = float(np.mean([r["robust"] for r in rs]))
                mean_std = float(np.mean([r["preferred_std"] for r in rs]))
                mean_snr = float(np.mean([r["signal_to_noise"] for r in rs]))
                master_rows.append({
                    "model": short, "dimension": dim,
                    "n_pairs": len(rs),
                    "robust_fraction": robust_frac,
                    "mean_preferred_std": mean_std,
                    "mean_signal_to_noise": mean_snr,
                })
            del rm
            clear_gpu()

    write_csv(master_rows, out / "e16_robustness.csv")

    summary = {}
    for r in master_rows:
        m = r["model"]
        summary.setdefault(m, {"per_dim": [], "mean_robust_fraction": 0.0,
                               "mean_signal_to_noise": 0.0})
        summary[m]["per_dim"].append(r)
    for m, s in summary.items():
        if s["per_dim"]:
            s["mean_robust_fraction"] = float(np.mean(
                [r["robust_fraction"] for r in s["per_dim"]]))
            s["mean_signal_to_noise"] = float(np.mean(
                [r["mean_signal_to_noise"] for r in s["per_dim"]]))
    save_json(summary, out / "e16_summary.json")

    _plot_robustness(master_rows, out / "figures" / "e16_robustness")
    return {"rows": master_rows, "summary": summary}


def _plot_robustness(rows: list[dict], path) -> None:
    if not rows:
        return
    setup_matplotlib()
    import matplotlib.pyplot as plt
    models = sorted({r["model"] for r in rows})
    dims = sorted({r["dimension"] for r in rows})
    grid = np.full((len(dims), len(models)), np.nan)
    for r in rows:
        i = dims.index(r["dimension"]); j = models.index(r["model"])
        grid[i, j] = r["robust_fraction"]
    fig, ax = plt.subplots(figsize=(2 + 2 * len(models), 1 + 0.4 * len(dims)))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_yticks(range(len(dims))); ax.set_yticklabels(dims)
    for i in range(len(dims)):
        for j in range(len(models)):
            v = grid[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="robust fraction (SNR > 1)")
    ax.set_title("E16 prompt-robustness fraction by dimension")
    fig.tight_layout()
    savefig(fig, path)
