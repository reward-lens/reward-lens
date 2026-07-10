"""
E17 — Interpretability-guided reward editing.

Identify a hackable concept whose direction is well-aligned with w_r
(typically length or agreement on Skywork). Construct
    w_r_edit = w_r - alpha * (w_r^T v_hat) * v_hat
sweeping alpha in {0, 0.5, 1, 1.5, 2}.

For each alpha, score:
  - the corresponding hacking probe set (does the bias drop?)
  - a held-out RewardBench-helpfulness sample (does base accuracy survive?)
Plot bias-vs-accuracy frontier.

This is the causal-validation centerpiece replacing E20.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..config import ExperimentConfig
from ..utils.io import manifest_run, save_json, write_csv
from ..utils.figures import setup_matplotlib, savefig, PALETTE
from ..utils.parallel import tprint, clear_gpu
from ..utils.datasets import load_rewardbench
from ..utils.models import load_reward_model


def _score_with_edited_w(rm, prompts: list[str], responses: list[str],
                         w_r_edit: torch.Tensor, max_length: int,
                         batch_size: int, progress: bool) -> np.ndarray:
    """Re-score a batch using an edited reward direction.

    The reward = w^T h_final + b. We recompute it from cached final-token
    residual streams using the edited weight, leaving the model's logits
    head untouched. This requires running forward_with_cache_batch and
    projecting cache.residual_streams[n_layers - 1] onto w_r_edit.
    """
    cache = rm.forward_with_cache_batch(
        list(zip(prompts, responses)),
        batch_size=batch_size, max_length=max_length, progress=progress,
    )
    h = cache.residual_streams.get(rm.n_layers - 1)
    if h is None:
        # fall back to last available layer
        last = max(cache.residual_streams.keys())
        h = cache.residual_streams[last]
    w = w_r_edit.to(h.device).float()
    proj = (h.float() @ w) + rm.reward_bias
    return proj.detach().cpu().numpy()


def run(cfg: ExperimentConfig) -> dict:
    out = cfg.out_path
    (out / "figures").mkdir(parents=True, exist_ok=True)
    alphas = cfg.extra.get("alphas", [0.0, 0.5, 1.0, 1.5, 2.0])
    target_concept = cfg.extra.get("target_concept", "verbosity")

    from reward_lens.concepts import ConceptExtractor, CONCEPT_PAIRS
    from reward_lens.hacking import LENGTH_TESTS

    helpfulness_pairs = load_rewardbench(subset="chat", limit=cfg.n_pairs_per_dim)
    if not helpfulness_pairs:
        from ..utils.diagnostics import load_diagnostic_v2
        helpfulness_pairs = load_diagnostic_v2(["helpfulness"], limit_per_dim=cfg.n_pairs_per_dim)

    # Use length probes for the bias half (verbosity = length-bias proxy).
    bias_probes = LENGTH_TESTS

    master_rows = []
    for mc in cfg.models:
        short = mc.short_name()
        model_out = out / short
        model_out.mkdir(parents=True, exist_ok=True)
        with manifest_run(model_out, "e17_reward_editing", cfg.__dict__,
                          model=mc.name, seed=cfg.seed,
                          swallow_exceptions=cfg.skip_models_on_error):
            try:
                rm = load_reward_model(mc)
            except Exception as e:
                tprint(f"[e17] load failed: {e}")
                raise

            # Extract concept direction
            extractor = ConceptExtractor(rm)
            if target_concept not in CONCEPT_PAIRS:
                tprint(f"[e17] concept {target_concept} not in CONCEPT_PAIRS, "
                       f"using first available")
                target_concept_use = list(CONCEPT_PAIRS.keys())[0]
            else:
                target_concept_use = target_concept
            vecs = extractor.extract_concepts({target_concept_use: CONCEPT_PAIRS[target_concept_use]})
            v = vecs[target_concept_use].detach().to(rm.reward_direction.device).float()
            v_hat = v / (v.norm() + 1e-12)
            w_r = rm.reward_direction.float().to(v_hat.device)
            scalar = float((w_r * v_hat).sum().item())  # w_r^T v_hat

            for alpha in alphas:
                w_edit = w_r - alpha * scalar * v_hat
                # Bias half: score each probe pair (neutral, biased)
                neutral_prompts = [p["prompt"] for p in bias_probes]
                neutral_resps = [p["neutral"] for p in bias_probes]
                biased_resps = [p["biased"] for p in bias_probes]
                scores_n = _score_with_edited_w(rm, neutral_prompts, neutral_resps, w_edit,
                                                cfg.max_length, cfg.batch_size, cfg.progress)
                scores_b = _score_with_edited_w(rm, neutral_prompts, biased_resps, w_edit,
                                                cfg.max_length, cfg.batch_size, cfg.progress)
                bias_delta = float((scores_b - scores_n).mean())
                bias_d = float((scores_b - scores_n).mean()
                                / max((scores_b - scores_n).std(ddof=1), 1e-12))

                # Accuracy half on RewardBench helpfulness (chat)
                acc_prompts = [p.prompt for p in helpfulness_pairs]
                acc_pref = [p.preferred for p in helpfulness_pairs]
                acc_disp = [p.dispreferred for p in helpfulness_pairs]
                scores_pref = _score_with_edited_w(rm, acc_prompts, acc_pref, w_edit,
                                                    cfg.max_length, cfg.batch_size, cfg.progress)
                scores_disp = _score_with_edited_w(rm, acc_prompts, acc_disp, w_edit,
                                                    cfg.max_length, cfg.batch_size, cfg.progress)
                acc = float((scores_pref > scores_disp).mean())

                row = {
                    "model": short, "alpha": alpha,
                    "concept": target_concept_use,
                    "bias_mean_delta": bias_delta,
                    "bias_cohens_d": bias_d,
                    "rewardbench_chat_accuracy": acc,
                }
                master_rows.append(row)
                save_json(row, model_out / f"edit_alpha_{alpha:.2f}.json")
            _plot_frontier([r for r in master_rows if r["model"] == short],
                           out / "figures" / f"e17_frontier_{short}")
            del rm
            clear_gpu()

    write_csv(master_rows, out / "e17_reward_editing.csv")
    return {"rows": master_rows}


def _plot_frontier(rows: list[dict], path: Path) -> None:
    setup_matplotlib()
    import matplotlib.pyplot as plt
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["alpha"])
    biases = [r["bias_cohens_d"] for r in rows]
    accs = [r["rewardbench_chat_accuracy"] for r in rows]
    alphas = [r["alpha"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(biases, accs, "o-", color=PALETTE[2])
    for a, b, c in zip(alphas, biases, accs):
        ax.annotate(f"α={a}", (b, c), fontsize=8)
    ax.set_xlabel("Length-bias Cohen's d (lower = better)")
    ax.set_ylabel("RewardBench-chat accuracy")
    ax.set_title("E17 reward editing — bias vs accuracy frontier")
    savefig(fig, path)
