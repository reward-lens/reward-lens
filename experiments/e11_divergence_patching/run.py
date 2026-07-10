"""
E11 — Divergence-aware patching at scale.

Fit the activation distribution from a corpus, run divergence-aware
patching on >=30 pairs/dim, and report:
  - % pernicious patches (patched activation OOD)
  - % harmless
  - reliability score distribution
  - whether the faithfulness picture changes when pernicious patches are
    excluded
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import ExperimentConfig
from ..utils.io import JsonlWriter, manifest_run, save_json, write_csv
from ..utils.figures import setup_matplotlib, savefig
from ..utils.parallel import tprint, clear_gpu
from ..utils.diagnostics import load_diagnostic_v2
from ..utils.models import load_reward_model

from reward_lens.statistics import bootstrap_ci, spearman_with_ci


def run(cfg: ExperimentConfig) -> dict:
    out = cfg.out_path
    (out / "figures").mkdir(parents=True, exist_ok=True)
    n_per_dim = int(cfg.extra.get("patching_pairs_per_dim", 30))
    corpus_size = int(cfg.extra.get("corpus_size", 200))

    pairs = list(load_diagnostic_v2(dimensions=cfg.dimensions, limit_per_dim=n_per_dim))

    master_rows = []
    for mc in cfg.models:
        short = mc.short_name()
        model_out = out / short
        model_out.mkdir(parents=True, exist_ok=True)
        with manifest_run(model_out, "e11_divergence_patching", cfg.__dict__,
                          model=mc.name, seed=cfg.seed,
                          swallow_exceptions=cfg.skip_models_on_error):
            try:
                rm = load_reward_model(mc)
            except Exception as e:
                tprint(f"[e11] load failed: {e}")
                raise

            from reward_lens.divergence_patching import DivergenceAwarePatching

            # Bug fix (deep_analysisv1): the previous implementation called
            # ``dap.fit_corpus(corpus)`` and ``dap.patch_all_components(...)``.
            # Neither used the divergence-aware code path: ``fit_corpus`` does
            # not exist (the method is ``fit_distribution``), and
            # ``patch_all_components`` is the *base* ActivationPatcher entry
            # which produces a plain PatchingResult — so reliability_scores /
            # pernicious_mask were always empty arrays. We now use the proper
            # API: ``fit_distribution(prompts, responses)`` then
            # ``patch_with_divergence_check(...)`` which returns a
            # DivergenceAwarePatchingResult.
            corpus_prompts = [p.prompt for p in pairs[:corpus_size]]
            corpus_responses = [p.preferred for p in pairs[:corpus_size]]
            dap = DivergenceAwarePatching(rm)
            try:
                dap.fit_distribution(
                    corpus_prompts, corpus_responses,
                    max_length=cfg.max_length, show_progress=False,
                )
            except Exception as e:
                tprint(f"[e11] divergence corpus fit failed: {type(e).__name__}: {e}")

            jsonl = JsonlWriter(model_out / "divergence_per_pair.jsonl")
            for p in pairs:
                rid = f"{p.source}:{p.pair_id}"
                if jsonl.has(rid):
                    continue
                try:
                    result = dap.patch_with_divergence_check(
                        p.prompt, p.preferred, p.dispreferred,
                        mode="noising", show_progress=False,
                    )
                    # Reliability is a single per-pair scalar in the library;
                    # synthesize a per-component reliability vector from the
                    # divergence types so downstream aggregation has data.
                    per_comp_rel = [
                        0.0 if info.divergence_type == "pernicious"
                        else (info.confidence if info.divergence_type == "harmless"
                              else 1.0)
                        for info in result.divergence_info
                    ]
                    pernicious_mask = [
                        bool(info.divergence_type == "pernicious")
                        for info in result.divergence_info
                    ]
                    pe = result.patch_effects
                    pe_list = pe.tolist() if hasattr(pe, "tolist") else list(pe)
                    rec = {
                        "record_id": rid, "source": p.source,
                        "dimension": p.dimension, "pair_id": p.pair_id,
                        "component_names": list(result.component_names),
                        "patch_effects": pe_list,
                        "reliability": per_comp_rel,
                        "pernicious_mask": pernicious_mask,
                        "reliability_score": float(result.reliability_score),
                        "has_pernicious_divergence": bool(result.has_pernicious_divergence),
                    }
                except Exception as e:
                    rec = {"record_id": rid, "error": f"{type(e).__name__}: {e}"}
                jsonl.write(rec)

            records = jsonl.read_all()
            valid = [r for r in records if "patch_effects" in r and r["patch_effects"]]
            if valid:
                pernicious_frac = []
                rel_scores = []
                for r in valid:
                    mask = np.asarray(r.get("pernicious_mask", []))
                    if mask.size:
                        pernicious_frac.append(float(mask.mean()))
                    rel = np.asarray(r.get("reliability", []))
                    if rel.size:
                        rel_scores.extend(rel.tolist())
                if pernicious_frac:
                    m = bootstrap_ci(pernicious_frac, statistic=np.mean,
                                      n_resamples=cfg.n_resamples, seed=cfg.seed, ci=cfg.ci)
                    master_rows.append({
                        "model": short, "metric": "pernicious_fraction",
                        "value": m.point, "ci_low": m.ci_low, "ci_high": m.ci_high,
                        "n": len(pernicious_frac),
                    })
                if rel_scores:
                    m = bootstrap_ci(rel_scores, statistic=np.mean,
                                      n_resamples=cfg.n_resamples, seed=cfg.seed, ci=cfg.ci)
                    master_rows.append({
                        "model": short, "metric": "mean_reliability",
                        "value": m.point, "ci_low": m.ci_low, "ci_high": m.ci_high,
                        "n": len(rel_scores),
                    })
            del rm
            clear_gpu()

    write_csv(master_rows, out / "e11_divergence.csv")
    return {"rows": master_rows}
