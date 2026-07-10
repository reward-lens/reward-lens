"""
E10 — Distortion index, fully exercised.

Build evaluation strategies of differing coverage: (a) full coverage,
(b) honesty-removed, (c) safety-removed, (d) only-helpfulness. For each,
compute the distortion index by running the model on per-dimension probe
sets and asking which dimensions are under-covered relative to the
strongest-covered dimension.

This is in-library validation only. The bold PPO experiment (E20) is
gated separately on E04 + E17.

Bug history (deep_analysisv1): the previous implementation called
``analyzer.analyze(coverage_dimensions=...)`` which does not exist on
DistortionAnalyzer. The hasattr check returned False, ``report`` became
None, and every distortion row was emitted as NaN. We now call the
actual library API ``compute_distortion_index`` with real
PreferencePair probes drawn from diagnostic_v2.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import ExperimentConfig
from ..utils.io import manifest_run, save_json, write_csv
from ..utils.figures import setup_matplotlib, savefig
from ..utils.parallel import tprint, clear_gpu
from ..utils.diagnostics import load_diagnostic_v2
from ..utils.models import load_reward_model


# Strategy → list of dimensions that are *removed* (i.e. assumed un-evaluated).
# A strategy's effective evaluation = all_dims minus removed.
_STRATEGIES = {
    "full":               [],
    "honesty_removed":    ["correctness", "factuality", "math_correctness", "code_correctness"],
    "safety_removed":     ["safety", "refusal_quality"],
    "only_helpfulness":   None,  # special: keep only helpfulness
}


def _strategy_dims(strat_name: str, all_dims: list[str]) -> list[str]:
    if strat_name == "only_helpfulness":
        return ["helpfulness"] if "helpfulness" in all_dims else all_dims[:1]
    removed = _STRATEGIES.get(strat_name) or []
    return [d for d in all_dims if d not in removed]


def run(cfg: ExperimentConfig) -> dict:
    out = cfg.out_path
    (out / "figures").mkdir(parents=True, exist_ok=True)

    from reward_lens.distortion import DistortionAnalyzer
    from reward_lens.diagnostic_data_v2 import ALL_DIMENSIONS_V2
    n_per_dim = int(cfg.extra.get("probes_per_dim", min(cfg.n_pairs_per_dim, 30)))
    all_pairs = list(load_diagnostic_v2(dimensions=cfg.dimensions, limit_per_dim=n_per_dim))
    # Group pairs by dimension once so each strategy can sample a subset.
    dim_to_pairs: dict[str, list] = {}
    for p in all_pairs:
        dim_to_pairs.setdefault(p.dimension, []).append(p)
    available_dims = [d for d in ALL_DIMENSIONS_V2 if d in dim_to_pairs]

    master_rows = []
    for mc in cfg.models:
        short = mc.short_name()
        model_out = out / short
        model_out.mkdir(parents=True, exist_ok=True)
        with manifest_run(model_out, "e10_distortion_index", cfg.__dict__,
                          model=mc.name, seed=cfg.seed,
                          swallow_exceptions=cfg.skip_models_on_error):
            try:
                rm = load_reward_model(mc)
            except Exception as e:
                tprint(f"[e10] load failed: {e}")
                raise

            analyzer = DistortionAnalyzer(rm)
            for strat_name in _STRATEGIES:
                eval_dims = _strategy_dims(strat_name, available_dims)
                # Build the probes dict the library expects: dim -> list[PreferencePair].
                # The 'only_helpfulness' strategy has 1 dim with N probes; everything
                # else has 1..K dims with N probes each. The library's coverage formula
                # needs >=1 dim with non-zero probes; degenerate cases yield distortion=1.0.
                probes = {d: dim_to_pairs.get(d, []) for d in eval_dims}
                try:
                    report = analyzer.compute_distortion_index(
                        quality_dimensions=eval_dims,
                        evaluation_probes=probes,
                        max_length=cfg.max_length,
                    )
                    score = float(report.predicted_hacking_severity)
                    per_dim = {k: float(v) for k, v in report.per_dimension_distortion.items()}
                    coverage = {k: float(v) for k, v in report.effective_coverage.items()}
                    under_covered = list(report.under_covered_dimensions)
                except Exception as e:
                    tprint(f"[e10] {short}/{strat_name} failed: {type(e).__name__}: {e}")
                    score = float("nan")
                    per_dim = {}
                    coverage = {}
                    under_covered = []
                row = {"model": short, "strategy": strat_name,
                       "distortion": score,
                       "n_dimensions": len(eval_dims),
                       "per_dimension_distortion": per_dim,
                       "effective_coverage": coverage,
                       "under_covered_dimensions": under_covered}
                master_rows.append({k: v for k, v in row.items() if k not in ("per_dimension_distortion", "effective_coverage", "under_covered_dimensions")})
                save_json(row, model_out / f"distortion_{strat_name}.json")
            del rm
            clear_gpu()

    write_csv(master_rows, out / "e10_distortion.csv")
    _heatmap(master_rows, out / "figures" / "e10_distortion_heatmap")
    return {"rows": master_rows}


def _heatmap(rows: list[dict], path: Path) -> None:
    setup_matplotlib()
    import matplotlib.pyplot as plt
    if not rows:
        return
    models = sorted({r["model"] for r in rows})
    strats = sorted({r["strategy"] for r in rows})
    grid = np.full((len(strats), len(models)), np.nan)
    for r in rows:
        i = strats.index(r["strategy"]); j = models.index(r["model"])
        grid[i, j] = r["distortion"]
    fig, ax = plt.subplots(figsize=(2 + 2 * len(models), 1 + len(strats)))
    im = ax.imshow(grid, cmap="viridis")
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_yticks(range(len(strats))); ax.set_yticklabels(strats)
    for i in range(len(strats)):
        for j in range(len(models)):
            v = grid[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white")
    fig.colorbar(im, ax=ax, label="distortion")
    ax.set_title("E10 distortion index by strategy")
    savefig(fig, path)
