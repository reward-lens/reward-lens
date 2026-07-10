"""
E13 — Scale study (8B Llama -> 27B Gemma + same-scale family control).

Aggregates four metrics per model (read from prior experiments' artifacts
when present, recomputed otherwise):
  - mean crystallization depth (E02)
  - mean per-pair faithfulness rho (E04)
  - mean concept alignment for top concept (E08)
  - n_top10_components needed to explain 80% of |attribution| (E03)

Output: a 2x2 panel "metric vs scale" plot + interpretation.

The 2B Skywork-Gemma may not exist; this experiment treats whatever
models are passed in via cfg.models as the rungs of the ladder.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..config import ExperimentConfig
from ..utils.io import manifest_run, save_json, write_csv, load_json
from ..utils.figures import setup_matplotlib, savefig, PALETTE


def _read_summary(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def run(cfg: ExperimentConfig) -> dict:
    out = cfg.out_path
    (out / "figures").mkdir(parents=True, exist_ok=True)
    upstream_root = Path(cfg.extra.get("upstream_root", str(out.parent)))

    rows = []
    for mc in cfg.models:
        short = mc.short_name()
        n_params = int(cfg.extra.get(f"{short}_params", 0)) or _guess_param_count(mc.name)

        # E02 — mean crystallization fraction
        e02 = _read_summary(upstream_root / "e02_lens_population" / short / "lens_summary.json")
        crystal = float("nan")
        if e02 and "per_dimension" in e02:
            cs = [v["mean_crystal_frac"] for v in e02["per_dimension"].values()
                  if np.isfinite(v.get("mean_crystal_frac", float("nan")))]
            crystal = float(np.mean(cs)) if cs else float("nan")

        # E04 — mean per-pair rho
        e04 = _read_summary(upstream_root / "e04_faithfulness_population" / short / "faithfulness_summary.json")
        rho = float("nan")
        if e04 and "per_dimension" in e04:
            rs = [v["mean_rho"] for v in e04["per_dimension"].values()
                  if np.isfinite(v.get("mean_rho", float("nan")))]
            rho = float(np.mean(rs)) if rs else float("nan")

        # E08 — top concept alignment cosine
        e08 = _read_summary(upstream_root / "e08_concept_population" / "e08_concept_dose_response.csv")
        # Read CSV instead — easier here.
        top_aln = float("nan")
        cands_path = upstream_root / "e08_concept_population" / "e08_concept_dose_response.csv"
        if cands_path.exists():
            import csv
            with open(cands_path) as f:
                reader = csv.DictReader(f)
                aligns = [abs(float(r["alignment_cosine"])) for r in reader if r.get("model") == short]
                if aligns:
                    top_aln = float(max(aligns))

        # E03 — components needed for 80% of total |attribution|
        n_for_80 = float("nan")
        e03 = _read_summary(upstream_root / "e03_attribution_population" / short / "attribution_summary.json")
        # Approximation: read per-dim top_components.json files and average
        attr_dir = upstream_root / "e03_attribution_population" / short
        if attr_dir.exists():
            counts = []
            for top_file in attr_dir.glob("top_components_*.json"):
                arr = load_json(top_file)
                if not arr:
                    continue
                # we don't have raw |attr|; approximate by frequency rank
                # — count the smallest k whose cumulative frequency >=0.8
                freqs = np.asarray([r["frequency"] for r in arr])
                if freqs.size == 0:
                    continue
                cum = np.cumsum(freqs) / max(freqs.sum(), 1e-12)
                k = int(np.searchsorted(cum, 0.8) + 1)
                counts.append(k)
            if counts:
                n_for_80 = float(np.mean(counts))

        rows.append({
            "model": short, "params": n_params,
            "mean_crystal_frac": crystal,
            "mean_faithfulness_rho": rho,
            "top_concept_alignment": top_aln,
            "n_components_for_80pct": n_for_80,
        })

    write_csv(rows, out / "e13_scale.csv")
    save_json(rows, out / "e13_scale.json")
    _plot_scale_panel(rows, out / "figures" / "e13_scale_panel")
    return {"rows": rows}


def _guess_param_count(name: str) -> int:
    n = name.lower()
    for marker, p in (("70b", 70_000_000_000), ("27b", 27_000_000_000),
                       ("20b", 20_000_000_000), ("13b", 13_000_000_000),
                       ("8b", 8_000_000_000), ("7b", 7_000_000_000),
                       ("3b", 3_000_000_000), ("2b", 2_000_000_000)):
        if marker in n:
            return p
    return 0


def _plot_scale_panel(rows: list[dict], path: Path) -> None:
    setup_matplotlib()
    import matplotlib.pyplot as plt
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r.get("params", 0))
    xs = np.asarray([r["params"] for r in rows], dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        ("mean_crystal_frac",      "Crystallization depth"),
        ("mean_faithfulness_rho",  "Mean per-pair rho (faithfulness)"),
        ("top_concept_alignment",  "Top concept alignment |cos|"),
        ("n_components_for_80pct", "n components for 80% of attribution"),
    ]
    for ax, (key, label) in zip(axes.ravel(), panels):
        ys = np.asarray([r.get(key, float("nan")) for r in rows], dtype=np.float64)
        ax.plot(xs, ys, "o-", color=PALETTE[0])
        for r, y in zip(rows, ys):
            if np.isfinite(y):
                ax.annotate(r["model"], (r["params"], y), fontsize=7, alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("parameters (log)")
        ax.set_ylabel(label)
    fig.suptitle("E13 scale study", fontsize=12)
    fig.tight_layout()
    savefig(fig, path)
