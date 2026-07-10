"""
Report generator — builds a 5-10 page REPORT.md aggregating across experiments.

Expects to find experiment outputs under ``run_root/<experiment_name>/...``.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .utils.io import load_json


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def build_report(run_root: Path) -> str:
    lines = []
    lines.append(f"# reward-lens v2 — REPORT")
    lines.append("")
    lines.append(f"Run root: `{run_root}`")
    lines.append("")
    args_file = run_root / "run_args.json"
    if args_file.exists():
        args = load_json(args_file)
        lines.append("## Setup")
        lines.append(f"- Models: {', '.join(args.get('models', []))}")
        lines.append(f"- git commit: `{args.get('git_commit','?')}`")
        lines.append("")

    # ---- E04 headline (the spine) ----
    e04 = _read_csv(run_root / "e04_faithfulness_population" / "e04_faithfulness.csv")
    if e04:
        lines.append("## E04 — Population-scale faithfulness (the spine)")
        lines.append("")
        lines.append("| model | dimension | n | mean ρ | 95% CI | frac<0 | q (BH) | sig? |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in e04:
            lines.append(
                f"| {r.get('model','')} | {r.get('dimension','')} | {r.get('n','')} | "
                f"{float(r.get('mean_rho',0) or 0):.3f} | "
                f"[{float(r.get('ci_low',0) or 0):.3f}, {float(r.get('ci_high',0) or 0):.3f}] | "
                f"{float(r.get('frac_negative',0) or 0):.2f} | "
                f"{r.get('q_value','-')} | "
                f"{r.get('significant_fdr05','-')} |"
            )
        lines.append("")
        # Headline interpretation
        rhos = [float(r["mean_rho"]) for r in e04 if r.get("mean_rho")]
        if rhos:
            avg = sum(rhos) / len(rhos)
            lines.append(f"**Aggregate mean per-pair ρ across cells:** {avg:.3f}")
            if avg > 0.5:
                lines.append("> Faithfulness *holds* at population scale: linear attribution "
                              "predicts causal patching.")
            elif avg < -0.1:
                lines.append("> Faithfulness *flips negative* at population scale: linear "
                              "attribution is anti-correlated with causal effect — the v1 "
                              "negative result generalises.")
            else:
                lines.append("> Faithfulness is *weak/null* at population scale: linear "
                              "attribution does not reliably predict causal effect.")
            lines.append("")

    # ---- E17 ----
    e17 = _read_csv(run_root / "e17_reward_editing" / "e17_reward_editing.csv")
    if e17:
        lines.append("## E17 — Interpretability-guided reward editing")
        lines.append("")
        lines.append("| model | α | concept | bias d | RB-chat acc |")
        lines.append("|---|---|---|---|---|")
        for r in e17:
            lines.append(f"| {r['model']} | {r['alpha']} | {r['concept']} | "
                          f"{float(r['bias_cohens_d']):.2f} | "
                          f"{float(r['rewardbench_chat_accuracy']):.3f} |")
        lines.append("")

    # ---- Other headline tables ----
    for label, csv_path in [
        ("E01 baseline accuracy",
         run_root / "e01_baseline_and_diagnostics" / "e01_accuracy.csv"),
        ("E02 crystallization",
         run_root / "e02_lens_population" / "e02_crystallization.csv"),
        ("E06 hacking effects",
         run_root / "e06_hacking_at_scale" / "e06_hacking_effects.csv"),
        ("E13 scale",
         run_root / "e13_scale_study" / "e13_scale.csv"),
        ("E14 cross-architecture",
         run_root / "e14_cross_architecture" / "e14_cross_arch.csv"),
        ("E15 top heads",
         run_root / "e15_head_path_patching" / "e15_top_heads.csv"),
    ]:
        rows = _read_csv(csv_path)
        if not rows:
            continue
        lines.append(f"## {label}")
        lines.append("")
        if rows:
            cols = list(rows[0].keys())
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("|" + "|".join(["---"] * len(cols)) + "|")
            for r in rows[:30]:  # cap
                lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
            lines.append("")

    # ---- known issues ----
    lines.append("## Known issues")
    lines.append("")
    lines.append("Consult per-experiment `manifest.json` for failures and "
                  "per-model `adapter_health_check.json` for adapter problems.")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    args = p.parse_args(argv)
    root = Path(args.runs)
    text = build_report(root)
    out = root / "REPORT.md"
    out.write_text(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
