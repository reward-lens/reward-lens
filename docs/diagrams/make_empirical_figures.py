#!/usr/bin/env python3
"""Regenerate every empirical figure in the docs from committed run artifacts.

Each figure is built only from data the repository actually produced: the
weekend RewardBench run (weekend_experiment/) and the at-scale v2 sweep
(outputs/v2_.../). No 8B model is loaded here; everything runs on CPU from
committed JSON/CSV. If a source file is missing the figure is skipped with a
notice rather than faked.

Run from the docs/ directory:  python3 diagrams/make_empirical_figures.py
Outputs land in content/assets/figures/*.svg.
"""
from __future__ import annotations
import csv
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.abspath(os.path.join(HERE, "..", "content", "assets", "figures"))
WEEKEND = os.path.join(REPO, "weekend_experiment")
V2 = os.path.join(REPO, "outputs", "v2_20260506_222648_unknown")
os.makedirs(OUT, exist_ok=True)

# --- validated palette (dataviz skill, light surface) -----------------------
BLUE = "#2a78d6"      # Skywork / preferred / "toward what the reward wants"
ORANGE = "#eb6834"    # ArmoRM
RED = "#e34948"       # dispreferred
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#ffffff"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]
SEQ_BLUE = LinearSegmentedColormap.from_list("seqblue", ["#cde2fb", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("bwr", [BLUE, "#f0efec", RED])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": BASELINE,
    "axes.linewidth": 0.9, "grid.color": GRID, "grid.linewidth": 0.8,
    "svg.fonttype": "path", "figure.dpi": 110,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    fig.tight_layout()
    p = os.path.join(OUT, name)
    fig.savefig(p, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def layer_of(component_name: str) -> int:
    m = re.search(r"_L(\d+)", component_name)
    return int(m.group(1)) if m else -1


def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


# --- 1. reward-lens curve: the margin forming across layers -----------------
def fig_lens_curve():
    d = load_json(os.path.join(WEEKEND, "skywork", "lens_results.json"))
    if not d:
        print("  skip lens-curve (no data)"); return
    pair = d["helpfulness"]["pairs"][0]
    curve = np.array(pair["differential_curve"])       # embed + 32 layers
    layers = np.array(pair["layers"])
    cryst = pair["crystallization_layer"]
    final = curve[-1]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.axhline(0, color=BASELINE, lw=1)
    ax.axhline(final / 2, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax.plot(layers, curve, color=BLUE, lw=2.2, zorder=3)
    ax.scatter(layers, curve, color=BLUE, s=14, zorder=4)
    ax.axvline(cryst, color=ORANGE, lw=1.4, ls="--", zorder=2)
    ax.annotate(f"crystallizes at\nlayer {cryst} of 32",
                xy=(cryst, final / 2), xytext=(cryst - 15, final * 0.62),
                color=ORANGE, fontsize=10, ha="left",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax.annotate("half the final margin", xy=(2, final / 2 + 0.4),
                color=MUTED, fontsize=9, ha="left")
    ax.set_xlabel("layer"); ax.set_ylabel(r"margin  $\Delta = w_r^\top(h_{\rm chosen}-h_{\rm rejected})$")
    ax.set_title("The margin stays near zero for two-thirds of the network, then forms late",
                 fontsize=11.5, loc="left")
    ax.grid(axis="y", alpha=0.7)
    save(fig, "lens-curve.svg")


# --- 2. crystallization by dimension, two models ----------------------------
def fig_crystallization():
    sky = load_json(os.path.join(WEEKEND, "skywork", "lens_results.json"))
    arm = load_json(os.path.join(WEEKEND, "armo", "lens_results.json"))
    if not sky or not arm:
        print("  skip crystallization (no data)"); return
    dims = ["helpfulness", "safety", "correctness", "verbosity"]
    sky_m = [sky[d]["mean_crystallization_frac"] for d in dims]
    sky_s = [sky[d]["std_crystallization_frac"] for d in dims]
    arm_m = [arm[d]["mean_crystallization_frac"] for d in dims]
    arm_s = [arm[d]["std_crystallization_frac"] for d in dims]
    x = np.arange(len(dims)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.bar(x - w / 2, sky_m, w, yerr=sky_s, color=BLUE, label="Skywork",
           error_kw=dict(ecolor=MUTED, lw=1, capsize=3))
    ax.bar(x + w / 2, arm_m, w, yerr=arm_s, color=ORANGE, label="ArmoRM",
           error_kw=dict(ecolor=MUTED, lw=1, capsize=3))
    ax.axhline(1.0, color=BASELINE, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(dims)
    ax.set_ylabel("crystallization depth (fraction of layers)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Skywork decides late and consistently; ArmoRM earlier and noisier",
                 fontsize=11.5, loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.7)
    save(fig, "crystallization-by-dim.svg")


# --- 3. attribution vs patching: the anti-correlation -----------------------
def fig_faithfulness_scatter():
    d = load_json(os.path.join(WEEKEND, "skywork", "faithfulness_results.json"))
    if not d:
        print("  skip faithfulness scatter (no data)"); return
    hp = d["helpfulness"]
    attr = np.array(hp["attribution_values"])
    patch = np.array(hp["patching_values"])
    names = hp["component_names"]
    rho = hp["spearman_rho"]
    layers = np.array([layer_of(n) for n in names])
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
    sc = ax.scatter(attr, patch, c=layers, cmap=SEQ_BLUE, s=42,
                    edgecolor="white", linewidth=0.6, zorder=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("layer (early → late)", color=INK2)
    cb.outline.set_edgecolor(BASELINE)
    ax.set_xlim(-0.4, attr.max() * 1.12)
    ax.set_ylim(-2.5, patch.max() * 1.12)
    # label the two extremes of the story, into empty space
    i_attr = int(np.argmax(attr)); i_patch = int(np.argmax(patch))
    ax.annotate(f"{names[i_patch]}  (early)", (attr[i_patch], patch[i_patch]),
                xytext=(attr[i_patch] + 1.0, patch[i_patch] - 1.5), color=INK2, fontsize=9.5, ha="left",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.annotate(f"{names[i_attr]}  (late)", (attr[i_attr], patch[i_attr]),
                xytext=(attr[i_attr] - 1.4, patch[i_attr] + 4.0), color=INK2, fontsize=9.5, ha="right",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.set_xlabel("attribution  (observational: signed share of the reward)")
    ax.set_ylabel("patch effect  (causal: change in margin)")
    ax.set_title(f"Attribution does not predict causal importance   (Spearman ρ = {rho:.2f})",
                 fontsize=11.5, loc="left")
    ax.text(0.97, 0.60,
            "Every point is one component.\n"
            "The mass hugs both axes:\n"
            "a component matters to one\n"
            "method or the other, never both.",
            transform=ax.transAxes, fontsize=9.5, color=MUTED, va="center", ha="right")
    save(fig, "attribution-vs-patching.svg")


# --- 4 & 5. canonical-pair attribution and patching bars --------------------
# Committed scalars from ran-notebook/Reward_Lens_Intro_Demo.ipynb (sky-blue pair).
CANON_ATTR = [("mlp_L31", 3.993), ("mlp_L30", 1.321), ("mlp_L29", 0.856),
              ("mlp_L28", 0.625), ("attn_L31", 0.505), ("mlp_L27", 0.451),
              ("mlp_L26", 0.393), ("mlp_L25", 0.334), ("mlp_L22", 0.330),
              ("mlp_L23", 0.306)]
CANON_PATCH = [("mlp_L0", 17.406), ("mlp_L6", 15.656), ("mlp_L4", 8.781),
               ("mlp_L7", 8.281), ("mlp_L5", 6.531), ("mlp_L9", 6.031),
               ("attn_L11", 5.969), ("mlp_L10", 5.500), ("attn_L8", 4.062),
               ("attn_L9", 3.969)]


def _hbar(items, color, title, xlabel, name):
    labels = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.barh(range(len(vals)), vals, color=color, height=0.66)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels, fontfamily="DejaVu Sans Mono", fontsize=9)
    for i, v in enumerate(vals):
        ax.text(v, i, f"  {v:.2f}", va="center", ha="left", color=INK2, fontsize=8.5)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, max(vals) * 1.16)
    ax.set_title(title, fontsize=11.5, loc="left")
    ax.grid(axis="x", alpha=0.6)
    save(fig, name)


def fig_canonical_bars():
    _hbar(CANON_ATTR, BLUE,
          "Attribution credits the last MLPs", "differential contribution to the margin",
          "attribution-bars.svg")
    _hbar(CANON_PATCH, ORANGE,
          "Patching says the early layers carry the cause", "patch effect on the margin",
          "patching-bars.svg")


# --- 6. hacking effect sizes, two models, sign flips ------------------------
def fig_hacking_effects():
    path = os.path.join(V2, "e06_hacking_at_scale", "e06_hacking_effects.csv")
    if not os.path.exists(path):
        print("  skip hacking effects (no data)"); return
    rows = list(csv.DictReader(open(path)))
    sky_name = "Skywork-Reward-Llama-3.1-8B-v0.2"
    arm_name = "ArmoRM-Llama3-8B-v0.1"
    order = ["length", "confidence", "formatting", "repetition", "sycophancy"]
    def d_of(model, dim):
        for r in rows:
            if r["model"] == model and r["dimension"] == dim:
                try: return float(r["cohens_d"])
                except: return np.nan
        return np.nan
    sky = [d_of(sky_name, k) for k in order]
    arm = [d_of(arm_name, k) for k in order]
    y = np.arange(len(order))[::-1]; h = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.axvline(0, color=BASELINE, lw=1.2, zorder=1)
    ax.barh(y + h / 2, sky, h, color=BLUE, label="Skywork", zorder=2)
    ax.barh(y - h / 2, arm, h, color=ORANGE, label="ArmoRM", zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(order)
    ax.set_xlabel("Cohen's d   (positive = the reward model prefers the hackable variant)")
    ax.set_title("The same surface feature flips sign between models", fontsize=11.5, loc="left")
    ax.legend(frameon=False, loc="lower right")
    # mark the three genuine flips
    for k in ("confidence", "formatting", "repetition"):
        yy = y[order.index(k)]
        ax.text(ax.get_xlim()[1], yy, "flip  ", color=RED, fontsize=9, va="center", ha="right",
                fontweight="bold")
    ax.grid(axis="x", alpha=0.6)
    save(fig, "hacking-effects.svg")


# --- 7. concept dose-response ----------------------------------------------
def fig_concept_dose():
    d = load_json(os.path.join(WEEKEND, "skywork", "concepts_report.json"))
    if not d:
        print("  skip concept dose (no data)"); return
    inter = d["interventions"]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.axhline(0, color=BASELINE, lw=1); ax.axvline(0, color=GRID, lw=0.8)
    # emphasize the strongest concept; mute the rest into one bundle
    ranked = sorted(inter.items(), key=lambda kv: -abs(kv[1].get("causal_slope", 0)))
    for name, info in ranked:
        dr = info.get("dose_response", [])
        if not dr:
            continue
        xs = [p["strength"] for p in dr]; ys = [p["delta_reward"] for p in dr]
        top = (name == ranked[0][0])
        ax.plot(xs, ys, color=BLUE if top else GRID, lw=2.6 if top else 1.4,
                zorder=3 if top else 2, marker="o" if top else None, ms=5)
        if top:
            ax.text(xs[-1], ys[-1], f"  {name}", color=INK, fontsize=10.5,
                    va="center", fontweight="bold")
    ax.text(1.9, 1.35, "other concepts", color=MUTED, fontsize=9.5, va="center", ha="left")
    slope = ranked[0][1]["causal_slope"]; cos = ranked[0][1]["alignment_cosine"]
    ax.set_xlabel("strength added along the concept direction")
    ax.set_ylabel("change in reward")
    ax.set_title(f"Push a concept into the activation, the reward follows\n"
                 f"'{ranked[0][0]}' moves reward {slope:.2f} per unit  (cosine with $w_r$ = {cos:.2f})",
                 fontsize=11, loc="left")
    ax.grid(alpha=0.5)
    save(fig, "concept-dose-response.svg")


# --- 8. cross-model formation overlay --------------------------------------
def fig_cross_model():
    sky = load_json(os.path.join(WEEKEND, "skywork", "lens_results.json"))
    arm = load_json(os.path.join(WEEKEND, "armo", "lens_results.json"))
    corr = load_json(os.path.join(WEEKEND, "comparison", "cross_model_results.json"))
    if not sky or not arm:
        print("  skip cross-model (no data)"); return
    def mean_formation(res):
        curves = []
        for p in res["helpfulness"]["pairs"]:
            c = np.array(p["differential_curve"], dtype=float)
            if abs(c[-1]) < 1e-6:
                continue
            curves.append(c / c[-1])              # normalize by final margin (library convention)
        return np.mean(curves, axis=0)
    sky_c, arm_c = mean_formation(sky), mean_formation(arm)
    frac = np.linspace(0, 1, len(sky_c))
    r = float(np.corrcoef(sky_c, arm_c)[0, 1])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.plot(frac, sky_c, color=BLUE, lw=2.4, label="Skywork")
    ax.plot(frac, arm_c, color=ORANGE, lw=2.4, label="ArmoRM")
    ax.set_xlabel("fractional depth (layer / total layers)")
    ax.set_ylabel("mean normalized margin (0 → final)")
    ax.set_ylim(-0.1, 1.08)
    ax.set_title(f"Both leave the decision to the late layers   (curve correlation r = {r:.2f})",
                 fontsize=11.5, loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.5)
    save(fig, "cross-model-overlay.svg")


# --- 9. ArmoRM 19-objective cosine heatmap ----------------------------------
def fig_armo_cosine():
    hits = glob.glob(os.path.join(V2, "*e18*", "*", "armo_obj_cosine.json"))
    if not hits:
        print("  skip armo cosine (no data)"); return
    m = np.array(load_json(hits[0])["cosine_matrix"])
    n = m.shape[0]
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    norm = TwoSlopeNorm(vmin=min(-0.2, m[~np.eye(n, dtype=bool)].min()), vcenter=0, vmax=1.0)
    im = ax.imshow(m, cmap=DIVERGING, norm=norm)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("cosine between objective directions", color=INK2)
    cb.outline.set_edgecolor(BASELINE)
    ax.set_xticks(range(0, n, 2)); ax.set_yticks(range(0, n, 2))
    ax.set_xlabel("objective index"); ax.set_ylabel("objective index")
    ax.set_title("ArmoRM's 19 objectives are mostly aligned, but not one direction",
                 fontsize=11, loc="left")
    save(fig, "armo-objective-cosine.svg")


if __name__ == "__main__":
    print(f"writing figures to {OUT}")
    fig_lens_curve()
    fig_crystallization()
    fig_faithfulness_scatter()
    fig_canonical_bars()
    fig_hacking_effects()
    fig_concept_dose()
    fig_cross_model()
    fig_armo_cosine()
    print("done.")
