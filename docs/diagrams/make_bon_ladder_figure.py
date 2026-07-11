#!/usr/bin/env python3
"""Build the best-of-n-ladder empirical figure (dual theme), computed on CPU.

The claim this figure argues: you can price optimization pressure in nats before you
spend it. The x-axis of every best-of-n frontier is the exact divergence
``KL(bo_n || base) = ln n - (n-1)/n`` (Beirami et al. 2401.01879), a function of ``n``
alone -- no model, no data, no fit. So before a single rollout you already know what a
given amount of selection pressure costs.

Left panel: that exact KL price across the default doubling ladder ``DEFAULT_NS``.

Right panel: what those nats buy, on a synthetic bank of base-policy samples whose one
exploitable feature the susceptibility spectrum flags in advance. As the ladder climbs,
the proxy reward the sampler optimizes keeps rising, the flagged feature runs away with
it, and the gold objective peaks and turns back over -- the classic over-optimization
curve, read against the same exact nat ruler as the left panel.

Everything is a real computation on CPU. ``bon_kl``, ``bon_ladder``, ``susceptibility``
and ``flag_hack_modes`` come straight from ``reward_lens.loops``; the feature drift under
best-of-n selection uses the library's own order-statistic weights (validated in-script
to reproduce ``bon_ladder``'s expected reward to machine precision). The bank is
synthetic and its data-generating process is stated here in full; no plotted point is
hand-entered.

Run:  python3 diagrams/make_bon_ladder_figure.py         (from docs/, torch-free, CPU)
Out:  content/assets/figures/best-of-n-ladder-{light,dark}.svg  (transparent background)
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "content", "assets", "figures"))
sys.path.insert(0, HERE)
from rl_palette import apply_style  # noqa: E402  (sibling module, single-source palette)

from reward_lens.loops import (  # noqa: E402
    DEFAULT_NS,
    bon_kl,
    bon_ladder,
    flag_hack_modes,
    susceptibility,
)

# --- the synthetic base-policy bank (data-generating process stated in full) -------
# A reward model that rewards genuine quality q and, mistakenly, a spurious feature h.
# The gold objective wants quality and is hurt by the spurious feature. The spurious
# feature has a heavy right tail: a rare "exploit" that only reliably surfaces once the
# selection pressure (n) is large. Nothing here is fit; these are declared constants.
SEED = 20260711
N_PROMPTS = 64          # independent prompts (across-prompt SEM comes from these)
M = 4096                # base-policy samples scored per prompt
A_QUALITY = 1.5         # the proxy's weight on genuine quality
B_HACK = 0.5            # the proxy's (mistaken) weight on the spurious feature
C_GOLD_PENALTY = 0.9    # how much the spurious feature hurts the gold objective
NOISE = 0.25            # idiosyncratic reward noise


def build_bank(seed: int = SEED):
    """Return per-prompt banks (N_PROMPTS, M): proxy reward, gold objective, and the two
    observable features (genuine quality, spurious/hackable)."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((N_PROMPTS, M))                       # genuine quality (light tail)
    h = rng.standard_exponential((N_PROMPTS, M)) - 1.0            # spurious feature (heavy right tail), mean 0
    eps = rng.standard_normal((N_PROMPTS, M))
    r = A_QUALITY * q + B_HACK * h + NOISE * eps                  # proxy reward the RM assigns
    g = q - C_GOLD_PENALTY * h                                    # gold objective (the true target)
    return r, g, q, h


def bon_expected_column(r_bank: np.ndarray, col_bank: np.ndarray, n: int) -> np.ndarray:
    """E[column value | sample selected as best-of-n on the reward], per prompt.

    Uses exactly the order-statistic weights the library's ``expected_bon_reward`` uses --
    the probability the maximum of ``n`` draws is the k-th order statistic, ``(k/m)^n -
    ((k-1)/m)^n`` -- but sorts by reward and reads a different column of the selected
    sample. Fed the reward column it reproduces ``bon_ladder`` exactly (asserted below).
    """
    out = np.empty(r_bank.shape[0], dtype=np.float64)
    m = r_bank.shape[1]
    k = np.arange(1, m + 1, dtype=np.float64)
    weights = (k / m) ** n - ((k - 1.0) / m) ** n
    for j in range(r_bank.shape[0]):
        order = np.argsort(r_bank[j])                            # ascending by reward
        out[j] = float(np.dot(col_bank[j][order], weights))
    return out


def compute():
    """All real numbers for the figure. Returns a dict of arrays and scalars."""
    r, g, q, h = build_bank()
    ns = np.asarray(DEFAULT_NS)
    kl = bon_kl(ns)                                              # exact KL price, library identity

    # proxy-reward frontier from the library
    ladder_ev = bon_ladder(r, DEFAULT_NS)
    ladder = ladder_ev.value

    # validate the column helper against the library on the reward column
    mine = np.array([bon_expected_column(r, r, int(n)).mean() for n in ns])
    max_diff = float(np.max(np.abs(mine - ladder.expected_reward)))
    assert max_diff < 1e-9, f"column helper disagrees with bon_ladder by {max_diff}"

    # feature / gold drift under best-of-n, per prompt then averaged, with across-prompt SEM
    gold_pp = np.array([bon_expected_column(r, g, int(n)) for n in ns])   # (n_ns, n_prompts)
    hack_pp = np.array([bon_expected_column(r, h, int(n)) for n in ns])
    sd_r, sd_g, sd_h = r.std(), g.std(), h.std()
    base_r, base_g, base_h = r.mean(), g.mean(), h.mean()

    def z(mean_arr, base, sd):
        return (mean_arr - base) / sd

    proxy = z(ladder.expected_reward, base_r, sd_r)
    gold = z(gold_pp.mean(1), base_g, sd_g)
    hack = z(hack_pp.mean(1), base_h, sd_h)
    proxy_sem = ladder.reward_sem / sd_r
    gold_sem = gold_pp.std(1, ddof=1) / np.sqrt(N_PROMPTS) / sd_g
    hack_sem = hack_pp.std(1, ddof=1) / np.sqrt(N_PROMPTS) / sd_h

    # susceptibility spectrum + hack-mode flag, all before any optimization
    chi_ev = susceptibility(
        r.ravel(), np.column_stack([q.ravel(), h.ravel()]), ["quality", "spurious"]
    )
    spec = chi_ev.value
    gold_cov = [
        float(np.cov(q.ravel(), g.ravel(), ddof=0)[0, 1]),
        float(np.cov(h.ravel(), g.ravel(), ddof=0)[0, 1]),
    ]
    flagged = flag_hack_modes(spec, gold_cov)

    peak_i = int(np.argmax(gold))
    return dict(
        ns=ns, kl=kl, proxy=proxy, gold=gold, hack=hack,
        proxy_sem=proxy_sem, gold_sem=gold_sem, hack_sem=hack_sem,
        peak_i=peak_i, kl_peak=float(kl[peak_i]), n_peak=int(ns[peak_i]),
        gold_peak=float(gold[peak_i]), gold_end=float(gold[-1]),
        proxy_end=float(proxy[-1]), hack_end=float(hack[-1]),
        chi=dict(zip(spec.feature_names, [float(x) for x in spec.chi])),
        teacher_variance=float(spec.teacher_variance),
        gold_cov=dict(zip(spec.feature_names, gold_cov)),
        flagged=flagged, trust=str(chi_ev.trust), gauge=str(chi_ev.gauge),
        max_diff=max_diff,
    )


def build_figure(D, theme: str):
    c = apply_style(theme)
    plt.rcParams.update({
        "font.size": 11.0, "axes.titlesize": 12.0, "axes.labelsize": 10.5,
        "figure.dpi": 120,
        # fixed salt -> deterministic element ids, so a regenerated SVG is byte-stable
        # (no spurious git churn when the data has not changed)
        "svg.hashsalt": f"reward-lens-best-of-n-ladder-{theme}",
    })
    fig = plt.figure(figsize=(11.6, 4.85))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.94, 1.06], wspace=0.22,
                          left=0.062, right=0.84, top=0.80, bottom=0.125)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    ns, kl = D["ns"], D["kl"]
    kl_peak, n_peak = D["kl_peak"], D["n_peak"]

    # headline (the claim), left-aligned above both panels
    fig.text(0.062, 0.955, "You can price optimization pressure in nats before you spend it",
             ha="left", va="top", color=c["ink"], fontsize=14.0, fontweight="bold")

    # ---------------- Panel A: the exact price ----------------
    axL.set_title("The price: an exact KL, known in advance", loc="left", color=c["ink"], pad=8)
    axL.semilogx(ns, kl, color=c["gauge"], lw=2.3, zorder=3, solid_capstyle="round")
    axL.scatter(ns, kl, s=26, facecolor=c["gauge"], edgecolor="none", zorder=4)
    # tie-point shared with panel B (the safe budget)
    axL.scatter([n_peak], [kl_peak], s=95, facecolor="none",
                edgecolor=c["trust_adjudicated"], linewidth=2.0, zorder=5)
    axL.plot([ns[0], n_peak], [kl_peak, kl_peak], color=c["muted"], lw=0.9, ls=":", zorder=2)
    axL.annotate(f"{kl_peak:.2f} nats\n$n={n_peak}$", xy=(n_peak, kl_peak),
                 xytext=(72, 1.72), color=c["ink2"], fontsize=9.5,
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=c["muted"], lw=0.9))
    # closed form + the "no model, no data" point
    axL.text(0.045, 0.955, r"$\mathrm{KL}(bo_n\,\Vert\,\pi_0)=\ln n-\dfrac{n-1}{n}$",
             transform=axL.transAxes, color=c["ink"], fontsize=12.5, va="top", ha="left")
    axL.text(0.045, 0.815, "no model. no data. no fit.",
             transform=axL.transAxes, color=c["gauge"], fontsize=9.5, va="top", ha="left")
    axL.text(9200, 0.30,
             "powers of two, evenly spaced:\neach doubling of $n$ adds $\\approx\\ln 2 = 0.69$ nats",
             color=c["muted"], fontsize=9.0, ha="right", va="bottom", linespacing=1.3)
    axL.set_xlabel("best-of-$n$   ($n$, log scale)")
    axL.set_ylabel("KL from the base policy  (nats)")
    axL.set_xlim(0.9, 13000)
    axL.set_ylim(-0.15, 8.7)
    axL.set_xticks([1, 4, 16, 64, 256, 1024, 4096])
    axL.set_xticklabels(["1", "4", "16", "64", "256", "1024", "4096"])
    axL.grid(axis="y", alpha=0.55, lw=0.8)

    # ---------------- Panel B: what the nats buy ----------------
    axR.set_title("The frontier: proxy runs, true objective turns over  (zero RL)",
                  loc="left", color=c["ink"], pad=8)
    # over-optimized region (past the gold peak)
    axR.axvspan(kl_peak, 8.35, color=c["tier_vulnerability"], alpha=0.06, lw=0, zorder=0)
    axR.axhline(0, color=c["baseline"], lw=1.1, zorder=1)          # the base policy
    axR.axvline(kl_peak, color=c["muted"], lw=1.0, ls="--", zorder=2)

    series = [
        ("proxy", D["proxy"], D["proxy_sem"], c["chosen"], "-", 2.3, "o"),
        ("hack", D["hack"], D["hack_sem"], c["accent"], (0, (5, 2)), 2.1, "^"),
        ("gold", D["gold"], D["gold_sem"], c["trust_adjudicated"], "-", 2.7, "s"),
    ]
    for key, y, sem, col, ls, lw, mk in series:
        axR.fill_between(kl, y - sem, y + sem, color=col, alpha=0.16, lw=0, zorder=2)
        axR.plot(kl, y, color=col, lw=lw, ls=ls, zorder=4, solid_capstyle="round")
        axR.scatter(kl[-1], y[-1], s=34, facecolor=col, edgecolor="none", marker=mk, zorder=5)

    # gold peak marker + annotation
    axR.scatter([kl_peak], [D["gold_peak"]], s=60, facecolor=c["trust_adjudicated"],
                edgecolor="none", zorder=6)
    axR.annotate(f"gold peaks at {kl_peak:.1f} nats  ($n={n_peak}$)",
                 xy=(kl_peak, D["gold_peak"]), xytext=(kl_peak + 0.3, D["gold_peak"] + 0.88),
                 color=c["ink2"], fontsize=9.5, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=c["muted"], lw=1.0))
    axR.text(8.28, 2.42, "over-optimized:\npaying nats to lose ground", color=c["tier_vulnerability"],
             fontsize=9.0, ha="right", va="center", linespacing=1.25)

    # direct end-labels (ink text; the colored end-marker carries identity)
    def endlabel(y, text, dy):
        axR.annotate(text, xy=(kl[-1], y), xytext=(8.62, y + dy),
                     color=c["ink"], fontsize=9.5, ha="left", va="center",
                     annotation_clip=False,
                     arrowprops=dict(arrowstyle="-", color=c["line"], lw=0.8))
    endlabel(D["proxy_end"], "proxy reward", 0.5)
    endlabel(D["hack_end"], "flagged feature\n($\\chi>0$)", -0.66)
    endlabel(D["gold_end"], "gold objective\n(the true target)", -0.5)

    axR.set_xlabel("KL from the base policy  (nats)   —   the exact price from the left panel")
    axR.set_ylabel("shift from base policy  (base-policy SDs)")
    axR.set_xlim(-0.2, 9.4)
    axR.set_ylim(-1.15, 4.35)
    axR.set_xticks([0, 2, 4, 6, 8])
    axR.grid(axis="y", alpha=0.55, lw=0.8)

    out = os.path.join(OUT, f"best-of-n-ladder-{theme}.svg")
    # Date=None drops the creation-timestamp metadata, so the SVG is byte-deterministic
    fig.savefig(out, format="svg", transparent=True, metadata={"Date": None})
    plt.close(fig)
    print(f"  wrote {os.path.relpath(out, OUT)}")


def main():
    os.makedirs(OUT, exist_ok=True)
    D = compute()
    print("best-of-n-ladder  —  computed on CPU from reward_lens.loops")
    print(f"  helper vs bon_ladder max|diff| = {D['max_diff']:.2e}  (order-stat read validated)")
    chi = {k: round(v, 3) for k, v in D["chi"].items()}
    gcov = {k: round(v, 3) for k, v in D["gold_cov"].items()}
    print(f"  susceptibility chi = {chi}   (teacher variance {D['teacher_variance']:.3f})")
    print(f"  Cov0(feature, gold) = {gcov}")
    print(f"  flagged hack modes = {D['flagged']}   (trust {D['trust']}, gauge {D['gauge']})")
    print(f"  gold peak: n={D['n_peak']}, KL={D['kl_peak']:.3f} nats, +{D['gold_peak']:.2f} SD; "
          f"end n=10000: gold {D['gold_end']:+.2f} SD, proxy {D['proxy_end']:+.2f} SD, "
          f"hack {D['hack_end']:+.2f} SD")
    for theme in ("light", "dark"):
        build_figure(D, theme)
    print("done.")


if __name__ == "__main__":
    main()
