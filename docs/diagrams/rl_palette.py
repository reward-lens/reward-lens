"""AUTO-GENERATED from palette.json by emit_palette.py. Do not edit by hand.

The house matplotlib palette, keyed by semantic role. Import this instead of
hardcoding hex, so every empirical figure stays in step with the TikZ figures
and the site CSS.

    from rl_palette import palette, apply_style
    c = palette("dark")
    apply_style("dark")            # sets rcParams for a transparent, theme-matched figure
    ax.plot(x, y, color=c["chosen"])
"""
from __future__ import annotations

LIGHT = {
    "ink": "#16181D",
    "ink2": "#4B5563",
    "muted": "#6B7280",
    "line": "#D7DAE0",
    "surface": "#FFFFFF",
    "surface2": "#F4F5F7",
    "baseline": "#C3C2B7",
    "tier_observational": "#0D9488",
    "tier_causal": "#B45309",
    "tier_vulnerability": "#BE123C",
    "trust_exploratory": "#94A3B8",
    "trust_calibrated": "#3B82F6",
    "trust_registered": "#7C3AED",
    "trust_adjudicated": "#15803D",
    "signal": "#A21CAF",
    "gauge": "#0891B2",
    "chosen": "#2A78D6",
    "rejected": "#E34948",
    "series_b": "#EB6834",
    "accent": "#E85D2C",
}

DARK = {
    "ink": "#E6E8EB",
    "ink2": "#B7BDC7",
    "muted": "#9AA0AA",
    "line": "#3A3D44",
    "surface": "#1D1E22",
    "surface2": "#26272C",
    "baseline": "#55565E",
    "tier_observational": "#2DD4BF",
    "tier_causal": "#F59E0B",
    "tier_vulnerability": "#FB7185",
    "trust_exploratory": "#64748B",
    "trust_calibrated": "#60A5FA",
    "trust_registered": "#A78BFA",
    "trust_adjudicated": "#22C55E",
    "signal": "#E879F9",
    "gauge": "#22D3EE",
    "chosen": "#5598E7",
    "rejected": "#F0716F",
    "series_b": "#F98A5C",
    "accent": "#FB8C5A",
}

SEQUENTIAL = {"light": ['#CDE2FB', '#0D366B'], "dark": ['#0D366B', '#CDE2FB']}
DIVERGING = {"light": ['#2A78D6', '#F0EFEC', '#E34948'], "dark": ['#5598E7', '#2A2B31', '#F0716F']}


def palette(theme: str = "light") -> dict:
    """Return the role->hex map for a theme ("light" or "dark")."""
    return dict(LIGHT if theme == "light" else DARK)


def apply_style(theme: str = "light") -> dict:
    """Set matplotlib rcParams for a theme-matched, transparent figure and return the palette.

    Backgrounds are transparent, not hardcoded white, so the SVG sits correctly on either
    the light or the dark page. Text and spines take the theme ink color.
    """
    import matplotlib as mpl

    c = palette(theme)
    mpl.rcParams.update({
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "text.color": c["ink"],
        "axes.edgecolor": c["line"],
        "axes.labelcolor": c["ink"],
        "xtick.color": c["muted"],
        "ytick.color": c["muted"],
        "grid.color": c["line"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "DejaVu Sans", "Arial"],
        "svg.fonttype": "path",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return c
