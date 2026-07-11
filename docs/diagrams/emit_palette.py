#!/usr/bin/env python3
"""Generate every color consumer from palette.json, the single source of truth.

Writes three files from `palette.json`:
  - `_palette.tex`                          the LaTeX color definitions every TikZ figure loads
  - `rl_palette.py`                         the matplotlib palette the empirical figures import
  - `../content/assets/stylesheets/_palette.css`  the CSS custom properties the site loads

Run it after editing palette.json:
    python emit_palette.py

Check for drift in CI or before a commit (nonzero exit if any consumer is stale):
    python emit_palette.py --check

No third-party dependencies. Runs anywhere Python does.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PALETTE = HERE / "palette.json"
OUT_TEX = HERE / "_palette.tex"
OUT_PY = HERE / "rl_palette.py"
OUT_CSS = HERE.parent / "content" / "assets" / "stylesheets" / "_palette.css"

BANNER = "AUTO-GENERATED from palette.json by emit_palette.py. Do not edit by hand."


def load() -> dict:
    with PALETTE.open() as fh:
        return json.load(fh)


def _hex(value: str) -> str:
    return value.lstrip("#").upper()


def render_tex(data: dict) -> str:
    roles = data["roles"]
    ramps = data.get("ramps", {})

    def defs(theme: str) -> str:
        lines = [f"  \\definecolor{{{r['tex']}}}{{HTML}}{{{_hex(r[theme])}}}%" for r in roles.values()]
        return "\n".join(lines)

    seq = ramps.get("sequential", {})
    div = ramps.get("diverging", {})
    chosen = roles["chosen"]
    rejected = roles["rejected"]

    def ramp_defs(theme: str) -> str:
        out = []
        if seq:
            lo, hi = seq[theme]
            out.append(f"  \\definecolor{{rlSeqLo}}{{HTML}}{{{_hex(lo)}}}%")
            out.append(f"  \\definecolor{{rlSeqHi}}{{HTML}}{{{_hex(hi)}}}%")
        if div:
            out.append(f"  \\definecolor{{rlDivMid}}{{HTML}}{{{_hex(div['mid'][theme])}}}%")
        return "\n".join(out)

    return (
        f"% {BANNER}\n"
        "% Each figure sets \\rltheme to light or dark before \\input; the preamble loads this file.\n"
        "\\usepackage{etoolbox}\n"
        "\\providecommand{\\rltheme}{light}%\n"
        "\\ifdefstring{\\rltheme}{dark}{%\n"
        f"{defs('dark')}\n"
        f"{ramp_defs('dark')}\n"
        "}{%\n"
        f"{defs('light')}\n"
        f"{ramp_defs('light')}\n"
        "}%\n"
    )


def render_py(data: dict) -> str:
    roles = data["roles"]
    ramps = data.get("ramps", {})
    light = {k: v["light"].upper() for k, v in roles.items()}
    dark = {k: v["dark"].upper() for k, v in roles.items()}
    seq = ramps.get("sequential", {})
    div_mid = ramps.get("diverging", {}).get("mid", {})

    def dump(d: dict) -> str:
        return "{\n" + "".join(f'    "{k}": "{v}",\n' for k, v in d.items()) + "}"

    seq_light = seq.get("light", ["#CDE2FB", "#0D366B"])
    seq_dark = seq.get("dark", ["#0D366B", "#CDE2FB"])
    div_light = [light["chosen"], (div_mid.get("light", "#F0EFEC")).upper(), light["rejected"]]
    div_dark = [dark["chosen"], (div_mid.get("dark", "#2A2B31")).upper(), dark["rejected"]]

    return f'''"""{BANNER}

The house matplotlib palette, keyed by semantic role. Import this instead of
hardcoding hex, so every empirical figure stays in step with the TikZ figures
and the site CSS.

    from rl_palette import palette, apply_style
    c = palette("dark")
    apply_style("dark")            # sets rcParams for a transparent, theme-matched figure
    ax.plot(x, y, color=c["chosen"])
"""
from __future__ import annotations

LIGHT = {dump(light)}

DARK = {dump(dark)}

SEQUENTIAL = {{"light": {seq_light!r}, "dark": {seq_dark!r}}}
DIVERGING = {{"light": {div_light!r}, "dark": {div_dark!r}}}


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
    mpl.rcParams.update({{
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
    }})
    return c
'''


def render_css(data: dict) -> str:
    roles = data["roles"]

    def block(theme: str) -> str:
        lines = [f"  --rl-{k.replace('_', '-')}: {v[theme].upper()};" for k, v in roles.items()]
        return "\n".join(lines)

    return (
        f"/* {BANNER} */\n"
        "/* Consumed by extra.css. Light values are the default; the slate scheme overrides. */\n"
        ":root {\n"
        f"{block('light')}\n"
        "}\n\n"
        '[data-md-color-scheme="slate"] {\n'
        f"{block('dark')}\n"
        "}\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail if any consumer is stale, write nothing")
    args = ap.parse_args()

    data = load()
    targets = {OUT_TEX: render_tex(data), OUT_PY: render_py(data), OUT_CSS: render_css(data)}

    if args.check:
        stale = []
        for path, content in targets.items():
            current = path.read_text() if path.exists() else None
            if current != content:
                stale.append(path)
        if stale:
            print("palette drift: these files are out of date with palette.json:", file=sys.stderr)
            for p in stale:
                print(f"  {p}", file=sys.stderr)
            print("run `python emit_palette.py` to regenerate.", file=sys.stderr)
            return 1
        print("palette consumers are in step with palette.json.")
        return 0

    for path, content in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        print(f"wrote {path.relative_to(HERE.parent.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
