#!/usr/bin/env bash
#
# Compile every TikZ source in diagrams/tikz/*.tex into a light and a dark SVG under
# content/assets/figures/ (<name>-light.svg and <name>-dark.svg). The committed SVGs mean
# the site builds in CI with no LaTeX toolchain.
#
# Run from the diagrams/ directory:
#     ./build_figures.sh                 # build every source, both themes
#     ./build_figures.sh reward-projection trust-ladder   # build only these
#
# Dependencies (local, one-time):
#   - lualatex or xelatex (for the real Inter font) or pdflatex (falls back to Fira Sans)
#   - one PDF->SVG converter: pdf2svg (preferred), or dvisvgm, or inkscape
#     Debian/Ubuntu:  sudo apt-get install texlive-luatex texlive-fonts-extra pdf2svg
#     The Inter font (fonts-inter) makes the figures match the site exactly.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$HERE/tikz"
OUT_DIR="$HERE/../content/assets/figures"
export TEXINPUTS="$HERE:$SRC_DIR:${TEXINPUTS:-}"

mkdir -p "$OUT_DIR"

# One source of truth for color: regenerate the LaTeX/CSS/matplotlib palettes first.
if command -v python3 >/dev/null 2>&1; then
  python3 "$HERE/emit_palette.py"
fi

# Engine: prefer one that can load the system Inter via fontspec.
engine=""
for e in lualatex xelatex pdflatex; do
  if command -v "$e" >/dev/null 2>&1; then engine="$e"; break; fi
done
if [ -z "$engine" ]; then
  echo "error: need lualatex, xelatex, or pdflatex on PATH." >&2
  exit 1
fi

converter=""
for c in pdf2svg dvisvgm inkscape; do
  if command -v "$c" >/dev/null 2>&1; then converter="$c"; break; fi
done
if [ -z "$converter" ]; then
  echo "error: need one of pdf2svg / dvisvgm / inkscape on PATH." >&2
  exit 1
fi

echo "engine: $engine   converter: $converter"

# Build the requested sources, or every source if none named.
shopt -s nullglob
if [ "$#" -gt 0 ]; then
  tex_files=(); for n in "$@"; do tex_files+=("$SRC_DIR/${n%.tex}.tex"); done
else
  tex_files=("$SRC_DIR"/*.tex)
fi
if [ ${#tex_files[@]} -eq 0 ]; then
  echo "no .tex sources to build in $SRC_DIR."
  exit 0
fi

convert_one() { # $1 pdf  $2 svg
  case "$converter" in
    pdf2svg)  pdf2svg "$1" "$2" ;;
    dvisvgm)  dvisvgm --pdf "$1" -o "$2" >/dev/null 2>&1 ;;
    inkscape) inkscape "$1" --export-type=svg --export-filename="$2" >/dev/null 2>&1 ;;
  esac
}

fail=0
for tex in "${tex_files[@]}"; do
  [ -f "$tex" ] || { echo "  skip (missing): $tex"; continue; }
  name="$(basename "$tex" .tex)"
  for theme in light dark; do
    work="$(mktemp -d)"
    job="$name-$theme"
    if "$engine" -interaction=nonstopmode -halt-on-error \
        -output-directory "$work" -jobname "$job" \
        "\def\rltheme{$theme}\input{$tex}" >"$work/log" 2>&1; then
      convert_one "$work/$job.pdf" "$OUT_DIR/$job.svg"
      echo "  $job.svg"
    else
      echo "ERROR building $job:" >&2
      tail -20 "$work/log" >&2
      fail=1
    fi
    rm -rf "$work"
  done
done

[ "$fail" -eq 0 ] || { echo "one or more figures failed." >&2; exit 1; }
echo "done -> $OUT_DIR"
echo "embed dual-theme with:"
echo "  ![Alt](assets/figures/<name>-light.svg#only-light)"
echo "  ![Alt](assets/figures/<name>-dark.svg#only-dark)"
