#!/usr/bin/env bash
#
# Compile every TikZ source in diagrams/tikz/*.tex into an SVG under
# content/assets/figures/. SVGs are committed, so the site builds (and deploys
# on ReadTheDocs / GitHub Pages) with no LaTeX toolchain in CI.
#
# Run from the docs/ directory:
#     ./diagrams/build_figures.sh
#
# Dependencies (local only, one-time):
#     - a LaTeX distribution providing pdflatex   (TeX Live / MacTeX)
#     - one PDF->SVG converter: pdf2svg  (preferred), or dvisvgm, or inkscape
#         Debian/Ubuntu:  sudo apt-get install texlive-pictures pdf2svg
#         macOS (brew):   brew install --cask mactex-no-gui && brew install pdf2svg

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$HERE/tikz"
OUT_DIR="$HERE/../content/assets/figures"

mkdir -p "$OUT_DIR"

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "error: pdflatex not found. Install TeX Live (texlive-pictures)." >&2
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

shopt -s nullglob
tex_files=("$SRC_DIR"/*.tex)
if [ ${#tex_files[@]} -eq 0 ]; then
  echo "no .tex sources in $SRC_DIR — nothing to build."
  exit 0
fi

for tex in "${tex_files[@]}"; do
  name="$(basename "$tex" .tex)"
  work="$(mktemp -d)"
  echo "compiling $name ..."
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$work" "$tex" >/dev/null
  case "$converter" in
    pdf2svg)  pdf2svg "$work/$name.pdf" "$OUT_DIR/$name.svg" ;;
    dvisvgm)  dvisvgm --pdf "$work/$name.pdf" -o "$OUT_DIR/$name.svg" >/dev/null 2>&1 ;;
    inkscape) inkscape "$work/$name.pdf" --export-type=svg --export-filename="$OUT_DIR/$name.svg" >/dev/null 2>&1 ;;
  esac
  rm -rf "$work"
  echo "  -> $OUT_DIR/$name.svg"
done

echo "done. Embed with:  ![caption](assets/figures/<name>.svg)"
