# reward-lens docs

The documentation site, built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).
Self-contained in this folder. The content is authored per
[`AUTHORING_PROMPT.md`](AUTHORING_PROMPT.md) — read that first if you are the one
writing (or generating) the pages.

## Layout

```
docs/
  mkdocs.yml              # site config: theme, plugins, nav
  requirements-docs.txt   # build deps (no torch needed)
  AUTHORING_PROMPT.md     # the brief: what to write and how
  content/                # docs_dir — every page lives here
    index.md              # Home / Why reward-lens
    concepts/  getting-started/  tutorials/  how-to/
    tools/  theory/  reference/  contributing/  caveats.md
    assets/               # css, js (mathjax), figures/
  diagrams/               # figure sources + pipeline (not served)
    tikz/                 # TikZ sources -> SVG
    build_figures.sh      # compile TikZ into content/assets/figures/
```

## Build and preview

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r docs/requirements-docs.txt

# live preview at http://127.0.0.1:8000
mkdocs serve -f docs/mkdocs.yml

# one-off build into docs/_site
mkdocs build -f docs/mkdocs.yml
```

The API reference is generated from `src/` by griffe via static analysis, so you
do **not** need `torch`/`transformers` installed to build the docs.

## Figures

```bash
cd docs
./diagrams/build_figures.sh     # needs a local LaTeX toolchain + pdf2svg
```

See [`diagrams/README.md`](diagrams/README.md) for the three figure kinds (TikZ
geometry, inline Mermaid, matplotlib-from-real-runs) and when to use each.

## Deploy

The repo is not currently a git repository. To publish on GitHub Pages:

```bash
git init && git add -A && git commit -m "docs"
git remote add origin https://github.com/suhailnadaf509/reward-lens.git
mkdocs gh-deploy -f docs/mkdocs.yml   # pushes built site to the gh-pages branch
```

A ReadTheDocs build also works (point it at `docs/mkdocs.yml`). `site_url` in
`mkdocs.yml` currently assumes GitHub Pages; change it to match your host.
