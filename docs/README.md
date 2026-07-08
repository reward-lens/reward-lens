# reward-lens docs

The documentation site, built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).
Self-contained in this folder.

## Layout

```
docs/
  mkdocs.yml              # site config: theme, plugins, nav
  requirements-docs.txt   # build deps (no torch needed)
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

Documentation publishes to GitHub Pages automatically. The
[`Deploy Docs to GitHub Pages`](../.github/workflows/docs.yml) workflow builds
the site with MkDocs on every push to the default branch and serves the built
HTML directly, so Pages never falls back to rendering a README.

One-time repository setup: **Settings → Pages → Build and deployment → Source →
GitHub Actions**. After that, every push to `main` rebuilds and redeploys.

A ReadTheDocs build also works (point it at `docs/mkdocs.yml`). `site_url` in
`mkdocs.yml` assumes GitHub Pages; change it to match your host.
