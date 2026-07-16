# Building the manuscript on Overleaf

This `docs/` folder is the manuscript source. It is self-contained: the
two documents compile from the `.tex`, the `.bib`, and the figure PDFs in
`figures/`.

## Quick start

From the repository root:

```bash
bash make_overleaf.sh        # creates overleaf_bundle.zip
```

Then in Overleaf: **New Project → Upload Project →** select
`overleaf_bundle.zip`. Set the **main document** to `paper.tex`
(Menu → Main document) and compile with **pdfLaTeX**.
`supplementary.tex` is a second compilable document in the same project.

## What Overleaf needs

- `paper.tex` — main manuscript (RevTeX 4-2, `prapplied`)
- `supplementary.tex` — supplementary material
- `ref.bib` — bibliography
- `figures/*.pdf` — all figures (referenced via `\includegraphics`)

The `revtex4-2` class is available on Overleaf by default. No custom
packages are required beyond a standard TeX Live.

## Not needed to compile (kept here for provenance)

- `collect_figures.sh` — copies regenerated figures from `../figures/phase5/`
  into `figures/` (used only when reproducing figures locally)
- `supplement_data/` — machine-generated data tables behind the
  supplementary numbers

## Local build (instead of Overleaf)

```bash
cd docs
pdflatex paper.tex && (bibtex paper || bibtexu paper) && pdflatex paper.tex && pdflatex paper.tex
pdflatex supplementary.tex && (bibtex supplementary || bibtexu supplementary) && pdflatex supplementary.tex && pdflatex supplementary.tex
```
