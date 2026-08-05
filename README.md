# Spin–valley model uncertainty in micromagnet-assisted shuttling

Simulation code and data accompanying the paper *"Spin–valley model
uncertainty in micromagnet-assisted shuttling: a periodic-reference
sensitivity atlas."*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21588003.svg)](https://doi.org/10.5281/zenodo.21588003)

How strongly do the conclusions of a valley-aware co-design analysis depend on
the **assumed form of the spin–valley coupling**? We evaluate four
phenomenological coupling ansätze (A, A_pocket, B_z, B_x) over a grid of
geometry and trajectory conditions, using a periodic micromagnet field as a
**controlled reference geometry** motivated by proposed shuttling-bus
architectures. We then quantify how the choice of ansatz changes (i) the
*ranking* of operating conditions by response magnitude and (ii) the
thresholded *classification* of each condition (robust / trade-off /
both-worsen / single-channel / below-threshold).

The four profiles are **diagnostic ansätze**, not microscopically derived
spin–valley Hamiltonians, and the atlas measures sensitivity to that modeling
choice rather than absolute shuttling performance. Throughout, `n_real` is the
number of independent charge-noise realizations averaged per condition — Monte
Carlo repetitions of the simulated noise, not experimental repetitions.

## Headline results

Over the full validation grid of 432 model-conditions, in the adopted
total-field convention `E_Z(x) = g μ_B [B_ext + B_z(x)]`:

- The **ranking** of operating conditions is largely conserved across ansätze:
  mean Spearman ρ ≈ **0.84**.
- The **thresholded classification** is not: mean exact-label agreement ≈
  **36%**, lowest for the pairs contrasting the two field-gradient components.
- Correcting the Zeeman convention alone moves ρ from **0.68 to 0.84** and
  agreement from **28.9% to 36.0%**. A bookkeeping choice changes design-level
  conclusions, which is itself a reported result.
- Whether a cross-ansatz robust operating point exists at all is
  convention-dependent. In the adopted atlas a targeted `n_real = 30` re-test
  confirms two survivors; in the legacy convention none survive.

Both layers are reproducible from this repository; see `REPRODUCIBILITY.md`.

## Install

```bash
pip install -r requirements.txt   # Python 3.12; JAX on CPU is sufficient
export PYTHONPATH=. JAX_ENABLE_X64=1
```

float64 is required — see `REPRODUCIBILITY.md`.

## Verify the headline numbers

The raw per-condition data from our run are included, so the merge step alone
reproduces the reported values without recomputation:

```bash
python reproduce/phase5_atlas_merge_validate.py --mode validate \
    --ez-convention total-local --profile-norm prefactor
# mean rank correlation: +0.837 (HIGH)
# mean quadrant agreement: 36.0%
```

The archived command prints `quadrant agreement`; this is the nine-category
classification agreement defined above.

The archived legacy layer merges with `--allow-legacy` and prints `+0.677` and
`28.9%`.

## Recompute the full atlas

Two cases, then a merge (a few minutes each on CPU):

```bash
python reproduce/phase5_sensitivity_atlas.py --mode validate --case case_i_center \
    --save-raw --ez-convention total-local --profile-norm prefactor
python reproduce/phase5_sensitivity_atlas.py --mode validate --case case_ii_edge \
    --save-raw --ez-convention total-local --profile-norm prefactor
python reproduce/phase5_atlas_merge_validate.py --mode validate \
    --ez-convention total-local --profile-norm prefactor
```

Full validation commands, expected values, and tolerances are documented in
`REPRODUCIBILITY.md`.

## Repository layout

```
geometry/  noise/  reproduce/  tests/   # field model, noise, analyses, checks
constants.py, field_landscape.py        # constants and analytic landscape
docs/                                   # manuscript sources and figures
figures/                                # generated figures, CSVs, raw .pkl data
REPRODUCIBILITY.md                      # reproduction recipe and reference values
```

## Build the manuscript

The manuscript targets *Physical Review Applied* and uses `revtex4-2`.

`docs/figures/` already contains the adopted manuscript figures, so the build
needs no preparation step:

```bash
cd docs
pdflatex paper.tex && (bibtex paper || bibtexu paper) && pdflatex paper.tex && pdflatex paper.tex
```

Run `docs/collect_figures.sh "__ez-total-local__norm-prefactor"` only after
regenerating the figure outputs yourself; called without that argument it
installs the archived legacy figures instead.

For a standalone source bundle — a new Overleaf project, or a submission
package — `bash make_overleaf.sh` produces `overleaf_bundle.zip` from the
repository root. Existing Overleaf projects can instead be updated by uploading
`paper.tex`, `supplementary.tex`, and `ref.bib` directly.

## Citation

Please cite the accompanying paper; see `CITATION.cff`. To cite this software
archive, use the version DOI
[10.5281/zenodo.21588784](https://doi.org/10.5281/zenodo.21588784), or the
concept DOI [10.5281/zenodo.21588003](https://doi.org/10.5281/zenodo.21588003)
to resolve to the latest release.

## License

Code under the MIT License (`LICENSES/MIT.txt`). Figures and data under
`figures/` may be reused under CC-BY 4.0 (`LICENSES/CC-BY-4.0.txt`) with
attribution to the accompanying paper.
