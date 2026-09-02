# Model-form versus finite-sampling uncertainty in micromagnet-assisted spin-valley shuttling

Simulation code and analysis outputs accompanying the paper *"Model-form versus
finite-sampling uncertainty in micromagnet-assisted spin-valley shuttling: a
periodic-reference sensitivity atlas."*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21588003.svg)](https://doi.org/10.5281/zenodo.21588003)

How strongly do the conclusions of a valley-aware co-design analysis depend on
the **assumed spatial form of the spin–valley coupling** — and how does that
dependence compare with the variation the same calculation shows when only the
random sample changes?

We evaluate four phenomenological coupling ansätze (A, A_pocket, B_z, B_x) over
a 108-condition grid of geometry and trajectory settings, using a periodic
micromagnet field as a **controlled reference geometry** motivated by proposed
shuttling-bus architectures. Two comparisons are placed on one scale:

- **cross-ansatz** — change the coupling model on a fixed set of noise
  realizations;
- **cross-seed** — change the noise sample under a fixed coupling model.

Both are measured as the fraction of conditions receiving different
nine-category thresholded labels, formed by classifying each of the two
response channels as improve, inactive, or worsen. Within each of five seed
blocks the four ansätze are driven by common random numbers, and each block is
evaluated at nested realization counts `n_real = 5 … 100`.

The four profiles are **diagnostic ansätze**, not microscopically derived
spin–valley Hamiltonians, and the study measures sensitivity to that modelling
choice rather than absolute shuttling performance. `n_real` counts Monte Carlo
repetitions of the simulated noise, not experimental repetitions.

## Headline results

| | `n_real = 30` | `n_real = 100` |
| --- | --- | --- |
| cross-ansatz disagreement | 0.394 | 0.396 |
| cross-seed disagreement | 0.389 | 0.302 |

`n_real = 30` is the pre-registered endpoint; `n_real = 100` is the endpoint of
a registered follow-up extension.

- Cross-ansatz disagreement **persists** across the eight registered
  checkpoints, staying within a 2.1-point band, while cross-seed disagreement
  falls monotonically from 0.540 to 0.302. The ordering reverses between
  `n_real = 20` and `30`.
- Magnitude **ranking** stays high across the family (mean Spearman ρ = 0.913 at
  `n_real = 100`) while classification agreement spans 0.483 to 0.817. Two
  ansatz pairs whose rank correlations differ by 0.003 differ by 33 percentage
  points in classification agreement.
- The balance is **observable dependent**: resolving the label into its two
  channels, the cross-ansatz to cross-seed ratio at `n_real = 100` is 4.75 for
  spin dephasing and 0.94 for valley excitation.
- Operating points selected for stability across noise samples alone replicate
  within their own ansatz (20 of 24 at `n_real = 30`) but **transport unevenly**
  across the family: of ten conditions, three are confirmed under all four
  ansätze and one under none.

Design conclusions obtained under one assumed coupling profile therefore remain
model-conditional.

## Install

```bash
pip install -r requirements.txt   # Python 3.12; JAX on CPU is sufficient
export PYTHONPATH=. JAX_ENABLE_X64=1
```

float64 is required, and pinning the numeric libraries to one thread is worth
roughly 4.8× on the atlas — see `REPRODUCIBILITY.md`.

## Verify a headline number

The closed analysis outputs are shipped, so the primary comparison is
checkable without any simulation:

```bash
python -c "
import json
a = json.load(open('data/t0/analysis/attribution_n100.json'))['by_n']
for n in ('30', '100'):
    v = a[n]
    print(f\"n_real={n:>4}  D_ansatz={v['D_ansatz']:.3f}\"
          f\"  D_seed={v['D_seed']:.3f}  ratio={v['R_D']:.2f}\")
"
# n_real=  30  D_ansatz=0.394  D_seed=0.389  ratio=1.01
# n_real= 100  D_ansatz=0.396  D_seed=0.302  ratio=1.31
```

`data/t0/analysis/` holds fourteen such files; `REPRODUCIBILITY.md` Sec. 3 maps
each one to the numbers it supplies.

## Regenerate the manuscript figures

Four commands, documented with expected output in `REPRODUCIBILITY.md` Sec. 2:

```bash
python reproduce/phase5_paper_figures.py \
    --ez-convention total-local --profile-norm prefactor
mkdir -p figures/t0
python reproduce/t0/make_fig4.py  data/t0/analysis figures/t0/sensitivity_atlas.pdf
python reproduce/t0/make_supp_robustness.py data/t0/analysis figures/t0/robustness_checks.pdf
docs/collect_figures.sh "__ez-total-local__norm-prefactor"
```

`collect_figures.sh` accepts only the adopted dataset suffix and fails rather
than installing anything else. Recomputing the underlying atlas from scratch,
the validation controls, and the tolerances are all in `REPRODUCIBILITY.md`.

## Repository layout

```
geometry/  noise/  reproduce/  tests/   field model, noise, analyses, checks
constants.py, field_landscape.py        constants and analytic landscape
data/t0/analysis/                       closed T0 analysis outputs (14 JSON)
reproduce/t0/                           Fig. 4 and the supplement robustness figure, and the
                                        provenance tools for the raw layer
docs/                                   manuscript sources and figures
figures/                                generated figures, CSVs, archival data
REPRODUCIBILITY.md                      commands, expected values, tolerances
```

The raw T0 production pickles are distributed in the Zenodo raw-data layer
rather than here; everything the manuscript reports about T0 is reproducible
from `data/t0/analysis/`.

## Build the manuscript

The manuscript targets the *Journal of Applied Physics* and uses `revtex4-2`
with the AIP substyle.
`docs/figures/` already contains the adopted figures, so the build needs no
preparation step:

```bash
cd docs
pdflatex paper.tex && (bibtex paper || bibtexu paper) && pdflatex paper.tex && pdflatex paper.tex
```

Run `docs/collect_figures.sh` only after regenerating the figure outputs
yourself. For a standalone source bundle — a new Overleaf project, or a
submission package — `bash make_overleaf.sh` produces `overleaf_bundle.zip`
from the repository root. Existing Overleaf projects can be updated by
uploading `paper.tex`, `supplementary.tex`, and `ref.bib` directly.

## Relation to the earlier release

`r31` is the immutable predecessor. Its atlas drew a separate noise realization
for each ansatz, so a cross-ansatz difference there also carried a sampling
difference; `r32` is a **scientific successor**, not a repackaging. The r31
headline values are historical reference, and the r31 dataset and its figures
remain in the r31 tar and Zenodo record rather than in this tree.
`REPRODUCIBILITY.md` Sec. 6 records the one comparison the paper still makes
against them.

## Citation

Please cite the accompanying paper; see `CITATION.cff`. To cite this software
archive, use the version DOI
[10.5281/zenodo.21588784](https://doi.org/10.5281/zenodo.21588784), or the
concept DOI [10.5281/zenodo.21588003](https://doi.org/10.5281/zenodo.21588003)
to resolve to the latest release.

## License

Code under the MIT License (`LICENSES/MIT.txt`). Figures and data under
`figures/` and `data/` may be reused under CC-BY 4.0
(`LICENSES/CC-BY-4.0.txt`) with attribution to the accompanying paper.
