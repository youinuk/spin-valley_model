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
| cross-ansatz disagreement | 0.388 | 0.395 |
| cross-seed disagreement | 0.385 | 0.298 |

`n_real = 30` is the pre-registered endpoint; `n_real = 100` is the endpoint of
a registered follow-up extension.

- Cross-ansatz disagreement **persists** across the eight registered
  checkpoints, staying within a 2.6-point band, while cross-seed disagreement
  falls monotonically from 0.542 to 0.298. The ordering reverses between
  `n_real = 20` and `30`.
- Magnitude **ranking** stays high across the family (mean Spearman ρ = 0.913 at
  `n_real = 100`) while classification agreement spans 0.467 to 0.839. Two
  ansatz pairs whose rank correlations differ by 0.011 differ by 37 percentage
  points in classification agreement.
- The balance is **observable dependent**: resolving the label into its two
  channels, the cross-ansatz to cross-seed ratio at `n_real = 100` is 5.60 for
  spin dephasing and 0.96 for valley excitation.
- Operating points selected for stability across noise samples alone replicate
  within their own ansatz (21 of 25 at `n_real = 30`) but **transport unevenly**
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

The canonical analysis outputs are shipped, so the primary comparison is
checkable without any simulation:

```bash
python -c "
import json
a = json.load(open('data/t0c/analysis/attribution_n100.json'))['by_n']
for n in ('30', '100'):
    v = a[n]
    print(f\"n_real={n:>4}  D_ansatz={v['D_ansatz']:.3f}\"
          f\"  D_seed={v['D_seed']:.3f}  ratio={v['R_D']:.2f}\")
"
# n_real=  30  D_ansatz=0.388  D_seed=0.385  ratio=1.01
# n_real= 100  D_ansatz=0.395  D_seed=0.298  ratio=1.33
```

`data/t0c/analysis/` holds fourteen such files; `REPRODUCIBILITY.md` Sec. 3
maps each one to the numbers it supplies.

`data/t0c/analysis/` is the canonical analysis layer for the current
manuscript. `data/t0/analysis/` ships alongside it as **superseded historical
provenance** for the pre-correction calculation: it is not a source of any
current number. The two layers may be compared with
`reproduce/t0/compare_t0_t0c.py`, which documents the valley-ordering
correction. That comparison is a correction audit and must not be read as a
scientific perturbation of one estimand — the two layers use opposite
valley-energy orderings, so the same symbol names different quantities in the
two columns.

## Regenerate the manuscript figures

Seven commands, documented with expected output in `REPRODUCIBILITY.md`
Sec. 2. Run them into a wiped `figures/` and all eight manuscript figures come
back; that is the acceptance test the release checker performs in Sec. 6.

```bash
# 1. Lift the per-condition responses out of the raw layer. Fig. 3 needs them
#    and they ship as data/t0c/responses_n100.csv.
python reproduce/t0/export_responses.py ../repro-runs/t0c-step3b \
    data/t0c/responses_n100.csv

# 2. Figs. 2, 3, 4 and S5 from one producer. --ez-convention and
#    --profile-norm have no defaults and are required whenever Fig. 2 is
#    requested, so a bare invocation cannot build Fig. 2 in the archival
#    convention beside three corrected figures.
PYTHONPATH=. python reproduce/phase5_paper_figures.py --figures all \
    --ez-convention total-local --profile-norm prefactor \
    --responses data/t0c/responses_n100.csv --analysis data/t0c/analysis

# 3. The four supplement validation figures. Each producer also prints the
#    numbers its caption quotes.
PYTHONPATH=. python reproduce/krzywda_B1_stationary.py
PYTHONPATH=. python reproduce/krzywda_B3_filter.py
PYTHONPATH=. python reproduce/oda_C1_lz_single.py
PYTHONPATH=. python geometry/prism_field.py

# 4. Install the eight manuscript figures into docs/figures/.
docs/collect_figures.sh "__ez-total-local__norm-prefactor"
```

`collect_figures.sh` accepts only the adopted dataset suffix and fails rather
than installing anything else. Recomputing the underlying atlas from scratch,
the validation controls, and the tolerances are all in `REPRODUCIBILITY.md`.

## Repository layout

```
geometry/  noise/  reproduce/  tests/   field model, noise, analyses, checks
constants.py, field_landscape.py        constants and analytic landscape
data/t0c/analysis/                      canonical corrected analysis outputs (14 JSON)
data/t0/analysis/                       superseded historical outputs (14 JSON)
reproduce/t0/                           provenance tools for the raw layer, the response
                                        exporter, and the T0/T0-C correction audit
docs/                                   manuscript sources and figures
figures/                                generated figures, CSVs, archival data
REPRODUCIBILITY.md                      commands, expected values, tolerances
```

The corrected production pickles are distributed in the Zenodo raw-data layer
rather than here; every current manuscript number is reproducible from
`data/t0c/analysis/`. `data/t0/analysis/` is retained only as superseded
historical provenance and is not a regeneration target: the convention that
produced it is no longer implemented in this tree.

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
