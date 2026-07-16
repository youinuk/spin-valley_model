# Spin–valley model uncertainty in micromagnet-assisted shuttling

Simulation code and data accompanying the paper *"Spin–valley model
uncertainty in micromagnet-assisted shuttling: a periodic-reference
sensitivity atlas."*

The study asks how strongly the conclusions of a valley-aware co-design
analysis depend on the **assumed form of the spin-valley coupling**. We
evaluate four phenomenological coupling ansätze (A, A_pocket, B_z, B_x)
over a grid of geometry and trajectory conditions, using a periodic
micromagnet field as a **controlled reference geometry** (motivated by
proposed shuttling-bus architectures, not a claim of a mature device),
and quantify how the choice of ansatz changes (i) the *ranking* of
operating conditions by response magnitude and (ii) the *quadrant*
classification of each condition (robust / trade-off / both-worsen).

**Adopted result (paper headline, total-field convention).** The atlas
is computed with the physically transparent total-field Zeeman energy
E_Z(x) = gμ_B[B_ext + B_z(x)]. On the full validate grid the ranking is
largely conserved across ansätze (mean Spearman ρ ≈ 0.84), whereas the
quadrant interpretation remains strongly ansatz-dependent (mean
agreement ≈ 36%), with the lowest agreement in pairs contrasting the two
magnetic-field-gradient components.

**Three result layers in this package.** To keep provenance auditable,
the repository deliberately carries three distinct layers:

1. *Archived legacy reproduction* — the original stray-field-only run
   (mean ρ = 0.677, agreement 28.9%), preserved byte-for-byte so the
   published archive stays reproducible.
2. *Adopted total-local result* — the paper's headline numbers
   (mean ρ = 0.837, agreement 36.0%), produced by re-running the atlas
   with `--ez-convention total-local --profile-norm prefactor`.
3. *Convention sensitivity / provenance* — the **difference** between
   layers 1 and 2 is itself part of the result: correcting the Zeeman
   convention alone moves ρ from 0.68 to 0.84 and dissolves the
   legacy "B_x ranking divergence," which is primary evidence that
   modeling choices, not only ansatz choice, control the conclusions.

**Convention-dependent robust region.** Whether any cross-ansatz robust
operating point exists is itself convention-dependent. In the legacy
convention the staged search finds none surviving high statistics; in
the adopted total-local atlas, four conditions are robust in ≥3/4
ansätze at n_real=5, and a targeted n_real=30 re-test
(`reproduce/phase5_robust_candidate_retest.py`) confirms that two
survive as robust in all four ansätze (edge pocket, v=10 m/s, deeper
well, λ ∈ {0.5, 1.0} µeV) while two others vanish as small-sample
fluctuations.

The atlas is a **diagnostic of model uncertainty**, not an optimization:
it does not rank absolute shuttling performance. The two response
observables ΔP_v (valley-leakage change) and Δχ_φ (dephasing change) are
baseline-relative diagnostics; absolute full-model proxies are reported
separately in the supplementary material.

## Requirements

- Python 3.12
- JAX + jaxlib (CPU is sufficient)
- NumPy, SciPy, Matplotlib

```bash
pip install jax jaxlib numpy scipy matplotlib
# On externally-managed environments add: --break-system-packages
```

**Important — float64 is mandatory.** The simulation multiplies
nanometre-scale positions (~10⁻⁹ m) by tesla-per-metre field gradients
(~10⁶ T/m). In float32 this product loses precision and the analytic
derivatives disagree with automatic differentiation. All scripts enable
64-bit mode at import (`jax.config.update("jax_enable_x64", True)`); do
not disable it.

All scripts are run from the repository root with `PYTHONPATH=.`.

## Reproducing the main result (sensitivity atlas)

The full validate grid has 432 model-conditions (4 ansätze × center/edge
× 9 geometries × v ∈ {5,10,20} m/s × λ ∈ {0.5,1.0} = 432 model-conditions,
each evaluated with n_real = 5). It is
split into two cases to keep each run within a single process; the raw
per-condition data are then merged.

```bash
# 1. compute each case (each ~4-10 min on CPU), saving raw per-condition data,
#    in the adopted total-field convention
PYTHONPATH=. python reproduce/phase5_sensitivity_atlas.py --mode validate \
    --case case_i_center --save-raw \
    --ez-convention total-local --profile-norm prefactor
PYTHONPATH=. python reproduce/phase5_sensitivity_atlas.py --mode validate \
    --case case_ii_edge --save-raw \
    --ez-convention total-local --profile-norm prefactor

# 2. merge the raw data and compute the six atlas metrics
PYTHONPATH=. python reproduce/phase5_atlas_merge_validate.py --mode validate \
    --ez-convention total-local --profile-norm prefactor
```

This writes the summary/sensitivity CSVs and metadata to `figures/phase5/`
(suffixed `__ez-total-local__norm-prefactor`) and prints the adopted
headline numbers:

```
mean rank correlation: +0.837 (HIGH)
mean quadrant agreement: 36.0%
```

The suffixed raw `.pkl` files from our run are included under
`figures/phase5/`, so step 2 alone reproduces the headline numbers
without recomputation.

**Archived legacy layer (provenance).** The original stray-field-only
run is preserved for byte-for-byte reproducibility. Its raws predate
config recording, so merging them requires `--allow-legacy`:

```bash
PYTHONPATH=. python reproduce/phase5_atlas_merge_validate.py --mode validate --allow-legacy
# prints: mean rank correlation +0.677 (MODERATE); mean quadrant agreement 28.9%
```

The **shift** from the legacy layer (0.677 / 28.9%) to the adopted layer
(0.837 / 36.0%) is itself reported in the paper as evidence that the
Zeeman-energy convention—not only the ansatz choice—moves design-level
conclusions.

Lighter modes are available for quick checks: `--mode preview` (fast
sanity), `--mode validate_lite` (edge-only), `--mode validate_mid`
(center+edge, v ∈ {5,20}). Metric values per mode in the **legacy
convention** (for provenance; add the `--ez-convention` flags for the
adopted values):

| mode          | mean rank ρ | mean quadrant agreement |
| ------------- | ----------: | ----------------------: |
| validate_lite |       0.553 |                   34.3% |
| validate_mid  |       0.671 |                   32.2% |
| validate (full, legacy) | 0.677 |             28.9% |
| **validate (full, adopted total-local)** | **0.837** | **36.0%** |

Under the adopted total-field convention the ranking is high across the
grid; the persistent gap between the high rank correlation and the much
lower quadrant agreement is the paper's central finding.

### Verifying script integrity

Each mode stores the SHA-256 of the computing script in its metadata.
To check that the released **adopted (total-local)** metadata matches the
released code without recomputing (this is the paper dataset; it must
report `OK`):

```bash
PYTHONPATH=. python reproduce/phase5_sensitivity_atlas.py \
  --mode validate --metadata-only \
  --ez-convention total-local --profile-norm prefactor
# -> validate metadata SHA: <hash>  -> OK
```

The archived **legacy** (stray-field) metadata intentionally predates
config recording, so the same check without the convention flags reports
`MISMATCH (expected for legacy)`. That is a provenance note, not an
error, and it is not the paper dataset:

```bash
PYTHONPATH=. python reproduce/phase5_sensitivity_atlas.py --mode validate --metadata-only
# -> MISMATCH (expected for legacy): archived raw predates config recording
```

## Figures

The paper and supplementary figures are regenerated with:

```bash
PYTHONPATH=. python reproduce/phase5_paper_figures.py          # Fig 1, Fig 2
PYTHONPATH=. python reproduce/phase5_atlas_figure.py --mode validate  # Fig 3 (atlas)
PYTHONPATH=. python reproduce/phase5_supp_robustness.py        # supplementary robustness
PYTHONPATH=. python reproduce/phase5_absolute_performance.py   # supplementary absolute table
```

Each saves both `.png` and a vector `.pdf` under `figures/`.

## Validation suite (supplementary material)

The simulator is validated against analytic limits and published
results. These reproduce the literature-reproduction checks (V1-V6) in
the supplementary material:

```bash
# V1-V3: charge-noise dephasing (stationary, shuttling, filter function)
PYTHONPATH=. python reproduce/krzywda_B1_stationary.py   # V1: gradient-halving T2* ratio
PYTHONPATH=. python reproduce/krzywda_B2_motional.py     # V2: motional-narrowing trend (up to ~64x)
PYTHONPATH=. python reproduce/krzywda_B3_filter.py       # V3: filter-function suppression

# V4: valley dynamics (Landau-Zener)
PYTHONPATH=. python reproduce/oda_C1_lz_single.py        # V4: valley Landau-Zener vs analytic
PYTHONPATH=. python reproduce/oda_C2v2_scan.py           # V4: deep-pocket velocity scan (official)
# note: oda_C2_pocket.py tests a legacy 10x-improvement heuristic and is
# expected to print "C2 PASS: False"; it is kept for the shared pocket
# machinery, not as a validation gate.

# V5-V6: micromagnet stray field and Fourier descriptor
PYTHONPATH=. python geometry/prism_field.py              # V5: single-prism field, dipole limit, autodiff
PYTHONPATH=. python geometry/periodic_array.py           # V6: periodic array, Fourier amplitudes
PYTHONPATH=. python geometry/fourier_field.py            # FourierField self-check (sign/harmonics)
```

The numerical-consistency checks (V0-V8) and the geometry-parameter
autodiff checks are covered by the unit tests (autodiff against finite
differences, plus the bounded-phase sanity check). Run them as separate
gates -- a single combined `pytest tests/ -q` can hang on some sandboxes
during JAX/subprocess teardown:

```bash
# fast convention gate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_convention_options.py -q -m "not slow"   # 14 passed, 1 deselected
# geometry / sanity gate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_geometry_autodiff.py tests/test_fourier_projection_geometry_autodiff.py tests/test_step1_sanity.py -q   # 3 passed
# slow end-to-end (subprocess; run on its own)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_convention_options.py::test_save_raw_e2e_tagged -q -s   # 1 passed
```

## Repository layout

```
constants.py                      # physical constants and default parameters
field_landscape.py                # analytic periodic field landscape
geometry/
  prism_field.py                  # single-prism stray field + dipole limit (V5/V6)
  periodic_array.py               # periodic micromagnet array, Fourier descriptor
  fourier_field.py                # Fourier-harmonic field model
noise/
  charge_noise.py                 # 1/f charge-noise generator and filter function
reproduce/
  phase5_sensitivity_atlas.py     # atlas computation kernel (six metrics)
  phase5_atlas_merge_validate.py  # merge case-split raw data -> metrics
  phase5_atlas_figure.py          # atlas figure (5 panels)
  phase5_paper_figures.py         # coupling-profile and quadrant-schematic figures
  phase5_supp_robustness.py       # rank-weighting / threshold / Cohen-kappa analyses
  phase5_absolute_performance.py  # absolute full-model performance table
  phase4p6_crossterm.py           # response engine: computes dP_v, dchi_phi
  phase4_step5_M2_observables.py  # M2 observables (leakage, dephasing)
  phase4_step4_M2_minimum.py      # nested-model (M1/M1V/M2) machinery
  krzywda_B1_stationary.py        # validation: gradient-halving T2* ratio
  krzywda_B2_motional.py          # validation: motional-narrowing trend
  krzywda_B3_filter.py            # validation: filter-function suppression
  oda_C1_lz_single.py             # validation: valley Landau-Zener vs analytic
  oda_C2_pocket.py                # pocket machinery (legacy heuristic; expected FAIL)
  oda_C2v2_scan.py                # V4 deep-pocket velocity scan (official)
  phase5_narrow_scope.py          # staged-falsification stages 3-5 (Table II)
  phase5_objective_analysis.py    # staged-falsification stage 2 (Table II)
tests/                            # pytest: autodiff and sanity checks
docs/                             # manuscript sources (Overleaf-ready)
  paper.tex                       # main manuscript (Physical Review Applied, REVTeX 4-2)
  supplementary.tex               # supplementary material (REVTeX 4-2)
  ref.bib                         # bibliography database
  figures/                        # journal-facing figure PDFs used by the .tex
  collect_figures.sh              # repopulate figures/ from pipeline outputs
  supplement_data/                # machine-generated data tables (robustness, absolute)
  BUILD.md                        # how to build (local / Overleaf)
figures/                          # generated figures (.png/.pdf), CSVs, raw .pkl data
make_overleaf.sh                  # build a self-contained Overleaf upload bundle
CITATION.cff, LICENSES/           # citation metadata; dual CC-BY-4.0 / MIT license
REPRODUCIBILITY.md                # exact reproduction recipe and reference values
requirements.txt                  # pinned dependencies
```

(An `internal/` folder with development notes is excluded from the
public release and the `SHA256SUMS.txt` manifest.)

## Building the manuscript

The manuscript targets *Physical Review Applied* and uses the
`revtex4-2` class (from the REVTeX bundle, e.g. `texlive-publishers`).
The manuscripts reference figures by journal-facing names under
`docs/figures/`; run the collection script once to populate that
directory from the pipeline outputs, then build inside `docs/`:

```bash
cd docs
./collect_figures.sh
pdflatex paper.tex && (bibtex paper || bibtexu paper) && pdflatex paper.tex && pdflatex paper.tex
pdflatex supplementary.tex && (bibtex supplementary || bibtexu supplementary) && pdflatex supplementary.tex && pdflatex supplementary.tex
```

For Overleaf, generate a self-contained upload bundle in one step:

```bash
bash make_overleaf.sh        # -> overleaf_bundle.zip
```

Then in Overleaf choose **New Project → Upload Project**, select
`overleaf_bundle.zip`, and set `paper.tex` as the main document. The
bundle contains only what compiles the manuscript (`paper.tex`,
`supplementary.tex`, `ref.bib`, `figures/*.pdf`). See `docs/BUILD.md`
for details. No source code or pipeline outputs are needed there.

## Notes on scope

The four λ_sv(x) profiles are **phenomenological diagnostic ansätze**,
not microscopically derived spin-valley Hamiltonians, and the four-level
model is a minimal diagnostic effective Hamiltonian rather than a
microscopic Si/SiGe H_sv. The atlas quantifies sensitivity to that
modelling choice; deriving a microscopic coupling is left to future work.

## Citation

If you use this code or data, please cite the accompanying paper (see
`docs/paper.tex` for the current title and author).

## License

Code is released under the MIT License (see `LICENSES/MIT.txt`). The
figures and data files under `figures/` may be reused under CC-BY 4.0
(see `LICENSES/CC-BY-4.0.txt`) with attribution to the accompanying
paper.
