# Reproducibility guide

This guide documents the **r31** scientific results and how to reproduce them.
Every command below is run from the released repository root.

`r32` changes the manuscript, documentation, and release packaging only. The
scientific code, retained datasets, and manuscript-facing generated figures are
unchanged from `r31`, so every scientific value and command in this guide
applies unchanged and the r31 results need no recomputation. Redundant raster
copies, in-tree compiled PDFs, and `docs/BUILD.md` are omitted; the r32 release
notes list every changed or removed path.

## 1. Environment

Python 3.12 with the pinned `requirements.txt`; JAX on CPU is sufficient. No
GPU is required.

```bash
export PYTHONPATH=.
export JAX_ENABLE_X64=1
export JAX_PLATFORM_NAME=cpu
export MPLBACKEND=Agg
```

**float64 is mandatory.** The simulation multiplies nanometre positions by
tesla-per-metre gradients; in float32 the analytic derivatives disagree with
automatic differentiation. All scripts enable 64-bit mode at import.

The Python analyses also run on Windows x86-64 with CPython 3.12. Keep line
endings as LF: a CRLF checkout changes recorded source hashes and makes
`sha256sum -c SHA256SUMS.txt` report spurious mismatches, and the included
`.gitattributes` enforces this. `docs/collect_figures.sh` and
`make_overleaf.sh` need `bash` on `PATH` (Git Bash or WSL2).

Fixed seeds make the workflow deterministic, but bit-identical output across
operating systems and BLAS builds is not assumed; use the tolerances in
Section 5.

## 2. Reproduce the convention comparison

The paper reports two atlas layers computed with the same response-model
pipeline under different Zeeman-energy bookkeeping conventions. Their
difference is itself a reported model-risk result, so both are reproducible
here.

### 2.1 Archived legacy data

The archived raw files predate config recording, so merge them without
recomputation:

```bash
python reproduce/phase5_atlas_merge_validate.py --mode validate --allow-legacy
```

Expected: mean Spearman rank correlation `0.6768746367906103`, mean
classification agreement `0.28858024691358025`. The archived scripts print
this quantity as `quadrant agreement`.

### 2.2 Fresh legacy recomputation

Fresh raw files contain a config block and merge without `--allow-legacy`:

```bash
python reproduce/phase5_sensitivity_atlas.py --mode validate --case case_i_center --save-raw --no-plots
python reproduce/phase5_sensitivity_atlas.py --mode validate --case case_ii_edge  --save-raw --no-plots
python reproduce/phase5_atlas_merge_validate.py --mode validate
```

Archived and fresh legacy raw files are not required to be byte-identical,
because the archived data predate config recording. Keys, numerical values
within tolerance, and the documented legacy-provenance state are the checks.

### 2.3 Adopted result (total-local convention)

The adopted result uses `E_Z(x) = g mu_B [B_ext + B_z(x)]` and `prefactor`
profile normalization. Tagged outputs leave the archived legacy dataset
untouched.

```bash
python reproduce/phase5_sensitivity_atlas.py --mode validate --case case_i_center --save-raw --no-plots --ez-convention total-local --profile-norm prefactor
python reproduce/phase5_sensitivity_atlas.py --mode validate --case case_ii_edge  --save-raw --no-plots --ez-convention total-local --profile-norm prefactor
python reproduce/phase5_atlas_merge_validate.py --mode validate --ez-convention total-local --profile-norm prefactor
```

Expected merged values (`n_conditions = 432`, `n_real = 5`):

| quantity | value | paper |
| --- | --- | --- |
| mean Spearman rank correlation | `0.8373806370891169` | `0.84` |
| mean classification agreement | `0.3595679012345679` | `36%` |

Pairwise rho / classification agreement: A–A_pocket `0.785 / 36.1%`,
A–B_z `0.913 / 45.4%`,
A–B_x `0.789 / 29.6%`, A_pocket–B_z `0.881 / 31.5%`, A_pocket–B_x
`0.855 / 44.4%`, B_z–B_x `0.801 / 28.7%`.

Thresholded-label category counts: `below_threshold 129`, `P_only_improve 106`,
`P_only_worsen 82`, `spin_trade 40`, `both_worsen 30`, `robust 30`,
`valley_trade 15`.

The reported agreement is exact equality of the full nine-category thresholded
label (four two-channel quadrants, four single-channel categories, and
below-threshold). Restricting each model-pair comparison to conditions for
which both ansätze receive one of the four two-channel labels gives
`47/105 = 44.8%` at the main floors and `7/22 = 31.8%` at the strict floors,
recomputable from the adopted summary CSV.

The shift from the legacy to the adopted convention, rho `0.68 -> 0.84`,
agreement `29% -> 36%`, and the disappearance of the legacy B_x ranking
divergence, is a reported result: the Zeeman-energy convention changes
design-level conclusions.

Lighter modes are available for quick checks: `--mode preview`,
`--mode validate_lite` (edge only), and `--mode validate_mid`
(center+edge, `v` in `{5,20}`).

### Verifying script integrity

Each mode stores the SHA-256 of the computing script in its metadata. The
adopted dataset must report `OK`:

```bash
python reproduce/phase5_sensitivity_atlas.py --mode validate --metadata-only --ez-convention total-local --profile-norm prefactor
```

The same check without the convention flags targets the archived legacy dataset
and reports `MISMATCH (expected for legacy)`. That is a provenance note, not an
error.

## 3. Post-processing, figures, and the robust retest

```bash
python reproduce/phase5_robust_candidate_retest.py
python reproduce/phase5_supp_robustness.py --dataset-suffix __ez-total-local__norm-prefactor
python reproduce/phase5_absolute_performance.py --ez-convention total-local --profile-norm prefactor
python reproduce/phase5_paper_figures.py --ez-convention total-local --profile-norm prefactor
python reproduce/phase5_atlas_figure.py --mode validate --dataset-suffix __ez-total-local__norm-prefactor
( cd docs && ./collect_figures.sh "__ez-total-local__norm-prefactor" )
```

The two figure generators produce Figs. 2–3 and Fig. 4 of the paper; the final
command installs the regenerated outputs under the manuscript-facing filenames
in `docs/figures/`. That directory already ships populated with those same
figures, so this step is only needed after regenerating them — and the suffix
argument is required, since without it the script installs the archived legacy
figures. Fig. 1 is drawn in TikZ inside `docs/paper.tex` and has no generator
script.

Adopted total-local references: weighting sweep `0.837 / 0.769 / 0.751 /
0.845`; threshold sweep `49.4% / 36.3% / 36.0% / 58.0%`; Cohen kappa A–B_x
`0.145`, B_z–B_x `0.135`. The robust-candidate retest uses `n_real = 30`; two
candidates survive as 4/4-robust under the adopted convention.

The staged falsification chain of Table II is generated by:

```bash
python reproduce/phase5_objective_analysis.py
python reproduce/phase5_narrow_scope.py --mode preview
python reproduce/phase5_narrow_scope.py --mode full
python reproduce/phase5_narrow_scope.py --mode targeted
```

## 4. Validation controls

```bash
# Test gates. Do not replace these with one combined `pytest tests/ -q`.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_convention_options.py -q -m "not slow"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_geometry_autodiff.py tests/test_fourier_projection_geometry_autodiff.py tests/test_step1_sanity.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_convention_options.py::test_save_raw_e2e_tagged -q -s

# Field, geometry, and physics benchmarks.
python field_landscape.py
python geometry/prism_field.py
python geometry/periodic_array.py
python geometry/fourier_field.py --full
python reproduce/krzywda_B1_stationary.py
python reproduce/krzywda_B2_motional.py
python reproduce/krzywda_B3_filter.py
python reproduce/oda_C1_lz_single.py
python reproduce/oda_C2v2_scan.py

# Numerical and physical controls.
python reproduce/v0p3_float_precision_check.py
python reproduce/phase5_gradient_kernel_hotspot_check.py
```

| control | expected result |
| --- | --- |
| analytic vs `jax.grad` field agreement | rel. error ~`4e-9` |
| prism far-field dipole limit | error ~`0.06%` |
| array periodicity deviation | ~`0.04%` |
| gradient-halving `T2*` ratio (V1) | `2.20` |
| motional-narrowing improvement (V2) | up to ~`64x` |
| filter / PSD-weighted suppression (V3) | ~`2e5` and `5.8e3` |
| Landau–Zener maximum error (V4) | `<0.28%` |
| deep-pocket scan (V4) | `P_e < 1e-5` at low `v`; order-of-magnitude for `v >= 30 m/s` |
| float64 precision check (V0.3) | PASS marker; float32 is intentionally inadequate |
| gradient-kernel hot-spot control | leakage ratio ~`252`, PASS marker |

The hot-spot control supports the Model-section statement that the
`E_v ≈ E_Z` enhancement is generated by the four-level dynamics without an
artificial resonance window in `lambda_sv(x)`: it preserves the Gaussian-pocket
shape while raising its minimum above the total-field `E_Z` range. Expected
leakage is ~`4.5e-1` with the crossing (`Ev_min = 5 ueV`) and ~`1.8e-3` without
it (`Ev_min = 70 ueV`), with lambda-zero leakage below `1e-3` in both
landscapes.

`reproduce/oda_C2_pocket.py` tests a retained legacy heuristic and is expected
to print `C2 PASS: False`; it is not a validation gate. The official V4 pocket
gate is `reproduce/oda_C2v2_scan.py`.

## 5. Comparison tolerances

- raw continuous values: `np.allclose(rtol=1e-7, atol=1e-10)`
- summary metrics: absolute difference below `5e-4`
- values quoted to three significant figures in the paper: exact agreement at
  that displayed precision
- category counts and row keys: exact match
- PDF and PNG byte hashes: not expected to match

## 6. Provenance and known limitation

> The released scripts' `--help` text points to `REPRODUCIBILITY.md, Sec. 7`,
> the section number in the archival r31 guide. In this condensed r32 guide,
> the same Zeeman-convention material is in this section.

**Zeeman convention.** An external-field inconsistency found during development
has been resolved. The archived legacy run used a stray-field-only Zeeman
energy: `Defaults.B_ext_T = 0.5` was defined but did not enter the atlas Zeeman
energy, giving a mean `E_Z` of about `3 ueV`. That state is retained for
archive reproducibility and is not the adopted convention. The adopted result
uses the total-local form above, with the full 432-condition atlas recomputed,
the paper values and figures regenerated, and the legacy archive preserved
through tagged filenames. The adopted profile normalization is `prefactor`; the
`final-peak` and `l2` variants remain optional sensitivity checks, and `l2` is
documented as divergent. Both conventions are exposed end to end:
`phase5_sensitivity_atlas.py` and `phase5_atlas_merge_validate.py` take
`--ez-convention` and `--profile-norm`, write non-legacy results to tagged
filenames, and embed a config block recording conventions, external field,
noise scale, seeds, script SHAs, and archive version. The merge step refuses
mismatched raw configs.

**Table II limitation.** The archived narrow-scope metadata records SHAs of two
earlier revisions of `phase5_narrow_scope.py` — `4d81427891c47a48` for the
stage-3 and stage-5 runs and `529c03aac5ba0ad1` for stage 4 — while the
included script is a later revision of the same analysis. Exact historical
revisions are not included for every stage. Table II is therefore approved as
archived provenance
when its table entries, retained outputs, and executable final-stage scripts
agree. It is not claimed as byte-exact forensic regeneration of every
historical development step. All principal atlas, robustness,
absolute-performance, and geometry-sensitivity gates reproduce independently of
this limitation.

**Scope of the validations.** These checks certify the field model, its
gradients, and the time-evolution engine, and they support the representative
parameters of Table I. None of them validates the spin–valley coupling profile
`lambda_sv(x)` itself, which the paper treats as a phenomenological diagnostic
ansatz rather than a microscopically derived coupling. Quantifying sensitivity
to that choice is the purpose of the atlas.
