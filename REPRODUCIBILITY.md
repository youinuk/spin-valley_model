# Reproducibility guide

This release reproduces the full sensitivity atlas, the reported pairwise
diagnostics, and the listed supplementary validations. (Full regeneration
of every historical stage of Table II additionally requires the exact
script revisions noted in Section 5.) Commands and expected reference
values follow. Run everything from the repository root with:

```bash
export PYTHONPATH=.
export JAX_ENABLE_X64=1
export JAX_PLATFORM_NAME=cpu
export MPLBACKEND=Agg
```

## 1. Environment

Python 3.12 with the pinned packages in `requirements.txt` (CPU-only JAX
is sufficient; no GPU is used). All results below were verified on CPU
with float64 enabled.

Verified host:

- python 3.12.3, Linux x86_64 (glibc 2.39)
- JAX on CPU (`JAX_PLATFORM_NAME=cpu`), float64 via
  `jax.config.update("jax_enable_x64", True)`
- LaTeX build: `pdflatex` + (`bibtex` or `bibtexu`); the `revtex4-2`
  class is required (e.g. `texlive-publishers`)

`requirements.txt` pins the direct dependencies. If you need a
byte-exact transitive environment, regenerate a full freeze on your host
with `pip freeze` after installing `requirements.txt`; the direct-pin set
above is what the reported numbers were produced with.

### Running on Windows

The pipeline is pure Python and uses `pathlib` throughout, so it runs on
Windows (x86-64). All pinned dependencies ship `win_amd64` wheels for
CPython 3.12, so `pip install -r requirements.txt` works as-is. Three
practical notes:

1. **Line endings must stay LF.** The scripts hash their own source, so a
   CRLF checkout changes the recorded script SHA and makes both
   `--metadata-only` and `sha256sum -c SHA256SUMS.txt` report spurious
   mismatches. The included `.gitattributes` enforces LF; do not override
   it with `core.autocrlf=true`. (If you extract the release tarball
   rather than cloning, the files are already LF.)

2. **Environment variables.** The `VAR=value cmd` form in this document is
   POSIX shell syntax. In PowerShell use, e.g.:

   ```powershell
   $env:PYTHONPATH="."; $env:JAX_ENABLE_X64="1"; $env:JAX_PLATFORM_NAME="cpu"; $env:MPLBACKEND="Agg"
   python reproduce/phase5_atlas_merge_validate.py --mode validate --ez-convention total-local --profile-norm prefactor
   $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
   python -m pytest tests/test_convention_options.py -q -m "not slow"
   ```

3. **The two `.sh` helpers need a POSIX shell.** `docs/collect_figures.sh`
   and `make_overleaf.sh` (and the test that invokes the former) require
   `bash` on `PATH` — Git for Windows ("Git Bash") or WSL both provide it.
   Everything else runs from PowerShell. For the manuscript, either
   install a TeX distribution with the `revtex4-2` class (MiKTeX/TeX Live)
   or use Overleaf via `docs/BUILD.md`.

Numbers are deterministic given the fixed seeds, but bit-identical
agreement across operating systems and BLAS builds is not guaranteed; the
tolerances in section 6 apply. VS Code with WSL2 and the Remote - WSL
extension is the safest setup because it stays closest to the Linux
reference environment and provides `bash`, `sha256sum`, and LaTeX with the
documented commands. Native Windows remains supported. A full CPU atlas
recalculation takes roughly four minutes per case (about eight minutes for
both cases) on the reference-class host; actual time depends on hardware.

## 2. Test suite and validations

```bash
# Fast gate (stable across environments; run the convention and geometry
# suites separately -- a single combined pytest invocation can hang on
# some sandboxes during JAX/subprocess teardown):
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_convention_options.py -q -m "not slow"   # 14 passed, 1 deselected
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_geometry_autodiff.py tests/test_fourier_projection_geometry_autodiff.py tests/test_step1_sanity.py -q   # 3 passed
# Slow end-to-end (subprocess; run on its own):
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_convention_options.py::test_save_raw_e2e_tagged -q -s   # 1 passed
# (Use the split commands above as the release gate; a single combined
#  `pytest tests/ -q` is not used as a release criterion.)
python field_landscape.py             # analytic vs jax.grad rel. err ~ 4e-9
python tests/test_step1_sanity.py     # bounded phase; quasi-static T2* within 1%
python geometry/prism_field.py        # far-field dipole error ~ 0.06%
python geometry/periodic_array.py     # periodicity deviation ~ 0.04%
python geometry/fourier_field.py
python reproduce/krzywda_B1_stationary.py   # T2* ratio ~ 2.20
python reproduce/krzywda_B2_motional.py     # motional narrowing up to ~64x
python reproduce/krzywda_B3_filter.py       # F ~ 2e5, chi ~ 5.8e3 suppression
python reproduce/oda_C1_lz_single.py        # max LZ error < 0.28%
python reproduce/oda_C2v2_scan.py           # OFFICIAL V4 deep-pocket check:
                                            # P_e<1e-5 at low v; LZ order-of-magnitude
                                            # agreement for v>=30 m/s
# (oda_C2_pocket.py tests a legacy 10x-improvement heuristic and is
#  EXPECTED to print "C2 PASS: False"; it is not a validation gate.)
```

## 3. Main atlas (432 model-conditions)

A. Archived-legacy reproduction (no recomputation; the archived raws
predate config recording, hence `--allow-legacy`):

```bash
python reproduce/phase5_atlas_merge_validate.py --mode validate --allow-legacy
```

B. Fresh legacy recomputation (provenance; freshly generated raws carry
a config block and merge WITHOUT `--allow-legacy`):

```bash
python reproduce/phase5_sensitivity_atlas.py --mode validate --case case_i_center  --save-raw --no-plots
python reproduce/phase5_sensitivity_atlas.py --mode validate --case case_ii_edge   --save-raw --no-plots
python reproduce/phase5_atlas_merge_validate.py --mode validate
```

Legacy reference values (both A and B; preserved for provenance):

- mean Spearman rank correlation = 0.6768746367906103 (legacy)
- mean quadrant agreement = 0.28858024691358025 (legacy)

C. Adopted result — paper headline (total-field convention; tagged
dataset, archive untouched):

```bash
python reproduce/phase5_sensitivity_atlas.py --mode validate --case case_i_center  --save-raw --no-plots --ez-convention total-local --profile-norm prefactor
python reproduce/phase5_sensitivity_atlas.py --mode validate --case case_ii_edge   --save-raw --no-plots --ez-convention total-local --profile-norm prefactor
python reproduce/phase5_atlas_merge_validate.py --mode validate --ez-convention total-local --profile-norm prefactor
```

Expected merged reference values (these are the paper's numbers):

- n_conditions = 432, n_real = 5
- mean Spearman rank correlation = 0.8373806370891169 (paper: 0.84)
- mean quadrant agreement = 0.3595679012345679 (paper: 36%)
- pairwise (rho / agreement): A-Ap 0.785/36.1%, A-Bz 0.913/45.4%,
  A-Bx 0.789/29.6%, Ap-Bz 0.881/31.5%, Ap-Bx 0.855/44.4%, Bz-Bx 0.801/28.7%
- quadrant category counts: below_threshold 129, P_only_improve 106,
  P_only_worsen 82, spin_trade 40, both_worsen 30, robust 30,
  valley_trade 15

The shift from B to C (ρ 0.68→0.84, agreement 29%→36%, and the
disappearance of the legacy B_x ranking divergence) is itself a reported
result: the Zeeman-energy convention moves design-level conclusions.

## 4. Post-processing

```bash
python reproduce/phase5_supp_robustness.py --dataset-suffix __ez-total-local__norm-prefactor
python reproduce/phase5_absolute_performance.py --ez-convention total-local --profile-norm prefactor
python reproduce/phase5_paper_figures.py --ez-convention total-local --profile-norm prefactor
python reproduce/phase5_atlas_figure.py --mode validate --dataset-suffix __ez-total-local__norm-prefactor
```

Robustness reference values (adopted total-local): weighting sweep
0.837 / 0.769 / 0.751 / 0.845; threshold sweep 49.4% / 36.3% / 36.0% /
58.0%; Cohen kappa A-Bx 0.145, Bz-Bx 0.135. Absolute full-model means as
tabulated in the supplement.

## 5. Staged falsification chain (Table II)

The stages of Table II are generated as follows (stage 1 uses the
archived phase-4 coarse outputs, regenerable with
`reproduce/phase4p6_crossterm.py`):

```bash
python reproduce/phase5_objective_analysis.py           # stage 2: objective-weight sweep
python reproduce/phase5_narrow_scope.py --mode preview  # stage 3: local refinement
python reproduce/phase5_narrow_scope.py --mode full     # stage 4: full local grid
python reproduce/phase5_narrow_scope.py --mode targeted # stage 5: n_real=30 re-test
```

Stage 6 (the four-ansatz family re-test) is the early four-model run of
`reproduce/phase5_sensitivity_atlas.py`, whose full-statistics successor
is the released 432-condition atlas of Section 3. Provenance note: the
archived narrow-scope metadata records the script SHA of an earlier
revision (529c03aa...) of `phase5_narrow_scope.py`; the included script
is the final revision (895add02...) of the same analysis.

## 5b. Float-precision check (V0.3)

```bash
python reproduce/v0p3_float_precision_check.py
# expected: float64 relative error ~1e-9; float32 relative error O(1)
```

## 6. Comparison tolerances

Raw continuous values: `np.allclose(rtol=1e-7, atol=1e-10)`. Summary
metrics: absolute difference < 5e-4; all three-significant-figure values
quoted in the paper must match exactly. Category counts and row keys must
match exactly; PDF/PNG byte hashes are not expected to match.

## 7. Zeeman-convention decision and provenance (RESOLVED)

An external-field inconsistency identified during development has been
resolved. The original legacy run used
a stray-field-only Zeeman energy (`Defaults.B_ext_T = 0.5` defined but
not entering the atlas Zeeman energy, giving mean E_Z ~ 3 ueV). This was
retained for archive reproducibility but is **not** the adopted paper
convention. The adopted paper result uses the **total-local** convention,
E_Z(x) = g mu_B [B_ext + B_z(x)], with the full 432-condition atlas
recomputed under it, the paper numbers, figures, and collected artifacts
regenerated, and the legacy archive preserved via tagged filenames. The
difference between the two conventions is itself reported as a result.

The profile-normalization axis remains available but is **optional** and
is not required before freezing the paper numbers: the adopted result
uses `--profile-norm prefactor`, and the alternative `final-peak` / `l2`
normalizations are provided for sensitivity checks only (the `l2`
variant is documented as divergent and is retained only as a reference
point). No `final-peak` full-atlas run is required for the paper.

The pipeline exposes the options end to end (defaults reproduce the
archived legacy behaviour):

- `phase5_sensitivity_atlas.py --ez-convention {stray-mean,total-local,total-mean}
  --profile-norm {prefactor,final-peak,l2}`; non-legacy runs write to
  suffixed filenames (`...__ez-<conv>__norm-<norm>...`) and embed a
  `config` block (conventions, B_ext, sigma_E, seeds, both script SHAs,
  archive version) in the raw pickles.
- `phase5_atlas_merge_validate.py` takes the same flags, refuses to merge
  raws whose configs disagree, requires `--allow-legacy` for archived
  raws that predate config recording, and stamps the metadata with the
  RAW-recorded atlas SHA (never the current script's) plus the full raw
  config.
- `phase5_paper_figures.py --ez-convention {legacy-50ueV,stray-mean,total-mean,total-local}
  --profile-norm ...` uses the same kernel as the simulator, including
  local E_Z(x); non-legacy outputs are suffixed.
- `phase5_atlas_figure.py / phase5_supp_robustness.py /
  phase5_absolute_performance.py` accept `--dataset-suffix` to read the
  suffixed datasets.
- `phase5_robust_candidate_retest.py` re-tests the cross-ansatz robust
  candidates flagged by the adopted atlas at `n_real=30` (total-local);
  two survive as 4/4-robust (edge pocket, v=10, deeper well).
- `tests/test_convention_options.py` covers global normalization,
  dataset-tag uniqueness, merge config validation, the Hamiltonian-level
  effect of the convention, and the flank-peak symmetry of the centred
  pocket.

The pinned direct dependencies are in `requirements.txt`; host and
runtime details are in section 1 above.
