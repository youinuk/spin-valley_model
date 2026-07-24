# Reproducibility guide

This guide documents the **r31** scientific results and their reproduction.
The immutable clean-room audit target is the r31 release archive and its
recorded SHA-256. Later documentation-only updates may be committed to Git
without creating a new scientific release archive, provided that they do not
alter or replace the immutable r31 tar used by the audit.

The release reproduces the full sensitivity atlas, the reported pairwise
diagnostics, and the listed supplementary validations. Full regeneration of
every historical stage of Table II additionally requires the exact script
revisions noted in Section 7.

## 1. Reproduction paths and version scope

Two complementary reproduction paths are provided.

### 1.1 Manual scientific reproduction

Run the individual Python analyses from the released repository root. In the
local audit workspace described below, that directory is:

```text
Spin-valley_model/repo/
```

The commands in Sections 4–10 are manual scientific-reproduction commands and
assume that this repository root is the current working directory.

### 1.2 Clean-room submission audit

Run the external Phase 1–5 shell harness from the **external audit workspace
root**. In the local layout used here, that directory is:

```text
Spin-valley_model/
```

It is the directory that contains `.venv/`, `dist/`, `repo/`, the Phase runners,
and the phase-result archives. The harness validates an immutable release tar
rather than trusting the mutable Git working tree. It records chain of custody,
environment, numerical gates, source integrity, regenerated artifacts, and
compact phase-result archives.

The shell infrastructure is release-aware, but the scientific expectations,
reference values, source-wiring checks, and paper-result gates are specific to
r31. A later scientific release must re-derive those expectations rather than
silently reusing the r31 criteria.

## 2. Reference platform and environment

Python 3.12 with the pinned packages in `requirements.txt` is required.
CPU-only JAX is sufficient; no GPU is used. The reported values were verified
with float64 enabled.

Verified reference host:

- Python 3.12.3, Linux x86_64 (glibc 2.39)
- JAX on CPU (`JAX_PLATFORM_NAME=cpu`), float64 enabled
- LaTeX build: `pdflatex` plus `bibtex` or `bibtexu`
- `revtex4-2.cls` available, for example through `texlive-publishers`

`requirements.txt` pins the direct dependencies. If a full transitive snapshot
is needed, record `pip freeze` after installing the pinned requirements. The
direct-pin set is the environment contract used for the reported results.

### 2.1 Clean-room audit platform

The Phase 1–5 shell runners target **Linux or WSL2** with GNU/POSIX command-line
utilities. They use tools and syntax such as `bash`, `sha256sum`, GNU `find`,
`lscpu`, process substitution, and POSIX environment assignment. Native
Windows PowerShell and Command Prompt are not supported for the submission-gate
harness.

### 2.2 Manual reproduction on native Windows

The individual Python analyses use `pathlib` and can be run on Windows x86-64
with CPython 3.12. The pinned dependencies provide `win_amd64` wheels. This
Windows path applies to the manual scientific commands, not to the external
Phase shell harness.

Practical notes:

1. **Keep line endings as LF.** A CRLF checkout changes recorded source hashes
   and can cause `--metadata-only` or `sha256sum -c SHA256SUMS.txt` to report
   spurious mismatches. The included `.gitattributes` enforces LF; do not
   override it with `core.autocrlf=true`. Extracting the release tar preserves
   the shipped LF files.

2. **Set environment variables using the host shell syntax.** For PowerShell:

   ```powershell
   $env:PYTHONPATH="."
   $env:JAX_PLATFORM_NAME="cpu"
   $env:JAX_ENABLE_X64="1"
   $env:MPLBACKEND="Agg"
   python reproduce/phase5_atlas_merge_validate.py --mode validate --ez-convention total-local --profile-norm prefactor
   $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
   python -m pytest tests/test_convention_options.py -q -m "not slow"
   ```

3. **Repository shell helpers still require Bash.** `docs/collect_figures.sh`
   and `make_overleaf.sh`, including tests that invoke them, require `bash` on
   `PATH`. Git Bash or WSL2 provides it. For the manuscript, install a TeX
   distribution containing `revtex4-2.cls`, or use Overleaf as described in
   `docs/BUILD.md`.

Fixed seeds make the numerical workflow deterministic, but bit-identical output
across operating systems, BLAS builds, or library implementations is not
assumed. Use the documented numerical tolerances. WSL2 most closely matches the
Linux reference environment.

## 3. External clean-room audit workspace

The local audit workspace keeps the immutable release, mutable Git checkout,
virtual environment, runners, and evidence outside one another:

```text
Spin-valley_model/
├── .venv/
├── dist/
│   └── spinshuttle-shuttling-atlas-release-r31.tar.gz
├── repo/                       # Git working tree and released-code root
├── r31-repro/                  # recreated from the immutable tar per phase
├── repro-runs/                 # temporary expanded evidence
├── run_phase1.sh
├── run_phase2.sh
├── run_phase3.sh
├── run_phase4.sh
├── run_phase5.sh               # added for the final submission audit
├── PHASE1_PROTOCOL.md
├── PHASE2_PROTOCOL.md
├── PHASE3_PROTOCOL.md
├── PHASE4_PROTOCOL.md
├── PHASE5_PROTOCOL.md
└── phase*-r31-results-<RUN_ID>.tar.gz
```

The audit is organized as follows:

1. **Phase 1 — release and execution foundation:** archive safety, manifest,
   pinned environment, tests, provenance, float64, and source integrity.
2. **Phase 2 — V1–V9 benchmarks:** literature and physics benchmarks,
   geometry, autodiff, periodicity, Fourier content, and field scales.
3. **Phase 3 — coupled-model validation:** M1/M1V/M2 wiring, separable limit,
   coupling ablation, CRN response engine, and hot-spot control.
4. **Phase 4 — paper-result reproduction:** Table II, legacy and adopted
   atlases, robust retest, robustness, absolute performance, and geometry
   sensitivity.
5. **Phase 5 — submission audit:** figures, tables, claim-to-evidence mapping,
   citations, manuscript consistency, and final PDF builds.

Each phase accepts the immutable release tar and, after Phase 1, the accepted
result archive from the preceding phase. Documentation and examples use
`<RUN_ID>` rather than a specific timestamped filename.

Local result archives retain timestamps:

```text
phase3-r31-results-YYYYMMDD_HHMMSS.tar.gz
```

This prevents accidental overwrite and preserves run-level provenance. These
local archives and temporary work trees should not be committed to ordinary Git
history. A sanitized, stable-name evidence bundle may be published separately
after the complete audit.

### Phase-3 maintenance note

The static wiring audit is derived from the released r31 source representation.
For a later scientific release, its source literals and expected structures
must be re-derived. A variable rename, comment change, or source reorganization
may require updating the audit even when the physical model is unchanged; the
checks must not be silently weakened or removed.

## 4. Manual test suite and supplementary validations

Run the commands in this and subsequent sections from the released repository
root (`repo/` in the external workspace):

```bash
export PYTHONPATH=.
export JAX_ENABLE_X64=1
export JAX_PLATFORM_NAME=cpu
export MPLBACKEND=Agg
```

```bash
# Fast convention gate.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_convention_options.py -q -m "not slow"

# Geometry, Fourier-projection, and foundation checks.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/test_geometry_autodiff.py \
  tests/test_fourier_projection_geometry_autodiff.py \
  tests/test_step1_sanity.py -q

# Slow end-to-end test, run separately.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/test_convention_options.py::test_save_raw_e2e_tagged -q -s

# Do not replace the split release gates above with one combined
# `pytest tests/ -q` invocation.
python field_landscape.py
python tests/test_step1_sanity.py
python geometry/prism_field.py
python geometry/periodic_array.py
python geometry/fourier_field.py --full
python reproduce/krzywda_B1_stationary.py
python reproduce/krzywda_B2_motional.py
python reproduce/krzywda_B3_filter.py
python reproduce/oda_C1_lz_single.py
python reproduce/oda_C2v2_scan.py
```

Expected reference behavior includes:

- analytic versus `jax.grad` field agreement at approximately `4e-9` relative
  error;
- prism far-field dipole error of approximately `0.06%`;
- periodicity deviation of approximately `0.04%`;
- stationary-gradient-halving ratio of approximately `2.20`;
- motional-narrowing improvement up to approximately `64x`;
- filter and PSD-weighted suppression of approximately `2e5` and `5.8e3`;
- Landau–Zener maximum error below approximately `0.28%`;
- official deep-pocket validation with `P_e < 1e-5` at low velocity and
  order-of-magnitude agreement for `v >= 30 m/s`.

`reproduce/oda_C2_pocket.py` tests a retained legacy 10x-improvement heuristic
and is expected to print `C2 PASS: False`; it is not a validation gate. The
official V4 pocket gate is `reproduce/oda_C2v2_scan.py`.

## 5. Main atlas: 432 model-conditions

### 5.1 Archived legacy reproduction

The archived raw files predate config recording, so merge them without
recomputation using `--allow-legacy`:

```bash
python reproduce/phase5_atlas_merge_validate.py --mode validate --allow-legacy
```

### 5.2 Fresh legacy recomputation

Fresh raw files contain a config block and merge without `--allow-legacy`:

```bash
python reproduce/phase5_sensitivity_atlas.py --mode validate --case case_i_center --save-raw --no-plots
python reproduce/phase5_sensitivity_atlas.py --mode validate --case case_ii_edge  --save-raw --no-plots
python reproduce/phase5_atlas_merge_validate.py --mode validate
```

Legacy reference values, preserved for provenance:

- mean Spearman rank correlation = `0.6768746367906103`
- mean quadrant agreement = `0.28858024691358025`

Archived legacy raw files are not required to be byte-identical to fresh raw
files because the archived data predate config recording. Keys, numerical
values within tolerance, and the documented expected legacy-provenance state
are the relevant checks.

### 5.3 Adopted paper result

The adopted result uses the total-local Zeeman convention and prefactor profile
normalization. Tagged outputs leave the archived legacy dataset untouched.

```bash
python reproduce/phase5_sensitivity_atlas.py --mode validate --case case_i_center --save-raw --no-plots --ez-convention total-local --profile-norm prefactor
python reproduce/phase5_sensitivity_atlas.py --mode validate --case case_ii_edge  --save-raw --no-plots --ez-convention total-local --profile-norm prefactor
python reproduce/phase5_atlas_merge_validate.py --mode validate --ez-convention total-local --profile-norm prefactor
```

Expected merged values:

- `n_conditions = 432`, `n_real = 5`
- mean Spearman rank correlation = `0.8373806370891169` (paper: `0.84`)
- mean quadrant agreement = `0.3595679012345679` (paper: `36%`)
- pairwise rho/agreement:
  - A–A_pocket: `0.785 / 36.1%`
  - A–B_z: `0.913 / 45.4%`
  - A–B_x: `0.789 / 29.6%`
  - A_pocket–B_z: `0.881 / 31.5%`
  - A_pocket–B_x: `0.855 / 44.4%`
  - B_z–B_x: `0.801 / 28.7%`
- quadrant category counts:
  - `below_threshold = 129`
  - `P_only_improve = 106`
  - `P_only_worsen = 82`
  - `spin_trade = 40`
  - `both_worsen = 30`
  - `robust = 30`
  - `valley_trade = 15`

The shift from legacy to adopted convention — rho `0.68 -> 0.84`, agreement
`29% -> 36%`, and disappearance of the legacy B_x ranking divergence — is a
reported result: the Zeeman-energy convention changes design-level conclusions.

## 6. Post-processing and robust retest

```bash
python reproduce/phase5_robust_candidate_retest.py
python reproduce/phase5_supp_robustness.py --dataset-suffix __ez-total-local__norm-prefactor
python reproduce/phase5_absolute_performance.py --ez-convention total-local --profile-norm prefactor
python reproduce/phase5_paper_figures.py --ez-convention total-local --profile-norm prefactor
python reproduce/phase5_atlas_figure.py --mode validate --dataset-suffix __ez-total-local__norm-prefactor
```

Adopted total-local robustness references:

- weighting sweep: `0.837 / 0.769 / 0.751 / 0.845`
- threshold sweep: `49.4% / 36.3% / 36.0% / 58.0%`
- Cohen kappa: A–B_x `0.145`, B_z–B_x `0.135`
- absolute full-model means: as tabulated in the supplement

The robust-candidate retest uses `n_real = 30`; two candidates survive as
4/4-robust under the adopted total-local convention.

## 7. Staged falsification chain: Table II

> **Note (script `--help` references).** The pipeline scripts inside the
> release tar say "see REPRODUCIBILITY.md, Sec. 7" for the
> **Zeeman-convention decision and provenance**. In the archival
> REPRODUCIBILITY.md shipped inside the tar that material is Section 7;
> in this reorganized guide it has moved to
> **[Section 10](#10-zeeman-convention-decision-and-provenance)**.
> This section covers the Table II staged chain instead.

The stages of Table II are generated as follows. Stage 1 uses the archived
phase-4 coarse outputs and is regenerable with
`reproduce/phase4p6_crossterm.py`.

```bash
python reproduce/phase5_objective_analysis.py
python reproduce/phase5_narrow_scope.py --mode preview
python reproduce/phase5_narrow_scope.py --mode full
python reproduce/phase5_narrow_scope.py --mode targeted
```

These correspond to the objective-weight sweep, local preview, full local grid,
and `n_real = 30` targeted retest. Stage 6 is the early four-model run of
`reproduce/phase5_sensitivity_atlas.py`; its full-statistics successor is the
released 432-condition atlas in Section 5.

The archived narrow-scope metadata records the SHA of an earlier revision
(`529c03aa...`) of `phase5_narrow_scope.py`; the included script is the final
revision (`895add02...`) of the same analysis. Exact historical revisions are
not included for every stage. Table II is therefore approved as archived
provenance when its table entries, retained outputs, and executable final-stage
scripts agree; it is not claimed as byte-exact full forensic regeneration of
every historical development step.

## 8. Additional numerical and physical controls

### 8.1 Float-precision check (V0.3)

```bash
python reproduce/v0p3_float_precision_check.py
```

The float64 branch must print its PASS marker and remain close to the reference
calculation; float32 is intentionally inadequate for this check.

### 8.2 Gradient-kernel hot-spot check

This supports the Model-section statement that the E_v approximately equal to
E_Z crossing enhancement is generated by the four-level dynamics without an
artificial resonance window in `lambda_sv(x)`. The control preserves the
Gaussian-pocket shape while raising its minimum above the total-field E_Z
range, and reports the deterministic charge-noise-off M1V branch.

```bash
PYTHONPATH=. python reproduce/phase5_gradient_kernel_hotspot_check.py
```

Expected behavior:

- lambda-zero leakage below `1e-3` in both landscapes;
- ansatz A leakage approximately `4.5e-1` with the crossing
  (`Ev_min = 5 ueV`);
- leakage approximately `1.8e-3` with the non-crossing pocket
  (`Ev_min = 70 ueV`);
- ratio approximately `252` and a final PASS marker.

## 9. Comparison tolerances

For the manual scientific comparisons:

- raw continuous values: `np.allclose(rtol=1e-7, atol=1e-10)`;
- summary metrics: absolute difference below `5e-4`;
- values quoted to three significant figures in the paper: exact agreement at
  that displayed precision;
- category counts and row keys: exact match;
- PDF and PNG byte hashes: not expected to match.

The clean-room Phase runners may define stricter or more specialized hard and
reference tolerances for individual gates. The corresponding Phase protocol is
the authority for those gate-specific thresholds; this section does not
replace them.

## 10. Zeeman-convention decision and provenance

> The release-tar pipeline scripts refer to this material as
> "REPRODUCIBILITY.md, Sec. 7" -- the section number of the archival copy
> inside the tar. Section numbers in this reorganized guide differ.

An external-field inconsistency identified during development has been
resolved. The archived legacy run used a stray-field-only Zeeman energy:
`Defaults.B_ext_T = 0.5` was defined but did not enter the atlas Zeeman energy,
giving mean E_Z of approximately `3 ueV`. This state is retained for archive
reproducibility but is not the adopted paper convention.

The adopted result uses the total-local convention,

```text
E_Z(x) = g mu_B [B_ext + B_z(x)]
```

with the full 432-condition atlas recomputed, the paper values and figures
regenerated, and the legacy archive preserved through tagged filenames. The
difference between the two conventions is itself a reported result.

The adopted profile normalization is `prefactor`. The `final-peak` and `l2`
variants remain optional sensitivity checks; the `l2` variant is documented as
divergent and is retained only as a reference point. No `final-peak` full-atlas
run is required for the paper.

The pipeline exposes the decisions end to end:

- `phase5_sensitivity_atlas.py --ez-convention {stray-mean,total-local,total-mean}
  --profile-norm {prefactor,final-peak,l2}` writes non-legacy results to tagged
  filenames and embeds a config block containing conventions, external field,
  noise scale, seeds, relevant script SHAs, and archive version.
- `phase5_atlas_merge_validate.py` accepts the same flags, refuses mismatched
  raw configs, requires `--allow-legacy` for archived raws that predate config
  recording, and stamps metadata with the raw-recorded atlas SHA rather than
  the current script SHA.
- `phase5_paper_figures.py --ez-convention
  {legacy-50ueV,stray-mean,total-mean,total-local} --profile-norm ...` uses the
  simulator kernel, including local E_Z(x), and tags non-legacy outputs.
- `phase5_atlas_figure.py`, `phase5_supp_robustness.py`, and
  `phase5_absolute_performance.py` accept `--dataset-suffix` to select the
  tagged data.
- `phase5_robust_candidate_retest.py` retests the adopted cross-ansatz robust
  candidates at `n_real = 30`.
- `tests/test_convention_options.py` covers normalization, dataset-tag
  uniqueness, merge-config validation, Hamiltonian-level convention effects,
  and centered-pocket flank symmetry.

## 11. Git and evidence publication policy

The mutable Git repository and immutable r31 audit tar have different roles.
A documentation-only Git update does not create a new r32 scientific release
and must not replace the r31 tar or its recorded SHA in the audit chain.

Commit to Git:

- source code and tests;
- `REPRODUCIBILITY.md` and build documentation;
- finalized Phase runners and protocols after the audit is complete;
- stable expected criteria and a final reproduction summary.

Do not commit ordinary local runtime state:

```gitignore
/.venv/
/r*-repro/
/repro-runs/
/phase*-r*-results-*.tar.gz
/phase*-r*-results-*.tar.gz.sha256
```

Keep timestamped phase archives locally as full audit evidence. After Phase 5,
a sanitized final evidence package may be published separately, for example as
a GitHub Release or Zenodo deposit, using a stable name such as:

```text
spinshuttle-r31-reproduction-evidence.tar.gz
```

The stable public bundle should record the accepted phase run IDs and hashes in
its internal manifest without hardcoding a particular timestamped result name
into the reusable runners or canonical protocols.
