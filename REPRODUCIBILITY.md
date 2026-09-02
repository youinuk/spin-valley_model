# Reproducibility guide

This guide documents the **r32** scientific results and how to reproduce them.
Every command is run from the released repository root.

`r32` is a **scientific successor** to `r31`, not a repackaging of it. The r31
atlas drew a separate noise realization for each coupling ansatz, so a
cross-ansatz difference there also carried a sampling difference. The r32 study
(T0) repeats the comparison with common random numbers shared across ansätze
within each seed block, at nested realization counts
`n_real = 5, 10, 20, 30, 40, 60, 80, 100`. The `n_real = 30` estimands and
candidate-selection rule were registered before T0 production. The
`n_real = 100` extension was registered before any `n > 30` data existed, with
no new estimand or threshold.

Consequences for anyone reproducing r31 numbers:

- The r31 headline values are **historical reference**, not current results.
  Section 6 records the one comparison the manuscript still makes against them.
- The r32 producer sources differ from r31 (atlas
  `9b78f64d5c78d85d` → `78daa7a9eabcac25`), so the r31 tagged atlas dataset is
  **not shipped here**. It remains in the immutable r31 tar and Zenodo record.
  Do not gate an r32 build on `--metadata-only` against r31 metadata; see
  Section 6.
- The raw T0 production pickles are **not in this tree** either. They are in the
  Zenodo raw-data layer. Every T0-successor analysis number in the manuscript,
  together with Fig. 4 and the supplement robustness figure, is reproducible from the analysis JSON shipped
  in `data/t0/analysis/`. Figs. 2 and 3, the field and geometry controls, and
  the document build do not come from that JSON; see Sections 2 and 4.

## 1. Environment

Python 3.12 with the pinned `requirements.txt`; JAX on CPU is sufficient. No GPU
is required.

```bash
export PYTHONPATH=.
export JAX_ENABLE_X64=1
export JAX_PLATFORM_NAME=cpu
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

**float64 is mandatory.** The simulation multiplies nanometre positions by
tesla-per-metre gradients; in float32 the analytic derivatives disagree with
automatic differentiation. All scripts enable 64-bit mode at import.

**Pin the numeric libraries to one thread.** `scipy.linalg.expm` on a 4×4 matrix
inside a multithreaded BLAS pool is almost all synchronization, and the atlas
calls it 500 times per simulation. Pinning does not change any value; it is
worth roughly 4.8× on the atlas.

The Python analyses also run on Windows x86-64 with CPython 3.12. Keep line
endings as LF: a CRLF checkout changes recorded source hashes and makes
`sha256sum -c SHA256SUMS.txt` report spurious mismatches; the included
`.gitattributes` enforces this. `docs/collect_figures.sh` and `make_overleaf.sh`
need `bash` on `PATH` (Git Bash or WSL2).

Fixed seeds make the workflow deterministic, but bit-identical output across
operating systems and BLAS builds is not assumed; use the tolerances in
Section 5.

## 2. Reproduce the manuscript figures

Four commands, in order, produce every figure in the paper and the supplement.

```bash
# Figs. 2 and 3 -- computed from the field and profile model at run time.
# The options are REQUIRED: the bare invocation defaults to the archival
# legacy-50ueV convention and does not produce the manuscript figure.
python reproduce/phase5_paper_figures.py \
    --ez-convention total-local --profile-norm prefactor

# Fig. 4 and the supplement robustness figure -- JSON in, PDF out. No pickles, no simulation, no JAX.
mkdir -p figures/t0
python reproduce/t0/make_fig4.py  data/t0/analysis figures/t0/sensitivity_atlas.pdf
python reproduce/t0/make_supp_robustness.py data/t0/analysis figures/t0/robustness_checks.pdf

# Install the eight manuscript figures into docs/figures/.
docs/collect_figures.sh "__ez-total-local__norm-prefactor"
```

`collect_figures.sh` resolves its own location, so it runs from the repository
root or from `docs/`. It accepts **only** the adopted dataset suffix, and it
fails rather than substituting anything if a source is missing. It does not
reference the r31 six-panel atlas or floor-sweep outputs; installing those
would revert Fig. 4 and the supplement robustness figure to superseded science.

Expected output: `collected 8 manuscript figures into …/docs/figures/`.

The generators emit PDF, which AIP accepts. AIP lists EPS as its preferred
format; if the production stage requests it, convert the installed PDFs rather
than regenerating, so the rendered content is unchanged.

Figure 1 is TikZ inside `docs/paper.tex` and needs no generator. Its panel (d)
coordinates are generated from `data/t0/analysis/attribution_n100.json`; the
mapping is stated in a comment beside the panel.

Build the documents in `docs/` with `pdflatex ×2 → bibtex → pdflatex ×2` for
both `paper.tex` and `supplementary.tex` (REVTeX 4.2 with the AIP `jap`
substyle and the `aipnum4-2` bibliography style; **no `p{}` or `>{}` array
columns** — they cause a clean-build fatal).

## 3. The T0 analysis layer

`data/t0/analysis/` holds the closed T0 outputs (14 JSON). Every T0-successor number in
the manuscript is traceable to one of these files.

| file | supplies |
| --- | --- |
| `attribution_n30.json` | pre-registered endpoint: `D_ansatz`, `D_seed`, the six ansatz-pair and ten seed-pair values, dilution diagnostics |
| `attribution_n100.json` | the eight registered checkpoints, per-channel continuous dispersions, union-conditioned sensitivity |
| `posthoc_n30.json`, `posthoc_n100.json` | three-state channel decomposition and the inclusion–exclusion identity `joint = D_P + D_chi − D_both` |
| `legacy_n30.json`, `legacy_n100.json` | pairwise rank correlation, classification agreement, Cohen's kappa |
| `ranking_n30.json`, `ranking_n100.json` | the six ranked quantities and the counterexample gap |
| `discovery_n30.json`, `holdout_n30.json` | the pre-registered candidate screen and the candidate table |
| `discovery_n100.json` | the candidate screen at the registered follow-up prefix; the reference input for the layer comparison |
| `confirmation_n100.json`, `candidate_delta.json` | extended confirmation at `n_real = 100` and the layer comparison |
| `legacy_n5_block1.json` | the matched-seeding comparison against the archived r31 estimate |

`reproduce/t0/` holds six files. Two produce figures from the table above:
`make_fig4.py` and `make_supp_robustness.py`. Three document the chain from raw
production evidence to that table: `t0_step4_analysis.py` computes the analysis
JSON from the raw pickles, and `t0_check_prefix.py` and
`t0_check_seed_pairing.py` are the nested-prefix and seed-pairing gates.
One of the three does run in-tree: with both discovery files shipped, the
registered layer comparison is reproducible without the raw layer,

```bash
python reproduce/t0/t0_step4_analysis.py candidate-delta \
    --ref data/t0/analysis/discovery_n30.json \
    --new data/t0/analysis/discovery_n100.json \
    --out candidate_delta.json
```

and its output should match the shipped `candidate_delta.json`.

**The remaining chain tools are not part of the default in-tree reproduction
path**, because
their input — the ten `n_real = 30` and ten `n_real = 100` raw pickles — is
distributed separately in the Zenodo raw-data layer. They become runnable
provenance checks once that layer is supplied.

One provenance fact is directly checkable from the released tree: the
5/10/20/30 prefix blocks of `attribution_n100.json` equal those of
`attribution_n30.json` exactly, although the two came from separate production
runs. That is the nested-prefix gate at the estimand level. A second
reproduction check, recorded in the release evidence rather than in this tree,
found the shipped `attribution_n30.json` numerically identical to an
independently computed review copy.

## 4. Validation controls

These certify the field model, its gradients, and the time-evolution engine.

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

Expected: `14 passed, 1 deselected` for the first gate, `3 passed` for the
geometry group, and the slow end-to-end test alone.

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
`eps_v ≈ E_Z` enhancement is generated by the four-level dynamics without an
artificial resonance window in `lambda_sv(x)`: it preserves the Gaussian-pocket
shape while raising its minimum above the total-field `E_Z` range. Expected
leakage is ~`4.5e-1` with the crossing (`eps_v_min = 5 ueV`) and ~`1.8e-3`
without it (`eps_v_min = 70 ueV`), with lambda-zero leakage below `1e-3` in both
landscapes.

`reproduce/oda_C2_pocket.py` tests a retained legacy heuristic and is expected
to print `C2 PASS: False`; it is not a validation gate. The official V4 pocket
gate is `reproduce/oda_C2v2_scan.py`.

## 5. Comparison tolerances

- raw continuous values: `np.allclose(rtol=1e-7, atol=1e-10)`
- summary metrics: absolute difference below `5e-4`
- values quoted to three significant figures in the paper: exact agreement at
  that displayed precision
- category counts, candidate counts, and row keys: exact match
- PDF and PNG byte hashes: **not expected to match.** The figure generators
  write TrueType-embedded PDFs (`pdf.fonttype = 42`) but do not pin
  non-scientific metadata such as `CreationDate`. Compare rasterized output.

## 6. Provenance and known limitations

**Zeeman convention.** An external-field inconsistency found during development
has been resolved. The archived legacy run used a stray-field-only Zeeman
energy: `Defaults.B_ext_T = 0.5` was defined but did not enter the atlas Zeeman
energy, giving a mean `E_Z` of about `3 ueV`. That state is retained for archive
reproducibility and is not the adopted convention. The adopted result uses the
total-local form `E_Z(x) = g mu_B [B_ext + B_z(x)]`. The adopted profile
normalization is `prefactor`; the `final-peak` and `l2` variants remain optional
sensitivity checks, and `l2` is documented as divergent. Both conventions are
exposed end to end: `phase5_sensitivity_atlas.py` and
`phase5_atlas_merge_validate.py` take `--ez-convention` and `--profile-norm`,
write non-legacy results to tagged filenames, and embed a config block recording
conventions, external field, noise scale, seeds, script SHAs, and archive
version. The merge step refuses mismatched raw configs.

**The r31 comparison the manuscript makes.** Repeating the archived comparison
with the seeding corrected, at matched ensemble size `n_real = 5`, matched seed
block and matched convention, raises the mean classification agreement from
`35.96%` to `56.94%` and the mean rank correlation from `0.837` to `0.917`.
`legacy_n5_block1.json` is that computation. It is the only place the r31 values
enter, and they enter as a historical descriptor.

**Why `--metadata-only` does not gate r32.** The gate compares the atlas
source's current bytes against the SHA recorded in the metadata of the data it
produced. It is a meaningful archive-provenance invariant only for a release
that ships the corresponding producer-linked dataset. r32 ships a post-T0
producer and does not ship the r31 n=5 tagged atlas dataset, so the gate has
nothing to check and would report a mismatch that means nothing. The relation
r32 must verify is instead

```
r32 producer bytes
  ↕  the raw T0 production evidence      (Zenodo raw-data layer)
  ↕  the closed analysis JSON            (data/t0/analysis/)
  ↕  the manuscript figures and numbers
```

**Archival r31 post-processing.** `phase5_robust_candidate_retest.py` (the
retired targeted re-test), `phase5_supp_robustness.py` (the r31 floor sweep),
`phase5_absolute_performance.py`, `phase5_atlas_figure.py` (the six-panel atlas)
and `phase5_narrow_scope.py` remain in the tree because r31 evidence and
released tests refer to them. They are **not** reproduction steps for this
manuscript and running them does not produce current figures.

**Scope of the validations.** These checks certify the field model, its
gradients, and the time-evolution engine, and they support the representative
parameters of Table I. None of them validates the spin–valley coupling profile
`lambda_sv(x)`, which the paper treats as a phenomenological diagnostic ansatz.
Quantifying sensitivity to that choice is the purpose of the study, and the
paper's conclusion is that design conclusions drawn under one assumed profile
remain model-conditional.
