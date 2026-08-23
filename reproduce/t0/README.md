# T0 successor analysis — reproduction of Fig. 4 and Fig. S1

Drop-in addition to the public tree. Adds the two manuscript figures that the
r31 figure scripts do not produce, together with the analysis outputs they read.

```
reproduce/t0/make_fig4.py        Fig. 4  (channel asymmetry, candidate transport,
                                          ranking against classification)
reproduce/t0/make_figS1.py       Fig. S1 (ranked-quantity gap, union-conditioned
                                          sensitivity)
data/t0/analysis/*.json          the closed T0 analysis outputs
```

## Reproduce

```bash
mkdir -p figures/t0
PYTHONPATH=. python reproduce/t0/make_fig4.py \
    data/t0/analysis figures/t0/sensitivity_atlas.pdf
PYTHONPATH=. python reproduce/t0/make_figS1.py \
    data/t0/analysis figures/t0/robustness_checks.pdf
docs/collect_figures.sh "__ez-total-local__norm-prefactor"
```

`collect_figures.sh` resolves its own location, so it may be run from the repo
root or from `docs/`. It accepts only the adopted dataset suffix.

Both scripts read JSON only. No pickle access, no simulation, no JAX. They
produce render-stable PDFs with TrueType-embedded fonts (`pdf.fonttype = 42`).
File hashes may differ between runs because of non-scientific PDF metadata such
as `CreationDate`; compare rasterized output, not checksums.

## What each file supplies

| file | supplies |
|---|---|
| `attribution_n30.json` | pre-registered endpoint: `D_ansatz`, `D_seed`, pairwise values, dilution diagnostics |
| `attribution_n100.json` | the eight registered checkpoints, continuous `S`, union-conditioned sensitivity |
| `posthoc_n30.json`, `posthoc_n100.json` | three-state channel decomposition and the inclusion–exclusion identity |
| `legacy_n30.json`, `legacy_n100.json` | pairwise rank correlation, classification agreement, Cohen's kappa |
| `ranking_n30.json`, `ranking_n100.json` | the six ranked quantities and the counterexample gap |
| `discovery_n30.json`, `holdout_n30.json` | pre-registered candidate screen and Table III |
| `confirmation_n100.json`, `candidate_delta.json` | extended confirmation and the layer comparison |
| `legacy_n5_block1.json` | matched-seeding comparison against the archived r31 estimate |

Every T0-successor analysis number reported in the manuscript is traceable to
one of these files; the mapping is recorded in the STEP 7 data ledger. Input
parameters, literature benchmarks, and quantities explicitly identified as
archival retain their original r31 provenance and are not covered here.

## Provenance

These are the accepted T0 outputs, reproduced on the pinned WSL2 machine. The
prefix blocks 5/10/20/30 of `attribution_n100.json` equal `attribution_n30.json`
exactly, across separate production runs, which checks the nested-prefix gate at
the estimand level.

The r31 outputs

```
figures/phase5/phase5_atlas_validate_figure*.pdf
figures/phase5/supp_robustness*.pdf
```

are the superseded six-panel atlas and floor-sweep figures. They are retained as
archival evidence for r31 and must never be installed as Fig. 4 or Fig. S1;
`docs/collect_figures.sh` no longer references them.

## Figures 2 and 3

Produced by the r31-era script, but only with the adopted convention:

```bash
PYTHONPATH=. python reproduce/phase5_paper_figures.py \
    --ez-convention total-local --profile-norm prefactor
```

The bare invocation defaults to `legacy-50ueV` and reproduces the archival
figure, not the manuscript figure.
