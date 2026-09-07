# `reproduce/t0/` — T0/T0-C provenance utilities

These are the public tools that connect the raw production evidence to the
shipped analysis layer. **No figure is drawn here.** The manuscript figures are
rendered by `reproduce/phase5_paper_figures.py`, one directory up.

```
t0_step4_analysis.py       raw pickles -> the analysis JSON; also the canonical
                           effect-size floors and the nine-category classifier,
                           which the figure renderer imports rather than copies
t0_check_prefix.py         C2-G8, the nested-prefix identity gate
t0_check_seed_pairing.py   C1-G4, the CRN seed-pairing structure
export_responses.py        raw -> data/t0c/responses_n100.csv, the per-condition
                           responses Fig. 3 is drawn from
compare_t0_t0c.py          the correction audit across the two shipped analysis
                           layers; not a perturbation series between them
```

`data/t0c/analysis/` is the canonical layer and the source of every current
manuscript number. `data/t0/analysis/` is superseded historical provenance,
retained for the correction audit.

**Everything else lives in one place, on purpose.** The commands, their expected
output, and the mapping from each analysis file to the numbers it supplies are
in `REPRODUCIBILITY.md`; the release gates are in `run_phase5.sh`. This file
used to carry its own copy of the command list and drifted so far from the tree
that it named a script which had never existed. A description repeated in three
places is a description that will disagree with itself.

## Post-hoc operational-floor sensitivity

`threshold_floor_sensitivity.py` is a pure post-processing check on the
corrected n=100 raw layer. It reconstructs the eight registered nested prefixes,
verifies the registered $D_{\rm ansatz}$/$D_{\rm seed}$ trajectory exactly, and
then independently rescales the two operational classification floors by
`0.5`, `0.75`, `1`, `1.5`, and `2`. It does not rerun the simulator and it does
not alter the registered classifier or primary results.

```bash
python reproduce/t0/threshold_floor_sensitivity.py \
    ../repro-runs/t0c-step3b \
    /tmp/threshold_floor_sensitivity.json
cmp /tmp/threshold_floor_sensitivity.json \
    data/t0c/analysis/threshold_floor_sensitivity.json
```

The shipped JSON is `data/t0c/analysis/threshold_floor_sensitivity.json`.

