What is in this folder
======================

Two generations of pipeline outputs coexist here, distinguished by a
filename tag. Both are needed; do not delete either set.

1. Tagged files: *__ez-total-local__norm-prefactor.*
   The ADOPTED dataset used for every number and figure in the paper:
   total-field Zeeman convention E_Z(x) = g mu_B [B_ext + B_z(x)] with
   prefactor profile normalization. Regenerate with the commands in
   REPRODUCIBILITY.md sections 3-4 (pass --ez-convention total-local
   --profile-norm prefactor).

2. Untagged files: phase5_atlas_*.{pkl,json,csv} without the tag
   The ARCHIVED LEGACY stray-field-only run. It is not the paper
   convention; it is retained deliberately because the paper reports the
   legacy -> adopted shift (mean Spearman rho 0.68 -> 0.84) as primary
   model-risk evidence, and REPRODUCIBILITY.md section 7 documents how to
   reproduce it (--allow-legacy path). These files predate config
   recording, hence the separate validation route.

Other files (per tag): merged summary CSVs, per-condition metadata JSON,
paper/supplementary figure PNG+PDF pairs, and the n_real=30 robust-candidate
retest JSON. The *_lite / *_mid variants are reduced-grid smoke-test
outputs referenced by the test suite.
