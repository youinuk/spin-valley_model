#!/usr/bin/env bash
# Collect pipeline figure outputs into docs/figures/ under the
# manuscript-facing names used by paper.tex and supplementary.tex.
#
# Usage (from anywhere):
#   docs/collect_figures.sh "__ez-total-local__norm-prefactor"
#
# --------------------------------------------------------------------------
# T0-C NOTE (2026-09)
#
# Fig. 4 (sensitivity_atlas.pdf) and Fig. S1 (robustness_checks.pdf) are NO
# LONGER produced by the r31 atlas/supplement figure scripts. The r31 outputs
#
#     figures/phase5/phase5_atlas_validate_figure*.pdf
#     figures/phase5/supp_robustness*.pdf
#
# are the six-panel atlas and the floor-sweep robustness figure. Both were
# computed under the r31 seeding scheme and neither matches the current
# manuscript. Copying them into docs/figures/ silently reverts two figures to
# superseded science, so they are deliberately NOT in the map below and this
# script never falls back to them.
#
# Current sources. Note the ANALYSIS LAYER: data/t0c/analysis is canonical.
# data/t0/analysis is the superseded pre-correction layer and produced figures
# whose annotations contradict their own captions; this header used to name it.
#
#   Figs. 2, 3, 4, S5   one producer:
#       PYTHONPATH=. python reproduce/phase5_paper_figures.py --figures all \\
#           --ez-convention total-local --profile-norm prefactor \\
#           --responses data/t0c/responses_n100.csv --analysis data/t0c/analysis
#
#   --figures {2,3,4,S5} regenerates one at a time. Only Fig. 2 needs the
#   simulator; 3, 4 and S5 read JSON and CSV and run without jax.
#
# The earlier header named a make_figS1.py that was never written. Its r31
# predecessor, the floor sweep in phase5_supp_robustness.py, was retired with
# its outputs in commit b3aa85b; phase5_supp_robustness.py remains in the tree
# as declared archival r31 post-processing and must not be substituted here.
#
# Fig. 3 is no longer the quadrant schematic. It is drawn from the shipped
# per-condition responses, so phase5_quadrant_schematic.pdf is not installed and
# is not referenced by paper.tex.
#
# phase5_paper_figures.py defaults to the archival legacy-50ueV convention and
# its bare invocation does NOT produce the manuscript figure. Only the adopted
# suffix is accepted below, so a legacy dataset cannot be installed by mistake.
# --------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

EXPECTED_SUFFIX="__ez-total-local__norm-prefactor"
if [[ "${1:-}" != "$EXPECTED_SUFFIX" ]]; then
  echo "Usage: $0 \"$EXPECTED_SUFFIX\"" >&2
  echo "" >&2
  echo "Only the adopted dataset suffix is accepted. The untagged (legacy)" >&2
  echo "outputs are archival r31 evidence and must not be installed as" >&2
  echo "manuscript figures." >&2
  exit 2
fi
SUFFIX="$EXPECTED_SUFFIX"

mkdir -p "$SCRIPT_DIR/figures"

declare -A MAP=(
  ["$REPO_ROOT/figures/phase5/phase5_coupling_profiles${SUFFIX}.pdf"]="coupling_profiles.pdf"
  ["$REPO_ROOT/figures/t0/quadrant_data.pdf"]="quadrant_data.pdf"
  ["$REPO_ROOT/figures/t0/sensitivity_atlas.pdf"]="sensitivity_atlas.pdf"
  ["$REPO_ROOT/figures/t0/robustness_checks.pdf"]="robustness_checks.pdf"
  ["$REPO_ROOT/figures/phase2/step2_B1_ratio.pdf"]="gradient_halving.pdf"
  ["$REPO_ROOT/figures/phase2/step2_B3_filter.pdf"]="filter_suppression.pdf"
  ["$REPO_ROOT/figures/phase2/step3_C1_lz.pdf"]="landau_zener.pdf"
  ["$REPO_ROOT/figures/phase3/phase3_prism_fields.pdf"]="prism_fields.pdf"
)

missing=0
for src in "${!MAP[@]}"; do
  if [ ! -s "$src" ]; then
    echo "MISSING or EMPTY: $src" >&2
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  echo "" >&2
  echo "collect_figures.sh: required source figure(s) missing." >&2
  echo "PYTHONPATH=. python reproduce/phase5_paper_figures.py --figures all \\" >&2
  echo "    --ez-convention total-local --profile-norm prefactor \\" >&2
  echo "    --responses data/t0c/responses_n100.csv --analysis data/t0c/analysis" >&2
  echo "Do NOT substitute figures/phase5/phase5_atlas_validate_figure*.pdf or" >&2
  echo "supp_robustness*.pdf; those are superseded r31 science." >&2
  exit 1
fi

for src in "${!MAP[@]}"; do
  cp "$src" "$SCRIPT_DIR/figures/${MAP[$src]}"
done
echo "collected ${#MAP[@]} manuscript figures into $SCRIPT_DIR/figures/"

# A stale copy left in docs/figures/ is invisible in the build and silently
# reverts a figure. paper.tex no longer references the schematic.
if [ -e "$SCRIPT_DIR/figures/quadrant_schematic.pdf" ]; then
  echo "" >&2
  echo "NOTE: docs/figures/quadrant_schematic.pdf is left over from the" >&2
  echo "schematic Fig. 3 and is no longer referenced. Delete it." >&2
fi

# Report what was installed, with timestamps, so a figure that was regenerated
# but not collected is visible here rather than in the built PDF.
echo ""
echo "installed:"
for src in "${!MAP[@]}"; do
  printf '  %-24s  <- %s\n' "${MAP[$src]}" "${src#$REPO_ROOT/}"
done | sort
