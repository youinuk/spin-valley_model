#!/usr/bin/env bash
# Collect pipeline figure outputs into docs/figures/ under the
# manuscript-facing names used by paper.tex and supplementary.tex.
#
# Usage (from anywhere):
#   docs/collect_figures.sh "__ez-total-local__norm-prefactor"
#
# --------------------------------------------------------------------------
# T0-SUCCESSOR NOTE (2026-08)
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
# Current sources:
#   Fig. 2, 3   reproduce/phase5_paper_figures.py
#                   --ez-convention total-local --profile-norm prefactor
#   Fig. 4      reproduce/t0/make_fig4.py  data/t0/analysis \
#                   figures/t0/sensitivity_atlas.pdf
#   Fig. S1     reproduce/t0/make_figS1.py data/t0/analysis \
#                   figures/t0/robustness_checks.pdf
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
  ["$REPO_ROOT/figures/phase5/phase5_quadrant_schematic.pdf"]="quadrant_schematic.pdf"
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
  echo "Fig. 2/3 : reproduce/phase5_paper_figures.py" >&2
  echo "           --ez-convention total-local --profile-norm prefactor" >&2
  echo "Fig. 4/S1: reproduce/t0/make_fig4.py and reproduce/t0/make_figS1.py" >&2
  echo "Do NOT substitute figures/phase5/phase5_atlas_validate_figure*.pdf or" >&2
  echo "supp_robustness*.pdf; those are superseded r31 science." >&2
  exit 1
fi

for src in "${!MAP[@]}"; do
  cp "$src" "$SCRIPT_DIR/figures/${MAP[$src]}"
done
echo "collected ${#MAP[@]} manuscript figures into $SCRIPT_DIR/figures/"
