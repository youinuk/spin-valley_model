#!/usr/bin/env bash
# Collect pipeline figure outputs into docs/figures/ under the
# manuscript-facing names used by paper.tex and supplementary.tex.
#
# Usage:
#   ./collect_figures.sh                # legacy archive figures
#   ./collect_figures.sh "__ez-total-local__norm-prefactor"
#                                        # post-decision dataset (run AFTER
#                                        # the full validate recomputation;
#                                        # preview/smoke datasets do not
#                                        # produce the validate-mode figures)
#
# The quadrant schematic is convention-independent and is always taken
# from the untagged file.
set -euo pipefail
SUFFIX="${1:-}"
mkdir -p figures

declare -A MAP=(
  ["../figures/phase5/phase5_coupling_profiles${SUFFIX}.pdf"]="figures/coupling_profiles.pdf"
  ["../figures/phase5/phase5_quadrant_schematic.pdf"]="figures/quadrant_schematic.pdf"
  ["../figures/phase5/phase5_atlas_validate_figure${SUFFIX}.pdf"]="figures/sensitivity_atlas.pdf"
  ["../figures/phase2/step2_B1_ratio.pdf"]="figures/gradient_halving.pdf"
  ["../figures/phase2/step2_B3_filter.pdf"]="figures/filter_suppression.pdf"
  ["../figures/phase2/step3_C1_lz.pdf"]="figures/landau_zener.pdf"
  ["../figures/phase3/phase3_prism_fields.pdf"]="figures/prism_fields.pdf"
  ["../figures/phase5/supp_robustness${SUFFIX}.pdf"]="figures/robustness_checks.pdf"
)

missing=0
for src in "${!MAP[@]}"; do
  if [ ! -f "$src" ]; then
    echo "MISSING: $src" >&2
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  echo "collect_figures.sh: required source figure(s) missing." >&2
  echo "For a suffixed dataset, run the FULL validate recomputation and its" >&2
  echo "post-processing first (atlas x2 -> merge -> atlas_figure -> " >&2
  echo "supp_robustness -> paper_figures), then re-run this script." >&2
  exit 1
fi
for src in "${!MAP[@]}"; do
  cp "$src" "${MAP[$src]}"
done
echo "collected ${#MAP[@]} manuscript figures into docs/figures/ (suffix='${SUFFIX}')"
