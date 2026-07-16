# Supplementary robustness analyses (full validate)

Recomputed from raw pkl. The key conclusion (quadrant is more ansatz-dependent than ranking, especially for B_x) holds across all variants.


## 1. Rank-correlation weighting sweep

Mean pairwise Spearman rho for different |R| magnitude definitions. The raw norm is biased toward the larger-scale dchi.


| weighting | mean Spearman ρ |
| --- | ---: |
| raw norm (w=1,1) | 0.837 |
| threshold-normalized | 0.769 |
| |dP_v| only | 0.751 |
| |dchi_phi| only | 0.845 |
| P_ref=1e-2, chi_ref=1 | 0.754 |
| P_ref=1e-4, chi_ref=1e-3 | 0.769 |

Conclusion: ranking remains high but weighting-dependent (rho ~ 0.75-0.85). Rank is a complementary, weighting-dependent diagnostic; the scale-insensitive quadrant is of primary interest.


## 2. Quadrant-threshold sweep

Mean quadrant agreement vs effect-size threshold (strict classification).


| threshold (P_min, chi_min) | mean quadrant agreement |
| --- | ---: |
| none | 49.4% |
| 1e-5, 1e-4 | 36.3% |
| 1e-4, 1e-3 (main) | 36.0% |
| 1e-3, 1e-2 | 58.0% |

Conclusion: quadrant agreement remains far from model-independent across thresholds, ranging from roughly 36% to 58% in this sweep. The conclusion that quadrant interpretation is ansatz-sensitive is threshold-robust.


## 3. Chance-corrected agreement (Cohen's kappa)

Agreement corrected for the marginal quadrant distribution.


| pair | raw agreement | Cohen's κ |
| --- | ---: | ---: |
| A vs A_pocket | 36.1% | 0.211 |
| A vs B_z | 45.4% | 0.334 |
| A vs B_x | 29.6% | 0.145 |
| A_pocket vs B_z | 31.5% | 0.154 |
| A_pocket vs B_x | 44.4% | 0.244 |
| B_z vs B_x | 28.7% | 0.135 |

Conclusion: chance-corrected agreement is modest across all pairs (kappa ~ 0.13-0.33); the two lowest values are the B_z vs B_x, A vs B_x pairs, both pairing the partial-x-Bx model with a partial-x-Bz one. This is a gradient-component contrast modulated by spatial localization, not a blanket property of any single ansatz.
