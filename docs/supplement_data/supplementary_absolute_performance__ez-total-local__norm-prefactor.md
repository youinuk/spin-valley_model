# Supplementary: absolute full-model performance proxies

The main-text (dP_v, dchi) are baseline-relative responses, so here we report the *absolute* full-model (M2) values and a phase-error proxy per ansatz. The setup matches the atlas (baseline geometry, center+edge, v=[5,10,20], n_real=5, lambda=1 ueV).


| ansatz | mean $P_v^{M2}$ | mean $\chi_\phi^{M2}$ | mean $p_\phi^{M2}$ |
| --- | ---: | ---: | ---: |
| A | 2.591e-01 | 1.954e-02 | 4.862e-03 |
| A\_pocket | 7.864e-02 | 1.061e-02 | 2.645e-03 |
| B\_z | 2.655e-01 | 1.723e-01 | 4.128e-02 |
| B\_x | 1.072e-01 | 1.694e-02 | 4.217e-03 |

Interpretation: the absolute leakage $P_v^{M2}$ and dephasing proxy $p_\phi^{M2}$ are of the same order of magnitude across ansatze. The quadrant interpretation of the response map is ansatz-sensitive, but the absolute error scale itself is comparable; this table separates response sensitivity from absolute performance (the main text claims no absolute optimization).
