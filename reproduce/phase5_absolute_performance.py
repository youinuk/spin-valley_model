"""
Absolute full-model performance proxies per ansatz.

The main text treats only the baseline-relative response (dP_v, dchi). This script
reports the *absolute* full-model values P_v^M2, chi_phi^M2 and the phase-error
proxy p_phi = (1-e^{-chi/2})/2 per ansatz, to give the supplement context for
"the response varies but the absolute error scale is around this level".

Same setup as the atlas (GEOM0 baseline, pocket center+edge, v=[5,10,20], n_real=5).

Usage:
  PYTHONPATH=. python reproduce/phase5_absolute_performance.py
Output: docs/supplement_data/supplementary_absolute_performance*.md
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from noise.charge_noise import OneOverFNoise
from geometry.fourier_field import fit_from_prism_array
from geometry.periodic_array import PeriodicPrismArray
from reproduce.phase4p6_crossterm import run_one_condition
from constants import Defaults

ROOT = Path(__file__).resolve().parents[1]
e_C = 1.602176634e-19
MODELS = ["A", "A_pocket", "B_z", "B_x"]
GEOM0 = dict(period_nm=150.0, depth_nm=-50.0, half_x_nm=25.0, half_z_nm=15.0)


def make_ff():
    arr = PeriodicPrismArray(
        period_a=GEOM0["period_nm"] * 1e-9, half_x=GEOM0["half_x_nm"] * 1e-9,
        half_y=25e-9, half_z=GEOM0["half_z_nm"] * 1e-9,
        cz=15e-9, N_periods_each_side=4, Ms=1.4e6,
    )
    return fit_from_prism_array(arr, GEOM0["depth_nm"] * 1e-9, N_harm=3)


def p_phi(chi):
    """phase-error proxy from circular phase variance χ."""
    return (1.0 - np.exp(-chi / 2.0)) / 2.0


def main():
    ff = make_ff()
    # same as the atlas: f_high=1e7 (phase5_sensitivity_atlas.py L207)
    noise = OneOverFNoise(sigma_total=Defaults.sigma_dx_m, alpha=1.0,
                          f_low=1e3, f_high=1e7)
    pocket_width = 30e-9
    # center (x_c=0) and edge (x_c=25 nm) exactly as the atlas (CASES L58)
    cases = {"center": 0.0, "edge": 25e-9}
    velocities = [5.0, 10.0, 20.0]
    n_real = 5
    lambda_uev = 1.0  # representative nonzero coupling

    # accumulate per-ansatz absolute M2 values
    acc = {m: {"P_v_M2": [], "chi_M2": []} for m in MODELS}
    for model in MODELS:
        for cname, x_c in cases.items():
            for v in velocities:
                T = (8 * pocket_width) / v
                R = run_one_condition(
                    ez_convention=EZ_CONVENTION, profile_norm=PROFILE_NORM,
                    v=v, case_label=cname, pocket_x_center=x_c,
                    lambda_0=lambda_uev * 1e-6 * e_C, coupling_model=model,
                    n_real=n_real, noise=noise, ff=ff,
                    Ev_baseline=100e-6 * e_C, Ev_min=5e-6 * e_C,
                    pocket_width=pocket_width, Delta_v=0.5e-6 * e_C,
                    N_max=500, base_seed=41, T_traj_for_noise=T,
                )
                acc[model]["P_v_M2"].append(R["M2"]["P_v_dia"]["mean"])
                acc[model]["chi_M2"].append(
                    R["M2"]["phase"]["var_circular"])
        print(f"  {model}: done")

    lines = ["# Supplementary: absolute full-model performance proxies\n",
             "The main-text (dP_v, dchi) are baseline-relative responses, so here we "
             "report the *absolute* full-model (M2) values and a phase-error proxy "
             "per ansatz. The setup matches the atlas "
             "(baseline geometry, center+edge, v=[5,10,20], n_real=5, lambda=1 ueV).\n",
             "\n| ansatz | mean $P_v^{M2}$ | mean $\\chi_\\phi^{M2}$ | "
             "mean $p_\\phi^{M2}$ |",
             "| --- | ---: | ---: | ---: |"]
    for m in MODELS:
        Pv = np.mean(acc[m]["P_v_M2"])
        chi = np.mean(acc[m]["chi_M2"])
        pp = p_phi(chi)
        disp = m.replace("_", r"\_")
        lines.append(f"| {disp} | {Pv:.3e} | {chi:.3e} | {pp:.3e} |")
    lines += ["\nInterpretation: the absolute leakage $P_v^{M2}$ and dephasing proxy "
              "$p_\\phi^{M2}$ are of the same order of magnitude across ansatze. The "
              "quadrant interpretation of the response map is ansatz-sensitive, but the "
              "absolute error scale itself is comparable; this table separates response "
              "sensitivity from absolute performance (the main text claims no absolute optimization).\n"]
    out = ROOT / "docs" / "supplement_data" / f"supplementary_absolute_performance{DATASET_SUFFIX}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"saved: {out}")
    print("\n".join(lines[3:]))


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--dataset-suffix", default="",
                    help="tag for the INPUT summary CSV")
    _p.add_argument("--mode", default="validate")
    _p.add_argument("--ez-convention", dest="ez_convention",
                    choices=["stray-mean", "total-local", "total-mean"],
                    default="stray-mean")
    _p.add_argument("--profile-norm", dest="profile_norm",
                    choices=["prefactor", "final-peak", "l2"],
                    default="prefactor")
    _a = _p.parse_args()
    DATASET_SUFFIX = _a.dataset_suffix
    EZ_CONVENTION = _a.ez_convention
    PROFILE_NORM = _a.profile_norm
    if (EZ_CONVENTION, PROFILE_NORM) != ("stray-mean", "prefactor") and not DATASET_SUFFIX:
        DATASET_SUFFIX = f"__ez-{EZ_CONVENTION}__norm-{PROFILE_NORM}"
    main()
