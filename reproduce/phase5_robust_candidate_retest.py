#!/usr/bin/env python3
"""Targeted high-statistics (n_real=30) re-test of the cross-ansatz robust
candidates that appear in the adopted total-local full-validate atlas.

At n_real=5 the atlas shows four physical conditions robust in >=3/4
ansaetze (one in 4/4). Following the staged-falsification logic, we re-run
exactly those conditions across all four ansaetze at n_real=30 in the
adopted total-field convention and re-classify. Robust means both responses
significant and improving: dP_v<0 and dchi_phi<0 with |dP_v|>=1e-4,
|dchi_phi|>=1e-3.

Deterministic w.r.t. the atlas: replicates evaluate() exactly (case_label
"atlas" for the CRN seed, base_seed=41), so realizations r=0..4 reproduce
the atlas and r=5..29 add statistics.
"""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduce.phase5_sensitivity_atlas import (
    make_ff, GEOM0, GEOM_STEP, CASES)
from reproduce.phase4p6_crossterm import run_one_condition
from noise.charge_noise import OneOverFNoise
from constants import Defaults

e_C = 1.602176634e-19
P_MIN, CHI_MIN = 1e-4, 1e-3
MODELS = ["A", "A_pocket", "B_z", "B_x"]

# the four candidates flagged at n_real=5 (case, v, lambda_uev, geom_label)
CANDIDATES = [
    ("case_i_center", 5.0, 0.5, "period_nm-"),
    ("case_ii_edge", 10.0, 0.5, "depth_nm+"),
    ("case_ii_edge", 10.0, 1.0, "depth_nm+"),
    ("case_ii_edge", 20.0, 1.0, "depth_nm+"),
]


def geom_from_label(label):
    g = dict(GEOM0)
    g["label"] = label
    if label == "baseline":
        return g
    # label like "period_nm-" / "depth_nm+"
    sign = +1 if label.endswith("+") else -1
    param = label[:-1]
    g[param] = GEOM0[param] + sign * GEOM_STEP[param]
    return g


def evaluate_n(model, x_c, v, lambda_uev, geom, n_real, noise,
               ez_convention="total-local", profile_norm="prefactor"):
    ff = make_ff(geom["period_nm"], geom["depth_nm"],
                 geom["half_x_nm"], geom["half_z_nm"])
    pocket_width = 30e-9
    T = (8 * pocket_width) / v
    R = run_one_condition(
        ez_convention=ez_convention, profile_norm=profile_norm,
        v=v, case_label="atlas", pocket_x_center=x_c,
        lambda_0=lambda_uev * 1e-6 * e_C, coupling_model=model,
        n_real=n_real, noise=noise, ff=ff,
        Ev_baseline=100e-6 * e_C, Ev_min=5e-6 * e_C, pocket_width=pocket_width,
        Delta_v=0.5e-6 * e_C, N_max=500, base_seed=41, T_traj_for_noise=T,
    )
    dP = R["M2"]["P_v_dia"]["mean"] - R["M1V"]["P_v_dia"]["mean"]
    dchi = R["M2"]["phase"]["var_circular"] - R["M1"]["phase"]["var_circular"]
    return dP, dchi


def is_robust(dP, dchi):
    return (abs(dP) >= P_MIN and abs(dchi) >= CHI_MIN and dP < 0 and dchi < 0)


def main(n_real=30):
    noise = OneOverFNoise(sigma_total=Defaults.sigma_dx_m, alpha=1.0,
                          f_low=1e3, f_high=1e7)
    out = {}
    t0 = time.time()
    for (case, v, lam, glabel) in CANDIDATES:
        x_c = CASES[case]
        geom = geom_from_label(glabel)
        rec = {}
        for model in MODELS:
            dP, dchi = evaluate_n(model, x_c, v, lam, geom, n_real, noise)
            rec[model] = {"dP_v": dP, "dchi_phi": dchi,
                          "robust": bool(is_robust(dP, dchi))}
        n_rob = sum(1 for m in MODELS if rec[m]["robust"])
        key = f"{case}|v{v}|lam{lam}|{glabel}"
        out[key] = {"n_robust": n_rob, "models": rec}
        print(f"[{time.time()-t0:.0f}s] {key}: {n_rob}/4 robust "
              f"({[m for m in MODELS if rec[m]['robust']]})", flush=True)
    outdir = Path(__file__).resolve().parents[1] / "figures" / "phase5"
    outpath = outdir / "phase5_robust_candidate_retest__ez-total-local__norm-prefactor.json"
    payload = {"n_real": n_real, "P_min": P_MIN, "chi_min": CHI_MIN,
               "ez_convention": "total-local", "profile_norm": "prefactor",
               "candidates": out}
    json.dump(payload, open(outpath, "w"), indent=2)
    print(f"\nwrote {outpath.name}")
    ge3 = [k for k, v in out.items() if v["n_robust"] >= 3]
    print(f"candidates robust in >=3/4 ansaetze at n_real={n_real}: {len(ge3)}")
    for k in ge3:
        print(f"  {k}: {out[k]['n_robust']}/4")
    return out


if __name__ == "__main__":
    nr = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    main(nr)
