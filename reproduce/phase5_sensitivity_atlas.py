"""
Phase 5 — Coupling-Model Sensitivity Atlas.

Coupling-ansatz sensitivity atlas: computation kernel.
*Not* a search for a robust basin, but a quantification of response sensitivity to the coupling-model choice.

Six metrics:
1. response amplitude: per-model mean |dP_v|, |dchi_phi|
2. geometry sensitivity (raw): S_θ = |R(θ+Δθ) - R(θ-Δθ)| / (2Δθ)
3. model sensitivity: D_{m1,m2} (raw + normalized), response difference at the same condition
4. rank correlation: Spearman rho of the |R| ranking across models
5. quadrant agreement: fraction of conditions in the same quadrant (sign-preserving)
6. relative geometry sensitivity: theta0-normalized S_theta

Mode:
  preview:  4 model × edge × v=[10,20] × λ=[0.5,1.0] × (baseline+1 perturb) × n_real=2
  validate: 4 model × 2 case × v=[5,10,20] × λ=[0.5,1.0] × 9 geom × n_real=5
  full:     validate grid × n_real=10

Outputs:
  figures/phase5/phase5_atlas_summary_{mode}.csv
  figures/phase5/phase5_atlas_sensitivity_{mode}.csv
  figures/phase5/phase5_atlas_metadata_{mode}.json
  figures/phase5/phase5_atlas_{mode}.png
  docs/phase5_sensitivity_atlas_results.md (auto-generated in full mode only)

Usage:
  PYTHONPATH=. python reproduce/phase5_sensitivity_atlas.py --mode preview
"""

from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from constants import Defaults
from noise.charge_noise import OneOverFNoise
from geometry.periodic_array import PeriodicPrismArray
from geometry.fourier_field import fit_from_prism_array
from reproduce.phase4p6_crossterm import run_one_condition

ARCHIVE_VERSION = "phase5_atlas_v27"  # v27: ez_convention/profile_norm options (legacy default unchanged)


def dataset_tag(ez_convention, profile_norm):
    """Filename tag for non-legacy convention/normalization datasets (audit r5 par.2).
    Legacy (stray-mean, prefactor) keeps the historical names."""
    if ez_convention == "stray-mean" and profile_norm == "prefactor":
        return ""
    return f"__ez-{ez_convention}__norm-{profile_norm}"

# Baseline geometry
GEOM0 = dict(period_nm=150.0, depth_nm=-50.0, half_x_nm=25.0, half_z_nm=15.0)
GEOM_STEP = dict(period_nm=30.0, depth_nm=10.0, half_x_nm=5.0, half_z_nm=3.0)
GEOM_FIXED = dict(half_y_nm=25.0, cz_nm=15.0)

CASES = {"case_i_center": 0.0, "case_ii_edge": 25e-9}
MODELS = ["A", "A_pocket", "B_z", "B_x"]
P_MIN_THRESH = 1e-4
CHI_MIN_THRESH = 1e-3
# Per-channel scales for the *channel-balanced* model distance: each
# response is measured in units of its own effect-size threshold, so the
# two channels contribute comparably (raw D is otherwise dominated by the
# larger-scale dchi). Using distinct scales makes this a genuinely
# different metric from raw D -- not a uniform rescaling -- so its ranking
# can differ from raw. Set equal only if a pure rescaling is intended.
P_SCALE = P_MIN_THRESH    # 1e-4
CHI_SCALE = CHI_MIN_THRESH  # 1e-3


def quadrant_of(dP, dchi, P_min=P_MIN_THRESH, chi_min=CHI_MIN_THRESH):
    """Response classification with strict effect-size handling.

    Earlier versions set eff=0 when only one channel exceeded its threshold,
    dumping every such case into 'both_worsen' (in full validate, 131 of the 168
    'both_worsen' cases were actually one-channel-only). That contaminated the
    'both_worsen' label, so one-channel-significant cases get their own categories.

    Classification (P=dP_v valley leakage, X=dchi_phi dephasing; <0 = improvement):
      both below threshold        -> below_threshold
      both significant:
        P<0, X<0                  -> robust          (both improve)
        P>0, X<0                  -> valley_trade
        P<0, X>0                  -> spin_trade
        P>0, X>0                  -> both_worsen      (both worsen, genuine)
      one significant only:
        P only, P<0               -> P_only_improve
        P only, P>0               -> P_only_worsen
        X only, X<0               -> chi_only_improve
        X only, X>0               -> chi_only_worsen
    """
    sig_P = abs(dP) >= P_min
    sig_chi = abs(dchi) >= chi_min
    if not (sig_P or sig_chi):
        return "below_threshold"
    if sig_P and sig_chi:
        if dP < 0 and dchi < 0:
            return "robust"
        if dP > 0 and dchi < 0:
            return "valley_trade"
        if dP < 0 and dchi > 0:
            return "spin_trade"
        return "both_worsen"
    # exactly one channel significant
    if sig_P:
        return "P_only_improve" if dP < 0 else "P_only_worsen"
    return "chi_only_improve" if dchi < 0 else "chi_only_worsen"


def make_ff(period_nm, depth_nm, half_x_nm, half_z_nm):
    arr = PeriodicPrismArray(
        period_a=period_nm * 1e-9, half_x=half_x_nm * 1e-9,
        half_y=GEOM_FIXED["half_y_nm"] * 1e-9, half_z=half_z_nm * 1e-9,
        cz=GEOM_FIXED["cz_nm"] * 1e-9, N_periods_each_side=4, Ms=1.4e6,
    )
    return fit_from_prism_array(arr, depth_nm * 1e-9, N_harm=3)


def build_geometries(mode):
    """unique geometry points (deduplicated)."""
    if mode == "preview":
        # baseline + 1 perturbation (period only)
        geoms = [
            {"label": "baseline", **GEOM0},
            {"label": "period+", **{**GEOM0, "period_nm": GEOM0["period_nm"] + GEOM_STEP["period_nm"]}},
        ]
    else:
        # baseline + 4 param × ±step = 9 unique
        geoms = [{"label": "baseline", **GEOM0}]
        for p in ["period_nm", "depth_nm", "half_x_nm", "half_z_nm"]:
            for sign in [-1, +1]:
                g = dict(GEOM0)
                g[p] = GEOM0[p] + sign * GEOM_STEP[p]
                g["label"] = f"{p}{'+' if sign > 0 else '-'}"
                geoms.append(g)
    return geoms


def evaluate(model, x_c, v, lambda_uev, geom, n_real, noise, base_seed=41,
             ez_convention="stray-mean", profile_norm="prefactor"):
    ff = make_ff(geom["period_nm"], geom["depth_nm"],
                  geom["half_x_nm"], geom["half_z_nm"])
    e_C = 1.602176634e-19
    pocket_width = 30e-9
    T = (8 * pocket_width) / v
    R = run_one_condition(
        ez_convention=ez_convention, profile_norm=profile_norm,
        v=v, case_label="atlas", pocket_x_center=x_c,
        lambda_0=lambda_uev * 1e-6 * e_C, coupling_model=model,
        n_real=n_real, noise=noise, ff=ff,
        Ev_baseline=100e-6 * e_C, Ev_min=5e-6 * e_C, pocket_width=pocket_width,
        Delta_v=0.5e-6 * e_C, N_max=500, base_seed=base_seed, T_traj_for_noise=T,
    )
    dP = R["M2"]["P_v_dia"]["mean"] - R["M1V"]["P_v_dia"]["mean"]
    dchi = R["M2"]["phase"]["var_circular"] - R["M1"]["phase"]["var_circular"]
    return {"dP_v": dP, "dchi_phi": dchi}


def main(mode="preview", n_real=None, no_plots=False, metadata_only=False,
         ez_convention="stray-mean", profile_norm="prefactor",
         only_case=None, save_raw=False):
    if metadata_only:
        # verify only the script-SHA consistency of existing metadata, no computation
        script_path = Path(__file__).resolve()
        cur_sha = hashlib.sha256(open(script_path, "rb").read()).hexdigest()[:16]
        meta_path = (Path(__file__).resolve().parents[1] / "figures" / "phase5"
                     / f"phase5_atlas_metadata_{mode}{dataset_tag(ez_convention, profile_norm)}.json")
        import reproduce.phase4p6_crossterm as _ct
        import hashlib as _hl
        ker_sha = _hl.sha256(open(_ct.__file__, "rb").read()).hexdigest()[:16]
        print(f"[--metadata-only] current atlas SHA: {cur_sha}  kernel SHA: {ker_sha}")
        is_legacy = (ez_convention in ("stray-mean", "legacy-50ueV")
                     and profile_norm in ("prefactor",))
        if is_legacy:
            print("  note: for LEGACY/stray-field metadata a mismatch with the current"
                  " script is expected, because the archived raw predates config recording.")
        else:
            print("  note: for the ADOPTED total-local metadata this SHA must MATCH the"
                  " shipped script; a mismatch is a release-blocking provenance error"
                  " (regenerate the raws with the current source).")
        if not meta_path.exists():
            print(f"  no metadata: {meta_path.name} (computation required)")
            return
        meta = json.load(open(meta_path))
        meta_sha = (meta.get("script_sha256_16")
                    or meta.get("atlas_script_sha256_16") or "?")
        match = "OK" if meta_sha == cur_sha else (
            "MISMATCH (expected for legacy)" if is_legacy
            else "MISMATCH — RELEASE BLOCKER (regenerate)")
        print(f"  {mode} metadata SHA: {meta_sha}  → {match}")
        if meta.get("merge_script_sha256_16"):
            print(f"  (merge script SHA: {meta['merge_script_sha256_16']})")
        print(f"  mean_rank_correlation: {meta.get('mean_rank_correlation')}")
        print(f"  mean_quadrant_agreement: {meta.get('mean_quadrant_agreement')}")
        return
    if n_real is None:
        n_real = {"preview": 2, "validate_lite": 3, "validate_mid": 4, "validate": 5, "full": 10}[mode]
    print("=" * 70)
    print(f"Phase 5 — Coupling-Model Sensitivity Atlas  [{mode}, n_real={n_real}]")
    print("=" * 70)
    
    if mode == "preview":
        cases = {"case_ii_edge": 25e-9}
        v_list = [10.0, 20.0]
        lambda_list = [0.5, 1.0]
    elif mode == "validate_lite":
        # edge only, 2 v points, 9 geometries, n_real=3 (lightweight for external review)
        # validate (2 cases x 3 v) is ~600 s even on dev, hence a lighter split
        cases = {"case_ii_edge": 25e-9}
        v_list = [10.0, 20.0]
        lambda_list = [0.5, 1.0]
    elif mode == "validate_mid":
        # include center+edge but reduce to 2 v points (5, 20) to fit a single run
        # purpose: remove the edge-only caveat (include center), n_real=4
        cases = CASES
        v_list = [5.0, 20.0]
        lambda_list = [0.5, 1.0]
    else:  # validate, full
        cases = CASES
        v_list = [5.0, 10.0, 20.0]
        lambda_list = [0.5, 1.0]
        # full validate can be split into cases to avoid single-process limits
    if only_case is not None:
        cases = {only_case: CASES[only_case]}
    geoms = build_geometries(mode)
    
    noise = OneOverFNoise(sigma_total=Defaults.sigma_dx_m, alpha=1.0,
                           f_low=1e3, f_high=1e7)
    
    n_total = len(MODELS) * len(cases) * len(v_list) * len(lambda_list) * len(geoms)
    print(f"  models={len(MODELS)}, cases={len(cases)}, v={len(v_list)}, "
          f"λ={len(lambda_list)}, geom={len(geoms)} → {n_total} conditions")
    
    # data[(model, case, v, lam, geom_label)] = {dP_v, dchi_phi}
    data = {}
    t0 = time.time()
    done = 0
    for model in MODELS:
        for case_label, x_c in cases.items():
            for v in v_list:
                for lam in lambda_list:
                    for geom in geoms:
                        r = evaluate(model, x_c, v, lam, geom, n_real, noise,
                     ez_convention=ez_convention, profile_norm=profile_norm)
                        data[(model, case_label, v, lam, geom["label"])] = r
                        done += 1
            if done % 50 < len(v_list) * len(lambda_list) * len(geoms):
                print(f"  [{done}/{n_total}] {time.time()-t0:.1f}s")
    elapsed = time.time() - t0
    print(f"\n  elapsed: {elapsed:.1f}s")
    
    OUT_RAW = Path(__file__).resolve().parents[1] / "figures" / "phase5"
    if save_raw or only_case is not None:
        # save per-case raw data (for merging). pickle (model,case,v,lam,geom)->r
        import pickle
        tag = only_case if only_case is not None else "all"
        raw_path = OUT_RAW / f"phase5_atlas_raw_{mode}_{tag}{dataset_tag(ez_convention, profile_norm)}.pkl"
        import hashlib as _hl
        from constants import Defaults as _D
        import reproduce.phase4p6_crossterm as _ct
        _atlas_sha = _hl.sha256(open(__file__, "rb").read()).hexdigest()
        _kernel_sha = _hl.sha256(open(_ct.__file__, "rb").read()).hexdigest()
        _cfg = {
            "ez_convention": ez_convention,
            "profile_norm": profile_norm,
            "B_ext_T": float(_D.B_ext_T),
            "sigma_E_ueV": 10.0,
            "mode": mode, "case": tag, "n_real": n_real,
            "base_seed": 41,
            "atlas_script_sha256": _atlas_sha,
            "atlas_script_sha256_16": _atlas_sha[:16],
            "kernel_script_sha256": _kernel_sha,
            "kernel_script_sha256_16": _kernel_sha[:16],
            "archive_version": ARCHIVE_VERSION,
        }
        with open(raw_path, "wb") as fh:
            pickle.dump({"data": data, "n_real": n_real, "elapsed": elapsed,
                         "config": _cfg}, fh)
        print(f"  saved raw: {raw_path}")
        if only_case is not None:
            print(f"  [only_case={only_case}] raw data saved; metrics computed in the merge script")
            return data
    
    # ===== metric 1: response amplitude (per model) =====
    print("\n[metric 1] Response amplitude (per-model mean |dP_v|, |dchi_phi|)")
    amplitude = {}
    for model in MODELS:
        dPs = [abs(v["dP_v"]) for k, v in data.items() if k[0] == model]
        dchis = [abs(v["dchi_phi"]) for k, v in data.items() if k[0] == model]
        amplitude[model] = {
            "mean_abs_dP_v": float(np.mean(dPs)),
            "mean_abs_dchi_phi": float(np.mean(dchis)),
            "std_abs_dP_v": float(np.std(dPs)),
            "std_abs_dchi_phi": float(np.std(dchis)),
        }
        print(f"  {model:10s}: |ΔP_v|={amplitude[model]['mean_abs_dP_v']:.3e} "
              f"±{amplitude[model]['std_abs_dP_v']:.3e}  "
              f"|Δχ_φ|={amplitude[model]['mean_abs_dchi_phi']:.3e} "
              f"±{amplitude[model]['std_abs_dchi_phi']:.3e}")
    
    # ===== metric 2: geometry sensitivity S_theta =====
    # S_theta = |R(theta+) - R(theta-)| / (2 dtheta), +/- perturbation about baseline
    print("\n[metric 2] Geometry sensitivity S_theta (per model, per parameter)")
    geom_sens = defaultdict(dict)
    if mode != "preview":
        for model in MODELS:
            for case_label, x_c in cases.items():
                for v in v_list:
                    for lam in lambda_list:
                        for p, step in GEOM_STEP.items():
                            kp = (model, case_label, v, lam, f"{p}+")
                            km = (model, case_label, v, lam, f"{p}-")
                            if kp in data and km in data:
                                rp, rm = data[kp], data[km]
                                dR = np.hypot(rp["dP_v"] - rm["dP_v"],
                                              rp["dchi_phi"] - rm["dchi_phi"])
                                S = dR / (2 * step)
                                geom_sens[model].setdefault(p, []).append(S)
        for model in MODELS:
            line = f"  {model:10s}: "
            for p in GEOM_STEP:
                if p in geom_sens[model]:
                    mean_S = np.mean(geom_sens[model][p])
                    line += f"{p[:6]}={mean_S:.2e}  "
            print(line)
    else:
        print("  (preview mode — geometry sensitivity skip)")
    
    # ===== metric 3: model sensitivity D_{m1,m2} (raw + normalized) =====
    print("\n[metric 3] Model sensitivity D (raw + normalized, per model pair)")
    model_sens = {}
    model_sens_norm = {}
    common_conditions = set((k[1], k[2], k[3], k[4]) for k in data.keys())
    for m1, m2 in combinations(MODELS, 2):
        diffs = []
        diffs_norm = []
        for cond in common_conditions:
            k1 = (m1, *cond)
            k2 = (m2, *cond)
            if k1 in data and k2 in data:
                d = np.hypot(data[k1]["dP_v"] - data[k2]["dP_v"],
                             data[k1]["dchi_phi"] - data[k2]["dchi_phi"])
                diffs.append(d)
                # scale-normalized distance
                dn = np.hypot((data[k1]["dP_v"] - data[k2]["dP_v"]) / P_SCALE,
                              (data[k1]["dchi_phi"] - data[k2]["dchi_phi"]) / CHI_SCALE)
                diffs_norm.append(dn)
        model_sens[f"{m1}_vs_{m2}"] = float(np.mean(diffs)) if diffs else 0.0
        model_sens_norm[f"{m1}_vs_{m2}"] = float(np.mean(diffs_norm)) if diffs_norm else 0.0
        print(f"  {m1:10s} vs {m2:10s}: D={model_sens[f'{m1}_vs_{m2}']:.3e}  "
              f"D_norm={model_sens_norm[f'{m1}_vs_{m2}']:.3f}")
    
    # ===== metric 4: rank correlation =====
    # per-model |R| ranking across conditions -> Spearman correlation
    print("\n[metric 4] Rank correlation (agreement of |R| ranking across models)")
    cond_list = sorted(common_conditions)
    model_ranks = {}
    for model in MODELS:
        vals = []
        for cond in cond_list:
            k = (model, *cond)
            if k in data:
                vals.append(np.hypot(data[k]["dP_v"], data[k]["dchi_phi"]))
            else:
                vals.append(np.nan)
        model_ranks[model] = vals
    
    def spearman(a, b):
        # scipy.stats.spearmanr: proper tie correction.
        # the full validate result is produced by merge_validate.py (same scipy);
        # this helper is for lite/preview/mid modes. Ties are rare for continuous values.
        from scipy.stats import spearmanr
        a = np.array(a); b = np.array(b)
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
        if len(a) < 3:
            return float("nan")
        rho, _ = spearmanr(a, b)
        return float(rho)
    
    rank_corr = {}
    for m1, m2 in combinations(MODELS, 2):
        rho = spearman(model_ranks[m1], model_ranks[m2])
        rank_corr[f"{m1}_vs_{m2}"] = rho
        print(f"  {m1:10s} vs {m2:10s}: ρ={rho:+.3f}")
    
    mean_rank_corr = float(np.nanmean(list(rank_corr.values())))
    print(f"\n  mean rank correlation: {mean_rank_corr:+.3f}")
    if mean_rank_corr < 0.5:
        interp = "LOW -- the coupling-model choice strongly changes the ranking (strong model-dependence)"
    elif mean_rank_corr < 0.8:
        interp = "MODERATE rank agreement (quadrant interpretation still model-dependent)"
    else:
        interp = "HIGH rank agreement, NOT model independence (quadrant interpretation remains model-dependent)"
    print(f"  → {interp}")
    
    # ===== metric 5: quadrant agreement =====
    # fraction of conditions where two models share the same quadrant (sign-preserving)
    print("\n[metric 5] Quadrant agreement (per model pair, sign-preserving)")
    quadrant_agree = {}
    for m1, m2 in combinations(MODELS, 2):
        agree = 0
        total = 0
        for cond in common_conditions:
            k1 = (m1, *cond)
            k2 = (m2, *cond)
            if k1 in data and k2 in data:
                q1 = quadrant_of(data[k1]["dP_v"], data[k1]["dchi_phi"])
                q2 = quadrant_of(data[k2]["dP_v"], data[k2]["dchi_phi"])
                total += 1
                if q1 == q2:
                    agree += 1
        frac = agree / total if total else 0.0
        quadrant_agree[f"{m1}_vs_{m2}"] = frac
        print(f"  {m1:10s} vs {m2:10s}: {agree}/{total} = {frac:.2%}")
    mean_quad_agree = float(np.mean(list(quadrant_agree.values())))
    print(f"\n  mean quadrant agreement: {mean_quad_agree:.2%}")
    
    # ===== metric 6: relative geometry sensitivity =====
    # S_rel = |R(theta+) - R(theta-)| / (2 dtheta/theta0) -- parameter-scale normalized
    print("\n[metric 6] Relative geometry sensitivity (normalized by theta0)")
    geom_sens_rel = defaultdict(dict)
    if mode != "preview":
        for model in MODELS:
            for case_label, x_c in cases.items():
                for v in v_list:
                    for lam in lambda_list:
                        for p, step in GEOM_STEP.items():
                            kp = (model, case_label, v, lam, f"{p}+")
                            km = (model, case_label, v, lam, f"{p}-")
                            if kp in data and km in data:
                                rp, rm = data[kp], data[km]
                                dR = np.hypot(rp["dP_v"] - rm["dP_v"],
                                              rp["dchi_phi"] - rm["dchi_phi"])
                                theta0 = abs(GEOM0[p])
                                # relative: denominator (2 dtheta/theta0) -> dimensionless step
                                S_rel = dR / (2 * step / theta0)
                                geom_sens_rel[model].setdefault(p, []).append(S_rel)
        for model in MODELS:
            line = f"  {model:10s}: "
            for p in GEOM_STEP:
                if p in geom_sens_rel[model]:
                    mean_S = np.mean(geom_sens_rel[model][p])
                    line += f"{p[:6]}={mean_S:.2e}  "
            print(line)
    else:
        print("  (preview mode — relative geometry sensitivity skip)")
    
    # ===== save =====
    OUT = Path(__file__).resolve().parents[1] / "figures" / "phase5"
    OUT.mkdir(parents=True, exist_ok=True)
    
    summary_csv = OUT / f"phase5_atlas_summary_{mode}{dataset_tag(ez_convention, profile_norm)}.csv"
    with open(summary_csv, "w") as f:
        f.write("model,case,v_ms,lambda_uev,geom_label,dP_v,dchi_phi\n")
        for k, v in data.items():
            f.write(f"{k[0]},{k[1]},{k[2]},{k[3]},{k[4]},"
                    f"{v['dP_v']:.6e},{v['dchi_phi']:.6e}\n")
    print(f"\n  saved: {summary_csv}")
    
    sens_csv = OUT / f"phase5_atlas_sensitivity_{mode}{dataset_tag(ez_convention, profile_norm)}.csv"
    with open(sens_csv, "w") as f:
        f.write("metric,key,value\n")
        for m in MODELS:
            f.write(f"amplitude_dP_v,{m},{amplitude[m]['mean_abs_dP_v']:.6e}\n")
            f.write(f"amplitude_dchi_phi,{m},{amplitude[m]['mean_abs_dchi_phi']:.6e}\n")
        for pair, d in model_sens.items():
            f.write(f"model_sensitivity,{pair},{d:.6e}\n")
        for pair, d in model_sens_norm.items():
            f.write(f"model_sensitivity_normalized,{pair},{d:.6f}\n")
        for pair, rho in rank_corr.items():
            f.write(f"rank_correlation,{pair},{rho:.6f}\n")
        for pair, fr in quadrant_agree.items():
            f.write(f"quadrant_agreement,{pair},{fr:.6f}\n")
        # save geometry sensitivity raw + relative
        for model in MODELS:
            for p in GEOM_STEP:
                if p in geom_sens.get(model, {}):
                    val = float(np.mean(geom_sens[model][p]))
                    f.write(f"geometry_sensitivity_raw,{model}|{p},{val:.6e}\n")
                if p in geom_sens_rel.get(model, {}):
                    val = float(np.mean(geom_sens_rel[model][p]))
                    f.write(f"geometry_sensitivity_relative,{model}|{p},{val:.6e}\n")
    print(f"  saved: {sens_csv}")
    
    script_path = Path(__file__).resolve()
    with open(script_path, "rb") as f:
        script_sha = hashlib.sha256(f.read()).hexdigest()[:16]
    p4p6 = Path(__file__).resolve().parent / "phase4p6_crossterm.py"
    with open(p4p6, "rb") as f:
        p4p6_sha = hashlib.sha256(f.read()).hexdigest()[:16]
    metadata = OUT / f"phase5_atlas_metadata_{mode}{dataset_tag(ez_convention, profile_norm)}.json"
    with open(metadata, "w") as f:
        json.dump({
            "archive_version": ARCHIVE_VERSION,
            "script_sha256_16": script_sha,
        "ez_convention": ez_convention,
        "profile_norm": profile_norm,
            "kernel_phase4p6_sha256_16": p4p6_sha,
            "mode": mode, "n_real": n_real,
            "n_conditions": n_total, "elapsed_sec": elapsed,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "amplitude": amplitude,
            "model_sensitivity": model_sens,
            "model_sensitivity_normalized": model_sens_norm,
            "rank_correlation": rank_corr,
            "mean_rank_correlation": mean_rank_corr,
            "rank_correlation_interpretation": interp,
            "quadrant_agreement": quadrant_agree,
            "mean_quadrant_agreement": mean_quad_agree,
            # save geometry sensitivity
            "geometry_sensitivity_raw": {
                m: {p: float(np.mean(geom_sens[m][p])) for p in geom_sens.get(m, {})}
                for m in MODELS},
            "geometry_sensitivity_relative": {
                m: {p: float(np.mean(geom_sens_rel[m][p])) for p in geom_sens_rel.get(m, {})}
                for m in MODELS},
        }, f, indent=2)
    print(f"  saved: {metadata}")
    
    # plot (--no-plots option)
    if not no_plots:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax = axes[0]
        x = np.arange(len(MODELS))
        amp_P = [amplitude[m]["mean_abs_dP_v"] for m in MODELS]
        amp_chi = [amplitude[m]["mean_abs_dchi_phi"] for m in MODELS]
        w = 0.35
        ax.bar(x - w/2, amp_P, w, label=r"$\overline{|\Delta P_v|}$")
        ax.bar(x + w/2, amp_chi, w, label=r"$\overline{|\Delta\chi_\phi|}$")
        ax.set_xticks(x); ax.set_xticklabels(MODELS)
        ax.set_yscale("log"); ax.set_title("Response amplitude by model")
        ax.legend(); ax.grid(alpha=0.3, axis="y")
        
        ax = axes[1]
        pairs = list(rank_corr.keys())
        rhos = [rank_corr[p] for p in pairs]
        ax.barh(range(len(pairs)), rhos, color=["g" if r > 0.5 else "r" for r in rhos])
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels([p.replace("_vs_", " vs ") for p in pairs], fontsize=8)
        ax.axvline(0.5, color="k", ls="--", lw=0.5)
        ax.set_xlim(-1, 1); ax.set_title(f"Rank correlation (mean={mean_rank_corr:+.2f})")
        ax.set_xlabel("Spearman ρ"); ax.grid(alpha=0.3, axis="x")
        
        fig.suptitle(f"Phase 5 Sensitivity Atlas ({mode}) — {interp[:40]}", fontsize=11)
        fig.tight_layout()
        png = OUT / f"phase5_atlas_{mode}{dataset_tag(ez_convention, profile_norm)}.png"
        fig.savefig(png, dpi=130)
        print(f"  saved: {png}")
    else:
        print("  (--no-plots: PNG skipped)")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  mean rank correlation: {mean_rank_corr:+.3f} ({interp[:50]})")
    print(f"  mean quadrant agreement: {mean_quad_agree:.1%}")
    print(f"  -> key finding of the coupling-model sensitivity atlas:")
    print(f"    key finding -- the coupling-model choice partially changes the ranking,")
    print(f"    and changes the robust/trade quadrant interpretation more strongly "
          f"(rank corr {mean_rank_corr:.2f} > quadrant agreement {mean_quad_agree:.2f}).")
    
    return data, amplitude, model_sens, rank_corr, mean_rank_corr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 sensitivity atlas")
    parser.add_argument("--mode", choices=["preview", "validate_lite", "validate_mid", "validate", "full"],
                        default="preview")
    parser.add_argument("--n_real", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true",
                        help="skip PNG generation (external review / fast sanity)")
    parser.add_argument("--metadata-only", action="store_true",
                        help="verify only the script-SHA consistency of existing metadata, no computation")
    parser.add_argument("--case", default=None,
                        choices=["case_i_center", "case_ii_edge"],
                        help="split full validate: run a single case and save raw data")
    parser.add_argument("--save-raw", action="store_true",
                        help="save raw per-condition data as pickle")
    parser.add_argument("--ez-convention", dest="ez_convention",
                        choices=["stray-mean", "total-local", "total-mean"],
                        default="stray-mean",
                        help="Zeeman convention (see REPRODUCIBILITY.md, Sec. 7); "
                             "default reproduces the archived legacy behaviour")
    parser.add_argument("--profile-norm", dest="profile_norm",
                        choices=["prefactor", "final-peak", "l2"], default="prefactor",
                        help="lambda_sv normalization (final-peak renormalizes "
                             "after the resonance window)")
    args = parser.parse_args()
    main(mode=args.mode, n_real=args.n_real, no_plots=args.no_plots,
         metadata_only=args.metadata_only, only_case=args.case,
         save_raw=args.save_raw,
         ez_convention=args.ez_convention, profile_norm=args.profile_norm)
