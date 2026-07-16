"""
Full validate merge -- combine the two per-case raw pkls (center, edge) and compute the six metrics.

Full validate (432 conditions) exceeds single-process limits, so it is run
per case (216 conditions each, ~222 s) and the raw pickles are merged here.

Input:  figures/phase5/phase5_atlas_raw_validate_case_{i_center,ii_edge}.pkl
Output: figures/phase5/phase5_atlas_{summary,sensitivity}_validate.csv
      figures/phase5/phase5_atlas_metadata_validate.json

Usage:
  PYTHONPATH=. python reproduce/phase5_atlas_merge_validate.py
"""

from __future__ import annotations
import pickle, json, hashlib
from pathlib import Path
from itertools import combinations
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr

# reuse constants/classification from the atlas module (single source of truth)
from reproduce.phase5_sensitivity_atlas import (
    MODELS, GEOM0, GEOM_STEP, P_SCALE, CHI_SCALE,
    P_MIN_THRESH, CHI_MIN_THRESH, quadrant_of, ARCHIVE_VERSION,
)

FIG = Path(__file__).resolve().parents[1] / "figures" / "phase5"


REQUIRED_CONFIG_KEYS = ["ez_convention", "profile_norm", "B_ext_T", "sigma_E_ueV",
                        "mode", "n_real", "atlas_script_sha256_16",
                        "kernel_script_sha256_16", "archive_version"]


def validate_raw_configs(cfgs, expect_ez, expect_norm, allow_legacy=False):
    """Cross-check per-case raw configs before merging (audit r5 par.4).

    cfgs: {case_tag: config-dict-or-None}. Raises ValueError on any
    inconsistency; returns the common config (or the string
    'legacy-unrecorded' when legacy raws are explicitly allowed)."""
    missing = [t for t, c in cfgs.items() if c is None]
    if missing:
        if not allow_legacy:
            raise ValueError(
                f"raw pkl(s) {missing} carry no config block (legacy archive); "
                "re-run the atlas, or pass --allow-legacy to merge them as "
                "'legacy-unrecorded' provenance")
        if expect_ez != "stray-mean" or expect_norm != "prefactor":
            raise ValueError("legacy raws can only be merged under the legacy "
                             "convention (stray-mean / prefactor)")
        return "legacy-unrecorded"
    vals = list(cfgs.values())
    for key in REQUIRED_CONFIG_KEYS:
        got = {t: c.get(key) for t, c in cfgs.items()}
        if len(set(got.values())) != 1:
            raise ValueError(f"raw config mismatch across cases for '{key}': {got}")
    if vals[0]["ez_convention"] != expect_ez or vals[0]["profile_norm"] != expect_norm:
        raise ValueError(
            f"raw config ({vals[0]['ez_convention']}/{vals[0]['profile_norm']}) does not "
            f"match the requested dataset ({expect_ez}/{expect_norm})")
    return vals[0]


def main(MODE="validate", ez_convention="stray-mean", profile_norm="prefactor",
         allow_legacy=False):
    from reproduce.phase5_sensitivity_atlas import dataset_tag
    dtag = dataset_tag(ez_convention, profile_norm)
    # ---- merge raw ----
    data = {}
    n_real = None
    total_elapsed = 0.0
    raw_cfgs = {}
    for tag in ["case_i_center", "case_ii_edge"]:
        p = FIG / f"phase5_atlas_raw_{MODE}_{tag}{dtag}.pkl"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing -- run --case {tag} first")
        d = pickle.load(open(p, "rb"))
        data.update(d["data"])
        n_real = d["n_real"]
        total_elapsed += d["elapsed"]
        raw_cfgs[tag] = d.get("config")
    raw_config = validate_raw_configs(raw_cfgs, ez_convention, profile_norm,
                                      allow_legacy=allow_legacy)
    print(f"merged: {len(data)} conditions (n_real={n_real}, cumulative {total_elapsed:.0f}s)")

    common = set((k[1], k[2], k[3], k[4]) for k in data.keys())

    # ---- metric 1: amplitude ----
    amplitude = {}
    for m in MODELS:
        dPs = [abs(data[k]["dP_v"]) for k in data if k[0] == m]
        dcs = [abs(data[k]["dchi_phi"]) for k in data if k[0] == m]
        amplitude[m] = {"mean_abs_dP_v": float(np.mean(dPs)),
                        "mean_abs_dchi_phi": float(np.mean(dcs))}

    # ---- metric 2/6: geometry sensitivity raw + relative ----
    geom_sens = defaultdict(dict)
    geom_sens_rel = defaultdict(dict)
    for m in MODELS:
        for p_name, step in GEOM_STEP.items():
            for cond in common:
                kp = (m, cond[0], cond[1], cond[2], f"{p_name}+")
                km = (m, cond[0], cond[1], cond[2], f"{p_name}-")
                if kp in data and km in data:
                    dR = np.hypot(data[kp]["dP_v"] - data[km]["dP_v"],
                                  data[kp]["dchi_phi"] - data[km]["dchi_phi"])
                    S = dR / (2 * step)
                    geom_sens[m].setdefault(p_name, []).append(S)
                    theta0 = abs(GEOM0[p_name])
                    geom_sens_rel[m].setdefault(p_name, []).append(dR / (2 * step / theta0))

    # ---- metric 3: model distance raw + normalized ----
    model_sens, model_sens_norm = {}, {}
    for m1, m2 in combinations(MODELS, 2):
        diffs, diffs_n = [], []
        for cond in common:
            k1, k2 = (m1, *cond), (m2, *cond)
            if k1 in data and k2 in data:
                diffs.append(np.hypot(data[k1]["dP_v"] - data[k2]["dP_v"],
                                      data[k1]["dchi_phi"] - data[k2]["dchi_phi"]))
                diffs_n.append(np.hypot((data[k1]["dP_v"] - data[k2]["dP_v"]) / P_SCALE,
                                        (data[k1]["dchi_phi"] - data[k2]["dchi_phi"]) / CHI_SCALE))
        model_sens[f"{m1}_vs_{m2}"] = float(np.mean(diffs))
        model_sens_norm[f"{m1}_vs_{m2}"] = float(np.mean(diffs_n))

    # ---- metric 4: rank correlation ----
    rank_corr = {}
    sorted_conds = sorted(common)
    for m1, m2 in combinations(MODELS, 2):
        r1, r2 = [], []
        for cond in sorted_conds:
            k1, k2 = (m1, *cond), (m2, *cond)
            if k1 in data and k2 in data:
                r1.append(np.hypot(data[k1]["dP_v"], data[k1]["dchi_phi"]))
                r2.append(np.hypot(data[k2]["dP_v"], data[k2]["dchi_phi"]))
        rho, _ = spearmanr(r1, r2)
        rank_corr[f"{m1}_vs_{m2}"] = float(rho)
    mean_rank_corr = float(np.nanmean(list(rank_corr.values())))

    # ---- metric 5: quadrant agreement ----
    quad_agree = {}
    for m1, m2 in combinations(MODELS, 2):
        agree = total = 0
        for cond in common:
            k1, k2 = (m1, *cond), (m2, *cond)
            if k1 in data and k2 in data:
                q1 = quadrant_of(data[k1]["dP_v"], data[k1]["dchi_phi"])
                q2 = quadrant_of(data[k2]["dP_v"], data[k2]["dchi_phi"])
                total += 1
                agree += (q1 == q2)
        quad_agree[f"{m1}_vs_{m2}"] = agree / total if total else 0.0
    mean_quad_agree = float(np.mean(list(quad_agree.values())))

    if mean_rank_corr < 0.4:
        interp = "LOW -- strong model-dependence"
    elif mean_rank_corr < 0.8:
        interp = "MODERATE rank agreement (quadrant interpretation still model-dependent)"
    else:
        interp = "HIGH rank agreement, NOT model independence (quadrant interpretation remains model-dependent)"

    # ---- save ----
    summary_csv = FIG / f"phase5_atlas_summary_{MODE}{dtag}.csv"
    with open(summary_csv, "w") as f:
        f.write("model,case,v_ms,lambda_uev,geom_label,dP_v,dchi_phi\n")
        for k in sorted(data.keys()):
            r = data[k]
            f.write(f"{k[0]},{k[1]},{k[2]},{k[3]},{k[4]},{r['dP_v']:.6e},{r['dchi_phi']:.6e}\n")

    sens_csv = FIG / f"phase5_atlas_sensitivity_{MODE}{dtag}.csv"
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
        for pair, fr in quad_agree.items():
            f.write(f"quadrant_agreement,{pair},{fr:.6f}\n")
        for m in MODELS:
            for p_name in GEOM_STEP:
                if p_name in geom_sens.get(m, {}):
                    f.write(f"geometry_sensitivity_raw,{m}|{p_name},{np.mean(geom_sens[m][p_name]):.6e}\n")
                if p_name in geom_sens_rel.get(m, {}):
                    f.write(f"geometry_sensitivity_relative,{m}|{p_name},{np.mean(geom_sens_rel[m][p_name]):.6e}\n")

    # script SHA (record both atlas and merge)
    if isinstance(raw_config, dict):
        atlas_sha = raw_config["atlas_script_sha256_16"]
    else:
        atlas_sha = hashlib.sha256(
            open(Path(__file__).resolve().parents[0] / "phase5_sensitivity_atlas.py", "rb").read()
        ).hexdigest()[:16]
    merge_sha = hashlib.sha256(open(Path(__file__).resolve(), "rb").read()).hexdigest()[:16]
    meta = {
        "archive_version": ARCHIVE_VERSION,
        "ez_convention": ez_convention,
        "profile_norm": profile_norm,
        "raw_config": raw_config,
        "atlas_sha_source": ("raw config" if isinstance(raw_config, dict)
                             else "current script on disk (legacy raw; "
                                  "generation SHA unrecorded)"),
        "mode": MODE,
        "n_real": n_real,
        "n_conditions": len(data),
        "atlas_script_sha256_16": atlas_sha,
        "merge_script_sha256_16": merge_sha,
        "mean_rank_correlation": mean_rank_corr,
        "rank_correlation_interpretation": interp,
        "mean_quadrant_agreement": mean_quad_agree,
        "rank_correlation": rank_corr,
        "quadrant_agreement": quad_agree,
        "model_sensitivity": model_sens,
        "model_sensitivity_normalized": model_sens_norm,
        "geometry_sensitivity_raw": {
            m: {p: float(np.mean(geom_sens[m][p])) for p in geom_sens.get(m, {})} for m in MODELS},
        "geometry_sensitivity_relative": {
            m: {p: float(np.mean(geom_sens_rel[m][p])) for p in geom_sens_rel.get(m, {})} for m in MODELS},
        "merged_from": ["case_i_center", "case_ii_edge"],
        "total_compute_seconds": round(total_elapsed, 1),
        "note": "full validate via case-split (avoids single-process limits)",
    }
    meta_path = FIG / f"phase5_atlas_metadata_{MODE}{dtag}.json"
    json.dump(meta, open(meta_path, "w"), indent=2, ensure_ascii=False)

    print(f"\n=== full validate (merged) ===")
    print(f"  mean rank correlation: {mean_rank_corr:+.3f} ({interp})")
    print(f"  mean quadrant agreement: {mean_quad_agree:.1%}")
    print(f"  saved: {summary_csv.name}, {sens_csv.name}, {meta_path.name}")
    print("\n  pairwise:")
    for pair in rank_corr:
        print(f"    {pair:20s} ρ={rank_corr[pair]:+.3f}  quad={quad_agree[pair]:.1%}")
    return meta


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="merge case-split atlas")
    parser.add_argument("--mode", default="validate",
                        choices=["validate", "validate_mid", "preview"])  # preview: smoke-chain e2e only
    parser.add_argument("--ez-convention", dest="ez_convention",
                        choices=["stray-mean", "total-local", "total-mean"],
                        default="stray-mean")
    parser.add_argument("--profile-norm", dest="profile_norm",
                        choices=["prefactor", "final-peak", "l2"],
                        default="prefactor")
    parser.add_argument("--allow-legacy", action="store_true",
                        help="permit merging archived raws that predate config recording")
    args = parser.parse_args()
    main(MODE=args.mode, ez_convention=args.ez_convention,
         profile_norm=args.profile_norm, allow_legacy=args.allow_legacy)
