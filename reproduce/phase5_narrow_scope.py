"""
Phase 5 -- narrow-scope pre-scan driver.

Implements the narrow-scope pre-scan protocol:
- baseline: edge ($x_c = +25$ nm), v=20 m/s, λ=1.0 μeV, period=150, depth=-50,
  half_x=25, half_z=15 nm
- geometry 1D slice: 3 points for each of 4 parameters
- trajectory scan: 5 velocities (15, 17.5, 20, 22.5, 25 m/s)
- A AND A_pocket cross-validation with effect-size thresholds
- automatic basin_size / basin_score computation
- automatic strong / weak / isolated classification

Mode:
  --mode preview: baseline + 4 parameters x (+/-1 step) only, v=20 fixed (~30 s)
  --mode full:    9 unique geometry × 5 v × 2 model (~3-5 min in dev env,
                  may take longer in other environments)

Outputs:
  figures/phase5/phase5_narrow_scope_summary_{mode}.csv
  figures/phase5/phase5_narrow_scope_metadata_{mode}.json
  figures/phase5/phase5_narrow_scope_{mode}.png
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
from pathlib import Path

import jax.numpy as jnp

from constants import hbar, Defaults
from noise.charge_noise import OneOverFNoise
from geometry.periodic_array import PeriodicPrismArray
from geometry.fourier_field import fit_from_prism_array
from reproduce.phase4p6_crossterm import run_one_condition

# baseline (plan §1)
BASELINE = {
    "x_c_nm": 25.0,     # edge pocket
    "v": 20.0,
    "lam_uev": 1.0,
    "period_nm": 150.0,
    "depth_nm": -50.0,
    "half_x_nm": 25.0,
    "half_y_nm": 25.0,
    "half_z_nm": 15.0,
    "cz_nm": 15.0,
}

# effect-size thresholds (same as the objective-analysis stage)
P_MIN_THRESH = 1e-4
CHI_MIN_THRESH = 1e-3


def make_ff(period_nm, depth_nm, half_x_nm, half_z_nm):
    """Build a FourierField for the given geometry."""
    arr = PeriodicPrismArray(
        period_a=period_nm * 1e-9,
        half_x=half_x_nm * 1e-9,
        half_y=BASELINE["half_y_nm"] * 1e-9,
        half_z=half_z_nm * 1e-9,
        cz=BASELINE["cz_nm"] * 1e-9,
        N_periods_each_side=4,
        Ms=1.4e6,
    )
    return fit_from_prism_array(arr, depth_nm * 1e-9, N_harm=3)


def evaluate_one_geom(geom_overrides, v, n_real, noise, base_seed):
    """Evaluate both A and A_pocket at one (geometry, v).
    
    return: {"A": dict, "A_pocket": dict} with delta_P_v, delta_chi_phi, etc.
    """
    g = dict(BASELINE)
    g.update(geom_overrides)
    
    ff = make_ff(g["period_nm"], g["depth_nm"], g["half_x_nm"], g["half_z_nm"])
    
    e_C = 1.602176634e-19
    Ev_baseline = 100e-6 * e_C
    Ev_min = 5e-6 * e_C
    pocket_width = 30e-9
    Delta_v = 0.5e-6 * e_C
    lambda_0 = g["lam_uev"] * 1e-6 * e_C
    x_c = g["x_c_nm"] * 1e-9
    
    T_traj_for_noise = (8 * pocket_width) / v
    
    out = {}
    for cm in ["A", "A_pocket"]:
        R = run_one_condition(
            v=v, case_label=f"narrow_{geom_overrides.get('label','')}",
            pocket_x_center=x_c,
            lambda_0=lambda_0, coupling_model=cm,
            n_real=n_real, noise=noise, ff=ff,
            Ev_baseline=Ev_baseline, Ev_min=Ev_min,
            pocket_width=pocket_width, Delta_v=Delta_v,
            N_max=500, base_seed=base_seed,
            T_traj_for_noise=T_traj_for_noise,
        )
        dP = R["M2"]["P_v_dia"]["mean"] - R["M1V"]["P_v_dia"]["mean"]
        dchi = R["M2"]["phase"]["var_circular"] - R["M1"]["phase"]["var_circular"]
        # SEM estimate: std of M2 divided by sqrt(n_real) approximates the Delta P_v SEM
        # (M1V is deterministic and has no sigma; only the M2 sample variance is used)
        P_sem = R["M2"]["P_v_dia"]["std"] / max(np.sqrt(n_real), 1.0)
        # Delta chi_phi is an ensemble observable (single value) -- no SEM estimate applied.
        out[cm] = {
            "delta_P_v": dP,
            "delta_chi_phi": dchi,
            "S_M2": R["M2"]["S_s"]["mean"],
            "P_v_M2_std": R["M2"]["P_v_dia"]["std"],
            "delta_P_v_sem": P_sem,
        }
    return out


def is_robust(r, P_min=P_MIN_THRESH, chi_min=CHI_MIN_THRESH, sem_aware=False):
    """Threshold passed AND robust quadrant.
    
    sem_aware=True: conservative SEM-aware criterion.
      Requires Delta P_v + 2*SEM < -P_min (signal sufficiently above noise).
    sem_aware=False: mean values only.
    """
    if sem_aware and "delta_P_v_sem" in r:
        # P_v is SEM-aware; Delta chi uses the mean only (previous definition kept).
        sem = r.get("delta_P_v_sem", 0.0)
        return (r["delta_P_v"] + 2 * sem < -P_min and
                abs(r["delta_chi_phi"]) >= chi_min and
                r["delta_chi_phi"] < 0)
    return (abs(r["delta_P_v"]) >= P_min and
            abs(r["delta_chi_phi"]) >= chi_min and
            r["delta_P_v"] < 0 and r["delta_chi_phi"] < 0)


def cross_robust(out, sem_aware=False):
    """Robust in both A AND A_pocket?"""
    return is_robust(out["A"], sem_aware=sem_aware) and is_robust(out["A_pocket"], sem_aware=sem_aware)


def build_scan_grid(mode):
    """Build the grid over unique geometries.
    
    The baseline is evaluated once; each parameter adds 2 points (+/-1 step).
    """
    if mode == "preview":
        # baseline + 4 parameters (+/-1 step) only, v=20 fixed
        geoms = [{"label": "baseline"}]
        for key, vals in [
            ("period_nm", [120, 180]),
            ("depth_nm", [-40, -60]),
            ("half_x_nm", [20, 30]),
            ("half_z_nm", [12, 18]),
        ]:
            for v_ in vals:
                geoms.append({"label": f"{key}={v_}", key: v_})
        v_list = [20.0]
    elif mode == "targeted":
        # Targeted stage: re-test full-mode candidates with enlarged n_real.
        # The full-mode CSV yields 2 naive cross-robust conditions; targeted picks
        # 3 nearby geometries (baseline / depth=-40 / half_x=30) and re-tests them
        # with strengthened statistics (geometry count and candidate count differ).
        geoms = [
            {"label": "baseline"},
            {"label": "depth_nm=-40", "depth_nm": -40},
            {"label": "half_x_nm=30", "half_x_nm": 30},
        ]
        v_list = [20.0, 25.0]   # baseline v(20) and v=25 where half_x=30 passed
    else:  # full
        geoms = [{"label": "baseline"}]
        for key, vals in [
            ("period_nm", [120, 180]),
            ("depth_nm", [-40, -60]),
            ("half_x_nm", [20, 30]),
            ("half_z_nm", [12, 18]),
        ]:
            for v_ in vals:
                geoms.append({"label": f"{key}={v_}", key: v_})
        v_list = [15.0, 17.5, 20.0, 22.5, 25.0]
    return geoms, v_list


def main(mode="preview", n_real=2):
    print("="*70)
    print(f"Phase 5 narrow-scope pre-scan  [mode={mode}, n_real={n_real}]")
    print(f"baseline: {BASELINE}")
    print("="*70)
    
    noise = OneOverFNoise(
        sigma_total=Defaults.sigma_dx_m, alpha=1.0,
        f_low=1e3, f_high=1e7,
    )
    
    geoms, v_list = build_scan_grid(mode)
    n_total = len(geoms) * len(v_list)
    print(f"  unique geometry: {len(geoms)}, v points: {len(v_list)}")
    print(f"  total evaluations: {n_total}")
    
    t0 = time.time()
    results = []   # list of (geom_label, geom_dict, v, A_dict, Apocket_dict)
    done = 0
    for geom_overrides in geoms:
        for v in v_list:
            out = evaluate_one_geom(geom_overrides, v, n_real, noise, base_seed=31)
            results.append({
                "geom_label": geom_overrides.get("label", "baseline"),
                "geom_overrides": {k: v_ for k, v_ in geom_overrides.items() if k != "label"},
                "v": v,
                "A": out["A"],
                "A_pocket": out["A_pocket"],
                "cross_robust": cross_robust(out),
                "cross_robust_sem": cross_robust(out, sem_aware=True),
            })
            done += 1
            cr = "✓" if results[-1]["cross_robust"] else "·"
            print(f"  [{done}/{n_total}] {geom_overrides.get('label','baseline'):20s} v={v:5.1f}  "
                  f"A: dP={out['A']['delta_P_v']:+.2e} dχ={out['A']['delta_chi_phi']:+.2e}  "
                  f"Ap: dP={out['A_pocket']['delta_P_v']:+.2e} dχ={out['A_pocket']['delta_chi_phi']:+.2e}  "
                  f"cross_robust={cr}")
    elapsed = time.time() - t0
    print(f"\n  elapsed: {elapsed:.1f}s")
    
    # ============================================================
    # Basin analysis
    # ============================================================
    print("\n" + "="*70)
    print("Basin analysis")
    print("="*70)
    
    # baseline (v=20) cross_robust?
    baseline_v20 = [r for r in results if r["geom_label"] == "baseline" and r["v"] == 20.0]
    baseline_cr = baseline_v20[0]["cross_robust"] if baseline_v20 else False
    print(f"  baseline (v=20) cross_robust: {baseline_cr}")
    
    # number of cross-robust neighbors excluding the baseline
    cr_neighbors_geom = sum(1 for r in results 
                              if r["geom_label"] != "baseline" and r["v"] == 20.0 and r["cross_robust"])
    total_geom_neighbors = sum(1 for r in results
                                 if r["geom_label"] != "baseline" and r["v"] == 20.0)
    cr_neighbors_v = sum(1 for r in results
                          if r["geom_label"] == "baseline" and r["v"] != 20.0 and r["cross_robust"])
    total_v_neighbors = sum(1 for r in results
                              if r["geom_label"] == "baseline" and r["v"] != 20.0)
    
    basin_size = cr_neighbors_geom + cr_neighbors_v
    total_neighbors = total_geom_neighbors + total_v_neighbors
    basin_score = basin_size / max(total_neighbors, 1)
    print(f"  cross-robust neighbors:")
    print(f"    geometry (v=20 fixed):  {cr_neighbors_geom}/{total_geom_neighbors}")
    print(f"    velocity (baseline fixed): {cr_neighbors_v}/{total_v_neighbors}")
    print(f"  basin_size  = {basin_size}")
    print(f"  basin_score = {basin_score:.2f}  (total neighbors: {total_neighbors})")
    
    # SEM-aware basin (conservative)
    n_cr_sem = sum(1 for r in results if r["cross_robust_sem"])
    n_cr_naive = sum(1 for r in results if r["cross_robust"])
    baseline_cr_sem = baseline_v20[0]["cross_robust_sem"] if baseline_v20 else False
    print(f"\n  SEM-aware criterion (Delta P_v + 2*SEM < -P_min):")
    print(f"    cross_robust (naive, mean only): {n_cr_naive}/{len(results)}")
    print(f"    cross_robust_sem (conservative):  {n_cr_sem}/{len(results)}")
    print(f"    baseline cross_robust_sem:        {baseline_cr_sem}")
    
    # per-parameter robust persistence (strong/weak/isolated)
    param_cr_count = {}
    for key in ["period_nm", "depth_nm", "half_x_nm", "half_z_nm"]:
        param_cr_count[key] = sum(1 for r in results
                                    if key in r["geom_overrides"] and r["v"] == 20.0 
                                    and r["cross_robust"])
    print("\n  parameter-wise cross-robust count (each has 2 neighbors at v=20):")
    for key, c in param_cr_count.items():
        print(f"    {key}: {c}/2")
    
    # verdict: baseline failure takes precedence; 4-category classification
    params_with_cr = sum(1 for c in param_cr_count.values() if c >= 1)
    params_with_2cr = sum(1 for c in param_cr_count.values() if c == 2)
    n_cr_total = sum(1 for r in results if r["cross_robust"])
    
    # 4-category classification:
    #   strong basin              : baseline robust + several robust neighbors
    #   weak basin                : baseline robust + neighbor 1-2
    #   isolated robust point     : baseline robust + neighbor 0
    #   off_baseline_sparse_hits  : baseline NOT robust + some robust neighbors
    #   no basin                  : baseline NOT robust + neighbor 0
    if not baseline_cr:
        if n_cr_total >= 1:
            basin_class = "off_baseline_sparse_hits"
        else:
            basin_class = "no_basin"
    else:
        # strong/weak/isolated classification only when the baseline passes
        if params_with_2cr >= 3:
            basin_class = "strong"
        elif params_with_cr >= 1 or cr_neighbors_v >= 1:
            basin_class = "weak"
        else:
            basin_class = "isolated_robust_point"
    print(f"\n  basin classification: **{basin_class}**")
    
    # baseline failure is treated as the highest-priority condition
    decision = {
        "strong":                   "proceed to full-scale Phase 5",
        "weak":                     "narrow-scope Phase 5 optimization conditionally possible",
        "isolated_robust_point":    "hold Phase 5 -- targeted long-run recommended",
        "off_baseline_sparse_hits": "hold Phase 5 -- baseline not reproduced; targeted long-run recommended",
        "no_basin":                 "hold Phase 5 -- no basin confirmed; redefine scope or coupling",
    }[basin_class]
    print(f"  decision: {decision}")
    
    # ============================================================
    # save
    # ============================================================
    FIG_DIR = Path(__file__).resolve().parents[1] / "figures" / "phase5"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # CSV
    csv_out = FIG_DIR / f"phase5_narrow_scope_summary_{mode}.csv"
    with open(csv_out, "w") as f:
        f.write("geom_label,v_ms,A_dP_v,A_dchi_phi,A_robust,"
                "Ap_dP_v,Ap_dchi_phi,Ap_robust,cross_robust\n")
        for r in results:
            f.write(f"{r['geom_label']},{r['v']},"
                    f"{r['A']['delta_P_v']:.6e},{r['A']['delta_chi_phi']:.6e},"
                    f"{is_robust(r['A'])},"
                    f"{r['A_pocket']['delta_P_v']:.6e},{r['A_pocket']['delta_chi_phi']:.6e},"
                    f"{is_robust(r['A_pocket'])},"
                    f"{r['cross_robust']}\n")
    print(f"\n  saved: {csv_out}")
    
    # metadata
    script_path = Path(__file__).resolve()
    with open(script_path, "rb") as f:
        script_sha = hashlib.sha256(f.read()).hexdigest()[:16]
    metadata_out = FIG_DIR / f"phase5_narrow_scope_metadata_{mode}.json"
    with open(metadata_out, "w") as f:
        json.dump({
            "archive_version": "phase5_v16",
            "script_sha256_16": script_sha,
            "mode": mode, "n_real": n_real,
            "n_evaluations": n_total,
            "elapsed_sec": elapsed,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "baseline": BASELINE,
            "threshold_P_v": P_MIN_THRESH,
            "threshold_chi_phi": CHI_MIN_THRESH,
            "baseline_cross_robust": baseline_cr,
            "basin_size": basin_size,
            "basin_score": basin_score,
            "basin_classification": basin_class,
            "decision": decision,
            "param_cross_robust_count": param_cr_count,
            # conservative SEM-aware verdict
            "cross_robust_naive_count": int(n_cr_naive),
            "cross_robust_sem_count": int(n_cr_sem),
            "baseline_cross_robust_sem": bool(baseline_cr_sem),
            "n_real_caveat": f"n_real >= 3 recommended for a meaningful SEM; current n_real={n_real}",
            # flat alias
            # flat alias fields
            "is_strong_basin": bool(basin_class == "strong"),
            "is_baseline_failed": bool(not baseline_cr),
            "phase5_entry_recommended": bool(basin_class in ("strong", "weak")),
        }, f, indent=2)
    print(f"  saved: {metadata_out}")
    
    # plot -- basin visualization
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    # left: Delta P_v vs Delta chi_phi scatter, color = parameter, star = baseline
    ax = axes[0]
    colors = {"baseline": "red", "period_nm": "blue", "depth_nm": "green",
              "half_x_nm": "orange", "half_z_nm": "purple"}
    seen_labels = set()
    for r in results:
        if r["geom_label"] == "baseline":
            cat = "baseline"
        else:
            cat = r["geom_label"].split("=")[0]
        c = colors.get(cat, "gray")
        marker = "*" if cat == "baseline" else ("o" if r["cross_robust"] else "x")
        ms = 16 if cat == "baseline" else 9
        legend = cat if cat not in seen_labels else None
        seen_labels.add(cat)
        # A model
        ax.plot(r["A"]["delta_P_v"], r["A"]["delta_chi_phi"], marker=marker,
                color=c, ms=ms, alpha=0.7, mfc=c if r["cross_robust"] else "none",
                label=legend)
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    # threshold lines
    ax.axhline(-CHI_MIN_THRESH, color="r", lw=0.5, alpha=0.5, ls="--")
    ax.axvline(-P_MIN_THRESH, color="r", lw=0.5, alpha=0.5, ls="--")
    ax.set_xlabel(r"$\Delta P_v$ (A model)")
    ax.set_ylabel(r"$\Delta\chi_\phi$ (A model)")
    ax.set_title(f"Narrow-scope scan (A model) — basin={basin_class}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # right: distribution of cross-robust points (geom_label x v)
    ax = axes[1]
    geom_labels = sorted(set(r["geom_label"] for r in results))
    vs_unique = sorted(set(r["v"] for r in results))
    grid = np.zeros((len(geom_labels), len(vs_unique)))
    for r in results:
        i = geom_labels.index(r["geom_label"])
        j = vs_unique.index(r["v"])
        grid[i, j] = 1 if r["cross_robust"] else 0
    ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(geom_labels)))
    ax.set_yticklabels(geom_labels, fontsize=8)
    ax.set_xticks(range(len(vs_unique)))
    ax.set_xticklabels([f"{v}" for v in vs_unique])
    ax.set_xlabel("v [m/s]")
    ax.set_title(f"cross_robust grid (green=robust, red=not)")
    for i in range(len(geom_labels)):
        for j in range(len(vs_unique)):
            ax.text(j, i, f"{int(grid[i,j])}", ha="center", va="center",
                     color="black", fontsize=8)
    
    fig.suptitle(f"Phase 5 narrow-scope ({mode}) — basin_size={basin_size}, "
                 f"basin_score={basin_score:.2f}, class={basin_class}", fontsize=11)
    fig.tight_layout()
    out = str(FIG_DIR / f"phase5_narrow_scope_{mode}.png")
    fig.savefig(out, dpi=130)
    print(f"  saved: {out}")
    
    return results, basin_size, basin_score, basin_class, decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 narrow-scope pre-scan")
    parser.add_argument("--mode", choices=["preview", "targeted", "full"],
                        default="preview",
                        help="preview (9 geom × 1 v, ~30s) | "
                             "targeted (3 geom × 2 v, n_real=30, ~5-10 min) | "
                             "full (9 geom × 5 v, n_real=2, ~3-5 min)")
    parser.add_argument("--n_real", type=int, default=None,
                        help="realizations per condition "
                             "(default: preview=2, targeted=30, full=2)")
    args = parser.parse_args()
    default_n_real = {"preview": 2, "targeted": 30, "full": 2}
    n_real = args.n_real if args.n_real is not None else default_n_real[args.mode]
    main(mode=args.mode, n_real=n_real)
