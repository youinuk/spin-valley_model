"""
Phase 5-0 -- objective sensitivity analysis + cross-model robust candidate extraction.

Reads phase4p6_summary_full_lite.csv and:
1. applies effect-size thresholds
2. extracts candidates robust in both A and A_pocket (cross-validation)
3. compares J_robust rankings across 5 weight sets
4. fills the go/no-go checklist for the next optimization stage
"""

import csv
from pathlib import Path
from collections import defaultdict

FIG_DIR = Path(__file__).resolve().parents[1] / "figures" / "phase4"
CSV_PATH = FIG_DIR / "phase4p6_summary_full_lite.csv"
RESULTS_OUT = Path(__file__).resolve().parents[1] / "docs" / "phase5_objective_results.md"


def load_full_lite():
    rows = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "cm": row["coupling_model"],
                "case": row["case"],
                "v": float(row["v_ms"]),
                "lam_uev": float(row["lambda_uev"]),
                "dP_v": float(row["delta_P_v"]),
                "dchi_phi": float(row["delta_chi_phi"]),
                "S_M2": float(row["S_M2"]),
            })
    return rows


def classify_with_threshold(rows, P_min, chi_min):
    """Apply effect-size thresholds."""
    cls = {"robust": [], "valley_trade": [], "spin_trade": [],
           "avoid": [], "below_threshold": []}
    for r in rows:
        if r["lam_uev"] == 0.0:
            continue
        dP, dchi = r["dP_v"], r["dchi_phi"]
        # threshold: classify into a quadrant only when |value| exceeds the floor
        sig_P = abs(dP) >= P_min
        sig_chi = abs(dchi) >= chi_min
        if not (sig_P or sig_chi):
            cls["below_threshold"].append(r)
            continue
        # only one channel significant: treat the other as 0
        eff_P = dP if sig_P else 0.0
        eff_chi = dchi if sig_chi else 0.0
        if eff_P < 0 and eff_chi < 0:
            cls["robust"].append(r)
        elif eff_P > 0 and eff_chi < 0:
            cls["valley_trade"].append(r)
        elif eff_P < 0 and eff_chi > 0:
            cls["spin_trade"].append(r)
        else:
            cls["avoid"].append(r)
    return cls


def cross_validate_robust(rows, P_min, chi_min):
    """Extract (case, v, lambda) that are robust in *both* A and A_pocket."""
    # condition key: (case, v, λ)
    per_cond = defaultdict(dict)
    for r in rows:
        if r["lam_uev"] == 0.0:
            continue
        key = (r["case"], r["v"], r["lam_uev"])
        per_cond[key][r["cm"]] = r
    
    common = []
    for key, by_model in per_cond.items():
        if not ("A" in by_model and "A_pocket" in by_model):
            continue
        ra, rap = by_model["A"], by_model["A_pocket"]
        # both pass the threshold AND fall in the robust quadrant
        def is_robust(r):
            sig_P = abs(r["dP_v"]) >= P_min
            sig_chi = abs(r["dchi_phi"]) >= chi_min
            if not (sig_P and sig_chi):
                return False
            return r["dP_v"] < 0 and r["dchi_phi"] < 0
        if is_robust(ra) and is_robust(rap):
            common.append({"case": key[0], "v": key[1], "lam_uev": key[2],
                            "A": ra, "A_pocket": rap})
    return common


def weight_sensitivity(rows, weight_sets, P_min, chi_min, normalize=True):
    """Weight sweep to assess ranking stability.
    
    J_robust = w_P * max(ΔP_v, 0) + w_φ * max(Δχ_φ, 0)
    
    Normalization:
    - normalized Delta P_v = Delta P_v / P_ref, P_ref = 1e-2
    - normalized Delta chi_phi = Delta chi_phi / chi_ref, chi_ref = 1.0 rad^2
    """
    P_ref = 1e-2
    chi_ref = 1.0
    
    rankings = {}
    for wname, (wP, wchi) in weight_sets.items():
        scored = []
        for r in rows:
            if r["lam_uev"] == 0.0:
                continue
            dP_t = r["dP_v"] / P_ref if normalize else r["dP_v"]
            dchi_t = r["dchi_phi"] / chi_ref if normalize else r["dchi_phi"]
            J = wP * max(dP_t, 0) + wchi * max(dchi_t, 0)
            scored.append((J, r))
        scored.sort(key=lambda x: x[0])
        # rank the top-K conditions (lower J = better)
        rankings[wname] = [(J, r["cm"], r["case"], r["v"], r["lam_uev"])
                            for J, r in scored[:10]]
    return rankings


def analyze_and_write():
    rows = load_full_lite()
    n_total_pos_lambda = sum(1 for r in rows if r["lam_uev"] != 0)
    
    # effect-size thresholds
    # noise-floor estimate for full_lite (lambda=0 results sit within the invariance-check tolerance)
    # conservatively use |Delta P_v| >= 1e-4, |Delta chi_phi| >= 1e-3
    P_min, chi_min = 1e-4, 1e-3
    
    # 1) classification with thresholds applied
    cls = classify_with_threshold(rows, P_min, chi_min)
    
    # 2) cross-validated robust (A AND A_pocket)
    common_robust = cross_validate_robust(rows, P_min, chi_min)
    
    # 3) weight sensitivity
    weight_sets = {
        "equal": (1.0, 1.0),
        "valley_heavy": (2.0, 1.0),
        "spin_heavy": (1.0, 2.0),
        "asym_low": (0.5, 1.5),
        "asym_high": (1.5, 0.5),
    }
    rankings = weight_sensitivity(rows, weight_sets, P_min, chi_min)
    
    # 4) ranking stability -- condition overlap in the top-5
    top5_sets = {}
    for wname, rk in rankings.items():
        top5_sets[wname] = set([(c, ca, v, l) for _, c, ca, v, l in rk[:5]])
    # conditions common to the top-5 of all weight sets
    common_top5 = set.intersection(*top5_sets.values())
    # agreement rate: equal vs valley_heavy
    overlap_eq_vh = len(top5_sets["equal"] & top5_sets["valley_heavy"]) / 5
    overlap_eq_sh = len(top5_sets["equal"] & top5_sets["spin_heavy"]) / 5
    
    # write the markdown report
    md = []
    md.append("# Phase 5-0 — Objective Sensitivity Analysis Results\n\n")
    md.append("**Status**: quantitative analysis based on the phase4p6 full_lite CSV.\n\n")
    md.append("Output: items 1-5 of the objective-design work list.\n\n")
    md.append("---\n\n")
    md.append("## 1. Classification with effect-size thresholds\n\n")
    md.append(f"effect-size threshold: $|\\Delta P_v| \\geq {P_min}$, $|\\Delta\\chi_\\phi| \\geq {chi_min:.0e}$ rad²\n\n")
    md.append(f"total conditions (λ>0): {n_total_pos_lambda}\n\n")
    md.append("| quadrant | count | % |\n|---|---:|---:|\n")
    for name, lst in cls.items():
        pct = 100 * len(lst) / n_total_pos_lambda
        md.append(f"| {name} | {len(lst)} | {pct:.1f}% |\n")
    md.append(f"\n-> **robust candidates (after thresholds)**: {len(cls['robust'])} / {n_total_pos_lambda}\n\n")
    md.append(f"Tiny negative values (e.g. $-10^{{-8}}$) fall into below_threshold as intended. ")
    md.append(f"Robust count shrinks from 16 (no threshold) to {len(cls['robust'])} after thresholds.\n\n")
    md.append("## 2. Cross-validated robust candidates (A AND A_pocket)\n\n")
    md.append("Only (case, v, lambda) robust in both A and A_pocket count as formal candidates.\n\n")
    if common_robust:
        md.append("| case | v (m/s) | λ (μeV) | A: ΔP_v | A: Δχ_φ | A_pocket: ΔP_v | A_pocket: Δχ_φ |\n")
        md.append("|---|---:|---:|---:|---:|---:|---:|\n")
        for c in common_robust:
            md.append(f"| {c['case']} | {c['v']} | {c['lam_uev']} | "
                       f"{c['A']['dP_v']:+.2e} | {c['A']['dchi_phi']:+.2e} | "
                       f"{c['A_pocket']['dP_v']:+.2e} | {c['A_pocket']['dchi_phi']:+.2e} |\n")
        md.append(f"\n**Number of cross-validated robust candidates**: {len(common_robust)}\n\n")
    else:
        md.append("**No cross-validated robust candidate** -- no (case, v, lambda) passes thresholds in both A and A_pocket.\n\n")
    
    md.append("## 3. Weight sensitivity sweep\n\n")
    md.append("Top-10 J_robust rankings compared across 5 weight sets (normalization: $P_{\\rm ref}=10^{-2}$, $\\chi_{\\rm ref}=1$ rad^2).\n\n")
    md.append("### Top-5 ranking comparison\n\n")
    md.append("| rank | equal (1,1) | valley_heavy (2,1) | spin_heavy (1,2) | asym_low (0.5,1.5) | asym_high (1.5,0.5) |\n")
    md.append("|---|---|---|---|---|---|\n")
    for i in range(5):
        row = ["|", f"{i+1}", "|"]
        for wname in ["equal", "valley_heavy", "spin_heavy", "asym_low", "asym_high"]:
            J, cm, ca, v, l = rankings[wname][i]
            row.append(f" {cm[:6]} {ca[5:9]} v={v} λ={l} ")
            row.append("|")
        md.append("".join(row) + "\n")
    md.append("\n### Ranking stability (top-5 overlap)\n\n")
    md.append(f"- equal vs valley_heavy: {overlap_eq_vh*100:.0f}%\n")
    md.append(f"- equal vs spin_heavy: {overlap_eq_sh*100:.0f}%\n")
    md.append(f"- top-5 common to all 5 weight sets: {len(common_top5)}/5\n\n")
    
    if overlap_eq_vh >= 0.6 and overlap_eq_sh >= 0.6:
        md.append("-> **stability PASS** (>=60%). Weight choice does not strongly affect the ranking.\n\n")
    else:
        md.append("-> **stability FAIL** (<60%). Ranking is weight-sensitive; the baseline weights need physical justification.\n\n")
    
    md.append("## 4. Go/no-go checklist\n\n")
    md.append("Criteria:\n\n")
    md.append("| condition | current status | met? |\n|---|---|:---:|\n")
    md.append(f"| weight sensitivity sweep done, top-5 >=80% stable | overlap eq/vh={overlap_eq_vh*100:.0f}%, eq/sh={overlap_eq_sh*100:.0f}% | {'OK' if (overlap_eq_vh >= 0.8 and overlap_eq_sh >= 0.8) else 'WARN'} |\n")
    md.append(f"| >=3 coupling-model-robust candidates | {len(common_robust)} | {'OK' if len(common_robust) >= 3 else 'NO'} |\n")
    md.append(f"| robust candidates pass |effect| thresholds (1e-4 / 1e-3) | passing: {len(cls['robust'])} | {'OK' if len(cls['robust']) > 0 else 'NO'} |\n")
    md.append(f"| J_robust evaluation cost estimated | full_lite ~50 s x 5 weights = ~4 min | approx |\n")
    md.append(f"| geometry parameter space/constraints specified | not yet | NO |\n\n")
    
    md.append("## 5. Verdict -- proceed to the main optimization?\n\n")
    n_pass = sum([
        overlap_eq_vh >= 0.8 and overlap_eq_sh >= 0.8,
        len(common_robust) >= 3,
        len(cls['robust']) > 0,
        True,   # cost estimate is approximate
    ])
    if n_pass >= 4:
        md.append("**Verdict**: proceed to the main optimization (4+ of 5 conditions met).\n\n")
    elif n_pass >= 2:
        md.append("**Verdict**: proceed with *narrow scope* only (2-3 of 5). Geometry parameter space must be defined.\n\n")
    else:
        md.append("**Verdict**: hold. Further validation required.\n\n")
    
    md.append("## 6. Next steps\n\n")
    if len(common_robust) >= 3 and overlap_eq_vh >= 0.6:
        md.append("1. Specify allowed ranges for half_x, half_z, period, depth\n")
        md.append("2. Define the narrow scope for the main optimization (cross-validated robust region)\n")
        md.append("3. (optional) one full-mode long run (n_real=30) to re-check candidate stability\n")
    else:
        md.append("1. **full-mode long run** (n_real=30) -- secure statistical stability\n")
        md.append("2. if cross-validated candidates are lacking, try resonance-window couplings\n")
        md.append("3. revisit thresholds -- compute the n_real bootstrap noise floor precisely\n")
    
    md.append("\n---\n")
    md.append("\n## Appendix A -- analysis metadata\n\n")
    md.append(f"- input: `figures/phase4/phase4p6_summary_full_lite.csv`\n")
    md.append(f"- threshold: P_min={P_min}, chi_min={chi_min}\n")
    md.append(f"- weight sets: {list(weight_sets.keys())}\n")
    md.append(f"- normalization: P_ref=1e-2, chi_ref=1.0\n")
    md.append(f"- script: `reproduce/phase5_objective_analysis.py`\n")
    
    with open(RESULTS_OUT, "w") as f:
        f.writelines(md)
    print(f"saved: {RESULTS_OUT}")
    
    # console summary
    print("\n=== Phase 5-0 analysis summary ===")
    print(f"  Total λ>0 conditions: {n_total_pos_lambda}")
    print(f"  Robust (threshold passed): {len(cls['robust'])}")
    print(f"  Below threshold: {len(cls['below_threshold'])}")
    print(f"  Cross-validated robust (A AND A_pocket): {len(common_robust)}")
    print(f"  Ranking stability: eq/vh = {overlap_eq_vh*100:.0f}%, eq/sh = {overlap_eq_sh*100:.0f}%")


if __name__ == "__main__":
    analyze_and_write()
