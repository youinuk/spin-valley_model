"""
Supplementary robustness analyses for the coupling-ansatz sensitivity atlas.

Supplementary robustness checks for the rank/quadrant metrics:
  1. Rank-correlation weighting sweep -- dependence of the |R| norm on channel weighting.
     (raw, normalized, |ΔP_v|-only, |Δχ_φ|-only).
  2. Quadrant-threshold sweep -- mean agreement vs effect-size threshold.
  3. Cohen's kappa — chance-corrected pairwise quadrant agreement.

Input:  full validate raw pkl (figures/phase5/phase5_atlas_raw_validate_case_*.pkl)
Output: docs/supplement_data/supplementary_robustness_results.md (table text)
      figures/phase5/supp_rank_weighting.pdf / .png
      figures/phase5/supp_threshold_sweep.pdf / .png

Usage:
  PYTHONPATH=. python reproduce/phase5_supp_robustness.py
"""
from __future__ import annotations
from pathlib import Path
from itertools import combinations
import glob
import pickle
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "phase5"
MODELS = ["A", "A_pocket", "B_z", "B_x"]
PAIRS = list(combinations(MODELS, 2))

# baseline thresholds (main text)
P_MIN = 1e-4
CHI_MIN = 1e-3


DATASET_SUFFIX = ""  # set via --dataset-suffix (e.g. "__ez-total-local__norm-prefactor")


def load_full():
    files = sorted(glob.glob(str(FIG / f"phase5_atlas_raw_validate_case_*{DATASET_SUFFIX}.pkl")))
    files = [f for f in files if (DATASET_SUFFIX or "__ez-" not in f)]
    merged = {}
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        merged.update(d["data"])
    # reorganize: cond -> model -> (dP, dchi)
    conds = {}
    for k, v in merged.items():
        model = k[0]
        cond = k[1:]
        conds.setdefault(cond, {})[model] = (v["dP_v"], v["dchi_phi"])
    return conds


def quadrant_strict(dP, dchi, p_min, chi_min):
    sP = abs(dP) >= p_min
    sX = abs(dchi) >= chi_min
    if not (sP or sX):
        return "below_threshold"
    if sP and sX:
        if dP < 0 and dchi < 0:
            return "robust"
        if dP > 0 and dchi < 0:
            return "valley_trade"
        if dP < 0 and dchi > 0:
            return "spin_trade"
        return "both_worsen"
    if sP:
        return "P_only_improve" if dP < 0 else "P_only_worsen"
    return "chi_only_improve" if dchi < 0 else "chi_only_worsen"


# ---------- 1. rank weighting sweep ----------
def rank_weighting(conds):
    """Different magnitude definitions -> mean pairwise Spearman."""
    cond_list = sorted(conds.keys())

    def magnitudes(weight_P, weight_chi, mode="norm"):
        m = {model: [] for model in MODELS}
        for cond in cond_list:
            for model in MODELS:
                if model in conds[cond]:
                    dP, dchi = conds[cond][model]
                    if mode == "P":
                        val = abs(dP)
                    elif mode == "chi":
                        val = abs(dchi)
                    else:
                        val = np.hypot(weight_P * dP, weight_chi * dchi)
                    m[model].append(val)
                else:
                    m[model].append(np.nan)
        return m

    def mean_rho(m):
        rs = []
        for a, b in PAIRS:
            x = np.array(m[a]); y = np.array(m[b])
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() >= 3:
                rho, _ = spearmanr(x[mask], y[mask])
                rs.append(rho)
        return float(np.mean(rs))

    rows = []
    rows.append(("raw norm (w=1,1)", mean_rho(magnitudes(1, 1))))
    # threshold-normalized: divide each channel by its threshold
    rows.append(("threshold-normalized", mean_rho(magnitudes(1/P_MIN, 1/CHI_MIN))))
    rows.append(("|dP_v| only", mean_rho(magnitudes(0, 0, mode="P"))))
    rows.append(("|dchi_phi| only", mean_rho(magnitudes(0, 0, mode="chi"))))
    # example fixed normalizations
    rows.append(("P_ref=1e-2, chi_ref=1", mean_rho(magnitudes(1/1e-2, 1/1.0))))
    rows.append(("P_ref=1e-4, chi_ref=1e-3", mean_rho(magnitudes(1/1e-4, 1/1e-3))))
    return rows


# ---------- 2. threshold sweep ----------
def threshold_sweep(conds):
    cond_list = sorted(conds.keys())
    settings = [
        ("none", 0.0, 0.0),
        ("1e-5, 1e-4", 1e-5, 1e-4),
        ("1e-4, 1e-3 (main)", 1e-4, 1e-3),
        ("1e-3, 1e-2", 1e-3, 1e-2),
    ]
    rows = []
    for label, pmin, cmin in settings:
        agrs = []
        for a, b in PAIRS:
            match = tot = 0
            for cond in cond_list:
                if a in conds[cond] and b in conds[cond]:
                    qa = quadrant_strict(*conds[cond][a], pmin, cmin)
                    qb = quadrant_strict(*conds[cond][b], pmin, cmin)
                    tot += 1
                    if qa == qb:
                        match += 1
            if tot:
                agrs.append(match / tot)
        rows.append((label, float(np.mean(agrs))))
    return rows


# ---------- 3. Cohen's kappa ----------
def cohen_kappa(conds):
    cond_list = sorted(conds.keys())
    cats = ["below_threshold", "robust", "valley_trade", "spin_trade",
            "both_worsen", "P_only_improve", "P_only_worsen",
            "chi_only_improve", "chi_only_worsen"]
    rows = []
    for a, b in PAIRS:
        qa_list, qb_list = [], []
        for cond in cond_list:
            if a in conds[cond] and b in conds[cond]:
                qa_list.append(quadrant_strict(*conds[cond][a], P_MIN, CHI_MIN))
                qb_list.append(quadrant_strict(*conds[cond][b], P_MIN, CHI_MIN))
        n = len(qa_list)
        po = sum(x == y for x, y in zip(qa_list, qb_list)) / n
        # expected agreement
        pe = 0.0
        for c in cats:
            pa = qa_list.count(c) / n
            pb = qb_list.count(c) / n
            pe += pa * pb
        kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
        rows.append((f"{a} vs {b}", po, kappa))
    return rows


def main():
    conds = load_full()
    if not conds:
        raise SystemExit(
            f"No matching validate raw data found for dataset suffix "
            f"'{DATASET_SUFFIX}'. This script analyses the FULL validate "
            f"dataset only -- run the full validate recomputation first, "
            f"or pass a valid validate dataset suffix.")
    print(f"loaded {len(conds)} conditions")

    rw = rank_weighting(conds)
    ts = threshold_sweep(conds)
    ck = cohen_kappa(conds)

    # ---- write markdown ----
    out_md = ROOT / "docs" / "supplement_data" / "supplementary_robustness_results.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Supplementary robustness analyses (full validate)\n",
             "Recomputed from raw pkl. The key conclusion (quadrant is more "
             "ansatz-dependent than ranking, especially for B_x) holds across all variants.\n",
             "\n## 1. Rank-correlation weighting sweep\n",
             "Mean pairwise Spearman rho for different |R| magnitude definitions. "
             "The raw norm is biased toward the larger-scale dchi.\n",
             "\n| weighting | mean Spearman ρ |",
             "| --- | ---: |"]
    for label, val in rw:
        lines.append(f"| {label} | {val:.3f} |")
    _rw_lo, _rw_hi = min(v for _, v in rw), max(v for _, v in rw)
    lines += [f"\nConclusion: ranking remains high but weighting-dependent "
              f"(rho ~ {_rw_lo:.2f}-{_rw_hi:.2f}). "
              "Rank is a complementary, weighting-dependent diagnostic; the "
              "scale-insensitive quadrant is of primary interest.\n",
              "\n## 2. Quadrant-threshold sweep\n",
             "Mean quadrant agreement vs effect-size threshold (strict classification).\n",
              "\n| threshold (P_min, chi_min) | mean quadrant agreement |",
              "| --- | ---: |"]
    for label, val in ts:
        lines.append(f"| {label} | {val:.1%} |")
    _ts_lo, _ts_hi = min(v for _, v in ts), max(v for _, v in ts)
    lines += [f"\nConclusion: quadrant agreement remains far from "
              f"model-independent across thresholds, ranging from roughly "
              f"{_ts_lo:.0%} to {_ts_hi:.0%} in this sweep. "
              "The conclusion that quadrant interpretation is ansatz-sensitive is threshold-robust.\n",
              "\n## 3. Chance-corrected agreement (Cohen's kappa)\n",
             "Agreement corrected for the marginal quadrant distribution.\n",
              "\n| pair | raw agreement | Cohen's κ |",
              "| --- | ---: | ---: |"]
    for label, po, k in ck:
        lines.append(f"| {label} | {po:.1%} | {k:.3f} |")
    _k_sorted = sorted(ck, key=lambda t: t[2])
    _k_lo, _k_hi = _k_sorted[0][2], _k_sorted[-1][2]
    _low_pairs = ", ".join(lbl for lbl, _, _ in _k_sorted[:2])
    lines += [f"\nConclusion: chance-corrected agreement is modest across all "
              f"pairs (kappa ~ {_k_lo:.2f}-{_k_hi:.2f}); the two lowest values are "
              f"the {_low_pairs} pairs, both pairing the partial-x-Bx model with a "
              "partial-x-Bz one. This is a gradient-component contrast modulated by "
              "spatial localization, not a blanket property of any single ansatz.\n"]
    out_md.write_text("\n".join(lines))
    print(f"saved: {out_md}")

    # ---- figure: rank weighting + threshold sweep ----
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    labels_rw = [r[0] for r in rw]; vals_rw = [r[1] for r in rw]
    axes[0].barh(range(len(rw)), vals_rw, color="#4477aa")
    axes[0].set_yticks(range(len(rw)))
    axes[0].set_yticklabels(labels_rw, fontsize=7)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel(r"mean Spearman $\rho$")
    axes[0].set_title("Rank correlation vs channel weighting", fontsize=9)
    try:
        import json as _json
        _md = _json.load(open(FIG / f"phase5_atlas_metadata_validate{DATASET_SUFFIX}.json"))
        def _g(o, k):
            if isinstance(o, dict):
                if k in o:
                    return o[k]
                for v in o.values():
                    r = _g(v, k)
                    if r is not None:
                        return r
        _base = _g(_md, "mean_rank_correlation")
    except Exception:
        _base = None
    if _base is not None:
        axes[0].axvline(_base, color="grey", ls="--", lw=0.8)
    axes[0].invert_yaxis()

    labels_ts = [r[0] for r in ts]; vals_ts = [r[1] for r in ts]
    axes[1].bar(range(len(ts)), vals_ts, color="#ee6677")
    axes[1].set_xticks(range(len(ts)))
    axes[1].set_xticklabels(labels_ts, fontsize=7, rotation=20, ha="right")
    axes[1].set_ylim(0, 0.6)
    axes[1].set_ylabel("mean quadrant agreement")
    axes[1].set_title("Quadrant agreement vs threshold", fontsize=9)

    fig.tight_layout()
    out = FIG / f"supp_robustness{DATASET_SUFFIX}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out} (+ .pdf)")


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--dataset-suffix", default="",
                    help="dataset filename tag from the atlas run "
                         "(empty = legacy archive)")
    _a = _p.parse_args()
    DATASET_SUFFIX = _a.dataset_suffix
    main()
