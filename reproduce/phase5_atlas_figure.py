"""
Phase 5 atlas -- figure (for the manuscript).

Five-panel layout:
1. rank correlation per model pair
2. quadrant agreement per model pair
3. raw vs normalized model distance
4. raw vs relative geometry sensitivity
5. example A vs A_pocket quadrant disagreement

Data:   figures/phase5/phase5_atlas_{sensitivity,summary}_<mode>.csv
Output: figures/phase5/phase5_atlas_<mode>_figure.png/.pdf

**caveat (stated in the figure title/caption)**: validate_lite is edge-only, n_real=3, preliminary.

Usage:
  PYTHONPATH=. python reproduce/phase5_atlas_figure.py
"""

from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET_SUFFIX = ""
FIG_DIR = Path(__file__).resolve().parents[1] / "figures" / "phase5"
MODELS = ["A", "A_pocket", "B_z", "B_x"]
GEOM_PARAMS = ["period_nm", "depth_nm", "half_x_nm", "half_z_nm"]


def load_sensitivity(mode="validate_lite"):
    """sensitivity CSV → dict by metric."""
    path = FIG_DIR / f"phase5_atlas_sensitivity_{mode}{DATASET_SUFFIX}.csv"
    data = defaultdict(dict)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["metric"]][row["key"]] = float(row["value"])
    return data


def load_summary(mode="validate_lite"):
    """summary CSV → list of condition dicts."""
    path = FIG_DIR / f"phase5_atlas_summary_{mode}{DATASET_SUFFIX}.csv"
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "model": row["model"], "case": row["case"],
                "v": float(row["v_ms"]), "lam": float(row["lambda_uev"]),
                "geom": row["geom_label"],
                "dP_v": float(row["dP_v"]), "dchi_phi": float(row["dchi_phi"]),
            })
    return rows


from reproduce.phase5_sensitivity_atlas import quadrant_of  # canonical strict classification


def main(mode="validate_lite"):
    sens = load_sensitivity(mode)
    summary = load_summary(mode)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    pairs = ["A_vs_A_pocket", "A_vs_B_z", "A_vs_B_x",
             "A_pocket_vs_B_z", "A_pocket_vs_B_x", "B_z_vs_B_x"]

    def _mlabel(model):
        return {"A": "A", "A_pocket": r"$\mathrm{A_{pocket}}$",
                "B_z": r"$\mathrm{B_z}$", "B_x": r"$\mathrm{B_x}$"}.get(model, model)

    def pair_disp(p, sep=" vs "):
        a, b = p.split("_vs_")
        return _mlabel(a) + sep + _mlabel(b)

    pair_labels = [pair_disp(p, "\nvs\n") for p in pairs]
    
    # ---- Panel 1: rank correlation ----
    ax = fig.add_subplot(gs[0, 0])
    rc = [sens["rank_correlation"][p] for p in pairs]
    colors = ["#2a9d8f" if r >= 0.5 else "#e76f51" for r in rc]
    ax.barh(range(len(pairs)), rc, color=colors)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([pair_disp(p) for p in pairs], fontsize=8)
    ax.axvline(0.5, color="k", ls="--", lw=0.6)
    mean_rc = np.mean(rc)
    ax.axvline(mean_rc, color="b", ls="-", lw=1.0, alpha=0.6,
               label=f"mean={mean_rc:.2f}")
    ax.set_xlim(0, 1); ax.set_xlabel(r"Spearman $\rho$")
    ax.set_title("(1) Rank correlation\n(partial ansatz-dependence)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")
    
    # ---- Panel 2: quadrant agreement ----
    ax = fig.add_subplot(gs[0, 1])
    qa = [sens["quadrant_agreement"][p] for p in pairs]
    colors = ["#2a9d8f" if q >= 0.5 else "#e76f51" for q in qa]
    ax.barh(range(len(pairs)), qa, color=colors)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([pair_disp(p) for p in pairs], fontsize=8)
    mean_qa = np.mean(qa)
    ax.axvline(mean_qa, color="b", ls="-", lw=1.0, alpha=0.6,
               label=f"mean={mean_qa:.0%}")
    ax.set_xlim(0, 1); ax.set_xlabel("quadrant agreement")
    ax.set_title("(2) Quadrant agreement\n(stronger ansatz-dependence)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")
    
    # ---- Panel 3: raw vs channel-balanced model distance ----
    # raw D is dominated by the larger-scale dchi; the channel-balanced D
    # measures each channel in units of its effect-size threshold. They are
    # different metrics (not a uniform rescaling), so we show each relative
    # to its own maximum to expose where the pair ranking differs.
    ax = fig.add_subplot(gs[0, 2])
    d_raw = np.array([sens["model_sensitivity"][p] for p in pairs], float)
    d_norm = np.array([sens["model_sensitivity_normalized"][p] for p in pairs], float)
    d_raw_rel = d_raw / d_raw.max()
    d_norm_rel = d_norm / d_norm.max()
    x = np.arange(len(pairs))
    w = 0.35
    ax.bar(x - w/2, d_raw_rel, w, label="raw D (rel.)", color="#264653")
    ax.bar(x + w/2, d_norm_rel, w, label="channel-balanced D (rel.)",
           color="#e9c46a")
    ax.set_xticks(x)
    ax.set_xticklabels([pair_disp(p, "/") for p in pairs],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("distance (relative to max)")
    ax.set_title(r"(3) Model distance" "\n" r"($\mathrm{B_z}$ vs $\mathrm{B_x}$ largest; ranking reorders)", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
    
    # ---- Panel 4: raw vs relative geometry sensitivity ----
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(len(GEOM_PARAMS))
    w = 0.2
    for i, m in enumerate(MODELS):
        raw_vals = [sens["geometry_sensitivity_raw"].get(f"{m}|{p}", 0) * 1e3
                    for p in GEOM_PARAMS]
        ax.bar(x + (i - 1.5) * w, raw_vals, w, label=_mlabel(m))
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_nm","").replace("half_x",r"$\mathrm{half_x}$").replace("half_z",r"$\mathrm{half_z}$") for p in GEOM_PARAMS], fontsize=8)
    ax.set_ylabel(r"$S_\theta^{raw} \times 10^3$")
    ax.set_title("(4a) Raw geometry sensitivity\n($\\mathrm{half_z}$ dominant)", fontsize=10)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3, axis="y")
    
    ax = fig.add_subplot(gs[1, 1])
    for i, m in enumerate(MODELS):
        rel_vals = [sens["geometry_sensitivity_relative"].get(f"{m}|{p}", 0)
                    for p in GEOM_PARAMS]
        ax.bar(x + (i - 1.5) * w, rel_vals, w, label=_mlabel(m))
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_nm","").replace("half_x",r"$\mathrm{half_x}$").replace("half_z",r"$\mathrm{half_z}$") for p in GEOM_PARAMS], fontsize=8)
    ax.set_ylabel(r"$S_\theta^{rel}$")
    ax.set_title("(4b) Relative geometry sensitivity\n(no shared hierarchy; $\\mathrm{B}_z$: period largest)", fontsize=10)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3, axis="y")
    
    # ---- Panel 5: example A vs A_pocket quadrant disagreement ----
    ax = fig.add_subplot(gs[1, 2])
    # compare A and A_pocket (dP_v, dchi) at the same (case,v,lam,geom)
    by_cond_A = {}
    by_cond_Ap = {}
    for r in summary:
        key = (r["case"], r["v"], r["lam"], r["geom"])
        if r["model"] == "A":
            by_cond_A[key] = r
        elif r["model"] == "A_pocket":
            by_cond_Ap[key] = r
    # arrows only for disagreement conditions
    n_shown = 0
    for key in by_cond_A:
        if key not in by_cond_Ap:
            continue
        rA, rAp = by_cond_A[key], by_cond_Ap[key]
        qA = quadrant_of(rA["dP_v"], rA["dchi_phi"])
        qAp = quadrant_of(rAp["dP_v"], rAp["dchi_phi"])
        if qA != qAp:
            ax.annotate("", xy=(rAp["dP_v"]*1e3, rAp["dchi_phi"]*1e3),
                        xytext=(rA["dP_v"]*1e3, rA["dchi_phi"]*1e3),
                        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5, lw=0.8))
            ax.plot(rA["dP_v"]*1e3, rA["dchi_phi"]*1e3, "o", color="#264653", ms=4)
            ax.plot(rAp["dP_v"]*1e3, rAp["dchi_phi"]*1e3, "s", color="#e76f51", ms=4)
            n_shown += 1
    ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
    ax.axhline(-1, color="r", lw=0.4, ls="--", alpha=0.5)
    ax.axvline(-0.1, color="r", lw=0.4, ls="--", alpha=0.5)
    ax.plot([], [], "o", color="#264653", label="A")
    ax.plot([], [], "s", color="#e76f51", label=_mlabel("A_pocket"))
    ax.set_xlabel(r"$\Delta P_v \times 10^3$")
    ax.set_ylabel(r"$\Delta\chi_\phi \times 10^3$")
    ax.set_title(f"(5) A $\\to$ $\\mathrm{{A_{{pocket}}}}$ quadrant shift\n({n_shown} disagreement conditions)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    
    caveat_map = {
        "validate_lite": "edge-only, n_real=3, validate_lite preliminary",
        "validate_mid": "center+edge, v=[5,20], n_real=4, validate_mid",
        "validate": "center+edge, v=[5,10,20], n_real=5, full validate",
        "preview": "edge-only, n_real=2, preview sanity",
    }
    caveat = caveat_map.get(mode, mode)
    caveat_disp = caveat.replace("n_real=", "realizations=")
    fig.suptitle(
        f"Coupling-Ansatz Sensitivity Atlas - {mode.upper()}\n"
        f"({caveat_disp})  |  "
        f"Ranking: mean rho={mean_rc:.2f} | "
        f"Quadrant: mean agreement {mean_qa:.0%}",
        fontsize=12, y=0.99)
    
    suffix = "preliminary_figure" if mode == "validate_lite" else f"{mode}_figure"
    out = FIG_DIR / f"phase5_atlas_{suffix}{DATASET_SUFFIX}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")  # vector for paper
    print(f"saved: {out} (+ .pdf)")
    
    # figure metadata (stale tracking)
    import hashlib, json, datetime
    script_path = Path(__file__).resolve()
    with open(script_path, "rb") as fh:
        fig_sha = hashlib.sha256(fh.read()).hexdigest()[:16]
    n_real_map = {"preview": 2, "validate_lite": 3, "validate_mid": 4, "validate": 5}
    meta = {
        "figure_script_sha256_16": fig_sha,
        "input_summary_csv": f"phase5_atlas_summary_{mode}{DATASET_SUFFIX}.csv",
        "input_sensitivity_csv": f"phase5_atlas_sensitivity_{mode}{DATASET_SUFFIX}.csv",
        "mode": mode,
        "n_real": n_real_map.get(mode),
        "mean_rank_correlation": round(mean_rc, 4),
        "mean_quadrant_agreement": round(mean_qa, 4),
        "quadrant_disagreement_conditions": n_shown,
        "caveat": caveat,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = FIG_DIR / f"phase5_atlas_figure_metadata_{mode}{DATASET_SUFFIX}.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"saved: {meta_path}")
    print(f"  mean rank correlation: {mean_rc:.3f}")
    print(f"  mean quadrant agreement: {mean_qa:.3f}")
    print(f"  panel 5 quadrant-disagreement conditions: {n_shown}")
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 5 atlas figure")
    parser.add_argument("--mode", default="validate_lite",
                        choices=["validate_lite", "validate_mid", "validate", "preview"])
    parser.add_argument("--dataset-suffix", default="",
                        help="dataset filename tag from the atlas run")
    args = parser.parse_args()
    DATASET_SUFFIX = args.dataset_suffix
    main(mode=args.mode)
