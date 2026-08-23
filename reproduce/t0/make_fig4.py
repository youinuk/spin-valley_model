"""Regenerate Fig. 4 for the T0 successor manuscript.

Three panels, all from the closed T0 analysis outputs:
  (a) channel-resolved three-state disagreement, ansatz against seed, n=30/100
  (b) candidate transport across the ansatz family, n=30 and n=100
  (c) rank correlation against classification agreement, six ansatz pairs, n=100

No pickle access: every number is read from the analysis JSON.
"""
import json
import sys
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np

ANS = "#d66014"   # bxorange, cross-ansatz
SEED = "#1f77b4"  # bzblue, cross-seed
TEAL = "#388e8e"  # robustteal
GREY = "#555555"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})


def load(p):
    with open(p) as fh:
        return json.load(fh)


def main(src, out):
    ph30 = load(f"{src}/posthoc_n30.json")["A_channel_three_state"]
    ph100 = load(f"{src}/posthoc_n100.json")["A_channel_three_state"]
    ho30 = load(f"{src}/holdout_n30.json")["cross_ansatz_transport"]
    cf100 = load(f"{src}/confirmation_n100.json")["cross_ansatz_transport"]
    lg100 = load(f"{src}/legacy_n100.json")

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.6))
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.24, top=0.87, wspace=0.55)
    # shift panel (c) right, keeping its width, so its two-line y-label clears
    # panel (b); bbox_inches="tight" absorbs the overhang on save
    box = axes[2].get_position()
    axes[2].set_position([box.x0 + 0.030, box.y0, box.width, box.height])

    # ---------------- (a) channel-resolved three-state disagreement ---------
    ax = axes[0]
    groups = [
        ("$\\Delta P_v$\n$30$", ph30["dP_v"]),
        ("$\\Delta P_v$\n$100$", ph100["dP_v"]),
        ("$\\Delta\\chi_\\phi$\n$30$", ph30["dchi_phi"]),
        ("$\\Delta\\chi_\\phi$\n$100$", ph100["dchi_phi"]),
    ]
    x = np.arange(len(groups))
    w = 0.36
    ax.bar(x - w / 2, [g[1]["D_ansatz"] for g in groups], w,
           color=ANS, label="cross-ansatz")
    ax.bar(x + w / 2, [g[1]["D_seed"] for g in groups], w,
           color="white", edgecolor=SEED, hatch="///", linewidth=0.8,
           label="cross-seed")
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_xlabel("$n_{\\rm real}$", labelpad=1)
    ax.set_ylabel("three-state disagreement\n(fraction of comparisons)")
    ax.set_ylim(0, 0.60)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax.set_title("(a) channel-resolved", loc="left")
    ax.legend(frameon=False, loc="upper right", handlelength=1.4)
    ax.axvline(1.5, color="0.85", lw=0.7, zorder=0)
    for xi, g in zip(x, groups):
        ax.text(xi, max(g[1]["D_ansatz"], g[1]["D_seed"]) + 0.012,
                f"{g[1]['ratio']:.2f}", ha="center", fontsize=10, color=GREY)

    # ---------------- (b) candidate transport ------------------------------
    ax = axes[1]
    k = np.arange(5)
    v30 = [ho30[str(i)] for i in k]
    v100 = [cf100[str(i)] for i in k]
    ax.bar(k - w / 2, v30, w, color=TEAL, label="$n{=}30$")
    ax.bar(k + w / 2, v100, w, color="white", edgecolor=TEAL, hatch="///",
           linewidth=0.8, label="$n{=}100$")
    ax.set_xticks(k)
    ax.set_xlabel("ans\u00e4tze confirming the candidate")
    ax.set_ylabel("conditions in $U^{\\rm disc}$ (count)")
    ax.set_ylim(0, 6.4)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_title("(b) candidate transport", loc="left")
    ax.legend(frameon=False, loc="upper left", handlelength=1.4,
              bbox_to_anchor=(-0.02, 1.03))
    ax.text(0.98, 0.62, f"$|U^{{\\rm disc}}|$ = {sum(v30)}, {sum(v100)}",
            transform=ax.transAxes, ha="right", fontsize=10, color=GREY)

    # ---------------- (c) ranking against classification -------------------
    ax = axes[2]
    rho = lg100["pairwise_rho"]
    agr = lg100["pairwise_agreement"]
    pairs = list(rho)
    ax.scatter([rho[p] for p in pairs], [agr[p] for p in pairs],
               s=26, facecolor="white", edgecolor=GREY, linewidth=0.9, zorder=3)
    hi = ["A/A_pocket", "A/B_z"]
    ax.scatter([rho[p] for p in hi], [agr[p] for p in hi],
               s=30, color=ANS, zorder=4)
    ax.plot([rho[hi[0]], rho[hi[1]]], [agr[hi[0]], agr[hi[1]]],
            color=ANS, lw=0.9, ls="--", zorder=2)
    lbl = {"A/A_pocket": ("A / A$_{\\rm pocket}$", 7, -3),
           "A/B_z": ("A / B$_z$", 7, -3),
           "A_pocket/B_z": ("A$_{\\rm pocket}$ / B$_z$", 7, 3)}
    for p, (t, dx, dy) in lbl.items():
        ax.annotate(t, (rho[p], agr[p]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=10, color=GREY)
    bx = ["A/B_x", "A_pocket/B_x", "B_z/B_x"]
    ax.annotate("three B$_x$ pairs\n(agreement identical)",
                (min(rho[p] for p in bx), agr[bx[0]]),
                textcoords="offset points", xytext=(-6, 30),
                fontsize=10, color=GREY, ha="left")
    ax.set_xlabel("Spearman $\\rho$ of $|\\mathcal{R}|$ (dimensionless)")
    ax.set_ylabel("classification agreement\n(fraction of conditions)")
    ax.set_xlim(0.82, 1.005)
    ax.set_ylim(0.40, 0.92)
    ax.set_title("(c) ranking against classification, $n{=}100$", loc="left")
    d_rho = abs(rho[hi[0]] - rho[hi[1]])
    d_agr = abs(agr[hi[0]] - agr[hi[1]])
    ax.text(0.8330, 0.418, f"$\\Delta\\rho$ = {d_rho:.3f},  {100*d_agr:.1f} pp",
            fontsize=10, color=ANS, ha="left", va="bottom")

    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.02)
    print("written:", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
