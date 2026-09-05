"""Regenerate Fig. 4 for the T0-C successor manuscript.

    python reproduce/t0/make_fig4.py data/t0c/analysis figures/t0/sensitivity_atlas.pdf

Pass the CANONICAL layer, data/t0c/analysis. Passing data/t0/analysis produces
a figure from the superseded pre-correction outputs; that has happened once
already, and the result was a panel (c) annotation reading 0.003 / 33.3 pp
under a caption reading 0.011 / 37.2. The script echoes the source path and the
values it drew, so the layer is visible in the build log.

Three panels, all from the analysis outputs:
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

from reproduce.figstyle import (apply as apply_style, COMPARISON_COLORS,
                                NEUTRAL, pair_colors, light_grid)

ANS, ANS_L = pair_colors("ansatz")
SEED, SEED_L = pair_colors("seed")
TEAL, TEAL_L = pair_colors("teal")
GREY = NEUTRAL["grey"]


def load(p):
    with open(p) as fh:
        return json.load(fh)


def main(src, out):
    apply_style()
    ph30 = load(f"{src}/posthoc_n30.json")["A_channel_three_state"]
    ph100 = load(f"{src}/posthoc_n100.json")["A_channel_three_state"]
    ho30 = load(f"{src}/holdout_n30.json")["cross_ansatz_transport"]
    cf100 = load(f"{src}/confirmation_n100.json")["cross_ansatz_transport"]
    lg100 = load(f"{src}/legacy_n100.json")

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5))
    fig.subplots_adjust(left=0.070, right=0.995, bottom=0.26, top=0.80, wspace=0.52)
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
    light_grid(ax, "y")
    ax.bar(x - w / 2, [g[1]["D_ansatz"] for g in groups], w,
           color=ANS, label="cross-ansatz")
    ax.bar(x + w / 2, [g[1]["D_seed"] for g in groups], w,
           color=SEED_L, edgecolor=SEED, linewidth=0.6, label="cross-seed")
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_xlabel("$n_{\\rm real}$", labelpad=1)
    ax.set_ylabel("three-state disagreement\n(fraction of comparisons)")
    ax.set_ylim(0, 0.60)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax.set_title("(a) channel-resolved\ndisagreement")
    ax.legend(loc="upper right")
    ax.axvline(1.5, color="0.88", lw=0.7, zorder=0)
    for xi, g in zip(x, groups):
        ax.text(xi, max(g[1]["D_ansatz"], g[1]["D_seed"]) + 0.014,
                f"{g[1]['ratio']:.2f}", ha="center", fontsize=8, color=GREY)

    # ---------------- (b) candidate transport ------------------------------
    ax = axes[1]
    k = np.arange(5)
    v30 = [ho30[str(i)] for i in k]
    v100 = [cf100[str(i)] for i in k]
    light_grid(ax, "y")
    ax.bar(k - w / 2, v30, w, color=TEAL, label="$n{=}30$")
    ax.bar(k + w / 2, v100, w, color=TEAL_L, edgecolor=TEAL, linewidth=0.6,
           label="$n{=}100$")
    ax.set_xticks(k)
    ax.set_xlabel("ans\u00e4tze confirming the candidate")
    ax.set_ylabel("conditions in $U^{\\rm disc}$ (count)")
    ax.set_ylim(0, 6.4)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_title("(b) candidate transport\nacross the family")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.03))
    ax.text(0.98, 0.62, f"$|U^{{\\rm disc}}|$ = {sum(v30)}, {sum(v100)}",
            transform=ax.transAxes, ha="right", fontsize=8, color=GREY)

    # ---------------- (c) ranking against classification -------------------
    ax = axes[2]
    rho = lg100["pairwise_rho"]
    agr = lg100["pairwise_agreement"]
    pairs = list(rho)
    light_grid(ax, "y")
    ax.scatter([rho[p] for p in pairs], [agr[p] for p in pairs],
               s=22, facecolor="white", edgecolor=GREY, linewidth=0.8, zorder=3)
    hi = ["A/A_pocket", "A/B_z"]
    ax.scatter([rho[p] for p in hi], [agr[p] for p in hi],
               s=26, color=ANS, zorder=4)
    ax.plot([rho[hi[0]], rho[hi[1]]], [agr[hi[0]], agr[hi[1]]],
            color=ANS, lw=0.9, ls="--", zorder=2)
    lbl = {"A/A_pocket": ("A / A$_{\\rm pocket}$", 7, -3),
           "A/B_z": ("A / B$_z$", 7, -3),
           "A_pocket/B_z": ("A$_{\\rm pocket}$ / B$_z$", 7, 3)}
    for p, (t, dx, dy) in lbl.items():
        ax.annotate(t, (rho[p], agr[p]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8, color=GREY)
    # The three B_x pairs sit together in the middle band. Under the archived
    # ordering their agreements happened to coincide exactly; under the
    # corrected one they do not, so the label states the grouping and not an
    # identity that is no longer true.
    bx = ["A/B_x", "A_pocket/B_x", "B_z/B_x"]
    ax.annotate("three B$_x$ pairs",
                (min(rho[p] for p in bx), min(agr[p] for p in bx)),
                textcoords="offset points", xytext=(-6, 30),
                fontsize=8, color=GREY, ha="left")
    ax.set_xlabel("Spearman $\\rho$ of $|\\mathcal{R}|$ (dimensionless)")
    ax.set_ylabel("classification agreement\n(fraction of conditions)")
    ax.set_xlim(0.82, 1.005)
    ax.set_ylim(0.40, 0.92)
    ax.set_title("(c) ranking against\nclassification, $n{=}100$")
    d_rho = abs(rho[hi[0]] - rho[hi[1]])
    d_agr = abs(agr[hi[0]] - agr[hi[1]])
    # axes coordinates, bottom left: in data coordinates this ran straight
    # through the A/A_pocket marker and its label
    ax.text(0.02, 0.99, f"$\\Delta\\rho$ = {d_rho:.3f},  {100*d_agr:.1f} pp",
            transform=ax.transAxes, fontsize=8.5, color=ANS,
            ha="left", va="top")

    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.02)
    print("written:", out)
    print(f"  source layer      : {src}")
    print(f"  panel (a) ratios  : "
          + ", ".join(f"{g[1]['ratio']:.2f}" for g in groups))
    print(f"  panel (c) d_rho   : {d_rho:.3f}")
    print(f"  panel (c) gap     : {100*d_agr:.1f} pp")
    print(f"  B_x pair agreement: "
          + ", ".join(f"{agr[p]:.3f}" for p in bx))
    print("  Check these against the caption before committing the figure.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
