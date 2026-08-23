"""Regenerate supplement Fig. S5 for the T0 successor manuscript.

Left  : the ranking/classification decoupling survives every alternative
        ranked quantity except the signed dephasing response.
Right : the cross-ansatz versus cross-seed contrast under union conditioning,
        which replaces the r31 floor sweep as the floor-sensitivity check.

All values read from the closed T0 analysis JSON; no pickle access.
"""
import json
import sys
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np

ANS = "#d66014"
SEED = "#1f77b4"
GREY = "#555555"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 10.5,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
})

QUANT = [
    ("hypot_raw", "released\n$|\\mathcal{R}|$"),
    ("norm2", "floor-\nnormalized"),
    ("abs_dP_v", "$|\\Delta P_v|$\nonly"),
    ("abs_dchi_phi", "$|\\Delta\\chi_\\phi|$\nonly"),
    ("dP_v", "$\\Delta P_v$\nsigned"),
    ("dchi_phi", "$\\Delta\\chi_\\phi$\nsigned"),
]


def load(p):
    with open(p) as fh:
        return json.load(fh)


def main(src, out):
    r30 = load(f"{src}/ranking_n30.json")
    r100 = load(f"{src}/ranking_n100.json")
    att = load(f"{src}/attribution_n100.json")["by_n"]
    ns = [5, 10, 20, 30, 40, 60, 80, 100]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.26, top=0.88, wspace=0.30)

    # ---------------- left: counterexample gap by ranked quantity -----------
    ax = axes[0]
    x = np.arange(len(QUANT))
    w = 0.36
    g30 = [r30["by_quantity"][k]["counterexample_gap"] for k, _ in QUANT]
    g100 = [r100["by_quantity"][k]["counterexample_gap"] for k, _ in QUANT]
    ax.bar(x - w / 2, g30, w, color=GREY, label="$n{=}30$")
    ax.bar(x + w / 2, g100, w, color="white", edgecolor=GREY, hatch="///",
           linewidth=0.8, label="$n{=}100$")

    a30 = r30["agreement"]
    a100 = r100["agreement"]
    cg30 = abs(a30["A/B_z"] - a30["A/A_pocket"])
    cg100 = abs(a100["A/B_z"] - a100["A/A_pocket"])
    ax.axhline(cg100, color=ANS, lw=0.9, ls="--")
    ax.text(len(QUANT) - 0.45, cg100 - 0.045,
            f"classification-agreement gap {cg100:.3f}",
            color=ANS, fontsize=10, ha="right")
    ax.axhline(cg30, color=ANS, lw=0.7, ls=":", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in QUANT])
    ax.set_ylabel("A/A$_{\\rm pocket}$ against A/B$_z$ gap")
    ax.set_ylim(0, 0.42)
    ax.set_title("(a) ranked quantity against classification", loc="left")
    ax.legend(frameon=False, loc="upper left", handlelength=1.4)

    # ---------------- right: union-conditioned sensitivity ------------------
    ax = axes[1]
    ua = [att[str(n)]["union_conditioned_sensitivity"]["ansatz"]["D_ansatz"] for n in ns]
    us = [att[str(n)]["union_conditioned_sensitivity"]["seed"]["D_seed"] for n in ns]
    da = [att[str(n)]["D_ansatz"] for n in ns]
    ds = [att[str(n)]["D_seed"] for n in ns]

    ax.plot(ns, ua, "-o", color=ANS, ms=3.4, lw=1.0,
            label="cross-ansatz, union-conditioned")
    ax.plot(ns, us, "--s", color=SEED, ms=3.4, lw=1.0, mfc="white",
            label="cross-seed, union-conditioned")
    ax.plot(ns, da, "-", color=ANS, lw=0.8, alpha=0.35)
    ax.plot(ns, ds, "--", color=SEED, lw=0.8, alpha=0.35)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) if n not in (40, 80) else "" for n in ns])
    ax.minorticks_off()
    ax.set_xlabel("realizations $n_{\\rm real}$ (log scale)")
    ax.set_ylabel("label disagreement")
    ax.set_ylim(0.25, 0.72)
    ax.set_title("(b) union-conditioned: $\\geq$1 two-channel label", loc="left")
    ax.legend(frameon=False, loc="upper right", handlelength=1.6)
    ax.text(0.03, 0.06, "faint lines: unconditioned estimands",
            transform=ax.transAxes, fontsize=10, color=GREY)

    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.02)
    print("written:", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
