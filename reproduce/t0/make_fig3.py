#!/usr/bin/env python3
"""Fig. 3 from data: the quadrant map with the counterexample ansatz pair.

    python reproduce/t0/make_fig3.py data/t0c/responses_n100.csv \
        figures/t0/quadrant_data.pdf

What it draws
-------------
Two panels, both at n_real = 100 and both on the same axes, showing every
condition that the two ans\"atze of a pair label differently, with an arrow from
the first ansatz's response to the second's.

    (a) A -> A_pocket   rank correlation 0.967, classification agreement 0.467
    (b) A -> B_z        rank correlation 0.978, classification agreement 0.839

That is the counterexample of the Results section drawn rather than asserted.
The two pairs have almost the same rank correlation and very different
classification agreement, and the difference is visible as the number and the
length of the arrows that cross a quadrant boundary.

Two restrictions, both stated on the panels.

Only relabelled conditions are drawn: an arrow that starts and ends in the same
category shows nothing, and the agreeing majority would bury the figure under
the below-threshold cluster.

Of those, only comparisons in which at least one member carries a two-channel
quadrant label are drawn. This is the union restriction already registered as a
sensitivity analysis in the Methods, and it is the set a design decision acts
on. It also thins panel (a) from 59 arrows to 35, which is the difference
between a readable panel and a tangle.

The tighter "both members two-channel" restriction is NOT used. It leaves 4
comparisons in panel (a) against 6 in panel (b), which reverses the contrast the
figure exists to show; a restriction chosen for legibility must not change the
direction of the result. Both counts are printed so the selection is visible.

Axes are symmetric-log. The effect-size floors are 1e-4 and 1e-3 while the
responses reach a few times 1e-2, so a linear axis puts the entire
below-threshold box inside one plotting pixel. The linear region of the symlog
axis is set to the floor, which makes the box exactly the linear core and shows
how small the floors are against the spread.
"""
import csv
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

FLOOR_P, FLOOR_CHI = 1e-4, 1e-3     # operational effect-size floors, Sec. III
BLOCK = 1                            # one seed block; see the caption note

C_A = "#1f6fb4"
C_AP = "#4fb3e8"
C_BZ = "#2f8f8f"
C_GREY = "#555555"

PAIRS = [("A", "A_pocket", C_A, C_AP, r"(a) A $\to$ A$_{\rm pocket}$"),
         ("A", "B_z", C_A, C_BZ, r"(b) A $\to$ B$_z$")]


def load(path):
    """Every block, keyed (ansatz, block, condition). Duplicates are refused."""
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            cond = (r["case"], r["v_m_per_s"], r["lambda_ueV"], r["geometry"])
            key = (r["ansatz"], int(r["block"]), cond)
            if key in out:
                raise SystemExit(f"BLOCKED: duplicate row for {key} in {path}; "
                                 "a dict would silently keep the last one")
            out[key] = (float(r["dP_v"]), float(r["dchi_phi"]))
    return out


def check_classifier(data, analysis):
    """Verify the figure's floor rule against the shipped analysis.

    The figure applies its own three-line threshold rule. If that rule ever
    drifts from the one the analysis layer used, the panels would show
    categories the paper does not report, and nothing would say so. Recomputing
    the six pairwise agreements over all five blocks and comparing them against
    legacy_n100.json catches exactly that: the numbers match only if the rule
    matches.
    """
    p = pathlib.Path(analysis) / "legacy_n100.json"
    if not p.exists():
        print(f"  classifier check SKIPPED: {p} not found")
        return False
    ref = json.loads(p.read_text()).get("pairwise_agreement")
    if not ref:
        print("  classifier check SKIPPED: no pairwise_agreement in "
              "legacy_n100.json")
        return False
    blocks = sorted({b for (_a, b, _c) in data})
    conds = sorted({c for (_a, _b, c) in data})
    worst = 0.0
    for pair, val in sorted(ref.items()):
        a1, a2 = pair.split("/")
        same = tot = 0
        for b in blocks:
            for c in conds:
                k1, k2 = (a1, b, c), (a2, b, c)
                if k1 not in data or k2 not in data:
                    continue
                tot += 1
                same += label(*data[k1]) == label(*data[k2])
        got = same / tot if tot else float("nan")
        worst = max(worst, abs(got - val))
        print(f"  agreement {pair:16s} figure {got:.6f}  analysis {val:.6f}")
    # One condition out of 540 comparisons shifts an agreement by 1/540, so
    # 1e-6 still catches a single flipped label while tolerating the rounding
    # in a hand-built fixture.
    if worst > 1e-6:
        raise SystemExit(f"BLOCKED: the figure's floor rule disagrees with the "
                         f"analysis layer by up to {worst:.2e}, which is more "
                         f"than one flipped condition ({1/540:.2e})")
    print(f"  classifier check PASS, worst difference {worst:.1e}")
    return True


def label(dp, dchi):
    """The nine-category label of Sec. III."""
    p = 0 if abs(dp) < FLOOR_P else (1 if dp > 0 else -1)
    c = 0 if abs(dchi) < FLOOR_CHI else (1 if dchi > 0 else -1)
    return (p, c)


def two_channel(lab):
    """True for the four decision-facing quadrants, both channels resolved."""
    return lab[0] != 0 and lab[1] != 0


def panel(ax, data, a1, a2, c1, c2, title):
    conds = sorted({c for (a, b, c) in data if a == a1 and b == BLOCK})
    drawn = relabelled = 0
    for c in conds:
        if (a1, BLOCK, c) not in data or (a2, BLOCK, c) not in data:
            continue
        p1, p2 = data[(a1, BLOCK, c)], data[(a2, BLOCK, c)]
        l1, l2 = label(*p1), label(*p2)
        if l1 == l2:
            continue
        relabelled += 1
        if not (two_channel(l1) or two_channel(l2)):
            continue
        drawn += 1
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle="->", color="0.70", lw=0.6,
                                    shrinkA=1.6, shrinkB=1.6, alpha=0.9))
        ax.plot(*p1, "o", color=c1, ms=2.8, zorder=4)
        ax.plot(*p2, "s", color=c2, ms=2.8, zorder=4)

    ax.axhline(0, color="k", lw=0.7, zorder=1)
    ax.axvline(0, color="k", lw=0.7, zorder=1)
    ax.add_patch(Rectangle((-FLOOR_P, -FLOOR_CHI), 2 * FLOOR_P, 2 * FLOOR_CHI,
                           facecolor="0.75", alpha=0.45, edgecolor="none",
                           zorder=2))
    ax.set_xscale("symlog", linthresh=FLOOR_P)
    ax.set_yscale("symlog", linthresh=FLOOR_CHI)
    ax._data_lim = max(
        [abs(v) for c in conds
         if (a1, BLOCK, c) in data and (a2, BLOCK, c) in data
         for pt in (data[(a1, BLOCK, c)], data[(a2, BLOCK, c)])
         for v in pt] or [1e-2])
    ax.set_xlabel(r"$\Delta P_v$")
    ax.set_title(title, loc="left", pad=4)
    ax.text(0.98, 0.03,
            f"{drawn} drawn of {relabelled} relabelled",
            transform=ax.transAxes, ha="right", fontsize=7, color=C_GREY)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return drawn, relabelled


def main(src, out, analysis=None):
    data = load(src)
    if analysis:
        check_classifier(data, analysis)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.linewidth": 0.7,
        "xtick.direction": "in", "ytick.direction": "in",
        "mathtext.fontset": "dejavusans",
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.1), sharey=True)
    counts = []
    for ax, (a1, a2, c1, c2, title) in zip(axes, PAIRS):
        counts.append(panel(ax, data, a1, a2, c1, c2, title))
    # one limit for both panels so the two are directly comparable, taken
    # from the data rather than fixed, so no arrow leaves the frame
    lim = 1.6 * max(ax._data_lim for ax in axes)
    for ax in axes:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    axes[0].set_ylabel(r"$\Delta\chi_\phi$")

    # quadrant names, placed once, outside the data
    for ax in axes:
        ax.text(0.03, 0.97, "spin-trade", transform=ax.transAxes, fontsize=7,
                color=C_GREY, va="top")
        ax.text(0.97, 0.97, "both-worsen", transform=ax.transAxes, fontsize=7,
                color=C_GREY, ha="right", va="top")
        ax.text(0.03, 0.10, "robust", transform=ax.transAxes, fontsize=7,
                color=C_GREY)
        ax.text(0.97, 0.10, "valley-trade", transform=ax.transAxes, fontsize=7,
                color=C_GREY, ha="right")

    handles = [
        plt.Line2D([], [], ls="", marker="o", color=C_A, ms=3.5, label="A"),
        plt.Line2D([], [], ls="", marker="s", color=C_AP, ms=3.5,
                   label=r"A$_{\rm pocket}$"),
        plt.Line2D([], [], ls="", marker="s", color=C_BZ, ms=3.5,
                   label=r"B$_z$"),
        plt.Line2D([], [], color="0.75", lw=0.9,
                   label="same condition, both ans\u00e4tze"),
        plt.Line2D([], [], ls="", marker="s", color="0.75", ms=5,
                   label="below-threshold zone"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.055),
               ncol=5, frameon=False, handlelength=1.6, columnspacing=1.3)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.92, bottom=0.24,
                        wspace=0.08)
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=400,
                bbox_inches="tight", pad_inches=0.02)
    print("written:", out)
    print(f"  source        : {src}")
    print(f"  seed block    : {BLOCK}")
    print(f"  floors        : |dP_v| >= {FLOOR_P}, |dchi_phi| >= {FLOOR_CHI}")
    print(f"  drawn         : (a) {counts[0][0]}, (b) {counts[1][0]}")
    print(f"  relabelled    : (a) {counts[0][1]}, (b) {counts[1][1]} of 108")
    print("  Drawn is the registered union restriction: at least one member")
    print("  carries a two-channel quadrant label.")
    print("  A/A_pocket has the lowest classification agreement in the family "
          "and A/B_z the highest,")
    print("  while their rank correlations differ by 0.011. That contrast is "
          "what this figure shows.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], pathlib.Path(sys.argv[2]),
                          sys.argv[3] if len(sys.argv) == 4 else None))
