#!/usr/bin/env python3
"""Regenerate supplementary Fig. S5 from the corrected analysis layer.

    python reproduce/t0/make_figS5.py data/t0c/analysis \
        figures/t0/robustness_checks.pdf

Two panels, both post-hoc diagnostics of Sec. S3:

  (a) the A/A_pocket against A/B_z gap under six alternative ranked
      quantities, at both endpoints, against the classification-agreement gap
      those quantities would have to reproduce;
  (b) both estimands restricted to comparisons in which at least one member
      carries a two-channel quadrant label, across the eight registered
      checkpoints, with the unconditioned estimands drawn faintly behind.

Why this file was rewritten
---------------------------
`collect_figures.sh` required `figures/t0/robustness_checks.pdf` and named
`make_figS1.py` as its producer, but no such script was present in the tree.
A shipped figure with no shipped producer is the failure `RELEASE_OPS.md`
Sec. 1 exists to prevent, and it is the third instance in this project after
`cmd_candidate_delta` and its recipe. This replacement reads only the analysis
JSON, so the figure regenerates from the public tree.

Pass the CANONICAL layer, `data/t0c/analysis`. The script prints every value it
draws so the layer and the numbers are visible in the build log; check them
against the supplement text before committing the figure.
"""
import json
import pathlib
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

# (JSON key, label). Horizontal bars, so the labels are ordinary left-aligned
# text and need no line breaks. Ordered as the supplement table is.
QUANTITIES = [
    ("hypot_raw",    r"released $|\mathcal{R}|$"),
    ("norm2",        r"floor-normalized"),
    ("abs_dP_v",     r"$|\Delta P_v|$ only"),
    ("abs_dchi_phi", r"$|\Delta\chi_\phi|$ only"),
    ("dP_v",         r"$\Delta P_v$, signed"),
    ("dchi_phi",     r"$\Delta\chi_\phi$, signed"),
]



def load(src, stem):
    p = pathlib.Path(src) / f"{stem}.json"
    if not p.exists():
        print(f"BLOCKED: {p} not found", file=sys.stderr)
        raise SystemExit(2)
    return json.loads(p.read_text())


def rate(v, depth=0):
    """A disagreement rate inside an entry, however the entry is nested.

    The union-conditioned record holds the rate beside a pair count, and the
    key names have changed once already, so take the one value in (0,1) that
    is not an integer count rather than assuming a shape.
    """
    if isinstance(v, float) and 0.0 < v < 1.0:
        return v
    if isinstance(v, dict) and depth < 2:
        for vv in v.values():
            r = rate(vv, depth + 1)
            if r is not None:
                return r
    return None


def union_pair(block):
    ucs = block.get("union_conditioned_sensitivity")
    if not isinstance(ucs, dict):
        return None, None
    a = s = None
    for k, v in ucs.items():
        if "ansatz" in k.lower() and a is None:
            a = rate(v)
        elif "seed" in k.lower() and s is None:
            s = rate(v)
    return a, s


def main(src, out):
    apply_style()
    r30, r100 = load(src, "ranking_n30"), load(src, "ranking_n100")
    att = load(src, "attribution_n100")["by_n"]

    def agr_gap(r):
        a = r["agreement"]
        return a["A/B_z"] - a["A/A_pocket"]

    gap30 = [r30["by_quantity"][q]["counterexample_gap"] for q, _ in QUANTITIES]
    gap100 = [r100["by_quantity"][q]["counterexample_gap"] for q, _ in QUANTITIES]
    cls30, cls100 = agr_gap(r30), agr_gap(r100)

    ns = sorted(att, key=int)
    u_a, u_s, d_a, d_s, kept = [], [], [], [], []
    for n in ns:
        a, s = union_pair(att[n])
        if a is None or s is None:
            continue
        kept.append(int(n))
        u_a.append(a)
        u_s.append(s)
        d_a.append(att[n]["D_ansatz"])
        d_s.append(att[n]["D_seed"])
    if len(kept) != len(ns):
        print(f"BLOCKED: union_conditioned_sensitivity present at "
              f"{len(kept)} of {len(ns)} checkpoints; panel (b) would be "
              f"incomplete and silently so", file=sys.stderr)
        raise SystemExit(2)

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.7))
    fig.subplots_adjust(left=0.150, right=0.985, bottom=0.22, top=0.78,
                        wspace=0.44)

    # ---------------- (a) ranked quantity against classification -----------
    # Horizontal bars. Every gap is well under a tenth of the classification
    # gap it would have to reproduce, so a vertical axis spends most of its
    # height empty and the six two-line category labels crowd the base. Laid
    # out this way the labels are ordinary text and the reference lines are
    # vertical, which is also how the eye reads "how far short".
    ax = axes[0]
    y = np.arange(len(QUANTITIES))[::-1]
    h = 0.36
    light_grid(ax, "x")
    ax.barh(y + h / 2, gap30, h, color=TEAL, label="$n = 30$")
    ax.barh(y - h / 2, gap100, h, color=TEAL_L, edgecolor=TEAL, linewidth=0.6,
            label="$n = 100$")
    ax.axvline(cls100, color=ANS, ls="--", lw=1.0)
    ax.axvline(cls30, color=ANS, ls=":", lw=1.0)
    # inside the axes, in the empty band between the bars and the reference
    # lines; above the panel it collided with the title
    # direct labels on the two reference lines: the dotted/dashed distinction
    # is not decodable from the figure alone
    # The two reference lines sit 0.03 apart on a 0.4 axis, so labels placed
    # above them always collide. Run each label up its own line instead.
    ax.text(cls30 * 0.955, 0.50, "classification gap",
            transform=ax.get_xaxis_transform(), ha="right", va="center",
            fontsize=8, color=ANS)
    for val, lab in ((cls30, f"$n{{=}}30$, {cls30:.3f}"),
                     (cls100, f"$n{{=}}100$, {cls100:.3f}")):
        ax.text(val, 0.50, lab, transform=ax.get_xaxis_transform(),
                ha="center", va="center", rotation=90, fontsize=7,
                color=ANS, backgroundcolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([l for _, l in QUANTITIES])
    ax.set_xlabel("A/A$_{\\rm pocket}$ against A/B$_z$ gap")
    ax.set_xlim(0, max(cls30, cls100) * 1.06)
    ax.set_ylim(-0.7, len(QUANTITIES) - 0.3)
    ax.set_title("(a) ranked quantity\nagainst classification")
    ax.legend(loc="lower right")

    # ---------------- (b) union-conditioned trajectory ---------------------
    ax = axes[1]
    ax.plot(kept, d_a, color=ANS, lw=1.0, alpha=0.28)
    ax.plot(kept, d_s, color=SEED, lw=1.0, ls="--", alpha=0.28)
    ax.plot(kept, u_a, "o-", color=ANS, ms=4, lw=1.4, label="cross-ansatz")
    ax.plot(kept, u_s, "s--", color=SEED, ms=4, lw=1.4, mfc="white",
            label="cross-seed")
    ax.set_xscale("log")
    shown = [k for k in kept if k in (5, 10, 20, 30, 60, 100)]
    ax.set_xticks(shown)
    ax.set_xticklabels([str(k) for k in shown])
    ax.minorticks_off()
    ax.set_xlabel("realizations $n_{\\rm real}$ (log scale)")
    ax.set_ylabel("label disagreement")
    ax.set_title("(b) union-conditioned\nlabel disagreement")
    light_grid(ax, "y")
    ax.text(0.03, 0.04, "faint: unconditioned", transform=ax.transAxes,
            fontsize=8, color=GREY)
    ax.legend(loc="upper right")

    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.02)
    print("written:", out)
    print(f"  source layer          : {src}")
    print(f"  classification gap    : n=30 {cls30:.4f}, n=100 {cls100:.4f}")
    print("  counterexample gap per ranked quantity (n=30, n=100):")
    for (q, _), a, b in zip(QUANTITIES, gap30, gap100):
        print(f"    {q:14s} {a:.4f}  {b:.4f}")
    print(f"  union-conditioned n=100: ansatz {u_a[-1]:.4f}, seed {u_s[-1]:.4f}")
    print(f"  union-conditioned ansatz range: "
          f"{min(u_a):.4f}-{max(u_a):.4f} over {len(kept)} checkpoints")
    print(f"  union-conditioned seed  : {u_s[0]:.4f} -> {u_s[-1]:.4f}")
    print("  Check these against the supplement text before committing.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
