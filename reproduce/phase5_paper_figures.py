"""Every manuscript and supplement figure produced by Phase 5.

    PYTHONPATH=. python reproduce/phase5_paper_figures.py --figures all \
        --ez-convention total-local --profile-norm prefactor \
        --responses data/t0c/responses_n100.csv \
        --analysis  data/t0c/analysis

    --figures {all,2,3,4,S5}   one figure at a time, for testing

    Fig. 2   figures/phase5/phase5_coupling_profiles<suffix>.pdf
    Fig. 3   figures/t0/quadrant_data.pdf
    Fig. 4   figures/t0/sensitivity_atlas.pdf
    Fig. S5  figures/t0/robustness_checks.pdf

Design notes that are not cosmetic
----------------------------------
The simulator imports live inside the Fig. 2 path. Fig. 2 draws the real
coupling profiles and needs geometry, the kernel module and `constants`, which
pull in jax and scipy. Figs. 3, 4 and S5 read JSON and CSV and nothing else,
and `--figures 4` must keep working on a machine that cannot import jax --
that is what the public claim "reproducible from the shipped analysis outputs"
means in terms of an actual dependency graph.

The classifier is imported from the canonical analyser, not reimplemented. A
second copy of the paper's central estimand living in plotting code is free to
drift from the one the analysis used, and nothing would report it. Fig. 3 goes
further and recomputes the six pairwise agreements from the CSV, requiring them
to equal `legacy_n100.json`; they match only if the rule matches.

Every caption-facing number is computed at run time and printed. None is
hardcoded here, in a docstring or in an annotation. That rule exists because
this file's predecessor carried a hardcoded "agreement identical" that the
corrected analysis layer contradicted, and re-running alone would not have
caught it.

The bare invocation defaults to --ez-convention legacy-50ueV, which reproduces
the archival r31 Fig. 2, NOT the current manuscript figure.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# matplotlib only
from reproduce.figstyle import (apply as apply_style, ANSATZ_COLORS,
                                COMPARISON_COLORS, NEUTRAL, pair_colors,
                                light_grid)
# numpy and stdlib only -- no simulator, no jax
from reproduce.t0.t0_step4_analysis import FLOOR_P, FLOOR_CHI, classify

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_PHASE5 = ROOT / "figures" / "phase5"
FIG_T0 = ROOT / "figures" / "t0"

e_C = 1.602176634e-19

# The registered checkpoint set. Fig. S5 refuses to draw without all eight.
EXPECTED_NS = ("5", "10", "20", "30", "40", "60", "80", "100")

ANS, ANS_L = pair_colors("ansatz")
SEED, SEED_L = pair_colors("seed")
TEAL, TEAL_L = pair_colors("teal")
GREY = NEUTRAL["grey"]
C_GREY = GREY

BLOCK = 1                      # the seed block Fig. 3 draws
FLOOR_CHI_ = FLOOR_CHI         # re-exported for readability below


def _load(src, stem):
    p = pathlib.Path(src) / f"{stem}.json"
    if not p.exists():
        raise SystemExit(f"BLOCKED: {p} not found")
    return json.loads(p.read_text())


# ===========================================================================
# Fig. 2 -- coupling profiles
# ===========================================================================
def _build_ff():
    """Baseline geometry, same as step 5 / Phase 4.6.

    The geometry and profile imports live here, not at module scope. Fig. 2 is
    the only panel that needs the simulator; Figs. 3, 4 and S5 read JSON and CSV
    and nothing else. Keeping these local means `--figures 4` regenerates with
    matplotlib and numpy alone, on a machine with no jax.
    """
    from geometry.periodic_array import PeriodicPrismArray
    from geometry.fourier_field import fit_from_prism_array
    a = 150e-9
    arr = PeriodicPrismArray(
        period_a=a, half_x=25e-9, half_y=25e-9, half_z=15e-9,
        cz=15e-9, N_periods_each_side=4, Ms=1.4e6,
    )
    return fit_from_prism_array(arr, -50e-9, N_harm=3), a


def Ev_of_x(x, pocket_x_center, pocket_width, Ev_baseline, depth):
    """Position-dependent valley detuning: baseline minus a Gaussian pocket."""
    return Ev_baseline - depth * np.exp(
        -(x - pocket_x_center) ** 2 / (2 * pocket_width ** 2))


def _fig_tag(ez_convention, profile_norm):
    if ez_convention == "legacy-50ueV" and profile_norm == "prefactor":
        return ""
    return f"__ez-{ez_convention}__norm-{profile_norm}"


def _EZ_of(x, ff, ez_convention):
    if ez_convention == "legacy-50ueV":
        return 50e-6 * e_C
    from constants import g_Si, mu_B, Defaults
    Bz_arr = np.asarray(ff.B_z(x))
    B_off = Defaults.B_ext_T if ez_convention.startswith("total") else 0.0
    if ez_convention == "total-local":
        return g_Si * mu_B * (B_off + Bz_arr)
    return g_Si * mu_B * (B_off + float(np.mean(Bz_arr)))


def _one_case(ff, xc, ez_convention, profile_norm, pocket_width, sigma_E):
    """eps_v(x), E_Z(x) and the four normalized profiles for one placement."""
    Ev_baseline = 100e-6 * e_C
    depth = 95e-6 * e_C                       # pocket floor ~5 ueV
    x = np.linspace(xc - 4 * pocket_width, xc + 4 * pocket_width, 800)
    E_Z = _EZ_of(x, ff, ez_convention)
    Ev_x = Ev_of_x(x, xc, pocket_width, Ev_baseline, depth)
    from reproduce.phase4p6_crossterm import lambda_sv_profile
    prof = {}
    for m in ("A", "A_pocket", "B_z", "B_x"):
        y = lambda_sv_profile(
            x, ff, m, 1.0, profile_norm=profile_norm,
            pocket_x_center=xc, sigma_lambda=pocket_width,
            eps_v_x=Ev_x, E_Z=E_Z, sigma_E=sigma_E)
        prof[m] = np.asarray(y) / max(float(np.max(np.abs(y))), 1e-30)
    return x, Ev_x, np.broadcast_to(np.asarray(E_Z), x.shape), prof


def make_fig2(ez_convention="legacy-50ueV", profile_norm="prefactor"):
    apply_style()
    ff, _ = _build_ff()
    pocket_width = 30e-9
    sigma_E = 10e-6 * e_C

    cases = [("(a) centred pocket, $x_c = 0$", 0.0),
             ("(b) edge pocket, $x_c = 25$ nm", 25e-9)]

    fig, axes = plt.subplots(
        2, 2, figsize=(6.9, 3.5), sharey="row",
        gridspec_kw={"height_ratios": [1.0, 1.6], "hspace": 0.18,
                     "wspace": 0.12})

    styles = [
        ("A",        ANSATZ_COLORS["A"],  "-",  r"A: $\propto|\partial_x B_z|$"),
        ("A_pocket", ANSATZ_COLORS["A_pocket"], "--", r"A$_{\rm pocket}$: A $\times$ pocket"),
        ("B_z",      ANSATZ_COLORS["B_z"], "-.", r"B$_z$: A $\times$ resonance"),
        ("B_x",      ANSATZ_COLORS["B_x"], ":",  r"B$_x$: $\propto|\partial_x B_x|\times$ resonance"),
    ]

    for col, (title, xc) in enumerate(cases):
        x, Ev_x, EZ_x, prof = _one_case(
            ff, xc, ez_convention, profile_norm, pocket_width, sigma_E)
        xnm = x * 1e9
        ev = Ev_x / e_C * 1e6
        ez = EZ_x / e_C * 1e6

        ax0, ax1 = axes[0, col], axes[1, col]

        # shade the proxy-crossing windows so the reader sees where the B
        # profiles are localized, and sees that the edge case moves them
        cross = np.where(np.diff(np.sign(ev - ez)))[0]
        for i in cross:
            ax0.axvline(xnm[i], color=NEUTRAL["EZ"], lw=0.6, ls=":", alpha=0.8)
            ax1.axvline(xnm[i], color=NEUTRAL["EZ"], lw=0.6, ls=":", alpha=0.8)
        # Label the verticals inside the figure. Unlabelled dotted lines around
        # the pocket read as a pocket extent, which is not what they are.
        if len(cross):
            ax0.annotate(r"$\delta = 0$", (xnm[cross[0]], 104),
                         textcoords="offset points", xytext=(3, 0),
                         fontsize=7, color=NEUTRAL["EZ"], va="center")

        ax0.plot(xnm, ev, color=NEUTRAL["eps"], lw=1.4, label=r"$\varepsilon_v(x)$")
        ax0.plot(xnm, ez, color=NEUTRAL["EZ"], ls="--", lw=1.1,
                 label=r"$E_Z(x)$ (proxy resonance)")
        ax0.set_ylim(0, 112)
        ax0.set_yticks([0, 50, 100])
        ax0.set_title(title, loc="left", pad=16)
        ax0.axvline(xc * 1e9, color="0.6", lw=0.7)

        for m, c, ls, _lab in styles:
            ax1.plot(xnm, prof[m], color=c, ls=ls, lw=1.3)
        ax1.set_ylim(-0.04, 1.10)
        ax1.set_yticks([0.0, 0.5, 1.0])
        ax1.set_xlabel("position along shuttling path $x$ (nm)")
        ax1.axvline(xc * 1e9, color="0.6", lw=0.7)

        if col == 0:
            ax0.set_ylabel(r"$\varepsilon_v$ ($\mu$eV)")
            ax1.set_ylabel(r"normalized $\lambda_{sv}(x)$")

    # legends outside the axes: detuning legend above the top row, profile
    # legend below the bottom row. Nothing overlaps a curve.
    h0, l0 = axes[0, 0].get_legend_handles_labels()
    fig.legend(h0, l0, loc="lower center", bbox_to_anchor=(0.5, 0.935),
               ncol=2, frameon=False, handlelength=1.8, columnspacing=1.6)

    handles = [plt.Line2D([], [], color=c, ls=ls, lw=1.3)
               for _m, c, ls, _l in styles]
    labels = [l for _m, _c, _ls, l in styles]
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.075), ncol=4, frameon=False,
               handlelength=2.0, columnspacing=1.4)

    fig.subplots_adjust(left=0.075, right=0.995, top=0.845, bottom=0.215)

    FIG_PHASE5.mkdir(parents=True, exist_ok=True)
    out = FIG_PHASE5 / "phase5_coupling_profiles.png"
    out = out.with_name(out.stem + _fig_tag(ez_convention, profile_norm) + out.suffix)
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved: {out} (+ .pdf)")
    for col, (title, xc) in enumerate(cases):
        x, Ev_x, EZ_x, prof = _one_case(
            ff, xc, ez_convention, profile_norm, pocket_width, sigma_E)
        xnm = x * 1e9
        print(f"  {title}")
        for m in prof:
            print(f"    {m:9s} peak at x = {xnm[int(np.argmax(prof[m]))]:+7.1f} nm")


# ===========================================================================
# Fig. 3 -- quadrant classification from data
# ===========================================================================

PAIRS = [("A", "A_pocket", ANSATZ_COLORS["A"], ANSATZ_COLORS["A_pocket"],
          r"(a) A $\to$ A$_{\rm pocket}$"),
         ("A", "B_z", ANSATZ_COLORS["A"], ANSATZ_COLORS["B_z"],
          r"(b) A $\to$ B$_z$")]

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
        if tot != 540:
            raise SystemExit(f"BLOCKED: {pair} has {tot} comparisons, expected "
                             f"540 (5 blocks x 108 conditions); rows are "
                             f"missing and a `continue` would hide it")
        got = same / tot
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


# The four decision-facing quadrant labels, as named by the analysis layer.
TWO_CHANNEL = {"robust", "both_worsen", "trade_Pimp_chiwor", "trade_Pwor_chiimp"}


def label(dp, dchi):
    """The nine-category label, from the canonical analyser.

    Imported rather than reimplemented: a second copy of the paper's central
    classifier living in plotting code is free to drift from the one the
    analysis used, and nothing would say so.
    """
    return classify(dp, dchi)


def two_channel(lab):
    return lab in TWO_CHANNEL


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


def make_fig3(src, analysis, out):
    apply_style()
    data = load(src)
    check_classifier(data, analysis)
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
        plt.Line2D([], [], ls="", marker="o", color=ANSATZ_COLORS["A"], ms=3.5, label="A"),
        plt.Line2D([], [], ls="", marker="s", color=ANSATZ_COLORS["A_pocket"], ms=3.5,
                   label=r"A$_{\rm pocket}$"),
        plt.Line2D([], [], ls="", marker="s", color=ANSATZ_COLORS["B_z"], ms=3.5,
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
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
    fig.savefig(out.with_suffix(".png"), dpi=400)
    print("written:", out)
    print(f"  source        : {src}")
    print(f"  seed block    : {BLOCK}")
    print(f"  floors        : |dP_v| >= {FLOOR_P}, |dchi_phi| >= {FLOOR_CHI}")
    print(f"  drawn         : (a) {counts[0][0]}, (b) {counts[1][0]}")
    print(f"  relabelled    : (a) {counts[0][1]}, (b) {counts[1][1]} of 108")
    print("  Drawn is the registered union restriction: at least one member")
    print("  carries a two-channel quadrant label.")
    lg = json.loads((pathlib.Path(analysis) / "legacy_n100.json").read_text())
    rho, agr = lg["pairwise_rho"], lg["pairwise_agreement"]
    print("  five-block aggregate, from legacy_n100.json:")
    for pair in ("A/A_pocket", "A/B_z"):
        print(f"    {pair:12s} rho {rho[pair]:.4f}  agreement {agr[pair]:.4f}")
    print(f"    rank-correlation gap {abs(rho['A/B_z']-rho['A/A_pocket']):.4f}, "
          f"agreement gap {100*(agr['A/B_z']-agr['A/A_pocket']):.1f} pp")
    print("  The panels show block %d; the aggregate above is what the caption "
          "quotes." % BLOCK)
    return 0


# ===========================================================================
# Fig. 4 -- channel structure, candidate transport, ranking
# ===========================================================================
def make_fig4(src, out):
    apply_style()
    ph30 = _load(src, "posthoc_n30")["A_channel_three_state"]
    ph100 = _load(src, "posthoc_n100")["A_channel_three_state"]
    ho30 = _load(src, "holdout_n30")["cross_ansatz_transport"]
    cf100 = _load(src, "confirmation_n100")["cross_ansatz_transport"]
    lg100 = _load(src, "legacy_n100")

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

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
    print("written:", out)
    print(f"  source layer      : {src}")
    print(f"  panel (a) ratios  : "
          + ", ".join(f"{g[1]['ratio']:.2f}" for g in groups))
    print(f"  panel (c) d_rho   : {d_rho:.3f}")
    print(f"  panel (c) gap     : {100*d_agr:.1f} pp")
    print(f"  B_x pair agreement: "
          + ", ".join(f"{agr[p]:.3f}" for p in bx))
    print("  Check these against the caption before committing the figure.")


# ===========================================================================
# Fig. S5 -- ranking robustness and union-conditioned trajectory
# ===========================================================================

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

def _rate_unused(v, depth=0):
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


# Accepted names for the rate inside a union-conditioned entry. A closed list,
# not a search: an earlier version took the first float in (0,1) anywhere in the
# record, which meant a schema change would be absorbed instead of reported.
_RATE_KEYS = ("D", "D_ansatz", "D_seed", "rate", "value", "disagreement")

# Which name was actually found, so the accepted list can be trimmed to the one
# the schema really uses instead of staying a guess.
_RATE_KEY_USED = {}


def _rate(entry, side):
    if isinstance(entry, float):
        _RATE_KEY_USED[side] = "(bare float)"
        return entry
    if isinstance(entry, dict):
        for k in _RATE_KEYS:
            if isinstance(entry.get(k), float):
                _RATE_KEY_USED[side] = k
                return entry[k]
    raise SystemExit(
        f"BLOCKED: cannot read the union-conditioned {side} rate. The entry is "
        f"{type(entry).__name__}"
        + (f" with keys {sorted(entry)}" if isinstance(entry, dict) else "")
        + f"; accepted rate keys are {list(_RATE_KEYS)}. Add the real key to "
        f"_RATE_KEYS rather than making the lookup search.")


def union_pair(block):
    """The union-conditioned pair, read from the schema as it is."""
    ucs = block.get("union_conditioned_sensitivity")
    if not isinstance(ucs, dict):
        raise SystemExit("BLOCKED: no union_conditioned_sensitivity record in "
                         "attribution_n100.json")
    try:
        a_entry, s_entry = ucs["ansatz"], ucs["seed"]
    except KeyError:
        raise SystemExit(f"BLOCKED: union_conditioned_sensitivity has keys "
                         f"{sorted(ucs)}, expected 'ansatz' and 'seed'")
    return _rate(a_entry, "ansatz"), _rate(s_entry, "seed")


def make_figS5(src, out):
    apply_style()
    r30, r100 = _load(src, "ranking_n30"), _load(src, "ranking_n100")
    att = _load(src, "attribution_n100")["by_n"]

    # The registered checkpoint set, checked before anything is drawn. Without
    # this a missing checkpoint produced a seven-point trajectory and exit 0.
    got = tuple(sorted(att, key=int))
    if got != EXPECTED_NS:
        raise SystemExit(f"BLOCKED: expected checkpoints {EXPECTED_NS}, "
                         f"got {got}")

    def agr_gap(r):
        a = r["agreement"]
        return a["A/B_z"] - a["A/A_pocket"]

    gap30 = [r30["by_quantity"][q]["counterexample_gap"] for q, _ in QUANTITIES]
    gap100 = [r100["by_quantity"][q]["counterexample_gap"] for q, _ in QUANTITIES]
    cls30, cls100 = agr_gap(r30), agr_gap(r100)

    kept = [int(n) for n in got]
    u_a, u_s, d_a, d_s = [], [], [], []
    for n in got:
        a, s = union_pair(att[n])
        u_a.append(a)
        u_s.append(s)
        d_a.append(att[n]["D_ansatz"])
        d_s.append(att[n]["D_seed"])

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

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
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
    print(f"  union-conditioned rate key: {_RATE_KEY_USED}")
    print("  Check these against the supplement text before committing.")
    return 0


# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", default="all",
                    choices=["all", "2", "3", "4", "S5"])
    ap.add_argument("--ez-convention", dest="ez_convention",
                    choices=["legacy-50ueV", "stray-mean", "total-mean",
                             "total-local"], default="legacy-50ueV")
    ap.add_argument("--profile-norm", dest="profile_norm",
                    choices=["prefactor", "final-peak", "l2"],
                    default="prefactor")
    ap.add_argument("--responses", default="data/t0c/responses_n100.csv")
    ap.add_argument("--analysis", default="data/t0c/analysis")
    a = ap.parse_args(argv)

    want = {"all": {"2", "3", "4", "S5"}}.get(a.figures, {a.figures})
    if "2" in want:
        make_fig2(ez_convention=a.ez_convention, profile_norm=a.profile_norm)
    if "3" in want:
        make_fig3(a.responses, a.analysis, FIG_T0 / "quadrant_data.pdf")
    if "4" in want:
        make_fig4(a.analysis, FIG_T0 / "sensitivity_atlas.pdf")
    if "S5" in want:
        make_figS5(a.analysis, FIG_T0 / "robustness_checks.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
