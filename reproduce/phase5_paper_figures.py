"""
Manuscript figures:
  Figure 1: comparison of the four coupling-ansatz lambda_sv(x) profiles (uses the actual simulator code)
  Figure 2: quadrant-classification schematic (dP_v vs dchi) -- ranking vs interpretation

Profiles are drawn by importing the actual lambda_sv_profile / FilterFunction,
so they match the manuscript Model definitions. The schematic matches the Method thresholds.

Usage:
  PYTHONPATH=. python reproduce/phase5_paper_figures.py
Output: figures/phase5/phase5_coupling_profiles.png/.pdf
      figures/phase5/phase5_quadrant_schematic.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from geometry.periodic_array import PeriodicPrismArray
from geometry.fourier_field import fit_from_prism_array
from reproduce.phase4p6_crossterm import lambda_sv_profile

FIG = Path(__file__).resolve().parents[1] / "figures" / "phase5"
FIG.mkdir(parents=True, exist_ok=True)

e_C = 1.602176634e-19


def build_ff():
    """Baseline geometry, same as step 5 / Phase 4.6."""
    a = 150e-9
    arr = PeriodicPrismArray(
        period_a=a, half_x=25e-9, half_y=25e-9, half_z=15e-9,
        cz=15e-9, N_periods_each_side=4, Ms=1.4e6,
    )
    return fit_from_prism_array(arr, -50e-9, N_harm=3), a


def Ev_of_x(x, pocket_x_center, pocket_width, Ev_baseline, depth):
    """Position-dependent valley splitting: baseline minus a Gaussian pocket."""
    return Ev_baseline - depth * np.exp(
        -(x - pocket_x_center) ** 2 / (2 * pocket_width ** 2))


def _fig_tag(ez_convention, profile_norm):
    if ez_convention == "legacy-50ueV" and profile_norm == "prefactor":
        return ""
    return f"__ez-{ez_convention}__norm-{profile_norm}"


def make_coupling_profiles(ez_convention="legacy-50ueV", profile_norm="prefactor"):
    ff, a = build_ff()
    pocket_x_center = 0.0
    pocket_width = 30e-9
    Ev_baseline = 100e-6 * e_C
    depth = 95e-6 * e_C            # pocket floor ~5 ueV
    sigma_E = 10e-6 * e_C
    lambda_0 = 1.0

    x = np.linspace(-4 * pocket_width, 4 * pocket_width, 600)
    if ez_convention == "legacy-50ueV":
        E_Z = 50e-6 * e_C         # archived figure convention (flank hot spot)
    else:
        from constants import g_Si, mu_B, Defaults
        Bz_arr = np.asarray(ff.B_z(x))
        B_off = Defaults.B_ext_T if ez_convention.startswith("total") else 0.0
        if ez_convention == "total-local":
            E_Z = g_Si * mu_B * (B_off + Bz_arr)      # local E_Z(x), same as simulator
        else:
            E_Z = g_Si * mu_B * (B_off + float(np.mean(Bz_arr)))
    Ev_x = Ev_of_x(x, pocket_x_center, pocket_width, Ev_baseline, depth)

    profiles = {}
    for m in ["A", "A_pocket", "B_z", "B_x"]:
        profiles[m] = lambda_sv_profile(
            x, ff, m, lambda_0, profile_norm=profile_norm,
            pocket_x_center=pocket_x_center, sigma_lambda=pocket_width,
            Ev_x=Ev_x, E_Z=E_Z, sigma_E=sigma_E)

    # ---- plot ----
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [1, 2.2]})
    xnm = x * 1e9

    # top: E_v(x) and resonance location
    ax0 = axes[0]
    ax0.plot(xnm, Ev_x / e_C * 1e6, color="#333333", lw=1.8,
             label=r"$E_v(x)$")
    _EZ_ueV = np.asarray(E_Z) / e_C * 1e6
    if _EZ_ueV.ndim == 0 or _EZ_ueV.size == 1:
        ax0.axhline(float(_EZ_ueV), color="#cc3311", ls="--", lw=1.2,
                    label=r"$E_Z$ (resonance)")
    else:  # total-local: local E_Z(x) curve
        ax0.plot(xnm, _EZ_ueV, color="#cc3311", ls="--", lw=1.2,
                 label=r"$E_Z(x)$ (resonance)")
    ax0.set_ylabel(r"$E_v$ ($\mu$eV)")
    ax0.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax0.set_title("Valley splitting profile and spin--valley resonance",
                  fontsize=9)
    ax0.grid(alpha=0.25)

    # bottom: 4 coupling profiles
    ax1 = axes[1]
    styles = {
        "A":        ("#0077bb", "-",  r"A: $\propto|\partial_x B_z|$ (gradient-only)"),
        "A_pocket": ("#33bbee", "--", r"$\mathrm{A_{pocket}}$: A $\times$ Gauss$(x_c)$"),
        "B_z":      ("#009988", "-.", r"$\mathrm{B_z}$: A $\times$ resonance$(E_v{=}E_Z)$"),
        "B_x":      ("#ee7733", ":",  r"$\mathrm{B_x}$: $\propto|\partial_x B_x|\times$ resonance"),
    }
    for m, (c, ls, lab) in styles.items():
        y = profiles[m]
        ax1.plot(xnm, y / max(np.max(np.abs(y)), 1e-30), color=c, ls=ls,
                 lw=2.0, label=lab)
    ax1.set_xlabel("position along shuttling path $x$ (nm)")
    ax1.set_ylabel(r"normalized $\lambda_{sv}(x)$")
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax1.set_title(r"Four phenomenological coupling ansatze $\lambda_{sv}(x)$",
                  fontsize=9)
    ax1.grid(alpha=0.25)
    # mark pocket center
    for ax in axes:
        ax.axvline(0, color="grey", lw=0.8, alpha=0.5)

    fig.tight_layout()
    out = FIG / "phase5_coupling_profiles.png"
    out = out.with_name(out.stem + _fig_tag(ez_convention, profile_norm) + out.suffix)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")  # vector for paper
    plt.close(fig)
    print(f"saved: {out} (+ .pdf)")
    # quantitative summary (caption basis)
    for m in profiles:
        pk = float(np.max(np.abs(profiles[m])))
        cm = xnm[int(np.argmax(np.abs(profiles[m])))]
        print(f"  {m}: peak |λ|={pk:.3e}, peak at x={cm:.1f} nm")
    return profiles


def make_quadrant_schematic():
    """dP_v vs dchi 4-quadrant schematic + ranking-vs-interpretation illustration."""
    fig, ax = plt.subplots(figsize=(5.2, 5.0))

    # thresholds (Method definitions)
    P_th, chi_th = 1.0, 1.0   # schematic units (actual dead-zone 1e-4, 1e-3)

    lim = 4
    # quadrant background colors
    cols = {
        "robust":      ("#0077bb", -1, -1),
        "valley-trade":("#ee7733",  1, -1),
        "spin-trade":  ("#ee3377", -1,  1),
        "both-worsen": ("#cc3311",  1,  1),
    }
    labels = {
        "robust":       ("robust\n" r"$\Delta P_v<0,\ \Delta\chi_\phi<0$", -2.2, -3.3),
        "valley-trade": ("valley-trade\n" r"$\Delta P_v{>}0,\Delta\chi_\phi{<}0$", 2.2, -3.3),
        "spin-trade":   ("spin-trade\n" r"$\Delta P_v{<}0,\Delta\chi_\phi{>}0$", -2.2, 3.3),
        "both-worsen":  ("both-worsen\n" r"$\Delta P_v{>}0,\Delta\chi_\phi{>}0$", 2.2, 3.3),
    }
    for name, (c, sx, sy) in cols.items():
        ax.add_patch(Rectangle((0 if sx > 0 else -lim, 0 if sy > 0 else -lim),
                               lim, lim, color=c, alpha=0.12, zorder=0))
        tx, ty = labels[name][1], labels[name][2]
        ax.text(tx, ty, labels[name][0], ha="center", va="center",
                fontsize=8.5, color=c, fontweight="bold")

    # dead zone (effect-size threshold)
    ax.add_patch(Rectangle((-P_th, -chi_th), 2 * P_th, 2 * chi_th,
                           color="grey", alpha=0.18, zorder=1))
    ax.text(0, 0, "dead\nzone", ha="center", va="center", fontsize=7,
            color="#555555")

    # the same condition lands in different quadrants for two models: schematic points
    # model 1 (filled) vs model 2 (open) -- similar ranking (radius) but different quadrant
    np.random.seed(3)
    pts_m1 = [(-2.4, -1.6), (1.8, -2.2), (-2.0, 1.9), (2.3, 1.5)]
    pts_m2 = [(-2.2, 1.7),  (2.0, 1.9),  (-1.9, -2.0), (2.1, -1.6)]
    for (x1, y1), (x2, y2) in zip(pts_m1, pts_m2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#444444",
                                    lw=1.0, alpha=0.6, ls="--"))
        ax.plot(x1, y1, "o", color="#222222", ms=7, zorder=5)
        ax.plot(x2, y2, "o", mfc="white", mec="#222222", mew=1.5, ms=7, zorder=5)

    ax.axhline(0, color="k", lw=1.0)
    ax.axvline(0, color="k", lw=1.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$\Delta P_v$  (valley leakage change)")
    ax.set_ylabel(r"$\Delta\chi_\phi$  (dephasing change)")
    ax.set_title("Quadrant interpretation can flip between ansatze\n"
                 "even when the magnitude ranking is similar", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    # legend
    ax.plot([], [], "o", color="#222222", ms=7, label="ansatz 1")
    ax.plot([], [], "o", mfc="white", mec="#222222", mew=1.5, ms=7,
            label="ansatz 2 (same condition)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=2, framealpha=0.9)

    fig.tight_layout()
    out = FIG / "phase5_quadrant_schematic.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")  # vector for paper
    plt.close(fig)
    print(f"saved: {out} (+ .pdf)")


if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("--ez-convention", dest="ez_convention",
                    choices=["legacy-50ueV", "stray-mean", "total-mean", "total-local"],
                    default="legacy-50ueV",
                    help="E_Z source for the coupling-profile figure "
                         "(see REPRODUCIBILITY.md, Sec. 7)")
    _p.add_argument("--profile-norm", dest="profile_norm",
                    choices=["prefactor", "final-peak", "l2"], default="prefactor")
    _args = _p.parse_args()
    ez_convention = _args.ez_convention
    profile_norm = _args.profile_norm
    # LaTeX rendering (consistent paper font) -- mathtext fallback on failure
    try:
        plt.rcParams.update({"text.usetex": False,
                             "font.family": "serif",
                             "mathtext.fontset": "dejavuserif"})
    except Exception:
        pass
    make_coupling_profiles(ez_convention=ez_convention, profile_norm=profile_norm)
    make_quadrant_schematic()
