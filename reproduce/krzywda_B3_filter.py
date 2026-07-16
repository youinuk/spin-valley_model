"""
B3 — Filter function notch.

Dephasing can be written as a weighted overlap of the form:
    chi(T) = (1/2) integral_0^infty S_omega(omega) F(omega, T) (1/omega^2) domega   [conventional def]
or a simpler variant:
    var(δφ(T)) = integral_0^infty S_omega(omega) F(omega, T) domega
    F(omega, T) = |G(omega, T)|^2,   G = FT of generating function

In this simulation:
    δφ(T) = (g μ_B / ℏ) ∫_0^T y(t) δx(t) dt
        y(t) = g_grad        (stationary, single gradient)
        y(t) = B_long'(x(t)) (shuttling)
This is a weighted integral, so define the filter function:
    G(omega, T) = ∫_0^T y(t) e^{-i omega t} dt
    F(omega, T) = |G(omega, T)|^2

Since S_omega = (g mu_B / hbar)^2 S_dx,
    var(δφ) = (g μ_B / ℏ)^2 ∫ S_dx(omega) F(omega, T) domega / (2π)

Key comparison:
- stationary dot, y(t)=g_grad constant: F = g_grad^2 * sinc^2(omega T/2) * T^2 --
  peaked at omega=0.
- shuttling dot, y(t)=-dB*k*sin(k v t): F shifts toward omega = k v = 2 pi v/a
  -> noise near omega=0 is barely seen. A notch forms.

B3 pass criterion: shuttling F is suppressed by >= 10x near omega -> 0 vs stationary F.
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constants import g_Si, mu_B, hbar, Defaults, FIG_DIR
from field_landscape import PeriodicField


def filter_function_stationary(T: float, g_grad: float, 
                                f_grid: np.ndarray) -> np.ndarray:
    """
    F(f, T) = |∫_0^T g_grad · e^{-i 2π f t} dt|^2
            = g_grad^2 · T^2 · sinc^2(π f T)
    where sinc(x) = sin(x)/x in numpy convention sinc(x)=sin(πx)/(πx).
    """
    return (g_grad * T)**2 * np.sinc(f_grid * T)**2


def filter_function_shuttling(T: float, v: float, field: PeriodicField,
                                f_grid: np.ndarray, n_t: int = 4000) -> np.ndarray:
    """
    Numerically F(f, T) = |integral_0^T B_long'(v t) e^{-i 2 pi f t} dt|^2.
    """
    t = np.linspace(0, T, n_t)
    dt = t[1] - t[0]
    k = 2 * np.pi / field.period
    y = -field.dB_long * k * np.sin(k * v * t)
    F = np.zeros(len(f_grid))
    for i, f in enumerate(f_grid):
        # numerical Fourier integral (trapezoidal)
        integrand = y * np.exp(-1j * 2 * np.pi * f * t)
        G = np.trapezoid(integrand, t)
        F[i] = np.abs(G)**2
    return F


def filter_function_shuttling_DD(T: float, v: float, field: PeriodicField,
                                  f_grid: np.ndarray, n_pulses: int,
                                  n_t: int = 8000) -> np.ndarray:
    """
    Dynamical decoupling -- n_pulses instantaneous pi-pulses spread evenly over T.
    y(t) -> s(t) y(t),  s(t) = ±1 sign function toggling at pulse times.

    CPMG-style pulse positions: t_k = T (2k-1)/(2 n_pulses).
    """
    t = np.linspace(0, T, n_t)
    dt = t[1] - t[0]
    k = 2 * np.pi / field.period
    y = -field.dB_long * k * np.sin(k * v * t)
    # sign function
    pulse_times = T * (2 * np.arange(1, n_pulses + 1) - 1) / (2 * n_pulses)
    s = np.ones_like(t)
    for tp in pulse_times:
        s[t >= tp] *= -1
    y_dd = s * y
    F = np.zeros(len(f_grid))
    for i, f in enumerate(f_grid):
        integrand = y_dd * np.exp(-1j * 2 * np.pi * f * t)
        G = np.trapezoid(integrand, t)
        F[i] = np.abs(G)**2
    return F


def run_B3_check() -> dict:
    print("="*60)
    print("B3 — Filter function notch + B3b PSD-weighted overlap")
    print("="*60)

    field = PeriodicField()
    T = 1e-6   # 1 us measurement window
    v = 10.0   # shuttling velocity, strong-narrowing regime from B2
    
    # mean absolute gradient (same value as used in B2)
    avg_abs_grad = (2.0 / np.pi) * field.dB_long * (2 * np.pi / field.period)

    # f grid -- log scale, wide [1 kHz, 1 GHz]
    f_grid = np.logspace(3, 9, 400)

    F_stat = filter_function_stationary(T, avg_abs_grad, f_grid)
    F_sh = filter_function_shuttling(T, v, field, f_grid)
    F_sh_dd = filter_function_shuttling_DD(T, v, field, f_grid, n_pulses=8)

    # key: suppression near omega -> 0
    # compare at the lowest frequency
    low_f_mask = f_grid < 1e6   # f < 1 MHz
    F_stat_low = np.mean(F_stat[low_f_mask])
    F_sh_low = np.mean(F_sh[low_f_mask])
    F_sh_dd_low = np.mean(F_sh_dd[low_f_mask])
    suppression_shuttle = F_stat_low / F_sh_low
    suppression_dd = F_stat_low / F_sh_dd_low
    
    print(f"  measurement window T = {T*1e6:.1f} us,  v = {v} m/s")
    print(f"  period a = {field.period*1e9:.0f} nm, f_drive = v/a = {v/field.period/1e6:.0f} MHz")
    print(f"\n  --- B3: F(f,T) low-frequency mean ---")
    print(f"  Stationary F (mean over f<1MHz):     {F_stat_low:.3e}")
    print(f"  Shuttling F (mean over f<1MHz):      {F_sh_low:.3e}")
    print(f"  Shuttling+DD F (8 pulses):           {F_sh_dd_low:.3e}")
    print(f"  Suppression (stat / shuttle):        {suppression_shuttle:.2e}")
    print(f"  Suppression (stat / shuttle+DD):     {suppression_dd:.2e}")
    
    # B3 pass criterion
    passed_B3 = suppression_shuttle >= 10.0
    dd_helps = suppression_dd > suppression_shuttle

    # ============================================================
    # B3b -- PSD-weighted overlap (the actual dephasing functional)
    # chi(T) ∝ ∫ S_dx(f) F(f, T) df
    # this value is directly tied to the real dephasing. Stronger than "there is a
    # filter notch": "the overlap with actual 1/f noise is reduced".
    # ============================================================
    from noise.charge_noise import OneOverFNoise
    from constants import Defaults
    noise = OneOverFNoise(
        sigma_total=Defaults.sigma_dx_m,
        alpha=1.0,
        f_low=1e3,
        f_high=1e9,
    )
    S_dx = noise.psd(f_grid)   # one-sided PSD [m²/Hz]
    
    # trapezoidal integration (log-spaced grid, so df_i = f_{i+1} - f_i)
    df = np.diff(f_grid, prepend=f_grid[0])
    chi_stat = float(np.sum(S_dx * F_stat * df))
    chi_sh   = float(np.sum(S_dx * F_sh   * df))
    chi_sh_dd= float(np.sum(S_dx * F_sh_dd* df))
    
    print(f"\n  --- B3b: PSD-weighted overlap chi(T) ~ ∫ S_dx(f) F(f,T) df ---")
    print(f"  Stationary:      chi = {chi_stat:.3e}")
    print(f"  Shuttling:       chi = {chi_sh:.3e}")
    print(f"  Shuttling+DD:    chi = {chi_sh_dd:.3e}")
    chi_supp = chi_stat / chi_sh
    chi_supp_dd = chi_stat / chi_sh_dd
    print(f"  Overlap reduction (stat/shuttle):    {chi_supp:.2e}")
    print(f"  Overlap reduction (stat/shuttle+DD): {chi_supp_dd:.2e}")
    
    # B3b pass criterion: the actual overlap drops by an order of magnitude or more
    passed_B3b = chi_supp >= 10.0

    print(f"\n  B3  PASS (F(f,T) low-f suppression ≥ 10x):       {passed_B3}")
    print(f"  B3b PASS (PSD-weighted overlap reduction ≥ 10x): {passed_B3b}")
    print(f"  DD strengthens the suppression:                   {dd_helps}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.loglog(f_grid, F_stat, label="stationary (single gradient)", lw=1.5)
    ax.loglog(f_grid, F_sh, label=f"shuttling v={v} m/s", lw=1.5)
    ax.loglog(f_grid, F_sh_dd, label="shuttling + DD (8 pulses)", lw=1.5)
    ax.axvline(v / field.period, color="gray", ls=":", alpha=0.7,
               label=f"f_drive = v/a = {v/field.period/1e6:.0f} MHz")
    ax.axvspan(1e3, 1e6, alpha=0.08, color="red", label="low-f region")
    ax.set_xlabel("f [Hz]"); ax.set_ylabel("F(f, T)  [arb]")
    ax.set_title(f"B3: Filter function (T={T*1e6:.0f} μs)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    # B3b: compare the integrand S_dx * F
    ax = axes[1]
    ax.loglog(f_grid, S_dx * F_stat, label="stationary", lw=1.5)
    ax.loglog(f_grid, S_dx * F_sh,   label="shuttling",  lw=1.5)
    ax.loglog(f_grid, S_dx * F_sh_dd, label="shuttling+DD", lw=1.5)
    ax.axvline(v / field.period, color="gray", ls=":", alpha=0.7)
    ax.set_xlabel("f [Hz]"); ax.set_ylabel(r"$S_{\delta x}(f) \cdot F(f,T)$")
    ax.set_title(f"B3b: PSD-weighted integrand (overlap)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = str(FIG_DIR / "step2_B3_filter.png")
    fig.savefig(out, dpi=130)
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")  # vector for supplement
    print(f"  saved: {out}")

    return {
        "F_stat_low_f": float(F_stat_low),
        "F_shuttling_low_f": float(F_sh_low),
        "F_shuttling_DD_low_f": float(F_sh_dd_low),
        "suppression_shuttle": float(suppression_shuttle),
        "suppression_DD": float(suppression_dd),
        "chi_stat": chi_stat,
        "chi_shuttle": chi_sh,
        "chi_shuttle_DD": chi_sh_dd,
        "chi_suppression": float(chi_supp),
        "chi_suppression_DD": float(chi_supp_dd),
        "passed_B3": bool(passed_B3),
        "passed_B3b": bool(passed_B3b),
        "dd_strengthens_suppression": bool(dd_helps),
    }


if __name__ == "__main__":
    res = run_B3_check()
    print()
    print(f"  Result: {res}")
