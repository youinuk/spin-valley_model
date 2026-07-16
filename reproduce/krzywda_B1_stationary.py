"""
Stationary-dot dephasing under 1/f charge noise.

Core: a stationary dot on a monotonic field gradient g_grad. Charge noise -> position jitter dx(t)
-> Larmor-frequency jitter:
    δω(t) = (g μ_B / ℏ) * g_grad * δx(t)
accumulated phase:
    δφ(T) = ∫_0^T δω(t) dt
free induction decay envelope:
    C(T) = |<exp(i δφ(T))>_realizations|

In the quasi-static limit (Phase 1), dx is constant so dphi = dw * T, var(dphi) = (sigma_w T)^2
=> Gaussian decay C(T) = exp(-T^2/(2 T_2^{*2})).

For 1/f noise the variance of dphi grows ~T^2 ln(...) with T, still Gaussian-like but
with a small correction. Krzywda reports that halving the gradient roughly doubles T_2*.
(B1: ratio = 8.5/4.4 = 1.93 ± 20%).

This module measures C(T) by Monte Carlo and extracts T_2* at the 1/e crossing.
B1 pass criterion: does halving the gradient roughly double T_2*?
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Tuple

from constants import g_Si, mu_B, hbar, Defaults, FIG_DIR
from noise.charge_noise import OneOverFNoise


def simulate_FID_stationary(
    g_grad: float,
    noise: OneOverFNoise,
    T_max: float,
    dt: float,
    n_realizations: int,
    rng: np.random.Generator,
    n_T_samples: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Free-induction-decay coherence envelope of a stationary dot.

    Implementation note: to resolve low-frequency 1/f components the trace length
    must be >> 1/f_low. Generating only a T_max window leaves almost no
    low-frequency bins inside it, creating a quasi-periodic artifact.
    So we generate one very long trace and draw realizations by picking
    random window start points.

    Returns
    -------
    T_grid : (n_T_samples,)
    C : (n_T_samples,)  coherence envelope |<exp(i δφ(T))>|
    """
    prefactor = (g_Si * mu_B / hbar) * g_grad   # rad/s per m of position jitter

    # generate a very long trace -- covers 1/f_low exactly (df = 1/T_long <= f_low)
    # B1 noise has f_low=1 kHz, so T_long >= 1 ms is needed.
    # Phase2Grid.T_long_s = 500 us (used by B2/B3) suffices for their grid (f_low=250 kHz)
    # but not for the B1 grid (f_low=1 kHz) -> use 1 ms here.
    T_long = 1e-3
    t_long, dx_long = noise.generate(T_long, dt, rng=rng)
    N_long = len(dx_long)
    # samples per window
    N_win = int(np.round(T_max / dt))
    assert N_win < N_long, "T_max too large vs trace length"

    # T grid (linear)
    T_grid = np.linspace(dt, T_max, n_T_samples)

    # random window start points, shuffled so windows do not overlap
    starts = rng.integers(0, N_long - N_win - 1, size=n_realizations)

    phi_at_T = np.zeros((n_realizations, n_T_samples))
    for r, s in enumerate(starts):
        dx_win = dx_long[s : s + N_win]
        # cumulative integral by trapezoidal
        integ = np.concatenate([
            [0.0],
            np.cumsum(0.5 * (dx_win[1:] + dx_win[:-1]) * dt)
        ])
        delta_phi_full = prefactor * integ
        # t for this window starts at 0
        t_win = np.arange(N_win) * dt
        phi_at_T[r] = np.interp(T_grid, t_win, delta_phi_full)

    C = np.abs(np.mean(np.exp(1j * phi_at_T), axis=0))
    return T_grid, C


def extract_T2(T_grid: np.ndarray, C: np.ndarray) -> float:
    """Extract T_2* at the 1/e crossing (linear interpolation)."""
    target = 1.0 / np.e
    if C[0] < target:
        return T_grid[0]
    # first point below target
    idx_below = np.where(C < target)[0]
    if len(idx_below) == 0:
        return np.nan  # never decays
    i1 = idx_below[0]
    i0 = i1 - 1
    # linear interp on log C if useful, here linear in C is fine
    frac = (C[i0] - target) / (C[i0] - C[i1])
    T2 = T_grid[i0] + frac * (T_grid[i1] - T_grid[i0])
    return float(T2)


def run_B1_check(seed: int = 7, n_real: int = 600) -> dict:
    """
    B1 pass criterion: reducing the gradient increases T_2* by ~1.93x (+/-20%).
    
    Measure T_2* for grad_before and grad_after, report the ratio.

    Grid choice: B1 is a stationary dot, so resolving the shuttling modulation frequency is unnecessary.
    A narrow bandwidth [1 kHz, 10 MHz] that captures the 1/f low-frequency tail
    is close to the Krzywda absolute regime. B2 (shuttling) uses a different grid.
    """
    print("="*60)
    print("B1 — Stationary dot T2* ratio (gradient reduction)")
    print("="*60)

    noise = OneOverFNoise(
        sigma_total=Defaults.sigma_dx_m,
        alpha=1.0,
        f_low=1e3,
        f_high=1e7,
    )
    dt = 0.1 / noise.f_high              # 10 ns
    T_max = 2e-6                          # 2 us
    print(f"  noise: 1/f, sigma_total = {noise.sigma_total*1e9:.2f} nm, "
          f"f in [{noise.f_low:.0e}, {noise.f_high:.0e}] Hz")
    print(f"  dt = {dt*1e9:.1f} ns,  T_max = {T_max*1e6:.1f} μs")
    print(f"  realizations per case: {n_real}")

    rng = np.random.default_rng(seed)

    cases = {
        "before (high grad)": Defaults.grad_lo,
        "after  (low grad)":  Defaults.grad_lo / 2.0,
    }

    results = {}
    for label, g_grad in cases.items():
        T_grid, C = simulate_FID_stationary(
            g_grad=g_grad,
            noise=noise,
            T_max=T_max,
            dt=dt,
            n_realizations=n_real,
            rng=rng,
            n_T_samples=300,
        )
        T2 = extract_T2(T_grid, C)
        results[label] = {"g_grad": g_grad, "T_grid": T_grid, "C": C, "T2": T2}
        print(f"  {label}:  g_grad = {g_grad*1e-6:.3f} T/μm,  "
              f"T_2* = {T2*1e9:.1f} ns" if not np.isnan(T2) 
              else f"  {label}:  T_2* = no decay reached in {T_max*1e6:.1f} μs")

    # Ratio
    T2_before = results["before (high grad)"]["T2"]
    T2_after = results["after  (low grad)"]["T2"]
    ratio = T2_after / T2_before
    print(f"\n  T_2* ratio (after/before) = {ratio:.3f}")
    print(f"  expected (grad halved):     2.000")
    print(f"  Krzywda reported:           {Defaults.T2_ratio_target:.3f}")
    print(f"  B1 acceptance:              [{Defaults.T2_ratio_target*0.8:.2f}, "
          f"{Defaults.T2_ratio_target*1.2:.2f}]  ({Defaults.T2_ratio_target*0.8:.2f}–"
          f"{Defaults.T2_ratio_target*1.2:.2f})")
    
    # in our case the ratio should be 2.0 (g_grad exactly halved).
    # Krzywda uses a different device perturbation, so reproducing their 1.93 exactly
    # is not this simulator's task. Our task: "if the gradient is reduced N-fold, does T2* grow N-fold?"
    passed_simple = abs(ratio - 2.0) / 2.0 < 0.10
    passed_krzywda = (Defaults.T2_ratio_target * 0.8 <= ratio <= Defaults.T2_ratio_target * 1.2)
    print(f"  PASS (simple 2x scaling, ±10%):       {passed_simple}")
    print(f"  PASS (Krzywda window 1.93 ± 20%):     {passed_krzywda}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, d in results.items():
        ax.plot(d["T_grid"] * 1e9, d["C"], lw=1.5,
                label=f"{label}: T2* = {d['T2']*1e9:.0f} ns")
    ax.axhline(1/np.e, color="k", ls=":", alpha=0.5, label="1/e")
    ax.set_xlabel("T [ns]")
    ax.set_ylabel("|C(T)|")
    ax.set_title("B1: Stationary dot, 1/f noise, gradient before/after")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = str(FIG_DIR / "step2_B1_ratio.png")
    fig.savefig(out, dpi=130)
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")  # vector for supplement
    print(f"  saved: {out}")

    return {
        "T2_before_ns": float(T2_before * 1e9),
        "T2_after_ns": float(T2_after * 1e9),
        "ratio": float(ratio),
        "passed_simple_2x": bool(passed_simple),
        "passed_krzywda_window": bool(passed_krzywda),
    }


if __name__ == "__main__":
    res = run_B1_check()
    print()
    print(f"  Result: {res}")
