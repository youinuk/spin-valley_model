"""
C2 -- traversal of a low-Ev pocket.

In real Si/SiGe shuttling, E_v(x) is not a monotonic sweep but has "dips"
due to alloy disorder (low-Ev pockets). The valley gap shrinks there, so the
spin-valley coupling Delta_sv has a large effect.

This check:
1. Introduce a Gaussian-dip model for E_v(x).
2. Constant-velocity shuttling -> measure valley excited population.
3. Oda-style velocity-profile optimization -> confirm reduced valley excited population.

Oda key idea:
- Traversing the pocket *fast* (diabatic) keeps the valley state fixed in the lab frame.
- But then the endpoints *do not match* the instantaneous eigenstate (excited state).
- Fix: adjust the velocity *only inside the pocket* so the phase closes cleanly (final state
  returns to the initial ground state).
- Normal velocity outside the pocket, a short fast trajectory only inside.

Here we make only a simple comparison:
- Profile A: constant v through entire trajectory
- Profile B: normal v outside the pocket, fast v inside (e.g. 5x faster)

C2 pass criterion: B reduces P_excited by 10x or more vs A.
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Callable, Tuple

from constants import hbar, Defaults, FIG_DIR
from reproduce.oda_C1_lz_single import (
    evolve_valley,
    instantaneous_eigenstates,
    populations_in_eigenbasis,
)


def Ev_pocket(x: float, Ev_baseline: float, Ev_min: float, 
              x_center: float, width: float) -> float:
    """
    E_v(x) = Ev_baseline - (Ev_baseline - Ev_min) * exp(-(x - x_center)² / (2 width²))
    """
    depth = Ev_baseline - Ev_min
    return Ev_baseline - depth * np.exp(-(x - x_center)**2 / (2 * width**2))


def simulate_pocket_traversal(
    x_of_t: Callable[[float], float],
    Ev_baseline: float,
    Ev_min: float,
    x_center: float,
    width: float,
    Delta_sv: float,
    t_grid: np.ndarray,
) -> dict:
    """
    Time-evolve along profile x(t) and measure the final valley excited population.
    """
    E_v_of_t = lambda t: Ev_pocket(x_of_t(t), Ev_baseline, Ev_min, x_center, width)
    Delta_of_t = lambda t: Delta_sv

    # initial ground state (instantaneous eigenstate at t=0)
    E0 = E_v_of_t(t_grid[0])
    _, vm0, _ = instantaneous_eigenstates(E0, Delta_sv)
    psi0 = vm0.astype(complex)

    psi_traj = evolve_valley(E_v_of_t, Delta_of_t, psi0, t_grid)
    
    # final state population in final-time eigenbasis
    E_final = E_v_of_t(t_grid[-1])
    p_g, p_e = populations_in_eigenbasis(psi_traj[-1], E_final, Delta_sv)
    
    # full trajectory of P_excited (tracked over time)
    p_e_t = np.zeros(len(t_grid))
    for i, t in enumerate(t_grid):
        Et = E_v_of_t(t)
        _, p_e_t[i] = populations_in_eigenbasis(psi_traj[i], Et, Delta_sv)
    
    return {
        "psi_traj": psi_traj,
        "p_g_final": p_g,
        "p_e_final": p_e,
        "p_e_t": p_e_t,
        "E_v_t": np.array([E_v_of_t(t) for t in t_grid]),
    }


def run_C2_check() -> dict:
    print("="*60)
    print("C2 — Deep pocket traversal (Oda-relevant regime)")
    print("="*60)

    # Oda-relevant scenario: pocket reaches near 0 -> a genuine avoided crossing
    # Delta_sv = 0.5 ueV becomes the dominant gap inside the pocket.
    e_C = 1.602176634e-19
    Ev_baseline = 100e-6 * e_C       # 100 μeV
    Ev_min = 0.0                     # 0 (touch zero) -> two genuine LZ crossings
    pocket_width = 30e-9
    Delta_sv = 0.5e-6 * e_C          # 0.5 ueV - standard spin-valley coupling
    
    # shuttling path
    x_start = -200e-9
    x_end   = +200e-9
    x_center = 0.0
    
    # Profile A (slow, adiabatic): traverse slowly
    v_slow = 1.0    # m/s -- LZ activation regime (P_LZ ~ 0.47, P_2cross ~ 0.5)
    # Profile B (fast, Oda-style): fast only near the pocket -> diabatic traversal
    v_baseline_B = v_slow
    v_fast_inside = 100.0   # m/s -- 100x faster inside the pocket

    print(f"  Pocket: E_v = {Ev_baseline/e_C*1e6:.0f} μeV → {Ev_min/e_C*1e6:.1f} μeV "
          f"(width {pocket_width*1e9:.0f} nm) — *touches zero, real LZ*")
    print(f"  Δ_sv = {Delta_sv/e_C*1e6:.2f} μeV")
    print(f"  Path: {x_start*1e9:.0f} → {x_end*1e9:.0f} nm")
    print(f"  Profile A: slow constant v = {v_slow} m/s  (adiabatic-ish)")
    print(f"  Profile B: v = {v_baseline_B} m/s outside, v = {v_fast_inside} m/s in pocket")

    dt = hbar / (50 * Delta_sv)
    print(f"  dt = {dt*1e12:.3f} ps")

    results = {}

    # Profile A: constant slow velocity
    T_A = (x_end - x_start) / v_slow
    N_A = int(np.ceil(T_A / dt))
    t_A = np.linspace(0, T_A, N_A)
    x_of_t_A = lambda t, v=v_slow, x0=x_start: x0 + v * t
    
    res_A = simulate_pocket_traversal(
        x_of_t_A, Ev_baseline, Ev_min, x_center, pocket_width,
        Delta_sv, t_A,
    )
    print(f"\n  Profile A (slow const v={v_slow} m/s):")
    print(f"    T = {T_A*1e9:.1f} ns,  N_steps = {N_A}")
    print(f"    P_excited(final) = {res_A['p_e_final']:.4e}")
    results["A_slow"] = {
        "T_total_ns": T_A*1e9, "P_excited": res_A["p_e_final"],
        "p_e_t": res_A["p_e_t"], "E_v_t": res_A["E_v_t"], "t": t_A,
    }
    
    # Profile B: fast only inside the pocket
    pocket_region_half = 3 * pocket_width
    x_arr = np.linspace(x_start, x_end, 10000)
    in_pocket = np.abs(x_arr - x_center) < pocket_region_half
    v_arr = np.where(in_pocket, v_fast_inside, v_baseline_B)
    # smooth transitions via tanh ramp to avoid sudden velocity jumps
    # (Oda style — smooth profile)
    smooth_width = 10e-9   # 10 nm transition
    edge_in = -pocket_region_half
    edge_out = +pocket_region_half
    ramp_in = 0.5 * (1 + np.tanh((x_arr - edge_in) / smooth_width))
    ramp_out = 0.5 * (1 - np.tanh((x_arr - edge_out) / smooth_width))
    smooth_factor = ramp_in * ramp_out   # ≈ 1 inside, ≈ 0 outside
    v_arr = v_baseline_B + (v_fast_inside - v_baseline_B) * smooth_factor
    
    dx = x_arr[1] - x_arr[0]
    dt_seg = dx / v_arr
    t_arr_B = np.concatenate([[0.0], np.cumsum(dt_seg[:-1])])
    T_B = t_arr_B[-1]
    N_B = int(np.ceil(T_B / dt))
    t_B = np.linspace(0, T_B, N_B)
    x_of_t_B_fn = lambda t, xa=x_arr, ta=t_arr_B: np.interp(t, ta, xa)
    
    res_B = simulate_pocket_traversal(
        x_of_t_B_fn, Ev_baseline, Ev_min, x_center, pocket_width,
        Delta_sv, t_B,
    )
    print(f"\n  Profile B (smooth tanh fast in pocket, max v={v_fast_inside} m/s):")
    print(f"    T = {T_B*1e9:.1f} ns,  N_steps = {N_B}")
    print(f"    P_excited(final) = {res_B['p_e_final']:.4e}")
    results["B_pocket_fast"] = {
        "T_total_ns": T_B*1e9, "P_excited": res_B["p_e_final"],
        "p_e_t": res_B["p_e_t"], "E_v_t": res_B["E_v_t"], "t": t_B,
    }

    P_A = res_A["p_e_final"]
    P_B = res_B["p_e_final"]
    improvement = P_A / max(P_B, 1e-15)
    print(f"\n  Improvement (P_A / P_B) = {improvement:.2f}×")
    print(f"  C2 pass criterion: improvement >= 10x")
    passed = improvement >= 10.0
    print(f"  C2 PASS: {passed}")
    print(f"  Oda target (P_e <= 10^-3):")
    print(f"    Profile A: {'PASS' if P_A <= 1e-3 else 'FAIL'} ({P_A:.2e})")
    print(f"    Profile B: {'PASS' if P_B <= 1e-3 else 'FAIL'} ({P_B:.2e})")

    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5))
    ax = axes[0]
    ax.plot(t_A*1e9, results["A_slow"]["E_v_t"]/e_C*1e6,
            label=f"A (slow v={v_slow} m/s)", lw=1.5)
    ax.plot(t_B*1e9, results["B_pocket_fast"]["E_v_t"]/e_C*1e6,
            label="B (smooth fast in pocket)", lw=1.5)
    ax.set_xlabel("t [ns]"); ax.set_ylabel(r"$E_v(x(t))$ [μeV]")
    ax.set_title("C2: deep pocket traversal — $E_v(t)$")
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.semilogy(t_A*1e9, np.maximum(results["A_slow"]["p_e_t"], 1e-15),
                label=f"A: final P_e = {P_A:.2e}", lw=1.5)
    ax.semilogy(t_B*1e9, np.maximum(results["B_pocket_fast"]["p_e_t"], 1e-15),
                label=f"B: final P_e = {P_B:.2e}", lw=1.5)
    ax.axhline(1e-3, color="k", ls=":", alpha=0.5, label="Oda target 10⁻³")
    ax.set_xlabel("t [ns]"); ax.set_ylabel(r"$P_\mathrm{excited}(t)$")
    ax.set_title("C2: valley excited population trajectory")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = str(FIG_DIR / "step3_C2_pocket.png")
    fig.savefig(out, dpi=130)
    print(f"  saved: {out}")
    
    return {
        "P_excited_A": float(P_A),
        "P_excited_B": float(P_B),
        "improvement": float(improvement),
        "passed_10x": bool(passed),
        "passed_Oda_target_A": bool(P_A <= 1e-3),
        "passed_Oda_target_B": bool(P_B <= 1e-3),
    }


if __name__ == "__main__":
    res = run_C2_check()
    print(f"\nC2 result: {res}")
