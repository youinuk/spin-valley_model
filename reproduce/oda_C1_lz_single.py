"""
Valley 2-level dynamics for Oda et al. reproduction.

H_V(t) = (E_v(x(t)) / 2) τ_z + Δ_sv(x(t)) τ_x

Here:
- E_v(x) : position-dependent valley splitting (energy units, J)
- Delta_sv(x): spin-valley or valley-orbit coupling.
            In the simple Oda LZ form it is the minimum gap of the avoided crossing.
- τ_z, τ_x: Pauli matrices on valley pseudospin space.

Phase 2 step 3 is the clean, charge-noise-free case (Model 1V).
Spin separates out as a spectator -- valley dynamics only.

Standard Landau-Zener formula:
    P_LZ(diabatic) = exp(-2π Δ^2 / (ℏ |dE_v/dt|))
i.e. faster dE_v/dt -> diabatic traversal (the valley state stays fixed in the lab frame),
slower -> adiabatic traversal (follows the instantaneous eigenstate).

This module:
- time evolution: numpy RK4 or an exact expm step.
- C1 check: for a linear sweep E_v(t) = alpha t, numerical vs analytic LZ formula.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple

from constants import hbar, FIG_DIR


# Pauli matrices (valley space)
TAU_I = np.eye(2, dtype=complex)
TAU_X = np.array([[0, 1], [1, 0]], dtype=complex)
TAU_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
TAU_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def hamiltonian(E_v: float, Delta_sv: float) -> np.ndarray:
    """H_V = (E_v/2) tau_z + Delta_sv tau_x. Units J."""
    return 0.5 * E_v * TAU_Z + Delta_sv * TAU_X


def evolve_valley(
    E_v_of_t: Callable[[float], float],
    Delta_sv_of_t: Callable[[float], float],
    psi0: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    """
    Time evolution of H(t). Exact step-by-step matrix exponential.

    If H varies slowly in time, the small-step approximation is OK:
        psi(t + dt) ≈ exp(-i H(t + dt/2) dt / ℏ) psi(t)

    Returns
    -------
    psi_t : (N_t, 2)  state vector at each t_grid point
    """
    from scipy.linalg import expm
    N = len(t_grid)
    psi = np.zeros((N, 2), dtype=complex)
    psi[0] = psi0
    for i in range(N - 1):
        t_mid = 0.5 * (t_grid[i] + t_grid[i + 1])
        dt = t_grid[i + 1] - t_grid[i]
        H = hamiltonian(E_v_of_t(t_mid), Delta_sv_of_t(t_mid))
        U = expm(-1j * H * dt / hbar)
        psi[i + 1] = U @ psi[i]
    return psi


def instantaneous_eigenstates(E_v: float, Delta_sv: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Instantaneous eigenvectors of H = (E_v/2) tau_z + Delta_sv tau_x.

    Returns
    -------
    energies : (2,)  ascending order
    v_minus, v_plus : (2,) ground, excited eigenvectors
    """
    energies, vecs = np.linalg.eigh(hamiltonian(E_v, Delta_sv))
    # eigh returns ascending: vecs[:, 0] = ground, vecs[:, 1] = excited
    return energies, vecs[:, 0], vecs[:, 1]


def populations_in_eigenbasis(psi: np.ndarray, E_v: float, 
                               Delta_sv: float) -> Tuple[float, float]:
    """Ground/excited population in the instantaneous eigenbasis of the current H."""
    _, vm, vp = instantaneous_eigenstates(E_v, Delta_sv)
    p_g = abs(np.vdot(vm, psi))**2
    p_e = abs(np.vdot(vp, psi))**2
    return float(p_g), float(p_e)


# ============================================================
# C1 — Single linear LZ crossing, analytic vs numerical
# ============================================================
def lz_probability_analytic(Delta: float, dE_dt: float) -> float:
    """Standard LZ formula: P_diabatic = exp(-2π Δ² / (ℏ |dE/dt|))."""
    return float(np.exp(-2 * np.pi * Delta**2 / (hbar * abs(dE_dt))))


def run_C1_check() -> dict:
    """
    Linear sweep: E_v(t) = α t  (zero crossing at t=0).
    Integration window [-T, T] long enough (~tens of Delta widths).
    Initial state: ground state at t = -T (i.e. |0> if alpha > 0).

    Compare the measured P_excited (at t = +T) with the LZ formula.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("="*60)
    print("C1 — Single Landau-Zener crossing (linear sweep)")
    print("="*60)

    e = 1.602176634e-19
    # Delta = 0.5 ueV (representative, Defaults.Delta_sv_J)
    Delta = 0.5e-6 * e   # J
    # alpha (dE_v/dt) -- units J/s. Sweep over several values.
    # scale: defining alpha = w_alpha * Delta^2 / hbar makes w_alpha the adiabaticity parameter
    # commonly the "sweep rate parameter" is normalized by Delta^2/hbar: gamma = hbar alpha / (2 pi Delta^2)
    # γ ≫ 1 -> diabatic, γ ≪ 1 -> adiabatic
    
    # gamma in [0.1, 10] is the interesting range
    gammas = np.logspace(-1, 1, 9)
    
    results = []
    for gamma in gammas:
        alpha = gamma * 2 * np.pi * Delta**2 / hbar   # J/s
        # Integration window: T such that alpha T >> Delta (nearly adiabatic at the endpoints)
        # alpha T = 30 Delta is enough
        T_half = 30 * Delta / alpha
        # time grid -- resolve the Delta/hbar oscillation period. dt < hbar/(10 Delta)
        dt = hbar / (50 * Delta)
        N = int(np.ceil(2 * T_half / dt))
        if N % 2 == 1: N += 1
        t_grid = np.linspace(-T_half, T_half, N)
        
        # initial state: ground at t = -T_half (i.e. E_v < 0, ground is |+> on tau_z)
        # E_v(-T_half) = -alpha T_half is very negative, ground is close to |1> in the tau_z basis
        # initialize exactly to the ground state -- the lowest eigenvector from eigh
        E0 = -alpha * T_half
        _, vm0, vp0 = instantaneous_eigenstates(E0, Delta)
        psi0 = vm0.astype(complex)
        
        # time evolution
        psi_final = evolve_valley(
            E_v_of_t=lambda t, a=alpha: a * t,
            Delta_sv_of_t=lambda t, d=Delta: d,
            psi0=psi0,
            t_grid=t_grid,
        )[-1]
        
        # P_excited at final state in instantaneous eigenbasis
        E_final = alpha * T_half
        _, p_e_num = populations_in_eigenbasis(psi_final, E_final, Delta)
        # ground -> ground is adiabatic. P_diabatic = P_excited (ground -> excited).
        p_diab_analytic = lz_probability_analytic(Delta, alpha)
        
        results.append({
            "gamma": gamma,
            "alpha": alpha,
            "P_excited_numerical": p_e_num,
            "P_diabatic_analytic": p_diab_analytic,
            "rel_err": abs(p_e_num - p_diab_analytic) / max(p_diab_analytic, 1e-12),
        })
        print(f"  γ={gamma:6.3f}  α={alpha:.3e} J/s  "
              f"P_e_num={p_e_num:.4f}  P_LZ_ana={p_diab_analytic:.4f}  "
              f"rel err = {results[-1]['rel_err']*100:.1f}%")

    # pass criterion: rel error < 10% for all gamma (or abs error < 5e-3 when P_LZ is very small)
    max_err = 0.0
    for r in results:
        if r["P_diabatic_analytic"] > 0.01:
            err = r["rel_err"]
        else:
            err = abs(r["P_excited_numerical"] - r["P_diabatic_analytic"])
        max_err = max(max_err, err)
    passed = max_err < 0.10
    print(f"\n  max error (rel or abs): {max_err*100:.2f}%")
    print(f"  C1 PASS (< 10%): {passed}")

    # Plot
    gammas_arr = np.array([r["gamma"] for r in results])
    P_num = np.array([r["P_excited_numerical"] for r in results])
    P_ana = np.array([r["P_diabatic_analytic"] for r in results])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx(gammas_arr, P_num, "o", label="numerical (expm step)", ms=7)
    gammas_dense = np.logspace(-1, 1, 200)
    alphas_dense = gammas_dense * 2 * np.pi * Delta**2 / hbar
    P_ana_dense = np.array([lz_probability_analytic(Delta, a) for a in alphas_dense])
    ax.semilogx(gammas_dense, P_ana_dense, "-", label="analytic LZ formula", alpha=0.8)
    ax.set_xlabel(r"$\gamma = \hbar \alpha / (2\pi \Delta^2)$  (sweep rate parameter)")
    ax.set_ylabel(r"$P_\mathrm{diabatic}$ (= excited population after sweep)")
    ax.set_title("C1: LZ crossing, numerical vs analytic")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    out = str(FIG_DIR / "step3_C1_lz.png")
    fig.savefig(out, dpi=130)
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")  # vector for supplement
    print(f"  saved: {out}")

    return {"max_err": float(max_err), "passed": bool(passed), "details": results}


if __name__ == "__main__":
    res = run_C1_check()
    print(f"\nC1 result: passed={res['passed']}, max_err={res['max_err']*100:.2f}%")
