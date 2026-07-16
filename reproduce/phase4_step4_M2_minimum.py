"""
Minimal three-model comparison (M1 vs M1V vs M2).

This nested-model machinery is used by the response engine
(phase4p6_crossterm.py). Note on naming: the Delta_sv term here (the
I_s (x) tau_x term) is a valley off-diagonal (avoided-crossing)
coupling, not spin-valley coupling; it equals Delta_v in the paper.

Purpose: check whether the two loss channels couple over a shared
trajectory x(t):
  1. Can both channels be turned on simultaneously over the same x(t)?
  2. Do the B_z, B_x, E_v time axes align?
  3. Are P_leak and <dphi^2> computed together over a velocity sweep?
  4. Is it a simple superposition, or is a cross-term visible?

Models:
- M1:  charge noise on, valley off
       -> measure <dphi^2>(T),  P_valley = N/A
- M1V: charge noise off, valley on
       -> <dphi^2> = 0,  measure P_leak (Oda-style)
- M2:  both on
       -> measure <dphi^2> (should match M1), measure P_leak (should match M1V under first-order separation)
       -> if a cross-term exists, both differ

Hamiltonian (spin x valley 4-level Hilbert space):
  H = (E_z(t)/2) sigma_z (x) I_v  +  (Omega_x(t)/2) sigma_x (x) I_v
      + (E_v(t)/2) I_s (x) tau_z  +  Delta_sv I_s (x) tau_x
  E_z(t) = g mu_B B_z(x(t) + dx(t))         <-- charge noise via dx(t)
  Omega_x(t) = g mu_B B_x(x(t) + dx(t))     <-- transverse Zeeman term
  E_v(t) = Ev_pocket(x(t))                  <-- position-dependent valley splitting
  Delta_sv = constant valley off-diagonal coupling (= Delta_v in the paper)

This is a small subset check, *not* a full simulation (n_real ~ 50,
run on both pocket cases). Pass criterion: all three models terminate
normally and cases (i) and (ii) show a qualitatively different trade-off.
"""

from __future__ import annotations
import numpy as np
import matplotlib

def _stable_seed(*parts):
    """Deterministic across processes (unlike built-in hash())."""
    import hashlib
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import jax.numpy as jnp

from constants import g_Si, mu_B, hbar, Defaults
from noise.charge_noise import OneOverFNoise
from geometry.periodic_array import PeriodicPrismArray
from geometry.fourier_field import fit_from_prism_array
from reproduce.oda_C2_pocket import Ev_pocket, simulate_pocket_traversal
from reproduce.oda_C1_lz_single import (
    instantaneous_eigenstates, populations_in_eigenbasis,
)


# 4-level Hamiltonian (spin × valley)
# build sigma_z (x) I_v etc. directly
SIG_I = np.eye(2, dtype=complex)
SIG_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIG_Z = np.array([[1, 0], [0, -1]], dtype=complex)
TAU_X = SIG_X.copy()
TAU_Z = SIG_Z.copy()


def kron(a, b): return np.kron(a, b)


def hamiltonian_4lvl(Ez, Omx, Ev, Delta_sv):
    """4-level spin × valley Hamiltonian."""
    H = 0.5 * Ez * kron(SIG_Z, SIG_I)
    H = H + 0.5 * Omx * kron(SIG_X, SIG_I)
    H = H + 0.5 * Ev * kron(SIG_I, TAU_Z)
    H = H + Delta_sv * kron(SIG_I, TAU_X)
    return H


def simulate_M2_single(
    v: float,
    ff,                        # FourierField
    pocket_x_center: float,
    Ev_baseline: float,
    Ev_min: float,
    pocket_width: float,
    Delta_sv: float,
    noise: OneOverFNoise,
    dt: float, T: float,
    n_realizations: int,
    enable_charge_noise: bool,
    enable_valley: bool,
    rng: np.random.Generator,
):
    """
    M1/M1V/M2 simulation:
    - x(t) = x_start + v t
    - dx(t): charge noise (0 if enable_charge_noise=False)
    - valley Hamiltonian: E_v, Delta_sv (zeroed out if enable_valley=False)

    Returns
    -------
    dict with:
        <delta_phi^2>(T_final)  (charge-noise dephasing)
        P_leak(T_final)         (valley LZ leakage)
    """
    from scipy.linalg import expm
    
    # x(t) -- traverse centered on the pocket
    x_start = pocket_x_center - 4 * pocket_width
    x_end   = pocket_x_center + 4 * pocket_width
    T_traj  = (x_end - x_start) / v
    
    # time grid -- Delta_sv resolution
    dt_traj = hbar / (30 * Delta_sv) if enable_valley else 5e-12
    # too many steps blow up expm cost -- cap at 5000.
    N = max(int(T_traj / dt_traj), 500)
    N = min(N, 5000)
    t_grid = np.linspace(0, T_traj, N)
    
    # deterministic path
    x_traj = x_start + v * t_grid
    
    # accumulate results
    delta_phi_sq_realizations = []
    P_leak_realizations = []
    
    # noise trace (long trace + window sampling)
    if enable_charge_noise:
        # the noise generator wants dt at Nyquist resolution. Too small a dt_traj causes OOM.
        # generate at a separate dt_noise (~10 ns) and interpolate onto t_grid.
        dt_noise = 1e-8                                # 10 ns
        T_long = max(50.0 / noise.f_low, 50 * T_traj)   # long enough
        T_long = min(T_long, 5e-3)                      # memory cap 5 ms
        t_noise_arr, dx_long = noise.generate(T_long, dt_noise, rng=rng)
        N_long = len(dx_long)
        # one realization = a T_traj-length window
        N_win_noise = int(T_traj / dt_noise) + 2
        starts = rng.integers(0, max(N_long - N_win_noise - 1, 1),
                               size=n_realizations)
    else:
        starts = [0] * n_realizations
        dx_long = None
        dt_noise = None
    
    for r in range(n_realizations):
        # one charge-noise realization -- via interpolation
        if enable_charge_noise:
            s = starts[r]
            # interpolate the [s, s+N_win_noise] window of dx_long onto t_grid
            t_win_noise = (np.arange(N_win_noise) * dt_noise)
            dx_win = dx_long[s : s + N_win_noise]
            dx_t = np.interp(t_grid, t_win_noise, dx_win)
            x_traj_r = x_traj + dx_t
        else:
            x_traj_r = x_traj
        
        # B_z(x(t) + δx(t)), B_x(x(t) + δx(t)) — FourierField evaluation
        Bz_t = np.asarray(ff.B_z(jnp.asarray(x_traj_r)))
        Bx_t = np.asarray(ff.B_x(jnp.asarray(x_traj_r)))
        
        # Zeeman energy: g μ_B B
        Ez_t = g_Si * mu_B * Bz_t
        Omx_t = g_Si * mu_B * Bx_t
        
        # E_v(x(t)) -- vectorized (Ev_pocket is Gaussian)
        depth = Ev_baseline - Ev_min
        Ev_t = Ev_baseline - depth * np.exp(-(x_traj - pocket_x_center)**2 / (2 * pocket_width**2))
        if not enable_valley:
            Ev_t = np.zeros_like(Ev_t)
            Delta_eff = 0.0
        else:
            Delta_eff = Delta_sv
        
        # 4-level time evolution (initial state = ground)
        E0 = Ez_t[0]; Omx0 = Omx_t[0]; Ev0 = Ev_t[0]
        H0 = hamiltonian_4lvl(E0, Omx0, Ev0, Delta_eff)
        eig0, vecs0 = np.linalg.eigh(H0)
        psi = vecs0[:, 0].astype(complex)
        
        # step-by-step
        for i in range(N - 1):
            t_mid = 0.5 * (t_grid[i] + t_grid[i+1])
            dt_step = t_grid[i+1] - t_grid[i]
            # mid-point Hamiltonian
            Ez_mid = 0.5*(Ez_t[i]+Ez_t[i+1])
            Omx_mid = 0.5*(Omx_t[i]+Omx_t[i+1])
            Ev_mid = 0.5*(Ev_t[i]+Ev_t[i+1])
            H = hamiltonian_4lvl(Ez_mid, Omx_mid, Ev_mid, Delta_eff)
            U = expm(-1j * H * dt_step / hbar)
            psi = U @ psi
        
        # final-state analysis:
        # valley leakage: sum over |spin x valley_excited> (indices 2,3)
        # 4-state index: |0_s, 0_v〉=0, |0_s, 1_v〉=1, |1_s, 0_v〉=2, |1_s, 1_v〉=3
        # in the eigenstates of I_s (x) tau_z, valley_excited is |x_s, 1_v>.
        # in general, project onto the valley_excited state of the final-time eigenbasis.
        # here we measure the population in the valley-sector eigenstate of H at final time.
        if enable_valley:
            E_final = Ev_t[-1]
            Ez_final = Ez_t[-1]
            Omx_final = Omx_t[-1]
            Hf = hamiltonian_4lvl(Ez_final, Omx_final, E_final, Delta_eff)
            eig_f, vecs_f = np.linalg.eigh(Hf)
            # ground state of full 4-level
            pop_ground = abs(np.vdot(vecs_f[:, 0], psi))**2
            # leakage = 1 - ground
            P_leak = 1.0 - float(pop_ground)
        else:
            P_leak = 0.0
        P_leak_realizations.append(P_leak)
        
        # charge-noise dephasing: spin coherence loss
        # simplification: decay of <sigma_x> or <sigma_y>
        # in the 4-level system the decay of |<spin sigma_x>|^2 directly measures dephasing
        # σ_x ⊗ I_v
        sigx_full = kron(SIG_X, SIG_I)
        spin_coh = float(abs(np.vdot(psi, sigx_full @ psi)))
        # initially <sigma_x> = 0 in the ground state, so another measure is needed
        # more direct: dephasing is the *phase variance* = <dphi^2>
        # here we compute dphi directly:
        # dphi = integral (g_grad . dx(t)) dt holds for simple stationary cases,
        # but here the phase is embedded in the 4-level evolution.
        # alternative: inner product of the initial ground state |0_s, 0_v> (or nearby) with the final state,
        # whose phase = (deterministic phase) + (charge-noise induced phase shift)
        # the accurate dephasing measure is |<exp(i dphi)>|.
        # in this minimal comparison we measure the *variance* of the *final-state ground population* across realizations.
        delta_phi_sq_realizations.append(1.0 - abs(np.vdot(vecs_f[:, 0], psi))**2 if enable_valley else
                                          1.0 - abs(np.vdot(vecs0[:, 0], psi))**2)
    
    P_leak_mean = float(np.mean(P_leak_realizations))
    P_leak_std = float(np.std(P_leak_realizations))
    pop_loss_mean = float(np.mean(delta_phi_sq_realizations))
    pop_loss_std = float(np.std(delta_phi_sq_realizations))
    
    return {
        "P_leak_mean": P_leak_mean,
        "P_leak_std": P_leak_std,
        "pop_loss_mean": pop_loss_mean,
        "pop_loss_std": pop_loss_std,
        "T_traj_ns": T_traj * 1e9,
    }


def main(n_real: int = 10, seed: int = 17, v_list=None):
    print("="*70)
    print(f"Phase 4 step 4 -- minimal Model 2 comparison (n_real={n_real})")
    print("="*70)
    
    # Baseline
    a = 150e-9
    arr = PeriodicPrismArray(
        period_a=a, half_x=25e-9, half_y=25e-9, half_z=15e-9,
        cz=15e-9, N_periods_each_side=4, Ms=1.4e6,
    )
    ff = fit_from_prism_array(arr, -50e-9, N_harm=3)
    
    e_C = 1.602176634e-19
    Ev_baseline = 100e-6 * e_C
    Ev_min = 5e-6 * e_C        # deep but not zero (less extreme)
    pocket_width = 30e-9
    Delta_sv = 0.5e-6 * e_C
    
    noise = OneOverFNoise(
        sigma_total=Defaults.sigma_dx_m, alpha=1.0,
        f_low=1e3, f_high=1e7,
    )
    
    # two cases
    cases = {
        "case_i_center":  0.0,
        "case_ii_edge":   +25e-9,
    }
    
    # velocity sweep
    if v_list is None:
        v_list = [1.0, 5.0, 10.0]   # m/s -- very small set
    
    print(f"  n_real = {n_real}, v_list = {v_list} m/s, Δ_sv = {Delta_sv/e_C*1e6:.1f} μeV")
    print(f"  pocket: Ev_min = {Ev_min/e_C*1e6:.1f} ueV (set less extreme)")
    print(f"  note: this step is sanity-level, not a full Monte Carlo. n_real={n_real}.")
    
    rng = np.random.default_rng(seed)
    results = {}
    
    for case_label, x_c in cases.items():
        results[case_label] = {}
        print(f"\n  --- {case_label}: pocket at x_c = {x_c*1e9:+.0f} nm ---")
        for v in v_list:
            results[case_label][v] = {}
            for model, en_cn, en_val in [("M1", True, False),
                                          ("M1V", False, True),
                                          ("M2", True, True)]:
                rng_use = np.random.default_rng(seed + _stable_seed(case_label, v, model) % 10000)
                # M1V has no stochastic part -> n_real can be reduced
                n_r = 1 if model == "M1V" else n_real
                res = simulate_M2_single(
                    v=v, ff=ff,
                    pocket_x_center=x_c,
                    Ev_baseline=Ev_baseline, Ev_min=Ev_min,
                    pocket_width=pocket_width,
                    Delta_sv=Delta_sv,
                    noise=noise, dt=1e-12, T=None,
                    n_realizations=n_r,
                    enable_charge_noise=en_cn, enable_valley=en_val,
                    rng=rng_use,
                )
                results[case_label][v][model] = res
                print(f"    v = {v:4.1f} m/s, {model:3s}:  "
                      f"P_leak = {res['P_leak_mean']:.3e}, "
                      f"pop_loss = {res['pop_loss_mean']:.3e}")
    
    # Plot -- per case P_leak(v), pop_loss(v)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for case_idx, case_label in enumerate(cases.keys()):
        col = case_idx
        # P_leak (M1V, M2)
        ax = axes[0, col]
        for model, color, ms in [("M1V", "red", "o"), ("M2", "purple", "s")]:
            ps = [results[case_label][v][model]["P_leak_mean"] for v in v_list]
            ax.semilogy(v_list, np.maximum(ps, 1e-12), color=color, marker=ms,
                        lw=1.5, ms=8, label=model)
        ax.set_xlabel("v [m/s]"); ax.set_ylabel("P_leak (1 - ground pop)")
        ax.set_title(f"{case_label}: valley leakage")
        ax.legend(); ax.grid(alpha=0.3, which="both")
        # pop_loss (M1, M2) -- honest label
        ax = axes[1, col]
        for model, color, ms in [("M1", "blue", "o"), ("M2", "purple", "s")]:
            ps = [results[case_label][v][model]["pop_loss_mean"] for v in v_list]
            ax.semilogy(v_list, np.maximum(ps, 1e-12), color=color, marker=ms,
                        lw=1.5, ms=8, label=model)
        # this measurement is a mixed observable (spin phase + valley leakage + ...)
        ax.set_xlabel("v [m/s]"); ax.set_ylabel("mixed ground-pop loss (NOT pure dephasing)")
        ax.set_title(f"{case_label}: mixed observable")
        ax.legend(); ax.grid(alpha=0.3, which="both")
    
    fig.suptitle("Phase 4 step 4: M1 / M1V / M2 minimum comparison", fontsize=12)
    fig.tight_layout()
    FIG_DIR_P4 = Path(__file__).resolve().parents[1] / "figures" / "phase4"
    FIG_DIR_P4.mkdir(parents=True, exist_ok=True)
    out = str(FIG_DIR_P4 / "phase4_step4_M2_minimum.png")
    fig.savefig(out, dpi=130)
    print(f"\n  saved: {out}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4 step 4 -- minimal Model 2 comparison")
    parser.add_argument("--full", action="store_true",
                        help="full validation (n_real=10, slower; default is quick n_real=1)")
    args = parser.parse_args()
    if args.full:
        results = main(n_real=10)
    else:
        # quick default: n_real=1, 3 v points
        # purpose is a simple sanity check, so keep it light.
        results = main(n_real=1, v_list=[1.0, 5.0, 10.0])
    print(f"\nSummary keys: {list(results.keys())}")
