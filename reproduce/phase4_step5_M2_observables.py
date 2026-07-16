"""
Separated observables + lambda_sv ablation.

This module computes the M2 observables (valley leakage and dephasing)
used by the response engine, and runs the lambda_sv ablation that turns
the coupling on and off.

Naming: the I_s (x) tau_x term is a valley off-diagonal coupling
(Delta_v), not spin-valley coupling. The added lambda_sv sigma_x (x)
tau_x term is a phenomenological spin-flip/valley-flip coupling ansatz,
lambda_sv(x) = lambda_max * |d_x B_z|/max. The default valley-leakage
observable is P_v_dia (diabatic basis); circular statistics are used
for the phase variance.

Scientific goal:
  The base Hamiltonian H_s (x) I + I (x) H_v is separable, so a
  cross-term cannot arise from it in principle. Adding the minimal
  diagnostic coupling ansatz lambda_sv sigma_x (x) tau_x and
  re-measuring isolates the coupling-induced cross-term. This term is a
  phenomenological ansatz, not a microscopic/standard coupling (see the
  hamiltonian_4lvl_with_coupling docstring and the paper Model section).

Ablation pass criterion:
  lambda_sv -> 0 limit: S_s = 0 (returns to separable)
  lambda_sv > 0:        S_s > 0 (nonseparable spin-valley dynamics)
  the coupling-dependent difference in the cross-terms dP, dphi_var must
  appear, larger in the edge case than the center case.

[Definitions]
4-state basis: |s, v> = (s*2 + v) in {0, 1, 2, 3}

reduced density matrix:
    ρ_s = Tr_v(ρ)      shape (2,2)  — spin sector
    ρ_v = Tr_s(ρ)      shape (2,2)  — valley sector

Separated observables:
    1. Valley leakage:
       P_v_dia = rho_v[1,1]              (diabatic basis: default measurement)
       P_v_ad  = ⟨v_exc(t_f)|ρ_v|v_exc(t_f)⟩   (valley sector eigenbasis)
    2. Spin phase:
       φ_s = arg(ρ_s[0,1])
       phase_variance_linear  = Var(φ)
       phase_variance_circular = -2 ln|<e^{i phi}>|
    3. Spin-valley entanglement:
       S = -Tr(rho_s log_2 rho_s)   <- key cross-term indicator
       purity = Tr(ρ_s²)

[Initial state]
|ψ_0⟩ = |+⟩_s ⊗ |0⟩_v
spin coherence is measurable. rho_s(0) = |+><+|, off-diagonal = 1/2.
"""

from __future__ import annotations
import hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import jax.numpy as jnp
from scipy.linalg import expm

from constants import g_Si, mu_B, hbar, Defaults
from noise.charge_noise import OneOverFNoise
from geometry.periodic_array import PeriodicPrismArray
from geometry.fourier_field import fit_from_prism_array
from reproduce.phase4_step4_M2_minimum import (
    SIG_I, SIG_X, SIG_Z, TAU_X, TAU_Z, kron,
)


def stable_seed(*items, base: int = 0) -> int:
    """Deterministic seed across Python processes."""
    s = "|".join(map(str, items)).encode()
    h = int(hashlib.md5(s).hexdigest()[:8], 16)
    return base + h % 10_000_000


def hamiltonian_4lvl_with_coupling(Ez, Omx, Ev, Delta_v, lambda_sv):
    """Minimal 4-level spin × valley effective Hamiltonian (diagnostic).

    H = (Ez/2)  σ_z ⊗ I_v   ─ longitudinal spin Zeeman (g μ_B B_z)
      + (Ex/2)  σ_x ⊗ I_v   ─ transverse Zeeman energy from B_x(x)
                              (Omx arg = g μ_B B_x, an ENERGY, not a Rabi freq)
      + (ε_v/2) I_s ⊗ τ_z   ─ diabatic valley detuning (two-level model)
      + Δ_v     I_s ⊗ τ_x   ─ off-diagonal valley coupling
      + λ_sv    σ_x ⊗ τ_x   ─ phenomenological spin-flip/valley-flip
                              coupling ansatz (minimal real Pauli product)

    IMPORTANT (provenance / claim caution):
    - This is neither a microscopic Si/SiGe H_sv nor a standard 4-level
      Hamiltonian. It combines a standard ingredient (Zeeman) + a generic
      valley two-level avoided-crossing + a phenomenological coupling ansatz.
    - The Omx factor is g mu_B B_x(x) [energy], not an AC/RWA EDSR Rabi drive.
    - The Ev factor should be read as the *diabatic valley detuning* eps_v(x).
      instantaneous valley gap = sqrt(eps_v^2 + 4 Delta_v^2). If Ev is read as the
      already-diagonalized valley splitting, Delta_v is an extra phenomenological mixing.
    - lambda_sv sigma_x (x) tau_x is only one possible minimal real coupling
      (sigma_x tau_y, sigma_y tau_x etc. are also possible). No unique microscopic form is claimed.
      Microscopically it would come from an intrinsic/synthetic SOC matrix element.
    - In the lambda_sv -> 0 limit it returns to a separable Hamiltonian (invariance check).
    """
    H = 0.5 * Ez * kron(SIG_Z, SIG_I)
    H = H + 0.5 * Omx * kron(SIG_X, SIG_I)
    H = H + 0.5 * Ev * kron(SIG_I, TAU_Z)
    H = H + Delta_v * kron(SIG_I, TAU_X)
    # phenomenological spin-flip/valley-flip coupling ansatz
    if lambda_sv != 0.0:
        H = H + lambda_sv * kron(SIG_X, TAU_X)
    return H


# ============================================================
# Partial trace utilities (4-level = 2 spin × 2 valley)
# ============================================================
def state_to_rho(psi):
    """Pure state |ψ⟩ → density matrix ρ = |ψ⟩⟨ψ|."""
    return np.outer(psi, psi.conj())


def partial_trace_valley(rho):
    """4×4 ρ → 2×2 ρ_s = Tr_v(ρ).
    Index order: |s, v⟩ = (s * 2) + v.
    ρ_s[s, s'] = Σ_v ρ[(s,v), (s',v)]"""
    rho4 = rho.reshape(2, 2, 2, 2)   # [s, v, s', v']
    return np.trace(rho4, axis1=1, axis2=3)


def partial_trace_spin(rho):
    """4×4 ρ → 2×2 ρ_v = Tr_s(ρ)."""
    rho4 = rho.reshape(2, 2, 2, 2)
    return np.trace(rho4, axis1=0, axis2=2)


def von_neumann_entropy(rho2):
    """S = -Tr(ρ log ρ) for 2×2 ρ. base = 2 (bits).
    
    Numerical error can give a small negative value, so clip to 0.
    """
    evals = np.linalg.eigvalsh(rho2).real
    evals = evals[evals > 1e-15]
    S = float(-np.sum(evals * np.log2(evals)))
    return max(S, 0.0)   # clip


# ============================================================
# Single trajectory with separated observables
# ============================================================
def simulate_separated(
    v, ff, pocket_x_center,
    Ev_baseline, Ev_min, pocket_width,
    Delta_v,                # renamed from Delta_sv
    lambda_sv_max,          # NEW: max strength of the coupling ansatz
    noise: OneOverFNoise,
    enable_charge_noise: bool, enable_valley: bool,
    enable_spin_valley_coupling: bool,    # NEW: lambda_sv ablation switch
    n_realizations: int,
    rng: np.random.Generator,
    N_max: int = 3000,
):
    """Single (case, v, model, coupling) simulation.

    [Model -- minimal 4-level diagnostic; the core definition is in
     the hamiltonian_4lvl_with_coupling docstring]
    Hamiltonian = H_s ⊗ I + I ⊗ H_v + (λ_sv(x))(σ_x ⊗ τ_x)
        H_s = (Ez/2)σ_z + (E_perp/2)σ_x
            ─ longitudinal + transverse Zeeman energies from local
              micromagnet field (E_perp = g μ_B B_x, not an EDSR Rabi drive)
        H_v = (ε_v/2)τ_z + Δ_v τ_x
            ─ generic diabatic valley-detuning avoided-crossing model
              (valley is not Zeeman; instantaneous gap sqrt(eps_v^2 + 4 Delta_v^2))
        λ_sv(x)
            ─ phenomenological spin-flip/valley-flip coupling profile;
              A-ansatz default uses |∂B_z/∂x|, atlas variants use
              A_pocket, B_z, B_x ansätze.
                = 0  if enable_spin_valley_coupling=False

    Returns dict with separated observables.
    """
    x_start = pocket_x_center - 4 * pocket_width
    x_end   = pocket_x_center + 4 * pocket_width
    T_traj  = (x_end - x_start) / v

    # time grid -- Delta_v resolution
    if enable_valley:
        dt_traj = hbar / (30 * Delta_v)
    else:
        dt_traj = 1e-11
    N = max(int(T_traj / dt_traj), 500)
    N = min(N, N_max)
    t_grid = np.linspace(0, T_traj, N)
    x_traj = x_start + v * t_grid

    # noise trace
    if enable_charge_noise:
        dt_noise = 1e-8
        T_long = max(50.0 / noise.f_low, 50 * T_traj)
        T_long = min(T_long, 5e-3)
        _, dx_long = noise.generate(T_long, dt_noise, rng=rng)
        N_long = len(dx_long)
        N_win_noise = int(T_traj / dt_noise) + 2
        starts = rng.integers(0, max(N_long - N_win_noise - 1, 1),
                               size=n_realizations)
        t_win_noise = np.arange(N_win_noise) * dt_noise
    else:
        starts = [0] * n_realizations
        dx_long = None
        dt_noise = None
        N_win_noise = None
        t_win_noise = None
        N_win_noise = None
        t_win_noise = None

    # initial state: |+>_s (x) |0>_v
    # In our basis order |s, v⟩ = (s*2 + v):
    #   |0,0⟩=0, |0,1⟩=1, |1,0⟩=2, |1,1⟩=3
    psi_0 = np.zeros(4, dtype=complex)
    psi_0[0] = 1.0 / np.sqrt(2)   # |0_s, 0_v⟩
    psi_0[2] = 1.0 / np.sqrt(2)   # |1_s, 0_v⟩

    # Vectorized E_v(x)
    depth = Ev_baseline - Ev_min
    Ev_t_base = Ev_baseline - depth * np.exp(
        -(x_traj - pocket_x_center)**2 / (2 * pocket_width**2)
    )

    Delta_v_eff = Delta_v if enable_valley else 0.0
    Ev_t = Ev_t_base if enable_valley else np.zeros_like(Ev_t_base)

    # lambda_sv(x) profile -- option A: gradient-dependent coupling
    # λ_sv(x) = lambda_sv_max · |∂Bz/∂x|(x) / max|∂Bz/∂x|
    # if the valley channel is off, lambda_sv is meaningless (the 4-level splits into two sectors)
    if enable_spin_valley_coupling and enable_valley:
        dBz_dx_t = np.asarray(ff.dBz_dx(jnp.asarray(x_traj)))
        # normalization denominator: peak gradient in one cell (deterministic trajectory)
        norm_grad = float(np.max(np.abs(dBz_dx_t)))
        lambda_sv_t = lambda_sv_max * np.abs(dBz_dx_t) / max(norm_grad, 1e-30)
    else:
        lambda_sv_t = np.zeros_like(t_grid)

    # store results
    P_v_dia_realizations = []
    P_v_ad_realizations  = []
    phase_realizations   = []
    spin_coh_realizations = []
    spin_purity_realizations = []
    spin_entropy_realizations = []
    ground_loss_realizations = []

    for r in range(n_realizations):
        if enable_charge_noise:
            s = starts[r]
            dx_win = dx_long[s : s + N_win_noise]
            dx_t = np.interp(t_grid, t_win_noise, dx_win)
            x_traj_r = x_traj + dx_t
        else:
            x_traj_r = x_traj

        Bz_t = np.asarray(ff.B_z(jnp.asarray(x_traj_r)))
        Bx_t = np.asarray(ff.B_x(jnp.asarray(x_traj_r)))
        Ez_t = g_Si * mu_B * Bz_t
        Omx_t = g_Si * mu_B * Bx_t

        psi = psi_0.copy()
        for i in range(N - 1):
            dt_step = t_grid[i+1] - t_grid[i]
            Ez_mid = 0.5*(Ez_t[i]+Ez_t[i+1])
            Omx_mid = 0.5*(Omx_t[i]+Omx_t[i+1])
            Ev_mid = 0.5*(Ev_t[i]+Ev_t[i+1])
            lam_mid = 0.5*(lambda_sv_t[i]+lambda_sv_t[i+1])
            H = hamiltonian_4lvl_with_coupling(Ez_mid, Omx_mid, Ev_mid,
                                                Delta_v_eff, lam_mid)
            U = expm(-1j * H * dt_step / hbar)
            psi = U @ psi

        # final density matrix
        rho = state_to_rho(psi)
        rho_s = partial_trace_valley(rho)
        rho_v = partial_trace_spin(rho)

        # Valley leakage (diabatic: |v=1> population in the fixed basis) -- default observable
        P_v_dia = float(rho_v[1, 1].real)
        # Valley leakage (adiabatic: instantaneous eigenbasis of the *valley sector*)
        # rotate rho_v into the excited eigenstate of H_v(t_f) = (Ev/2)tau_z + Delta_v tau_x
        Hv_final = 0.5 * Ev_t[-1] * TAU_Z + Delta_v_eff * TAU_X
        eig_v, vecs_v = np.linalg.eigh(Hv_final)
        # vecs_v[:,1] = valley excited (higher energy)
        v_exc = vecs_v[:, 1]
        P_v_ad = float(np.real(np.vdot(v_exc, rho_v @ v_exc)))

        # Spin phase from off-diagonal
        rho_s_off = rho_s[0, 1]
        phase = float(np.angle(rho_s_off))
        spin_coh = float(abs(rho_s_off))   # |<exp(iφ)>| per realization
        spin_purity = float(np.real(np.trace(rho_s @ rho_s)))
        spin_entropy = von_neumann_entropy(rho_s)

        P_v_dia_realizations.append(P_v_dia)
        P_v_ad_realizations.append(P_v_ad)
        phase_realizations.append(phase)
        spin_coh_realizations.append(spin_coh)
        spin_purity_realizations.append(spin_purity)
        spin_entropy_realizations.append(spin_entropy)
        # full 4-level ground state population (for diagnostic only)
        Hf_full = hamiltonian_4lvl_with_coupling(
            Ez_t[-1], Omx_t[-1], Ev_t[-1], Delta_v_eff, lambda_sv_t[-1])
        eig_f, vecs_f = np.linalg.eigh(Hf_full)
        pop_ground = abs(np.vdot(vecs_f[:, 0], psi))**2
        ground_loss_realizations.append(1.0 - float(pop_ground))

    # ensemble statistics
    phases = np.array(phase_realizations)
    # circular statistics (phase can cross the +/-pi boundary)
    R_mag = float(abs(np.mean(np.exp(1j * phases))))
    coh_envelope = R_mag
    phase_var_linear = float(np.var(phases))
    # Gaussian phase noise approximation: Var(φ) ≈ -2 ln|<e^{iφ}>|
    phase_var_circular = float(-2 * np.log(max(R_mag, 1e-12)))

    return {
        "n_real": n_realizations,
        "N_steps": N,
        "T_traj_ns": T_traj * 1e9,
        # Valley
        "P_v_dia_mean": float(np.mean(P_v_dia_realizations)),
        "P_v_dia_std":  float(np.std(P_v_dia_realizations)),
        "P_v_ad_mean":  float(np.mean(P_v_ad_realizations)),
        "P_v_ad_std":   float(np.std(P_v_ad_realizations)),
        # Spin phase (linear + circular)
        "phase_mean": float(np.mean(phases)),
        "phase_variance_linear": phase_var_linear,
        "phase_variance_circular": phase_var_circular,
        "coherence_envelope": coh_envelope,
        "spin_coherence_per_real_mean": float(np.mean(spin_coh_realizations)),
        # Entanglement
        "spin_purity_mean": float(np.mean(spin_purity_realizations)),
        "spin_entropy_mean": float(np.mean(spin_entropy_realizations)),
        # Mixed
        "ground_loss_mean": float(np.mean(ground_loss_realizations)),
    }


def main(n_real: int = 5, seed: int = 23, v_list=None, N_max: int = 3000,
         lambda_sv_max: float = None):
    """
    Phase 4 step 5: separated observables + spin-valley coupling ablation.

    For each (case, v): runs M1, M1V, M2 with both:
      - coupling OFF  (λ_sv = 0):  baseline separable, S_s = 0 expected
      - coupling ON   (λ_sv > 0):  cross-term measurable

    This ablation shows the cross-term *originates* in the coupling term.
    """
    print("="*70)
    print(f"Phase 4 step 5 -- Separated observables + lambda_sv ablation (n_real={n_real})")
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
    Ev_min = 5e-6 * e_C
    pocket_width = 30e-9
    Delta_v = 0.5e-6 * e_C   # renamed from Delta_sv
    # max strength of the coupling ansatz -- comparable to Delta_v
    if lambda_sv_max is None:
        lambda_sv_max = 1.0e-6 * e_C   # 1 μeV — coupling regime

    noise = OneOverFNoise(
        sigma_total=Defaults.sigma_dx_m, alpha=1.0,
        f_low=1e3, f_high=1e7,
    )

    cases = {
        "case_i_center": 0.0,
        "case_ii_edge":  +25e-9,
    }
    if v_list is None:
        v_list = [1.0, 5.0, 10.0]

    print(f"  n_real = {n_real}, v_list = {v_list} m/s")
    print(f"  Δ_v (valley mixing) = {Delta_v/e_C*1e6:.2f} μeV")
    print(f"  λ_sv,max (phenomenological coupling ansatz) = {lambda_sv_max/e_C*1e6:.2f} μeV")
    print(f"  pocket Ev_min = {Ev_min/e_C*1e6:.1f} μeV")
    print(f"  observable separation: rho_s = Tr_v(rho), rho_v = Tr_s(rho)")
    print(f"  Ablation: coupling OFF (λ_sv=0) vs coupling ON (λ_sv>0)")
    print(f"  initial state: |+>_s (x) |0>_v")

    results = {}   # results[case][v][model][coupling_state]
    coupling_states = [("OFF", False), ("ON", True)]
    for case_label, x_c in cases.items():
        results[case_label] = {}
        print(f"\n  === {case_label}: pocket at x_c = {x_c*1e9:+.0f} nm ===")
        for v in v_list:
            results[case_label][v] = {}
            for model, en_cn, en_val in [("M1", True, False),
                                          ("M1V", False, True),
                                          ("M2", True, True)]:
                results[case_label][v][model] = {}
                for coupling_label, coupling_on in coupling_states:
                    # M1 has valley off, so coupling is meaningless -> skip ON
                    if model == "M1" and coupling_on:
                        # just copy the OFF result
                        results[case_label][v][model]["ON"] = results[case_label][v][model]["OFF"]
                        continue
                    # stable seed
                    rng_use = np.random.default_rng(
                        stable_seed(case_label, v, model, coupling_label, base=seed))
                    n_r = 1 if (model == "M1V" and not en_cn) else n_real
                    res = simulate_separated(
                        v=v, ff=ff, pocket_x_center=x_c,
                        Ev_baseline=Ev_baseline, Ev_min=Ev_min,
                        pocket_width=pocket_width,
                        Delta_v=Delta_v, lambda_sv_max=lambda_sv_max,
                        noise=noise,
                        enable_charge_noise=en_cn, enable_valley=en_val,
                        enable_spin_valley_coupling=coupling_on,
                        n_realizations=n_r, rng=rng_use, N_max=N_max,
                    )
                    results[case_label][v][model][coupling_label] = res
                    print(f"    v={v:4.1f} m/s, {model:3s}, λ_sv {coupling_label:3s}:  "
                          f"P_v_dia={res['P_v_dia_mean']:.3e}  "
                          f"φ_var_circ={res['phase_variance_circular']:.3e}  "
                          f"S_s={res['spin_entropy_mean']:.4f}")

    # Cross-term analysis (with ablation)
    print("\n" + "="*70)
    print("Cross-term quantification -- coupling OFF vs ON (novelty-(a) candidate signal)")
    print("="*70)
    for case_label in cases.keys():
        print(f"\n  {case_label}:")
        print(f"    {'v':>5}  |  {'λ_sv':>4}  | "
              f"{'Δφ_var (M2-M1)':>18} | "
              f"{'ΔP_dia (M2-M1V)':>18} | "
              f"{'S_s(M2)':>10}")
        for v in v_list:
            R = results[case_label][v]
            for cl in ["OFF", "ON"]:
                dphi = R["M2"][cl]["phase_variance_circular"] - R["M1"][cl]["phase_variance_circular"]
                dP   = R["M2"][cl]["P_v_dia_mean"] - R["M1V"][cl]["P_v_dia_mean"]
                S_M2 = R["M2"][cl]["spin_entropy_mean"]
                print(f"    {v:5.1f}  |  {cl:>4}  | "
                      f"{dphi:>18.3e} | "
                      f"{dP:>18.3e} | "
                      f"{S_M2:>10.4f}")
    
    # key ablation summary
    print("\n" + "="*70)
    print("Ablation summary -- entanglement-entropy change for lambda_sv ON vs OFF")
    print("="*70)
    print("  (for a separable Hamiltonian, S_s = 0 when OFF and S_s > 0 when ON)")
    for case_label in cases.keys():
        for v in v_list:
            S_off = results[case_label][v]["M2"]["OFF"]["spin_entropy_mean"]
            S_on  = results[case_label][v]["M2"]["ON"]["spin_entropy_mean"]
            print(f"    {case_label}, v={v:4.1f} m/s:  "
                  f"S_s(λ=0) = {S_off:.4f},  S_s(λ>0) = {S_on:.4f},  "
                  f"ΔS_s = {S_on - S_off:+.4f}")

    # Plot -- 2 cases x 2 rows (phase variance, P_v_dia), OFF/ON comparison per row
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for case_idx, case_label in enumerate(cases.keys()):
        col = case_idx
        # row 0: phase variance, M1 vs M2 with coupling OFF/ON
        ax = axes[0, col]
        for model, color in [("M1", "blue"), ("M2", "purple")]:
            for cl, ls in [("OFF", "--"), ("ON", "-")]:
                ys = [results[case_label][v][model][cl]["phase_variance_circular"]
                      for v in v_list]
                ax.semilogy(v_list, np.maximum(ys, 1e-12), color=color, ls=ls,
                            marker="o" if cl == "OFF" else "s",
                            lw=1.5, ms=7, label=f"{model} λ_sv {cl}")
        ax.set_xlabel("v [m/s]"); ax.set_ylabel(r"$\langle\delta\phi^2\rangle_\mathrm{circ}$ (rad²)")
        ax.set_title(f"{case_label}: spin phase variance")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
        # row 1: P_v_dia, M1V vs M2 with coupling OFF/ON
        ax = axes[1, col]
        for model, color in [("M1V", "red"), ("M2", "purple")]:
            for cl, ls in [("OFF", "--"), ("ON", "-")]:
                ys = [results[case_label][v][model][cl]["P_v_dia_mean"] for v in v_list]
                ax.semilogy(v_list, np.maximum(ys, 1e-12), color=color, ls=ls,
                            marker="o" if cl == "OFF" else "s",
                            lw=1.5, ms=7, label=f"{model} λ_sv {cl}")
        ax.set_xlabel("v [m/s]"); ax.set_ylabel(r"$P_{v,\mathrm{exc}}$ (diabatic basis)")
        ax.set_title(f"{case_label}: valley leakage")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    fig.suptitle(f"Phase 4 step 5: separated observables + λ_sv ablation (n_real={n_real})",
                 fontsize=12)
    fig.tight_layout()
    FIG_DIR_P4 = Path(__file__).resolve().parents[1] / "figures" / "phase4"
    FIG_DIR_P4.mkdir(parents=True, exist_ok=True)
    out = str(FIG_DIR_P4 / "phase4_step5_separated.png")
    fig.savefig(out, dpi=130)
    print(f"\n  saved: {out}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4 step 5 -- separated observables")
    parser.add_argument("--full", action="store_true",
                        help="full validation (n_real=20, slower; default quick n_real=3)")
    args = parser.parse_args()
    if args.full:
        results = main(n_real=20, N_max=4000)
    else:
        # quick: n_real=3, N_max=2000
        results = main(n_real=3, N_max=2000)
    print(f"\nSummary keys: {list(results.keys())}")
