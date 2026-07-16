"""
Response engine: cross-term computation with common random numbers
(CRN), a lambda sweep, and coupling-model sensitivity.

This module computes the two response observables dP_v and dchi that the
sensitivity atlas is built on.

CRN structure and references:
  - M1 (charge noise on, valley off) and M2 (both on) *share the same charge-noise trace*
  - M1V (charge noise off, valley on) is a *deterministic charge-off reference*,
    computed once with charge noise disabled rather than sharing a trace
  - cross-term signals:
      dP_v  = <P_v^M2>_dx - P_v^M1V    (effect of charge noise on valley leakage)
      dchi  = chi^M2 - chi^M1          (effect of valley on charge-noise dephasing)

Coupling models: A: ~|d_x B_z|, A_pocket: A * exp(-(x-x_c)^2/(2 sigma^2)).
The lambda sweep, n_real, and v_list are mode-dependent:

  | mode          | n_real | v list           | lambda list [ueV] | model        | N_max |
  |---------------|-------:|------------------|-------------------|--------------|------:|
  | quick         |      1 | [1, 5, 10]       | [0, 0.5, 1]       | A            |   500 |
  | validate_lite |      5 | [5, 10]          | [0, 0.25, 0.5, 1] | A            |   500 |
  | validate      |     10 | [1, 5, 10]       | [0, 0.25, 0.5, 1] | A            |  1000 |
  | sensitivity   |      2 | [5, 10]          | [0, 1]            | A + A_pocket |   500 |
  | full          |     30 | [0.5,1,2,5,10,20]| [0,0.25,0.5,1,2]  | A + A_pocket |  2000 |

Interpretation limits: quick mode uses n_real=1, so ensemble
observables (dchi, lambda^2 scaling) are not statistically
interpretable; use validate_lite/validate/full for those. Both dP_v
(valley) and dchi (spin) must be viewed together to make a trade-off
claim.
"""

from __future__ import annotations
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# archive_version is the *Phase 4.6 computation kernel* version.
# It may differ from the package version; the same kernel means the same
# outputs on re-run. Bump only when the computation logic changes.
ARCHIVE_VERSION = "phase4_v13"  # v13: ez_convention/profile_norm kernel options; nominal-coordinate kernel E_Z for total-*

import jax.numpy as jnp
from scipy.linalg import expm

from constants import g_Si, mu_B, hbar, Defaults
from noise.charge_noise import OneOverFNoise
from geometry.periodic_array import PeriodicPrismArray
from geometry.fourier_field import fit_from_prism_array
from reproduce.phase4_step5_M2_observables import (
    hamiltonian_4lvl_with_coupling,
    partial_trace_valley, partial_trace_spin,
    von_neumann_entropy, state_to_rho, stable_seed,
    TAU_X, TAU_Z,
)


def lambda_sv_profile(x, ff, model_name, lambda_0, pocket_x_center=None,
                       sigma_lambda=None, Ev_x=None, E_Z=None, sigma_E=None,
                       profile_norm="prefactor"):
    """λ_sv(x) for coupling model.
    
    model_name:
      "A":         λ_0 · |∂Bz/∂x|(x) / max|∂Bz/∂x|        (gradient-only)
      "A_pocket":  λ_0 · |∂Bz/∂x|(x) / max · exp(-(x-x_c)²/(2σ_λ²))  (pocket-weighted)
      "B_z":       λ_0 · |∂Bz/∂x|(x)/max · exp(-(E_v(x)-E_Z)²/(2σ_E²))  (resonance, Bz)
      "B_x":       λ_0 · |∂Bx/∂x|(x)/max · exp(-(E_v(x)-E_Z)²/(2σ_E²))  (resonance, Bx)
    
    B model: coupling is enhanced near the spin-valley resonance (E_v ~ E_Z).
    The narrower the resonance window sigma_E, the smaller the active region.
    """
    if model_name in ("A", "A_pocket", "B_z"):
        dBz_dx = np.asarray(ff.dBz_dx(jnp.asarray(x)))
        norm = float(np.max(np.abs(dBz_dx)))
        base = lambda_0 * np.abs(dBz_dx) / max(norm, 1e-30)
    elif model_name == "B_x":
        dBx_dx = np.asarray(ff.dBx_dx(jnp.asarray(x)))
        norm = float(np.max(np.abs(dBx_dx)))
        base = lambda_0 * np.abs(dBx_dx) / max(norm, 1e-30)
    else:
        raise ValueError(f"unknown coupling model: {model_name}")
    
    if model_name == "A":
        prof = base
    elif model_name == "A_pocket":
        assert pocket_x_center is not None and sigma_lambda is not None
        gauss = np.exp(-(x - pocket_x_center)**2 / (2 * sigma_lambda**2))
        prof = base * gauss
    else:  # B_z / B_x resonance windows
        assert Ev_x is not None and E_Z is not None and sigma_E is not None
        resonance = np.exp(-(np.asarray(Ev_x) - E_Z)**2 / (2 * sigma_E**2))
        prof = base * resonance

    # Optional renormalization of the FINAL kernel, applied to ALL models
    # (audit r5 §7): "final-peak" fixes max|prof| = lambda_0; "l2" fixes the
    # RMS of the kernel over the evaluation window to lambda_0.
    if profile_norm == "final-peak":
        m = float(np.max(np.abs(prof)))
        if m > 0:
            prof = lambda_0 * prof / m
    elif profile_norm == "l2":
        rms = float(np.sqrt(np.mean(np.abs(prof)**2)))
        if rms > 0:
            prof = lambda_0 * prof / rms
    return prof
    raise ValueError(f"unknown coupling model: {model_name}")


def simulate_one_real(v, ff, pocket_x_center,
                       Ev_baseline, Ev_min, pocket_width,
                       Delta_v, lambda_0, coupling_model,
                       enable_charge_noise, enable_valley,
                       dx_t,                       # shared noise realization (CRN)
                       N_max=3000,
                       ez_convention="stray-mean", profile_norm="prefactor"):
    """Single realization -- the caller supplies dx_t (charge-noise trace).
    
    This is the core of CRN: the caller fixes the noise from (case, v, coupling, real_index)
    only; the model (M1/M1V/M2) differs solely in whether dx_t is used.
    """
    x_start = pocket_x_center - 4 * pocket_width
    x_end   = pocket_x_center + 4 * pocket_width
    T_traj  = (x_end - x_start) / v
    
    # BUG FIX: if M1 and M2 use different dt_traj the accumulated Trotter error differs,
    # so the lambda=0 separable check fails (dchi ~ -0.05 was seen in a full_lite CSV).
    # Force a common *Delta_v-based grid* for all models.
    dt_traj = hbar / (30 * Delta_v)   # valley-resolution grid, common to M1/M1V/M2
    N = max(int(T_traj / dt_traj), 500)
    N = min(N, N_max)
    t_grid = np.linspace(0, T_traj, N)
    x_traj = x_start + v * t_grid
    
    # interpolate the charge noise onto t_grid
    if enable_charge_noise:
        # dx_t is a long trace supplied by the caller (must be long enough)
        # we use the first N_step samples
        if len(dx_t) >= N:
            dx_use = dx_t[:N]
        else:
            # pad if too short (this case should not occur)
            dx_use = np.concatenate([dx_t, np.zeros(N - len(dx_t))])
        x_traj_r = x_traj + dx_use
    else:
        x_traj_r = x_traj
    
    Bz_t = np.asarray(ff.B_z(jnp.asarray(x_traj_r)))
    Bx_t = np.asarray(ff.B_x(jnp.asarray(x_traj_r)))
    # Zeeman convention (see REPRODUCIBILITY.md, Sec. 7):
    #   "stray-mean"  (legacy): stray field only; resonance E_Z = mean(Ez_t)
    #   "total-local": B_ext included; resonance uses local E_Z(x)
    #   "total-mean" : B_ext included; resonance E_Z = mean(Ez_t)
    if ez_convention in ("total-local", "total-mean"):
        Bz_t = Bz_t + Defaults.B_ext_T
    Ez_t = g_Si * mu_B * Bz_t
    Omx_t = g_Si * mu_B * Bx_t
    
    depth = Ev_baseline - Ev_min
    if enable_valley:
        Ev_t = Ev_baseline - depth * np.exp(-(x_traj - pocket_x_center)**2 / (2 * pocket_width**2))
        Delta_v_eff = Delta_v
    else:
        Ev_t = np.zeros_like(x_traj)
        Delta_v_eff = 0.0
    
    if enable_valley and lambda_0 > 0:
        # --- Coordinate convention for the coupling kernel (audit r5 par.8) ---
        # The SPIN Hamiltonian uses the noisy coordinate (Ez_t, Omx_t above):
        # charge noise displaces the dot relative to the lab-frame magnet.
        # The valley landscape Ev_t and the kernel lambda_sv are evaluated at
        # the NOMINAL coordinate (they ride with the confined dot). For the
        # total-* conventions the kernel E_Z is therefore ALSO built from the
        # nominal coordinate, so that Ev(x) and E_Z(x) inside the resonance
        # window share one coordinate. The legacy convention keeps its exact
        # archived numerics: mean of the noisy-coordinate Ez_t.
        if ez_convention == "total-local":
            Bz_nom = np.asarray(ff.B_z(jnp.asarray(x_traj))) + Defaults.B_ext_T
            E_Z_baseline = g_Si * mu_B * Bz_nom      # local E_Z(x) at nominal coords
        elif ez_convention == "total-mean":
            Bz_nom = np.asarray(ff.B_z(jnp.asarray(x_traj))) + Defaults.B_ext_T
            E_Z_baseline = float(np.mean(g_Si * mu_B * Bz_nom))
        else:
            E_Z_baseline = float(np.mean(Ez_t))      # legacy: mean Zeeman (noisy coords)
        sigma_E_default = 10e-6 * 1.602176634e-19   # 10 μeV — narrow resonance window
        lambda_t = lambda_sv_profile(x_traj, ff, coupling_model, lambda_0,
                                      pocket_x_center=pocket_x_center,
                                      sigma_lambda=pocket_width,
                                      Ev_x=Ev_t, E_Z=E_Z_baseline,
                                      sigma_E=sigma_E_default,
                                      profile_norm=profile_norm)
    else:
        lambda_t = np.zeros_like(t_grid)
    
    # initial state |+>_s (x) |0>_v
    psi = np.zeros(4, dtype=complex)
    psi[0] = 1.0 / np.sqrt(2)
    psi[2] = 1.0 / np.sqrt(2)
    
    for i in range(N - 1):
        dt_step = t_grid[i+1] - t_grid[i]
        Ez_mid = 0.5*(Ez_t[i]+Ez_t[i+1])
        Omx_mid = 0.5*(Omx_t[i]+Omx_t[i+1])
        Ev_mid = 0.5*(Ev_t[i]+Ev_t[i+1])
        lam_mid = 0.5*(lambda_t[i]+lambda_t[i+1])
        H = hamiltonian_4lvl_with_coupling(Ez_mid, Omx_mid, Ev_mid,
                                            Delta_v_eff, lam_mid)
        U = expm(-1j * H * dt_step / hbar)
        psi = U @ psi
    
    # observables
    rho = state_to_rho(psi)
    rho_s = partial_trace_valley(rho)
    rho_v = partial_trace_spin(rho)
    P_v_dia = float(rho_v[1, 1].real)
    # valley sector adiabatic excited state
    Hv_final = 0.5 * Ev_t[-1] * TAU_Z + Delta_v_eff * TAU_X
    eig_v, vecs_v = np.linalg.eigh(Hv_final)
    v_exc = vecs_v[:, 1]
    P_v_ad = float(np.real(np.vdot(v_exc, rho_v @ v_exc)))
    phase = float(np.angle(rho_s[0, 1])) if abs(rho_s[0, 1]) > 1e-12 else 0.0
    spin_purity = float(np.real(np.trace(rho_s @ rho_s)))
    S_s = von_neumann_entropy(rho_s)
    
    return {
        "P_v_dia": P_v_dia, "P_v_ad": P_v_ad,
        "phase": phase, "spin_purity": spin_purity, "S_s": S_s,
    }


def run_one_condition(v, case_label, pocket_x_center,
                       lambda_0, coupling_model,
                       n_real, noise, ff,
                       Ev_baseline, Ev_min, pocket_width, Delta_v,
                       N_max, base_seed, T_traj_for_noise,
                      ez_convention="stray-mean", profile_norm="prefactor"):
    """CRN structure: for one (case, v, lambda, model) all *three models* use the same noise trace.
    
    For each realization r:
      seed = stable_seed(case, v, λ, coupling_model, r)
      -> generate dx_t from that seed
      ─> M1 (charge_noise=True, valley=False), M1V (False, True), M2 (True, True)
         all use the same dx_t
    """
    # generate noise trace -- N steps plus margin
    dt_noise = 1e-8
    # buffer ~ T_traj x 50
    T_long = max(50.0 / noise.f_low, 50 * T_traj_for_noise)
    T_long = min(T_long, 5e-3)
    N_buffer = int(T_long / dt_noise)
    
    # accumulate results: per-model lists of observables
    results = {"M1": [], "M1V": [], "M2": []}
    
    # M1V is charge-noise-off and therefore deterministic.
    # Compute it *once* outside the realization loop and reuse it.
    # It depends only on lambda_0, coupling_model, pocket_x_center, v.
    # Repeating n_real times would waste n_real-fold runtime.
    x_start_det = pocket_x_center - 4 * pocket_width
    x_end_det = pocket_x_center + 4 * pocket_width
    T_traj_det = (x_end_det - x_start_det) / v
    dt_traj_det = hbar / (30 * Delta_v)
    N_det = max(int(T_traj_det / dt_traj_det), 500)
    N_det = min(N_det, N_max)
    dx_zero_full = np.zeros(N_det)
    r1v_det = simulate_one_real(v, ff, pocket_x_center,
                                  Ev_baseline, Ev_min, pocket_width,
                                  Delta_v, lambda_0, coupling_model,
                                  False, True, dx_zero_full, N_max=N_max,
                                  ez_convention=ez_convention, profile_norm=profile_norm)
    
    for r in range(n_real):
        # CRN: the realization seed has no model dependence
        seed_r = stable_seed(case_label, v, lambda_0, coupling_model, r,
                              base=base_seed)
        rng_r = np.random.default_rng(seed_r)
        # generate one long trace
        _, dx_long = noise.generate(T_long, dt_noise, rng=rng_r)
        # noise start offset
        N_win_noise = int(T_traj_for_noise / dt_noise) + 2
        s = rng_r.integers(0, max(len(dx_long) - N_win_noise - 1, 1))
        t_win = np.arange(N_win_noise) * dt_noise
        dx_win = dx_long[s : s + N_win_noise]
        
        # compute t_grid (same logic as simulate_one_real)
        x_start = pocket_x_center - 4 * pocket_width
        x_end = pocket_x_center + 4 * pocket_width
        T_traj_this = (x_end - x_start) / v
        dt_traj = hbar / (30 * Delta_v)
        N_this = max(int(T_traj_this / dt_traj), 500)
        N_this = min(N_this, N_max)
        t_grid_this = np.linspace(0, T_traj_this, N_this)
        dx_t_interp = np.interp(t_grid_this, t_win, dx_win)
        
        # M1: charge_noise=True, valley=False, same dx_t
        r1 = simulate_one_real(v, ff, pocket_x_center,
                                Ev_baseline, Ev_min, pocket_width,
                                Delta_v, lambda_0, coupling_model,
                                True, False, dx_t_interp, N_max=N_max,
                               ez_convention=ez_convention, profile_norm=profile_norm)
        # M2: both on, same dx_t -- CRN
        r2 = simulate_one_real(v, ff, pocket_x_center,
                                Ev_baseline, Ev_min, pocket_width,
                                Delta_v, lambda_0, coupling_model,
                                True, True, dx_t_interp, N_max=N_max,
                               ez_convention=ez_convention, profile_norm=profile_norm)
        results["M1"].append(r1)
        results["M1V"].append(r1v_det)   # deterministic, identical across r
        results["M2"].append(r2)
    
    # ensemble statistics
    def aggregate(lst, key, circular=False):
        vals = np.array([r[key] for r in lst])
        if circular:
            R = float(abs(np.mean(np.exp(1j * vals))))
            return {"mean": float(np.mean(vals)),
                     "var_linear": float(np.var(vals)),
                     "var_circular": float(-2 * np.log(max(R, 1e-12))),
                     "R": R}
        return {"mean": float(np.mean(vals)),
                 "std": float(np.std(vals))}
    
    return {
        m: {
            "P_v_dia": aggregate(results[m], "P_v_dia"),
            "P_v_ad":  aggregate(results[m], "P_v_ad"),
            "phase":   aggregate(results[m], "phase", circular=True),
            "spin_purity": aggregate(results[m], "spin_purity"),
            "S_s": aggregate(results[m], "S_s"),
        } for m in ["M1", "M1V", "M2"]
    }


def main(n_real: int = 10, mode: str = "quick", skip_plots: bool = False,
         overwrite: bool = False):
    print("="*70)
    print(f"Phase 4.6 — Cross-term verification  [{mode}, n_real={n_real}]")
    print("="*70)
    
    # precomputed outputs are preserved by default.
    # if outputs exist and --overwrite is not given, write to a timestamped subdir.
    BASE_DIR = Path(__file__).resolve().parents[1] / "figures" / "phase4"
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    existing_metadata = BASE_DIR / f"phase4p6_metadata_{mode}.json"
    if existing_metadata.exists() and not overwrite:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        OUTPUT_DIR = BASE_DIR / f"run_{ts}_{mode}"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  [protect] existing {mode} run found at {BASE_DIR}/.")
        print(f"  [protect] saving new run to {OUTPUT_DIR}/ (use --overwrite to replace)")
    else:
        OUTPUT_DIR = BASE_DIR
        if overwrite and existing_metadata.exists():
            print(f"  [--overwrite] replacing existing {mode} run")
    
    # Baseline geometry (same as step 5)
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
    Delta_v = 0.5e-6 * e_C
    
    noise = OneOverFNoise(
        sigma_total=Defaults.sigma_dx_m, alpha=1.0,
        f_low=1e3, f_high=1e7,
    )
    
    cases = {
        "case_i_center": 0.0,
        "case_ii_edge":  +25e-9,
    }
    
    if mode == "quick":
        # sanity must finish within 1-2 minutes to be meaningful
        v_list = [1.0, 5.0, 10.0]
        lambda_list_uev = [0.0, 0.5, 1.0]
        coupling_models = ["A"]
        N_max = 500
    elif mode == "validate":
        # statistics check: at least three positive lambda points
        v_list = [1.0, 5.0, 10.0]
        lambda_list_uev = [0.0, 0.25, 0.5, 1.0]
        coupling_models = ["A"]
        N_max = 1000
    elif mode == "validate_lite":
        # generates a lightweight statistical output. The expm loop is the bottleneck and
        # may take several minutes in some environments.
        # for external reproducibility prefer quick (n_real=1, ~10 s).
        # the output itself is verifiable via SHA + invariance check without re-running.
        v_list = [5.0, 10.0]
        lambda_list_uev = [0.0, 0.25, 0.5, 1.0]
        coupling_models = ["A"]
        N_max = 500
    elif mode == "sensitivity":
        # lightweight mode to check A vs A_pocket
        v_list = [5.0, 10.0]
        lambda_list_uev = [0.0, 1.0]
        coupling_models = ["A", "A_pocket"]
        N_max = 500
    elif mode == "full_lite":
        # lightweight mode that also runs in external review environments
        v_list = [2.0, 5.0, 10.0, 20.0]
        lambda_list_uev = [0.0, 0.25, 0.5, 1.0]
        coupling_models = ["A", "A_pocket"]
        N_max = 500
    elif mode == "full_dev":
        # development-environment only; not guaranteed externally
        v_list = [2.0, 5.0, 10.0, 20.0]
        lambda_list_uev = [0.0, 0.25, 0.5, 1.0]
        coupling_models = ["A", "A_pocket"]
        N_max = 1000
    elif mode == "full":
        v_list = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        lambda_list_uev = [0.0, 0.25, 0.5, 1.0, 2.0]
        coupling_models = ["A", "A_pocket"]
        N_max = 2000
    else:
        raise ValueError(f"unknown mode: {mode}")
    
    print(f"  v_list: {v_list} m/s")
    print(f"  λ_list: {lambda_list_uev} μeV")
    print(f"  coupling models: {coupling_models}")
    print(f"  n_real per condition: {n_real}")
    print(f"  CRN: M1 and M2 share the same charge-noise trace. "
          f"M1V is charge-noise-off, a deterministic valley reference.")
    print(f"  Cross-term axes:")
    print(f"    ΔP_v   = ⟨P_v^M2⟩_δx - P_v^M1V  (charge noise → valley)")
    print(f"    Δχ_φ   = χ_circ^M2 - χ_circ^M1   (valley → spin dephasing)")
    
    # result structure: results[coupling_model][case][v][lambda] = {M1, M1V, M2}
    results = {}
    total_conditions = len(coupling_models) * len(cases) * len(v_list) * len(lambda_list_uev)
    done = 0
    t0 = time.time()
    for cm in coupling_models:
        results[cm] = {}
        for case_label, x_c in cases.items():
            results[cm][case_label] = {}
            for v in v_list:
                results[cm][case_label][v] = {}
                # compute T_traj (sets the noise buffer)
                T_traj_for_noise = (8 * pocket_width) / v
                for lam_uev in lambda_list_uev:
                    lambda_0 = lam_uev * 1e-6 * e_C
                    R = run_one_condition(
                        v=v, case_label=case_label, pocket_x_center=x_c,
                        lambda_0=lambda_0, coupling_model=cm,
                        n_real=n_real, noise=noise, ff=ff,
                        Ev_baseline=Ev_baseline, Ev_min=Ev_min,
                        pocket_width=pocket_width, Delta_v=Delta_v,
                        N_max=N_max, base_seed=29,
                        T_traj_for_noise=T_traj_for_noise,
                    )
                    results[cm][case_label][v][lam_uev] = R
                    done += 1
                    print(f"  [{done}/{total_conditions}] cm={cm}, {case_label}, "
                          f"v={v}, λ={lam_uev}μeV: "
                          f"S_M2={R['M2']['S_s']['mean']:.3f}, "
                          f"P_v_dia_M2={R['M2']['P_v_dia']['mean']:.3e}, "
                          f"P_v_dia_M1V={R['M1V']['P_v_dia']['mean']:.3e}")
    print(f"\n  Total time: {time.time()-t0:.1f} sec")
    
    # ============================================================
    # Cross-term analysis (after CRN) -- both axes needed for a trade-off
    # ============================================================
    # M1V vs M2 difference in P_v_dia = effect of charge noise on valley dynamics
    # M1 vs M2 difference in phase variance = effect of valley on charge-noise dephasing
    # both axes must be *non-zero* and *correlated* to call it a "trade-off".
    print("\n" + "="*70)
    print("Cross-term (CRN) -- two axes")
    print("  dP_v   = <P_v^M2>_dx - P_v^M1V  (effect of charge noise on valley leakage)")
    print("  dchi   = chi_circ^M2 - chi_circ^M1  (effect of valley on charge-noise dephasing)")
    print("="*70)
    for cm in coupling_models:
        print(f"\n  Coupling model: {cm}")
        for case_label in cases.keys():
            print(f"  --- {case_label} ---")
            print(f"    {'v':>5}  {'λ[μeV]':>7}  | "
                  f"{'ΔP_v_dia':>14}  {'Δχ_φ_circ':>14}  | "
                  f"{'S_M2':>7}  {'S_M1V':>7}")
            for v in v_list:
                for lam_uev in lambda_list_uev:
                    R = results[cm][case_label][v][lam_uev]
                    dP = R["M2"]["P_v_dia"]["mean"] - R["M1V"]["P_v_dia"]["mean"]
                    # spin-side cross-term: M2 vs M1 (both charge-noise on, valley differs)
                    dchi = R["M2"]["phase"]["var_circular"] - R["M1"]["phase"]["var_circular"]
                    print(f"    {v:5.1f}  {lam_uev:>7.2f}  | "
                          f"{dP:>14.3e}  {dchi:>14.3e}  | "
                          f"{R['M2']['S_s']['mean']:>7.4f}  "
                          f"{R['M1V']['S_s']['mean']:>7.4f}")
    
    # lambda^2 scaling check -- at small lambda, dP ~ lambda^2
    print("\n" + "="*70)
    print("lambda^2 scaling (perturbation regime): dP ~ lambda^2 at small lambda?")
    print("="*70)
    for cm in coupling_models:
        for case_label in cases.keys():
            for v in v_list:
                # log-log fit over lambda = 0.25, 0.5, 1.0
                lam_uev_subset = [l for l in lambda_list_uev if 0 < l <= 1.0]
                if len(lam_uev_subset) < 2:
                    continue
                dPs = []
                for lam_uev in lam_uev_subset:
                    R = results[cm][case_label][v][lam_uev]
                    dP = abs(R["M2"]["P_v_dia"]["mean"] - R["M1V"]["P_v_dia"]["mean"])
                    dPs.append(dP)
                dPs = np.array(dPs)
                lams = np.array(lam_uev_subset)
                # ideal log-log slope = 2
                if np.all(dPs > 1e-20):
                    log_l = np.log(lams)
                    log_p = np.log(dPs)
                    slope = float(np.polyfit(log_l, log_p, 1)[0])
                else:
                    slope = float('nan')
                print(f"  {cm}, {case_label}, v={v:4.1f}:  "
                      f"ΔP log-log slope vs λ = {slope:+.2f}  "
                      f"(perturbation theory: +2)")
    
    # ============================================================
    # Plots — λ sweep visualization
    # ============================================================
    # --no-plots can skip this (for cold matplotlib environments)
    if skip_plots:
        print("\n  [--no-plots] skipping plot generation; saving JSON/CSV/metadata only")
    else:
     for cm in coupling_models:
        fig, axes = plt.subplots(3, 2, figsize=(13, 12))
        for case_idx, case_label in enumerate(cases.keys()):
            col = case_idx
            # row 0: dP_v_dia vs lambda for each v (valley-side cross-term)
            ax = axes[0, col]
            for v in v_list:
                dPs = [
                    abs(results[cm][case_label][v][lam]["M2"]["P_v_dia"]["mean"] -
                        results[cm][case_label][v][lam]["M1V"]["P_v_dia"]["mean"])
                    for lam in lambda_list_uev
                ]
                ax.loglog(np.maximum(lambda_list_uev, 1e-4), np.maximum(dPs, 1e-12),
                          marker="o", lw=1.5, label=f"v={v} m/s")
            if len(lambda_list_uev) > 1 and lambda_list_uev[1] > 0:
                lam_arr = np.array([l for l in lambda_list_uev if l > 0])
                ref = lam_arr**2 * 1e-3 / max(lam_arr.max()**2, 1e-12)
                ax.loglog(lam_arr, ref, "k:", alpha=0.4, label="$\\propto \\lambda^2$")
            ax.set_xlabel("$\\lambda_0$ [μeV]")
            ax.set_ylabel("|ΔP_v| (M2 - M1V)")
            ax.set_title(f"{case_label}: valley cross-term")
            ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
            # row 1: dchi vs lambda -- spin-side cross-term
            ax = axes[1, col]
            for v in v_list:
                dchis = [
                    abs(results[cm][case_label][v][lam]["M2"]["phase"]["var_circular"] -
                        results[cm][case_label][v][lam]["M1"]["phase"]["var_circular"])
                    for lam in lambda_list_uev
                ]
                ax.semilogy(lambda_list_uev, np.maximum(dchis, 1e-12),
                            marker="s", lw=1.5, label=f"v={v} m/s")
            ax.set_xlabel("$\\lambda_0$ [μeV]")
            ax.set_ylabel("|Δχ_φ_circ| (M2 - M1)")
            ax.set_title(f"{case_label}: spin dephasing cross-term")
            ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
            # row 2: S_s_M2 vs lambda -- correlation (deterministic, not a cross-term)
            ax = axes[2, col]
            for v in v_list:
                S_vals = [results[cm][case_label][v][lam]["M2"]["S_s"]["mean"]
                          for lam in lambda_list_uev]
                ax.plot(lambda_list_uev, S_vals, marker="o", lw=1.5, label=f"v={v} m/s")
            ax.set_xlabel("$\\lambda_0$ [μeV]")
            ax.set_ylabel("$S_s(\\rho_s)$ [bits]")
            ax.set_title(f"{case_label}: entanglement (deterministic, not cross-term)")
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.suptitle(f"Phase 4.6 ({cm}): CRN cross-term — two axes + entanglement, "
                     f"n_real={n_real}, mode={mode}", fontsize=12)
        fig.tight_layout()
        FIG_DIR_P4 = OUTPUT_DIR
        
        out = str(FIG_DIR_P4 / f"phase4p6_{cm}_crossterm_{mode}.png")
        fig.savefig(out, dpi=130)
        print(f"  saved: {out}")
    
    # trade-off scatter -- x=dP_v, y=dchi, color=v, marker=lambda
    # this figure is needed to discuss a "trade-off"
    if not skip_plots and n_real >= 5:   # meaningless for quick (n=1) / sensitivity (n=2)
        fig, axes = plt.subplots(len(coupling_models), len(cases),
                                  figsize=(7*len(cases), 5*len(coupling_models)),
                                  squeeze=False)
        v_arr = np.array(v_list)
        cmap = plt.get_cmap("viridis")
        markers = ["o", "s", "^", "D", "v"]
        for row, cm in enumerate(coupling_models):
            for col, case_label in enumerate(cases.keys()):
                ax = axes[row, col]
                for v_idx, v in enumerate(v_list):
                    color = cmap(v_idx / max(len(v_list)-1, 1))
                    for lam_idx, lam_uev in enumerate(lambda_list_uev):
                        marker = markers[lam_idx % len(markers)]
                        R = results[cm][case_label][v][lam_uev]
                        dP = R["M2"]["P_v_dia"]["mean"] - R["M1V"]["P_v_dia"]["mean"]
                        dchi = R["M2"]["phase"]["var_circular"] - R["M1"]["phase"]["var_circular"]
                        ax.plot(dP, dchi, marker=marker, color=color,
                                ms=10, alpha=0.8,
                                label=f"v={v} λ={lam_uev}" if (row==0 and col==0) else None)
                ax.axhline(0, color="k", lw=0.5, alpha=0.3)
                ax.axvline(0, color="k", lw=0.5, alpha=0.3)
                ax.set_xlabel(r"$\Delta P_v$ (valley)")
                ax.set_ylabel(r"$\Delta\chi_\phi$ (spin dephasing) [rad²]")
                ax.set_title(f"{cm} — {case_label}")
                ax.grid(alpha=0.3)
                if row==0 and col==0:
                    ax.legend(fontsize=6, ncol=2, loc="best")
        fig.suptitle(f"Phase 4.6 trade-off scatter — n_real={n_real}, mode={mode}\n"
                     f"(color = v, marker = λ; checking if ΔP_v and Δχ_φ are correlated)",
                     fontsize=11)
        fig.tight_layout()
        out = str(FIG_DIR_P4 / f"phase4p6_tradeoff_{mode}.png")
        fig.savefig(out, dpi=130)
        print(f"  saved: {out}")
    
    # save results JSON -- allows re-analysis after a long run
    import json
    def _jsonify(o):
        if isinstance(o, dict):
            return {str(k): _jsonify(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_jsonify(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    FIG_DIR_P4 = OUTPUT_DIR
    
    json_out = FIG_DIR_P4 / f"phase4p6_results_{mode}.json"
    with open(json_out, "w") as f:
        json.dump({
            "mode": mode, "n_real": n_real,
            "v_list": v_list, "lambda_list_uev": lambda_list_uev,
            "coupling_models": coupling_models, "N_max": N_max,
            "results": _jsonify(results),
        }, f, indent=2)
    print(f"  saved: {json_out}")
    
    # save summary CSV -- makes trade-off / paper figures easy to build
    csv_out = FIG_DIR_P4 / f"phase4p6_summary_{mode}.csv"
    with open(csv_out, "w") as f:
        f.write("mode,coupling_model,case,v_ms,lambda_uev,"
                "delta_P_v,delta_chi_phi,"
                "S_M2,S_M1V,"
                "P_v_M2,P_v_M1V,"
                "chi_M2,chi_M1\n")
        for cm in coupling_models:
            for case_label in cases.keys():
                for v in v_list:
                    for lam_uev in lambda_list_uev:
                        R = results[cm][case_label][v][lam_uev]
                        dP = R["M2"]["P_v_dia"]["mean"] - R["M1V"]["P_v_dia"]["mean"]
                        dchi = R["M2"]["phase"]["var_circular"] - R["M1"]["phase"]["var_circular"]
                        f.write(f"{mode},{cm},{case_label},{v},{lam_uev},"
                                f"{dP:.6e},{dchi:.6e},"
                                f"{R['M2']['S_s']['mean']:.6e},"
                                f"{R['M1V']['S_s']['mean']:.6e},"
                                f"{R['M2']['P_v_dia']['mean']:.6e},"
                                f"{R['M1V']['P_v_dia']['mean']:.6e},"
                                f"{R['M2']['phase']['var_circular']:.6e},"
                                f"{R['M1']['phase']['var_circular']:.6e}\n")
    print(f"  saved: {csv_out}")
    
    # response-map 4-quadrant summary, generated automatically.
    # apply a threshold so small negatives are not classified as robust.
    # quick mode (n_real=1, dchi==0) is analysis_valid=false.
    P_MIN_THRESH = 1e-4
    CHI_MIN_THRESH = 1e-3
    analysis_valid = (n_real >= 2)   # n_real=1 gives inaccurate ensemble observables
    print("\n" + "="*70)
    print("  response-map 4-quadrant classification")
    print(f"  threshold: |ΔP_v| ≥ {P_MIN_THRESH}, |Δχ_φ| ≥ {CHI_MIN_THRESH}")
    print(f"  analysis_valid = {analysis_valid} (n_real={n_real}: " +
          ("response-map statistically interpretable" if analysis_valid else "n_real=1: not interpretable") + ")")
    print("="*70)
    quadrants = {
        "robust_PvNeg_chiNeg": [],   # both improve -- robust-region candidate
        "PvPos_chiNeg":        [],   # valley worsens, spin improves
        "PvNeg_chiPos":        [],   # valley improves, spin worsens
        "both_worsen":         [],   # both worsen — avoid
        "below_threshold":     [],   # below-threshold quadrant
    }
    for cm in coupling_models:
        for case_label in cases.keys():
            for v in v_list:
                for lam_uev in lambda_list_uev:
                    if lam_uev == 0.0:
                        continue   # lambda=0 is always 0, no classification
                    R = results[cm][case_label][v][lam_uev]
                    dP = R["M2"]["P_v_dia"]["mean"] - R["M1V"]["P_v_dia"]["mean"]
                    dchi = R["M2"]["phase"]["var_circular"] - R["M1"]["phase"]["var_circular"]
                    entry = {
                        "cm": cm, "case": case_label, "v": v, "lam_uev": lam_uev,
                        "dP_v": dP, "dchi_phi": dchi,
                    }
                    # below-threshold cases go to the below_threshold quadrant
                    sig_P = abs(dP) >= P_MIN_THRESH
                    sig_chi = abs(dchi) >= CHI_MIN_THRESH
                    if not (sig_P or sig_chi):
                        quadrants["below_threshold"].append(entry)
                        continue
                    # if only one axis passes, treat the other as 0
                    eff_P = dP if sig_P else 0.0
                    eff_chi = dchi if sig_chi else 0.0
                    if eff_P < 0 and eff_chi < 0:
                        quadrants["robust_PvNeg_chiNeg"].append(entry)
                    elif eff_P > 0 and eff_chi < 0:
                        quadrants["PvPos_chiNeg"].append(entry)
                    elif eff_P < 0 and eff_chi > 0:
                        quadrants["PvNeg_chiPos"].append(entry)
                    else:
                        quadrants["both_worsen"].append(entry)
    n_total = sum(len(v) for v in quadrants.values())
    for qname, entries in quadrants.items():
        frac = len(entries) / n_total if n_total > 0 else 0
        print(f"  {qname:<30s}: {len(entries):>3d} ({100*frac:5.1f}%)")
    print(f"  TOTAL conditions (λ>0): {n_total}")
    if len(quadrants["robust_PvNeg_chiNeg"]) > 0:
        print(f"\n  ROBUST region candidates (top 5 by |dP_v + dchi_phi|):")
        rob = sorted(quadrants["robust_PvNeg_chiNeg"],
                      key=lambda e: -(abs(e["dP_v"]) + abs(e["dchi_phi"])))
        for e in rob[:5]:
            print(f"    {e['cm']:<10s} {e['case']:<15s} v={e['v']:>5.1f}  λ={e['lam_uev']:>5.2f}  "
                  f"ΔP_v={e['dP_v']:+.2e}  Δχ_φ={e['dchi_phi']:+.2e}")
    else:
        print(f"\n  (no robust region candidates in this mode)")
    
    # save 4-quadrant summary CSV
    summary_q_out = FIG_DIR_P4 / f"phase4p6_response_map_{mode}.csv"
    with open(summary_q_out, "w") as f:
        f.write("quadrant,coupling_model,case,v_ms,lambda_uev,dP_v,dchi_phi\n")
        for qname, entries in quadrants.items():
            for e in entries:
                f.write(f"{qname},{e['cm']},{e['case']},{e['v']},{e['lam_uev']},"
                        f"{e['dP_v']:.6e},{e['dchi_phi']:.6e}\n")
    print(f"  saved: {summary_q_out}")
    
    # lambda=0 invariance auto-check
    # for a separable Hamiltonian, dP_v, dchi, S_M2 should all be ~ 0
    print("\n" + "="*70)
    print("  lambda=0 invariance check")
    print("="*70)
    invariance_violations = []
    for cm in coupling_models:
        for case_label in cases.keys():
            for v in v_list:
                if 0.0 not in lambda_list_uev:
                    continue
                R = results[cm][case_label][v][0.0]
                dP = abs(R["M2"]["P_v_dia"]["mean"] - R["M1V"]["P_v_dia"]["mean"])
                dchi = abs(R["M2"]["phase"]["var_circular"] - R["M1"]["phase"]["var_circular"])
                S_M2 = abs(R["M2"]["S_s"]["mean"])
                # tolerances
                if dP > 1e-10 or dchi > 1e-6 or S_M2 > 1e-8:
                    invariance_violations.append({
                        "cm": cm, "case": case_label, "v": v,
                        "dP": dP, "dchi": dchi, "S_M2": S_M2,
                    })
    max_dP = max([abs(results[cm][cl][v][0.0]["M2"]["P_v_dia"]["mean"] -
                       results[cm][cl][v][0.0]["M1V"]["P_v_dia"]["mean"])
                  for cm in coupling_models for cl in cases.keys() for v in v_list
                  if 0.0 in lambda_list_uev], default=0.0)
    max_dchi = max([abs(results[cm][cl][v][0.0]["M2"]["phase"]["var_circular"] -
                         results[cm][cl][v][0.0]["M1"]["phase"]["var_circular"])
                    for cm in coupling_models for cl in cases.keys() for v in v_list
                    if 0.0 in lambda_list_uev], default=0.0)
    max_S = max([abs(results[cm][cl][v][0.0]["M2"]["S_s"]["mean"])
                 for cm in coupling_models for cl in cases.keys() for v in v_list
                 if 0.0 in lambda_list_uev], default=0.0)
    print(f"    max |ΔP_v|   = {max_dP:.3e}  (tol 1e-10)")
    print(f"    max |Δχ_φ|   = {max_dchi:.3e}  (tol 1e-6)")
    print(f"    max  S_M2    = {max_S:.3e}  (tol 1e-8)")
    invariance_pass = len(invariance_violations) == 0
    print(f"    invariance PASS: {invariance_pass}")
    if not invariance_pass:
        print(f"  WARNING: {len(invariance_violations)} violations — "
              f"DO NOT use this run for claims")
        for v_ in invariance_violations[:3]:
            print(f"    {v_}")
    
    # update metadata -- written to a separate file, not the CSV
    metadata_out = FIG_DIR_P4 / f"phase4p6_metadata_{mode}.json"
    import datetime, hashlib
    script_path = Path(__file__).resolve()
    with open(script_path, "rb") as f:
        script_sha = hashlib.sha256(f.read()).hexdigest()[:16]
    with open(metadata_out, "w") as f:
        json.dump({
            "archive_version": ARCHIVE_VERSION,
            "script_sha256_16": script_sha,
            "mode": mode, "n_real": n_real,
            "N_max": N_max,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "lambda0_invariance": {
                "pass": invariance_pass,
                "max_dP_v": max_dP, "max_dchi_phi": max_dchi, "max_S_M2": max_S,
                "violation_count": len(invariance_violations),
            },
            # record response-map analysis validity + thresholds
            "response_map_analysis": {
                "analysis_valid": analysis_valid,
                "threshold_P_v": P_MIN_THRESH,
                "threshold_chi_phi": CHI_MIN_THRESH,
                "robust_count": len(quadrants["robust_PvNeg_chiNeg"]),
                "below_threshold_count": len(quadrants["below_threshold"]),
            },
            # flat aliases for convenient external analysis
            "lambda0_invariance_pass": invariance_pass,
            "analysis_valid": analysis_valid,
            "robust_count": len(quadrants["robust_PvNeg_chiNeg"]),
        }, f, indent=2)
    print(f"  saved: {metadata_out}")
    
    # mode-dependent interpretation limits
    print("\n" + "="*70)
    if mode == "quick":
        print("  NOTE: quick mode uses n_real=1. dchi and lambda^2 scaling are not")
        print("        statistically interpretable (ensemble observables). Code sanity only.")
        print("        For statistics use --mode validate_lite (lightweight, external review)")
        print("        or --mode validate, --mode full_lite, --mode full.")
    elif mode == "sensitivity":
        print("  NOTE: sensitivity mode uses n_real=2. For checking A vs A_pocket")
        print("        structural differences. Quantitative claims need full_lite/full.")
    elif mode == "validate_lite":
        print(f"  NOTE: validate_lite (n_real={n_real}, v=[5,10]). Lightweight for external review.")
        print("        Quantitative conclusions need --mode full_lite or --mode full.")
    elif mode == "validate":
        print(f"  NOTE: validate (n_real={n_real}, v=[1,5,10]). Runtime is environment-dependent.")
        print("        May be slow in external review environments -- prefer validate_lite.")
    elif mode == "full_lite":
        print(f"  NOTE: full_lite (n_real={n_real}, A+A_pocket, 4 v points). Statistics + model")
        print(f"        comparison mode. Trade-off analysis possible. Lighter than full.")
    elif mode == "full":
        print(f"  NOTE: full mode (n_real={n_real}). Results saved to JSON.")
    print()
    # signed-quantity caveat
    print("  CAVEAT:")
    print("    - the lambda^2 slope is a |dP_v| log-log fit. Where the sign changes it")
    print("      reflects *magnitude only*; physical sign interpretation is separate.")
    print("    - dchi is signed. A negative value means valley coupling reduced the")
    print("      phase variance -- possibly one direction of a trade-off; do not over-generalize.")
    print("="*70)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4.6 — cross-term verification")
    parser.add_argument("--mode",
                        choices=["quick", "validate_lite", "validate",
                                  "sensitivity", "full_lite", "full_dev", "full"],
                        default="quick",
                        help="quick (sanity) | validate_lite | validate | "
                             "sensitivity (A vs A_pocket) | "
                             "full_lite (external review, n_real=5) | "
                             "full_dev (development only, n_real=10) | "
                             "full (n_real=30, long-run)")
    parser.add_argument("--n_real", type=int, default=None,
                        help="override n_real")
    parser.add_argument("--no-plots", action="store_true",
                        help="skip plot generation "
                             "(fast sanity in cold matplotlib-font-cache environments). "
                             "JSON, CSV, metadata are still saved.")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite existing figures/phase4/ outputs. "
                             "default writes to a timestamped subdir to preserve precomputed data.")
    args = parser.parse_args()
    
    default_n_real = {"quick": 1, "validate_lite": 5,
                       "validate": 10, "sensitivity": 2,
                       "full_lite": 5, "full_dev": 10, "full": 30}
    n_real = args.n_real if args.n_real is not None else default_n_real[args.mode]
    main(n_real=n_real, mode=args.mode, skip_plots=args.no_plots,
         overwrite=args.overwrite)
