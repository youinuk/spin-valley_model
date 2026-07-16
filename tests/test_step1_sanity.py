"""
Sanity tests for the deterministic-phase and quasi-static-dephasing limits.

Checks:
  (A) Prediction: under constant-velocity shuttling, phi_mod(T) returns to 0 each
      period (bounded oscillation).
        phi_mod(T) = (g μ_B dB / ℏ) · (a / 2π v) · sin(2π v T / a)
  (B) Stationary dot + quasi-static charge-noise ensemble -> position jitter produces
      Larmor-frequency jitter through the longitudinal gradient, so the coherence
      decays as a Gaussian:
        C(T) = <exp(-i δφ)> = exp(-½ <δφ²>)
"""

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constants import Defaults, FIG_DIR, g_Si, mu_B, hbar
from field_landscape import PeriodicField, LinearGradientField


# ============================================================
# (A) Bounded-oscillation check
# ============================================================
def integrate_phase_periodic(v_ms: float, T_total_s: float, n_steps: int = 20000):
    """Numerically integrate the modulation phase of constant-velocity shuttling x(t)=v*t.
    
    phi_mod(T) = ∫₀ᵀ (g μ_B / ℏ) · [B_long(x(t)) − B_ext] dt
               = ∫₀ᵀ (g μ_B / ℏ) · dB · cos(2π v t / a) dt
    """
    field = PeriodicField()
    t = jnp.linspace(0.0, T_total_s, n_steps)
    x_of_t = v_ms * t
    dB_only = field.B_long(x_of_t) - field.B_ext
    omega_mod = (g_Si * mu_B / hbar) * dB_only
    # trapezoidal integration
    phi_t = jnp.concatenate([
        jnp.array([0.0]),
        jnp.cumsum(0.5 * (omega_mod[1:] + omega_mod[:-1]) * (t[1:] - t[:-1]))
    ])
    # analytic expression
    a = field.period
    phi_analytic = (g_Si * mu_B * field.dB_long / hbar) * (a / (2 * jnp.pi * v_ms)) \
                   * jnp.sin(2 * jnp.pi * v_ms * t / a)
    return np.asarray(t), np.asarray(phi_t), np.asarray(phi_analytic)


def check_A_bounded_oscillation():
    """
    The real definition of bounded: |phi|_max does not grow with the number of shuttles N.
    
    Analytic expression:
        |phi_mod|_max(N) = (g μ_B dB / ℏ) · a / (2π v)
    This value is *independent* of N. So measure |phi|_max at N = 1, 5, 10, 50, 100
    and check that the spread is small. As a control, show that on a disorder/gradient
    landscape |phi|_max(N) grows linearly / as a square root with N.
    """
    print("="*60)
    print("(A) Bounded oscillation: is |phi|_max(N) independent of N?")
    print("="*60)
    v = Defaults.v_sweet_lo  # 20 m/s
    field = PeriodicField()
    T_period = field.period / v
    
    # analytic prediction
    phi_max_predicted = (g_Si * mu_B * field.dB_long / hbar) * field.period / (2 * jnp.pi * v)
    print(f"  v = {v} m/s,  period a = {field.period*1e9:.0f} nm,  T_period = {T_period*1e12:.2f} ps")
    print(f"  analytic prediction |phi|_max = {float(phi_max_predicted):.4f} rad  (independent of N)")

    # 1. measure |phi|_max on the periodic landscape as N grows 1 -> 100
    Ns = np.array([1, 2, 5, 10, 20, 50, 100])
    phi_max_periodic = []
    for N in Ns:
        t, phi_num, _ = integrate_phase_periodic(v, N * T_period, n_steps=400 * N)
        phi_max_periodic.append(np.max(np.abs(phi_num)))
    phi_max_periodic = np.array(phi_max_periodic)
    
    rel_spread = (phi_max_periodic.max() - phi_max_periodic.min()) / phi_max_periodic.mean()
    print(f"  periodic landscape, |phi|_max(N) =")
    for N, p in zip(Ns, phi_max_periodic):
        print(f"    N = {N:3d}:  |phi|_max = {p:.4f} rad  (predicted {float(phi_max_predicted):.4f})")
    print(f"  -> relative spread = {rel_spread*100:.2f} %  (small means bounded)")

    # 2. control: on a monotonic-gradient landscape |phi| should grow with N
    lin_field = LinearGradientField(grad=Defaults.grad_lo)
    phi_max_linear = []
    for N in Ns:
        T_tot = N * T_period
        n_steps = 400 * N
        t = jnp.linspace(0.0, T_tot, n_steps)
        x = v * t
        # B_long(x) − B_ext = grad · x
        domega = (g_Si * mu_B / hbar) * lin_field.grad * x
        phi_t = jnp.concatenate([
            jnp.array([0.0]),
            jnp.cumsum(0.5 * (domega[1:] + domega[:-1]) * (t[1:] - t[:-1]))
        ])
        phi_max_linear.append(float(jnp.max(jnp.abs(phi_t))))
    phi_max_linear = np.array(phi_max_linear)
    print(f"  control (monotonic gradient), |phi|_max(N):")
    for N, p in zip(Ns, phi_max_linear):
        print(f"    N = {N:3d}:  |phi|_max = {p:.3f} rad  (~N^2 or ~N means unbounded)")
    
    # figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # left: time trace (N=10)
    t_show, phi_show, phi_ana_show = integrate_phase_periodic(v, 10 * T_period, n_steps=4000)
    axes[0].plot(t_show * 1e9, phi_show, label="numerical", lw=1.2)
    axes[0].plot(t_show * 1e9, phi_ana_show, "--", label="analytic", lw=1.0, alpha=0.7)
    axes[0].axhline( float(phi_max_predicted), color="k", ls=":", alpha=0.4)
    axes[0].axhline(-float(phi_max_predicted), color="k", ls=":", alpha=0.4)
    axes[0].set_xlabel("t [ns]"); axes[0].set_ylabel(r"$\phi_\mathrm{mod}(t)$ [rad]")
    axes[0].set_title(f"Periodic landscape, v={v} m/s")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    # right: |phi|_max vs N -- periodic vs monotonic
    axes[1].plot(Ns, phi_max_periodic, "o-", label="periodic (bounded)", lw=1.5)
    axes[1].plot(Ns, phi_max_linear, "s-", label="linear gradient (grows)", lw=1.5)
    axes[1].axhline(float(phi_max_predicted), color="C0", ls=":", alpha=0.4)
    axes[1].set_xlabel("N (number of periods traversed)")
    axes[1].set_ylabel(r"$|\phi|_\mathrm{max}$ [rad]")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].legend(); axes[1].grid(alpha=0.3, which="both")
    axes[1].set_title("(A) Bounded vs unbounded growth")
    fig.tight_layout()
    out = str(FIG_DIR / "step1_A_bounded.png")
    fig.savefig(out, dpi=130)
    print(f"  figure saved: {out}")

    # pass criterion: relative spread of |phi|_max(N) on the periodic landscape < 5%
    passed = rel_spread < 0.05
    assert passed, f"rel_spread {rel_spread:.3f} exceeds 5%"
    return {
        "phi_max_predicted_rad": float(phi_max_predicted),
        "rel_spread_periodic": float(rel_spread),
        "passed": bool(passed),
    }


# ============================================================
# (B) Quasi-static charge noise → Gaussian T2* decay
# ============================================================
def quasi_static_T2_test():
    """
    Stationary dot under a monotonic gradient g_field, with position jitter dx ~ N(0, sigma_dx^2)
    quasi-static.
        δω = (g μ_B / ℏ) · g_field · δx
    coherence:
        C(T) = <exp(-i δω T)>_δx = exp(-½ σ_ω² T²)
    giving Gaussian decay with 1/e time T2* = sqrt(2) / sigma_w.
    """
    print("="*60)
    print("(B) Quasi-static charge noise → Gaussian T2* decay")
    print("="*60)
    g_field = Defaults.grad_hi  # 1 mT/nm
    sigma_dx = Defaults.sigma_dx_m  # 0.3 nm RMS

    sigma_omega = (g_Si * mu_B / hbar) * g_field * sigma_dx
    T2_predicted = np.sqrt(2) / sigma_omega
    print(f"  gradient g_field = {g_field*1e-6:.2f} T/um  (= {g_field*1e-9*1e3:.2f} mT/nm)")
    print(f"  position jitter sigma_dx = {sigma_dx*1e9:.2f} nm")
    print(f"  → σ_ω = {sigma_omega:.3e} rad/s")
    print(f"  -> predicted T2* = sqrt(2)/sigma_w = {T2_predicted*1e6:.3f} us")

    # Monte Carlo: N realizations of dx
    rng = np.random.default_rng(0)
    N = 20000
    dx_samples = rng.normal(0, sigma_dx, size=N)
    domega = (g_Si * mu_B / hbar) * g_field * dx_samples

    T_grid = np.linspace(0, 3 * T2_predicted, 200)
    coherence_num = np.array([
        np.abs(np.mean(np.exp(-1j * domega * T))) for T in T_grid
    ])
    coherence_analytic = np.exp(-0.5 * (sigma_omega * T_grid)**2)

    # extract T2* at the 1/e crossing
    idx_1e_num = np.argmin(np.abs(coherence_num - 1/np.e))
    idx_1e_ana = np.argmin(np.abs(coherence_analytic - 1/np.e))
    T2_num = T_grid[idx_1e_num]
    T2_ana = T_grid[idx_1e_ana]
    print(f"  extracted T2* (numerical MC):   {T2_num*1e6:.3f} us")
    print(f"  extracted T2* (analytic):       {T2_ana*1e6:.3f} us")
    print(f"  relative error:                 {abs(T2_num - T2_predicted)/T2_predicted*100:.2f} %")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(T_grid * 1e6, coherence_num, label=f"MC (N={N})", lw=1.5)
    ax.plot(T_grid * 1e6, coherence_analytic, "--", label="analytic Gaussian", lw=1.2)
    ax.axhline(1/np.e, color="k", ls=":", alpha=0.5, label="1/e")
    ax.axvline(T2_predicted*1e6, color="C2", ls=":", alpha=0.5,
               label=f"predicted T2*={T2_predicted*1e6:.2f} μs")
    ax.set_xlabel("T [μs]"); ax.set_ylabel("|C(T)|")
    ax.set_title("(B) Stationary dot, quasi-static charge noise")
    ax.legend(); ax.grid(alpha=0.3)
    out = str(FIG_DIR / "step1_B_T2star.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  figure saved: {out}")

    passed = abs(T2_num - T2_predicted) / T2_predicted < 0.10
    return {
        "T2_predicted_us": float(T2_predicted*1e6),
        "T2_numerical_us": float(T2_num*1e6),
        "passed": bool(passed),
    }


if __name__ == "__main__":
    res_A = check_A_bounded_oscillation()
    print()
    res_B = quasi_static_T2_test()
    print()
    print("="*60)
    print("Step 1 summary")
    print("="*60)
    print(f"  (A) bounded oscillation:   {'PASS' if res_A['passed'] else 'FAIL'}  "
          f"(predicted {res_A['phi_max_predicted_rad']:.3f} rad, relative spread of |phi|_max(N) {res_A['rel_spread_periodic']*100:.2f}%)")
    print(f"  (B) quasi-static T2*:      {'PASS' if res_B['passed'] else 'FAIL'}  "
          f"(predicted {res_B['T2_predicted_us']:.3f} us, numerical {res_B['T2_numerical_us']:.3f} us)")


def test_step1_bounded_oscillation():
    """pytest entry point (no return; assertion inside check function)."""
    check_A_bounded_oscillation()
