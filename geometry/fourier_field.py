"""
FourierField: a prism-derived differentiable periodic field.

Replaces the toy `PeriodicField` with:
    B_z(x) = B_0 + Σ_n A_n cos(n k x)
    B_x(x) = -sum_n C_n sin(n k x)            <-- prism sign convention
                k = 2π / a

Sign convention:
The toy `PeriodicField` used B_trans(x) = +b sin(kx), but the real
prism-array B_x peaks at the magnet *edges* with the opposite sign,
i.e. B_x(x) ~ -sin(kx) in the prism convention. This module follows
the prism sign throughout.

Harmonic order:
Which harmonics are retained depends on the duty cycle (magnet width / period):
  - duty 50% (200^3 nm / a=400 nm): odd orders only (1, 3, 5, ...)
  - duty != 50% (50^3 nm / a=150 nm): even orders also nonzero (1, 2, 3, ...)
N_harm = 3 fits both cases to ~1% residual.

Differentiability:
The field is plain cos/sin, so jax.grad applies directly.
The coefficients can be made learnable variables for inverse design.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field as dc_field
from typing import Tuple

from constants import FIG_DIR_PHASE3 as FIG_DIR_P3  # noqa: F401 (jax x64)


@dataclass(frozen=True)
class FourierField:
    """
    Fourier-prism field model with arbitrary harmonic content:
        B_z(x) = B0 + Σ_n A_n cos(n k x)
        B_x(x) = -Σ_n C_n sin(n k x)
        k = 2π / period

    Coefficients are passed as a tuple (immutable); use a dict/array to make them learnable.
    """
    period:    float                 # spatial period [m]
    B0:        float                 # DC offset of Bz [T]
    A_coeffs:  Tuple[float, ...]     # (A_1, A_2, A_3, ...)
    C_coeffs:  Tuple[float, ...]     # (C_1, C_2, C_3, ...)

    def k(self) -> float:
        return 2 * jnp.pi / self.period

    def B_z(self, x):
        kk = self.k()
        total = self.B0
        for n, A_n in enumerate(self.A_coeffs, start=1):
            total = total + A_n * jnp.cos(n * kk * x)
        return total

    def B_x(self, x):
        kk = self.k()
        total = jnp.zeros_like(jnp.asarray(x, dtype=float))
        for n, C_n in enumerate(self.C_coeffs, start=1):
            total = total - C_n * jnp.sin(n * kk * x)
        return total

    def dBz_dx(self, x):
        kk = self.k()
        total = jnp.zeros_like(jnp.asarray(x, dtype=float))
        for n, A_n in enumerate(self.A_coeffs, start=1):
            total = total - n * kk * A_n * jnp.sin(n * kk * x)
        return total

    def dBx_dx(self, x):
        kk = self.k()
        total = jnp.zeros_like(jnp.asarray(x, dtype=float))
        for n, C_n in enumerate(self.C_coeffs, start=1):
            total = total - n * kk * C_n * jnp.cos(n * kk * x)
        return total


def fit_from_prism_array(arr, z_eval: float, N_harm: int = 3,
                          n_samples: int = 200) -> FourierField:
    """
    Extract the Fourier descriptor from a PeriodicPrismArray via one-period sampling + FFT.

    Coefficient convention:
        Bz(x) = B0 + Σ A_n cos(n k x)
        Bx(x) = -sum C_n sin(n k x)   (prism sign convention)

    FFT: c_k = (1/N) Σ x_n exp(-2π i k n/N)
        x_n = c_0 + Σ_{k≥1} [2 Re(c_k) cos(...) - 2 Im(c_k) sin(...)]

    [note] This function uses numpy FFT and is *not* end-to-end
    differentiable with respect to geometry parameters. For inverse
    design use `fit_from_prism_array_jax` below.
    """
    a = arr.period_a
    x_arr = jnp.linspace(0.0, a, n_samples, endpoint=False)
    Bx_j, _, Bz_j = arr.B(x_arr, 0.0, z_eval)
    Bz_vals = np.asarray(Bz_j)
    Bx_vals = np.asarray(Bx_j)

    Bz_fft = np.fft.rfft(Bz_vals) / n_samples
    Bx_fft = np.fft.rfft(Bx_vals) / n_samples

    B0 = float(np.real(Bz_fft[0]))
    # Bz: cos expansion of an even function -> c_k real -> A_n = 2 Re(c_n)
    A_coeffs = tuple(float(2 * np.real(Bz_fft[n])) for n in range(1, N_harm + 1))
    # Bx: -sin expansion of an odd function:
    #   in the numpy convention c_k
    #   our convention Bx = -C_n sin(...) -> C_n = 2 Im(c_n)
    C_coeffs = tuple(float(2 * np.imag(Bx_fft[n])) for n in range(1, N_harm + 1))

    return FourierField(period=a, B0=B0, A_coeffs=A_coeffs, C_coeffs=C_coeffs)


def fit_from_prism_array_jax(arr, z_eval: float, N_harm: int = 3,
                              n_samples: int = 200):
    """
    *End-to-end JAX-differentiable* Fourier projection.

    Instead of the numpy FFT in fit_from_prism_array, use a direct projection integral:
        A_n = (2/N) Σ_j Bz(x_j; θ) cos(n k x_j)            (n ≥ 1)
        C_n = -(2/N) sum_j Bx(x_j; theta) sin(n k x_j)    (Bx = -C_n sin convention)
        B_0 = (1/N) Σ_j Bz(x_j; θ)
    Here theta = geometry parameters (half_x, half_z, period, depth, ...);
    if they are attributes of arr, their derivatives pass through.

    Returns
    -------
    dict {"B0": ..., "A": (A_1, ..., A_N), "C": (C_1, ..., C_N), "period": ...}
    
    All JAX-traceable scalars/tuples. To wrap in FourierField the caller
    must keep them un-unboxed (no float()) to preserve differentiability.
    """
    a = arr.period_a
    x_arr = jnp.linspace(0.0, a, n_samples, endpoint=False)
    Bx_j, _, Bz_j = arr.B(x_arr, 0.0, z_eval)
    kk = 2 * jnp.pi / a

    # B0 = mean(Bz)
    B0 = jnp.mean(Bz_j)
    
    # projection for n=1..N_harm
    n_arr = jnp.arange(1, N_harm + 1)
    # cos(n k x) basis, shape (N_harm, n_samples)
    cos_basis = jnp.cos(n_arr[:, None] * kk * x_arr[None, :])
    sin_basis = jnp.sin(n_arr[:, None] * kk * x_arr[None, :])
    # A_n = (2/N) Σ Bz cos(n k x)
    A = 2.0 * jnp.mean(Bz_j[None, :] * cos_basis, axis=1)
    # our convention: Bx = -sum C_n sin(n k x)
    # so in the Bx . sin integral: sum_j Bx_j sin(...) = -(N/2) C_n
    # → C_n = -(2/N) Σ Bx sin(...)
    C = -2.0 * jnp.mean(Bx_j[None, :] * sin_basis, axis=1)
    
    return {"B0": B0, "A": A, "C": C, "period": a}


# ============================================================
# Self-check: prism array vs FourierField fit
# ============================================================
def _main():
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from geometry.periodic_array import PeriodicPrismArray

    parser = argparse.ArgumentParser(description="FourierField (prism fit) self-check")
    parser.add_argument("--full", action="store_true",
                        help="full validation incl. jax.vmap(jax.grad) gradient check (slow)")
    args = parser.parse_args()
    quick = not args.full

    print("="*60)
    print(f"FourierField self-check  [{'QUICK' if quick else 'FULL'}]")
    print("="*60)

    Lx = 50e-9; Ly = 50e-9; Lz = 30e-9
    a = 150e-9
    zd = -50e-9
    N_harm = 3
    arr = PeriodicPrismArray(
        period_a=a, half_x=Lx/2, half_y=Ly/2, half_z=Lz/2,
        cz=Lz/2, N_periods_each_side=4, Ms=1.4e6,
    )
    print(f"  Baseline: {Lx*1e9:.0f}x{Ly*1e9:.0f}x{Lz*1e9:.0f} nm,  "
          f"a={a*1e9:.0f} nm,  z={zd*1e9:.0f} nm")
    print(f"  harmonic order: N_harm = {N_harm}")

    ff = fit_from_prism_array(arr, zd, N_harm=N_harm, n_samples=200)
    print(f"\n  fit result:")
    print(f"    B0    = {ff.B0*1e3:8.3f} mT")
    for n, A in enumerate(ff.A_coeffs, start=1):
        rel = A / ff.A_coeffs[0] * 100 if ff.A_coeffs[0] != 0 else 0.0
        print(f"    A_{n}   = {A*1e3:8.3f} mT   [A_{n}/A_1 = {rel:+6.1f}%]")
    for n, C in enumerate(ff.C_coeffs, start=1):
        rel = C / ff.C_coeffs[0] * 100 if ff.C_coeffs[0] != 0 else 0.0
        print(f"    C_{n}   = {C*1e3:8.3f} mT   [C_{n}/C_1 = {rel:+6.1f}%]")

    # check: prism vs FourierField at sample points
    # quick: 80 points, full: 300 points
    n_test = 80 if quick else 300
    x_test = jnp.linspace(0, a, n_test)
    Bx_arr_j, _, Bz_arr_j = arr.B(x_test, 0.0, zd)
    Bz_prism = np.asarray(Bz_arr_j)
    Bx_prism = np.asarray(Bx_arr_j)
    Bz_ff = np.asarray(ff.B_z(x_test))
    Bx_ff = np.asarray(ff.B_x(x_test))

    # max abs error  
    err_Bz_max = float(np.max(np.abs(Bz_prism - Bz_ff)))
    err_Bx_max = float(np.max(np.abs(Bx_prism - Bx_ff)))
    # normalize to the first-harmonic amplitude
    norm_Bz = sum(abs(a) for a in ff.A_coeffs)
    norm_Bx = sum(abs(c) for c in ff.C_coeffs)
    rel_Bz = err_Bz_max / norm_Bz if norm_Bz > 0 else float('inf')
    rel_Bx = err_Bx_max / norm_Bx if norm_Bx > 0 else float('inf')
    print(f"\n  Fit residual (N_harm={N_harm} FourierField vs prism array):")
    print(f"    max|dBz| = {err_Bz_max*1e3:.3f} mT  (rel to sum|A_n|: {rel_Bz*100:.2f}%)")
    print(f"    max|dBx| = {err_Bx_max*1e3:.3f} mT  (rel to sum|C_n|: {rel_Bx*100:.2f}%)")

    # Gradient residual -- charge-noise dephasing depends directly on dBz/dx,
    # and the transverse term depends on Bx(x), so dBx/dx is also checked.
    # quick: central finite difference on the prism field (light, eps=0.05 nm)
    # full:  jax.vmap(jax.grad) on the prism field (accurate but heavy)
    # geometry-parameter autodiff is checked separately in tests/test_geometry_autodiff.py,
    # so here we only check consistency of the *prism truth gradient*.
    if quick:
        eps = 0.05e-9
        _, _, Bz_p = arr.B(x_test + eps, 0.0, zd)
        _, _, Bz_m = arr.B(x_test - eps, 0.0, zd)
        Bx_p, _, _ = arr.B(x_test + eps, 0.0, zd)
        Bx_m, _, _ = arr.B(x_test - eps, 0.0, zd)
        dBz_dx_prism = np.asarray((Bz_p - Bz_m) / (2 * eps))
        dBx_dx_prism = np.asarray((Bx_p - Bx_m) / (2 * eps))
        method = "central FD (eps=0.05 nm)"
    else:
        @jax.jit
        def Bz_prism_scalar(x_scalar):
            _, _, bz = arr.B(x_scalar, 0.0, zd)
            return bz
        @jax.jit
        def Bx_prism_scalar(x_scalar):
            bx, _, _ = arr.B(x_scalar, 0.0, zd)
            return bx
        dBz_dx_prism_fn = jax.vmap(jax.grad(Bz_prism_scalar))
        dBx_dx_prism_fn = jax.vmap(jax.grad(Bx_prism_scalar))
        dBz_dx_prism = np.asarray(dBz_dx_prism_fn(x_test))
        dBx_dx_prism = np.asarray(dBx_dx_prism_fn(x_test))
        method = "jax.vmap(jax.grad)"
    dBz_dx_ff = np.asarray(ff.dBz_dx(x_test))
    dBx_dx_ff = np.asarray(ff.dBx_dx(x_test))
    err_dBz = float(np.max(np.abs(dBz_dx_prism - dBz_dx_ff)))
    err_dBx = float(np.max(np.abs(dBx_dx_prism - dBx_dx_ff)))
    norm_dBz = float(np.max(np.abs(dBz_dx_prism)))
    norm_dBx = float(np.max(np.abs(dBx_dx_prism)))
    rel_dBz = err_dBz / norm_dBz if norm_dBz > 0 else float('inf')
    rel_dBx = err_dBx / norm_dBx if norm_dBx > 0 else float('inf')
    print(f"\n  Gradient residual (prism via {method} vs FourierField):")
    print(f"    max|d(dBz/dx)| = {err_dBz*1e-6:.4f} mT/nm  "
          f"(rel to max|prism|: {rel_dBz*100:.2f}%)")
    print(f"    max|d(dBx/dx)| = {err_dBx*1e-6:.4f} mT/nm  "
          f"(rel to max|prism|: {rel_dBx*100:.2f}%)")
    
    # sign-convention check
    Bx_ff_quarter = float(ff.B_x(0.25 * a))
    Bx_prism_quarter = float(arr.B(0.25*a, 0.0, zd)[0])
    sign_match = (Bx_ff_quarter * Bx_prism_quarter) > 0
    print(f"\n  sign-convention check (x = a/4 = {0.25*a*1e9:.1f} nm):")
    print(f"    Bx (prism)        = {Bx_prism_quarter*1e3:+.3f} mT")
    print(f"    Bx (FourierField) = {Bx_ff_quarter*1e3:+.3f} mT")
    print(f"    sign match: {sign_match}")
    
    # JAX differentiation check
    dBz_auto = jax.grad(ff.B_z)(0.1 * a)
    dBz_ana = ff.dBz_dx(0.1 * a)
    rel_err = float(abs(dBz_auto - dBz_ana) / abs(dBz_ana + 1e-30))
    print(f"\n  JAX autodiff vs analytic gradient (FourierField):")
    print(f"    rel err (dBz/dx) = {rel_err:.3e}")

    # JAX projection fit vs numpy FFT fit agreement
    coef_jax = fit_from_prism_array_jax(arr, zd, N_harm=N_harm, n_samples=200)
    B0_jax = float(coef_jax["B0"])
    A_jax = tuple(float(a_) for a_ in coef_jax["A"])
    C_jax = tuple(float(c_) for c_ in coef_jax["C"])
    err_B0 = abs(B0_jax - ff.B0) / max(abs(ff.B0), 1e-20)
    err_A = max(abs(aj - an) / max(abs(an), 1e-20)
                 for aj, an in zip(A_jax, ff.A_coeffs))
    err_C = max(abs(cj - cn) / max(abs(cn), 1e-20)
                 for cj, cn in zip(C_jax, ff.C_coeffs))
    max_jax_vs_np = max(err_B0, err_A, err_C)
    pass_jax_proj = max_jax_vs_np < 1e-8
    print(f"\n  JAX projection vs numpy FFT (end-to-end differentiable fit):")
    print(f"    max rel err = {max_jax_vs_np:.3e}  "
          f"{'PASS' if pass_jax_proj else 'FAIL'} (< 1e-8)")

    # pass/fail
    pass_residual = rel_Bz < 0.05 and rel_Bx < 0.05
    pass_gradient = rel_dBz < 0.05 and rel_dBx < 0.05
    pass_sign = sign_match
    pass_autodiff = rel_err < 1e-6
    all_pass = pass_residual and pass_gradient and pass_sign and pass_autodiff and pass_jax_proj
    print(f"\n  Result:")
    print(f"    field residual <= 5%:         {pass_residual}")
    print(f"    gradient residual <= 5%:      {pass_gradient}")
    print(f"    Bx sign convention:           {pass_sign}")
    print(f"    JAX autodiff == analytic:     {pass_autodiff}")
    print(f"    JAX projection == numpy FFT:  {pass_jax_proj}")
    print(f"    Overall PASS:                 {all_pass}")

    # Plot -- within one cell: prism vs FourierField(N_harm) vs toy(1st only)
    from field_landscape import PeriodicField
    toy_1st = PeriodicField(B_ext=ff.B0, dB_long=ff.A_coeffs[0],
                             b_trans=ff.C_coeffs[0], period=a)
    Bz_toy = np.asarray(toy_1st.B_long(x_test))
    Bx_toy_oldsign = np.asarray(toy_1st.B_trans(x_test))   # +sin old convention

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.plot(np.asarray(x_test)*1e9, Bz_prism*1e3, label="prism (truth)", lw=2)
    ax.plot(np.asarray(x_test)*1e9, Bz_ff*1e3, "--",
            label=f"FourierField (N_harm={N_harm})", lw=1.5)
    ax.plot(np.asarray(x_test)*1e9, Bz_toy*1e3, ":", label="toy (1st only)", lw=1.2, alpha=0.7)
    ax.set_xlabel("x [nm]"); ax.set_ylabel("$B_z$ [mT]")
    ax.set_title(f"$B_z(x)$ at z={zd*1e9:.0f} nm, period={a*1e9:.0f} nm")
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(np.asarray(x_test)*1e9, Bx_prism*1e3, label="prism (truth)", lw=2)
    ax.plot(np.asarray(x_test)*1e9, Bx_ff*1e3, "--",
            label=f"FourierField (prism sign)", lw=1.5)
    ax.plot(np.asarray(x_test)*1e9, Bx_toy_oldsign*1e3, ":",
            label="toy (+sin: wrong sign)", lw=1.2, alpha=0.7)
    ax.set_xlabel("x [nm]"); ax.set_ylabel("$B_x$ [mT]")
    ax.set_title(f"$B_x(x)$ — sign convention check")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    
    from pathlib import Path
    FIG_DIR_P4 = Path(__file__).resolve().parents[1] / "figures" / "phase4"
    FIG_DIR_P4.mkdir(parents=True, exist_ok=True)
    out = str(FIG_DIR_P4 / "phase4_step0_fourier_fit.png")
    fig.savefig(out, dpi=130)
    print(f"\n  saved: {out}")

    return {
        "N_harm": N_harm,
        "B0_mT": ff.B0*1e3,
        "A_mT": tuple(A*1e3 for A in ff.A_coeffs),
        "C_mT": tuple(C*1e3 for C in ff.C_coeffs),
        "field_residual_pct": {"Bz": rel_Bz*100, "Bx": rel_Bx*100},
        "gradient_residual_pct": {"dBz_dx": rel_dBz*100, "dBx_dx": rel_dBx*100},
        "sign_match_with_prism": bool(sign_match),
        "autodiff_vs_analytic_rel_err": rel_err,
        "passed": bool(all_pass),
    }


if __name__ == "__main__":
    res = _main()
    print(f"\nSummary: {res}")
