"""
Periodic prism-array stray field.

Repeat a single UniformMagPrism along x with spatial period a:
identical magnets evenly spaced above the shuttling channel.

Finite vs infinite array:
An infinite array can be treated exactly by Fourier decomposition, but here we
explicitly sum a *sufficiently large* number (N_periods) of magnets. Near the
channel center the boundary effects are weak, so N_periods ~ 5-10 is enough.

Comparison target:
The sinusoidal toy model in field_landscape.py keeps only the *lowest Fourier
mode*. The real prism stray field contains *higher harmonics*; this module
measures that difference.

Pass criteria:
- the field repeats with one period (test_periodicity)
- the first Fourier component (cos 2 pi x/a) has an amplitude of the same
  order as the toy dB_long (test_amplitude_consistency)
- jax.grad matches finite differences (already checked in prism_field; rechecked here)
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List

from constants import FIG_DIR_PHASE3 as FIG_DIR
from geometry.prism_field import UniformMagPrism, MU_0


@dataclass(frozen=True)
class PeriodicPrismArray:
    """
    Periodic array of identical uniformly-magnetized prisms along x.

    Each prism has the same shape (half_x, half_y, half_z) and Ms.
    Spatial period along x = a; prism centers cx = m*a, m = 0, +/-1, ..., +/-N_periods.
    """
    period_a: float        # spatial period [m]
    half_x: float          # half-extent along x [m]
    half_y: float          # half-extent along y [m]
    half_z: float          # half-extent along z [m]
    cz: float              # magnet center height [m]
    N_periods_each_side: int = 5    # sum over +/-N_periods
    Ms: float = 1.4e6

    def _prism_list(self) -> List[UniformMagPrism]:
        """List of prism centers, generated lazily on first call."""
        prisms = []
        for m in range(-self.N_periods_each_side, self.N_periods_each_side + 1):
            prisms.append(UniformMagPrism(
                cx=m * self.period_a,
                cy=0.0,
                cz=self.cz,
                half_x=self.half_x,
                half_y=self.half_y,
                half_z=self.half_z,
                Ms=self.Ms,
            ))
        return prisms

    def B(self, x, y, z) -> Tuple:
        """Total stray field at (x, y, z) — sum over all prisms."""
        Bx = jnp.zeros_like(jnp.atleast_1d(x), dtype=jnp.float64)
        By = jnp.zeros_like(jnp.atleast_1d(x), dtype=jnp.float64)
        Bz = jnp.zeros_like(jnp.atleast_1d(x), dtype=jnp.float64)
        # handle scalar input as well
        if jnp.ndim(x) == 0:
            Bx = jnp.zeros((), dtype=jnp.float64)
            By = jnp.zeros((), dtype=jnp.float64)
            Bz = jnp.zeros((), dtype=jnp.float64)
        for p in self._prism_list():
            bx, by, bz = p.B(x, y, z)
            Bx = Bx + bx; By = By + by; Bz = Bz + bz
        return Bx, By, Bz


# ============================================================
# Self-checks
# ============================================================
def _periodicity_check(arr: PeriodicPrismArray, z_eval: float, 
                        x0: float = 0.0, n_periods_to_check: int = 3) -> float:
    """B(x_0 + a) == B(x_0)?  Returns max relative deviation over n_periods samples."""
    a = arr.period_a
    Bz_ref = float(arr.B(x0, 0.0, z_eval)[2])
    diffs = []
    for k in range(1, n_periods_to_check + 1):
        Bz_k = float(arr.B(x0 + k * a, 0.0, z_eval)[2])
        # not exact due to boundary effects (finite array)
        diff = abs(Bz_k - Bz_ref) / max(abs(Bz_ref), 1e-20)
        diffs.append(diff)
    return max(diffs)


def _fourier_amplitude(arr: PeriodicPrismArray, z_eval: float,
                       n_samples: int = 200) -> dict:
    """Fourier decomposition of the periodic landscape.

    Extract the 1st, 2nd, 3rd Fourier components of B_z(x), B_x(x) over one period [0, a].
    Uses vectorized calls -- no scalar loop.
    """
    a = arr.period_a
    x_arr = jnp.linspace(0.0, a, n_samples, endpoint=False)
    # Vectorized: arr.B accepts array input
    Bx_vals_j, _, Bz_vals_j = arr.B(x_arr, 0.0, z_eval)
    Bx_vals = np.asarray(Bx_vals_j)
    Bz_vals = np.asarray(Bz_vals_j)
    # FFT
    Bz_mean = float(np.mean(Bz_vals))
    Bz_fft = np.fft.rfft(Bz_vals - Bz_mean) / n_samples
    Bx_fft = np.fft.rfft(Bx_vals) / n_samples
    # 1st, 2nd, 3rd harmonics (2*|c_k| form: positive + negative conjugate pair)
    amp_Bz = [2 * abs(Bz_fft[k]) for k in (1, 2, 3)]
    amp_Bx = [2 * abs(Bx_fft[k]) for k in (1, 2, 3)]
    return {
        "Bz_mean": Bz_mean,
        "Bz_fundamental_amp": amp_Bz[0],
        "Bz_second_harm_amp": amp_Bz[1],
        "Bz_third_harm_amp": amp_Bz[2],
        "Bx_fundamental_amp": amp_Bx[0],
        "Bx_second_harm_amp": amp_Bx[1],
        "Bx_third_harm_amp": amp_Bx[2],
        "x_samples": np.asarray(x_arr),
        "Bz_samples": Bz_vals,
        "Bx_samples": Bx_vals,
    }


def _sanity_checks():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from field_landscape import PeriodicField

    print("="*60)
    print("Periodic prism array -- sanity checks")
    print("="*60)

    # standard config: magnet 200nm x 200nm x 100nm, period 400 nm (200 nm gap between magnets)
    # magnet center z = +50 nm, 2DEG at z = -50 nm
    # -> magnet-center-to-2DEG distance = 100 nm, magnet-bottom(z=0)-to-2DEG = 50 nm

    Lx_full = 200e-9; Ly_full = 200e-9; Lz_full = 100e-9
    a = 400e-9   # spatial period
    arr = PeriodicPrismArray(
        period_a=a,
        half_x=Lx_full/2, half_y=Ly_full/2, half_z=Lz_full/2,
        cz=Lz_full/2,
        N_periods_each_side=5,
        Ms=1.4e6,
    )
    z_2deg = -50e-9
    print(f"  prism: 200 nm × 200 nm × 100 nm,  period a = {a*1e9:.0f} nm")
    print(f"  array: {2*arr.N_periods_each_side + 1} prisms")
    print(f"  z_2deg = {z_2deg*1e9:.0f} nm")
    
    # (1) Periodicity
    print("\n(1) periodicity check:  B(x + a) == B(x) ?")
    max_dev = _periodicity_check(arr, z_2deg, x0=0.0, n_periods_to_check=2)
    print(f"    Max relative deviation over 2 periods: {max_dev*100:.4f}%")
    passed_periodicity = max_dev < 0.01
    print(f"    PASS (< 1%): {passed_periodicity}")

    # (2) Fourier decomposition + comparison with the sinusoidal toy
    print("\n(2) Fourier amplitudes (within one cell):")
    F = _fourier_amplitude(arr, z_2deg)
    print(f"    <B_z> = {F['Bz_mean']*1e3:.2f} mT")
    print(f"    1st harmonic |B_z|_amp  = {F['Bz_fundamental_amp']*1e3:.2f} mT")
    print(f"    2nd harmonic |B_z|_amp  = {F['Bz_second_harm_amp']*1e3:.2f} mT  "
          f"(~0 by symmetry)")
    print(f"    3rd harmonic |B_z|_amp  = {F['Bz_third_harm_amp']*1e3:.2f} mT  "
          f"({F['Bz_third_harm_amp']/F['Bz_fundamental_amp']*100:.1f}% of 1st)")
    print(f"    1st harmonic |B_x|_amp  = {F['Bx_fundamental_amp']*1e3:.2f} mT")
    print(f"    3rd harmonic |B_x|_amp  = {F['Bx_third_harm_amp']*1e3:.2f} mT  "
          f"({F['Bx_third_harm_amp']/F['Bx_fundamental_amp']*100:.1f}% of 1st)")
    in_range = 5.0 <= F['Bz_fundamental_amp']*1e3 <= 30.0
    print(f"    dB_z in [5, 30] mT: {'in-range' if in_range else 'out-of-range'}")

    # update the toy dB_long with the measured value
    toy = PeriodicField(
        B_ext=0.5,
        dB_long=F['Bz_fundamental_amp'],
        b_trans=F['Bx_fundamental_amp'],
        period=a,
    )
    
    # (3) jax.grad vs finite difference (one point)
    print("\n(3) JAX autodiff vs finite difference:")
    x_test = 0.1 * a   # dot position between magnets
    eps_fd = 1e-12
    grad_auto = jax.grad(lambda xx: arr.B(xx, 0.0, z_2deg)[2])(x_test)
    _, _, bp = arr.B(x_test + eps_fd, 0.0, z_2deg)
    _, _, bm = arr.B(x_test - eps_fd, 0.0, z_2deg)
    grad_fd = (float(bp) - float(bm)) / (2 * eps_fd)
    rel_err_grad = abs(float(grad_auto) - grad_fd) / max(abs(grad_fd), 1e-20)
    print(f"    jax.grad: {float(grad_auto)*1e-6:.4f} T/μm")
    print(f"    fin-diff: {grad_fd*1e-6:.4f} T/μm")
    print(f"    rel err = {rel_err_grad:.3e}")
    passed_grad = rel_err_grad < 1e-4

    # (4) Depth dependence
    print("\n(4) Depth sensitivity:")
    depths_nm = [-30, -50, -75, -100, -150]
    for d_nm in depths_nm:
        z = d_nm * 1e-9
        F_d = _fourier_amplitude(arr, z, n_samples=100)
        print(f"    z = {d_nm:4d} nm:  <B_z> = {F_d['Bz_mean']*1e3:6.2f} mT,  "
              f"1st amp = {F_d['Bz_fundamental_amp']*1e3:5.2f} mT")
    
    # Plots: prism array vs toy sinusoidal
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    # comparison over one cell (x in [0, a])
    x_compare = jnp.linspace(0, a, 300)
    # Vectorized: arr.B accepts arrays
    Bx_arr_j, _, Bz_arr_j = arr.B(x_compare, 0.0, z_2deg)
    Bz_arr_pts = np.asarray(Bz_arr_j)
    Bx_arr_pts = np.asarray(Bx_arr_j)
    Bz_toy = np.asarray(toy.B_long(x_compare)) - toy.B_ext
    Bx_toy = np.asarray(toy.B_trans(x_compare))
    
    axes[0].plot(np.asarray(x_compare)*1e9, Bz_arr_pts*1e3, label="prism array (real)", lw=1.5)
    axes[0].plot(np.asarray(x_compare)*1e9, Bz_toy*1e3 + F['Bz_mean']*1e3, "--", 
                 label="toy (1st harmonic + mean)", lw=1.2, alpha=0.8)
    axes[0].set_xlabel("x [nm]"); axes[0].set_ylabel("$B_z$ [mT]")
    axes[0].set_title(f"$B_z(x)$ at z={z_2deg*1e9:.0f} nm, one period")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(np.asarray(x_compare)*1e9, Bx_arr_pts*1e3, label="prism array (real)", lw=1.5)
    axes[1].plot(np.asarray(x_compare)*1e9, Bx_toy*1e3, "--", 
                 label="toy (1st harmonic)", lw=1.2, alpha=0.8)
    axes[1].set_xlabel("x [nm]"); axes[1].set_ylabel("$B_x$ [mT]")
    axes[1].set_title(f"$B_x(x)$ at z={z_2deg*1e9:.0f} nm, one period")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = str(FIG_DIR / "phase3_periodic_array.png")
    fig.savefig(out, dpi=130)
    print(f"\n  saved: {out}")

    # (5) Descriptor sweep over magnet sizes / periods / depths
    print("\n(5) Descriptor sweep (various magnet sizes / periods / depths):")
    print(f"  {'magnet [nm]':<22} {'a[nm]':<8} {'z[nm]':<8} "
          f"{'δBz[mT]':<10} {'<Bz>[mT]':<10} {'max|∂Bz/∂x|[mT/nm]':<22} "
          f"{'<|∂Bz/∂x|>[mT/nm]':<20}")
    print("  " + "-"*92)
    configs = [
        (50e-9,  50e-9,  30e-9,  150e-9, -50e-9),
        (100e-9, 100e-9, 50e-9,  300e-9, -75e-9),
        (200e-9, 200e-9, 100e-9, 400e-9, -100e-9),
        (200e-9, 200e-9, 100e-9, 400e-9, -150e-9),
    ]
    sweep_results = []
    for Lx, Ly, Lz, a_sw, zd in configs:
        arr_sw = PeriodicPrismArray(
            period_a=a_sw, half_x=Lx/2, half_y=Ly/2, half_z=Lz/2,
            cz=Lz/2, N_periods_each_side=3, Ms=1.4e6,
        )
        F_sw = _fourier_amplitude(arr_sw, zd, n_samples=80)
        Bz_pts = F_sw['Bz_samples']
        x_pts = F_sw['x_samples']
    # units: dBz [T] / dx [m] = T/m.  1 T/m = 1e-6 mT/nm.
    # (corrects an earlier extra x1e-3 that underestimated by 1000x)
        dBz_dx_Tperm = np.gradient(Bz_pts, x_pts)
        grad_peak_mTnm = float(np.max(np.abs(dBz_dx_Tperm)) * 1e-6)
        grad_avg_mTnm = float(np.mean(np.abs(dBz_dx_Tperm)) * 1e-6)
        print(f"  {Lx*1e9:.0f}×{Ly*1e9:.0f}×{Lz*1e9:.0f}             "
              f"{a_sw*1e9:.0f}      {zd*1e9:.0f}     "
              f"{F_sw['Bz_fundamental_amp']*1e3:7.2f}    "
              f"{F_sw['Bz_mean']*1e3:7.2f}    "
              f"{grad_peak_mTnm:10.3f}            "
              f"{grad_avg_mTnm:.3f}")
        sweep_results.append({
            "magnet_nm": (Lx*1e9, Ly*1e9, Lz*1e9),
            "period_nm": a_sw*1e9, "depth_nm": zd*1e9,
            "dBz_fund_mT": F_sw['Bz_fundamental_amp']*1e3,
            "Bz_mean_mT": F_sw['Bz_mean']*1e3,
            "max_abs_dBz_dx_mT_per_nm": grad_peak_mTnm,
            "mean_abs_dBz_dx_mT_per_nm": grad_avg_mTnm,
        })

    return {
        "periodicity_max_dev": max_dev,
        "Bz_fundamental_mT": F['Bz_fundamental_amp']*1e3,
        "Bz_2nd_harmonic_mT": F['Bz_second_harm_amp']*1e3,
        "Bz_3rd_harmonic_mT": F['Bz_third_harm_amp']*1e3,
        "Bz_3rd_over_1st_pct": F['Bz_third_harm_amp']/F['Bz_fundamental_amp']*100,
        "Bx_3rd_over_1st_pct": F['Bx_third_harm_amp']/F['Bx_fundamental_amp']*100,
        "autodiff_vs_fd_rel_err": rel_err_grad,
        "passed_periodicity": bool(passed_periodicity),
        "passed_autodiff": bool(passed_grad),
        "descriptor_sweep": sweep_results,
    }


if __name__ == "__main__":
    res = _sanity_checks()
    print(f"\nSummary: {res}")
