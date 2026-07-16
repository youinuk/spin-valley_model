"""
Geometry-parameter autodiff test.

Check that `jax.grad` matches finite differences not only for the *position* x
derivative but also for *geometry-parameter* derivatives. This underpins any
differentiable geometry/trajectory inverse design -- without it the
"differentiable inverse design" claim would not be safe.
Variables checked:
  - half_x  (magnet half-width)
  - half_z  (magnet half-thickness)
  - period_a (period)
  - z_eval  (2DEG depth)

For each variable, compare dB_z/d(var) between autodiff and central FD.
Pass criterion: |rel_err| < 1e-4.
"""

import numpy as np
import jax

# always first -- force jax x64
from constants import PROJECT_ROOT, FIG_DIR_PHASE3  # noqa: F401
from geometry.prism_field import UniformMagPrism
from geometry.periodic_array import PeriodicPrismArray


def Bz_function_of_half_x(half_x, x_eval=20e-9, z_eval=-50e-9):
    """Function form: return Bz at a fixed point as half_x varies."""
    p = UniformMagPrism(
        cx=0.0, cy=0.0, cz=15e-9,   # half of a 30nm-thick magnet
        half_x=half_x, half_y=25e-9, half_z=15e-9,
        Ms=1.4e6,
    )
    _, _, Bz = p.B(x_eval, 0.0, z_eval)
    return Bz


def Bz_function_of_half_z(half_z, x_eval=20e-9, z_eval=-50e-9):
    """half_z (magnet thickness) variation."""
    p = UniformMagPrism(
        cx=0.0, cy=0.0, cz=half_z,   # cz=half_z so the magnet bottom is at z=0
        half_x=25e-9, half_y=25e-9, half_z=half_z,
        Ms=1.4e6,
    )
    _, _, Bz = p.B(x_eval, 0.0, z_eval)
    return Bz


def Bz_function_of_period(period, x_eval=20e-9, z_eval=-50e-9):
    """Bz near the array center as the period varies."""
    arr = PeriodicPrismArray(
        period_a=period,
        half_x=25e-9, half_y=25e-9, half_z=15e-9,
        cz=15e-9, N_periods_each_side=3, Ms=1.4e6,
    )
    _, _, Bz = arr.B(x_eval, 0.0, z_eval)
    return Bz


def Bz_function_of_depth(z_eval, x_eval=20e-9):
    """Bz as the 2DEG depth (z_eval) varies.
    (This equals the position derivative, but checks a differently-named degree of freedom.)"""
    p = UniformMagPrism(
        cx=0.0, cy=0.0, cz=15e-9,
        half_x=25e-9, half_y=25e-9, half_z=15e-9,
        Ms=1.4e6,
    )
    _, _, Bz = p.B(x_eval, 0.0, z_eval)
    return Bz


def compare_autodiff_vs_fd(fn, var_value, fd_step, label: str) -> dict:
    """Compare autodiff and central FD for a single-variable function."""
    grad_auto = float(jax.grad(fn)(float(var_value)))
    fp = float(fn(var_value + fd_step))
    fm = float(fn(var_value - fd_step))
    grad_fd = (fp - fm) / (2 * fd_step)
    rel_err = abs(grad_auto - grad_fd) / max(abs(grad_fd), 1e-30)
    return {
        "label": label,
        "var_value": float(var_value),
        "fd_step": fd_step,
        "grad_auto": grad_auto,
        "grad_fd": grad_fd,
        "rel_err": rel_err,
    }


def run_geometry_autodiff_test() -> dict:
    print("="*70)
    print("Geometry-parameter autodiff test")
    print("="*70)

    tests = []

    # Test 1: half_x derivative
    # nominal 25 nm.  FD step 0.1 nm.
    print("\n  Test 1: dBz/d(half_x)")
    res = compare_autodiff_vs_fd(Bz_function_of_half_x, 25e-9, 0.1e-9, "half_x")
    print(f"    half_x = {res['var_value']*1e9:.1f} nm,  FD step = {res['fd_step']*1e9:.2f} nm")
    print(f"    grad_auto = {res['grad_auto']:.6e} T/m  (Bz [T] / half_x [m])")
    print(f"    grad_fd   = {res['grad_fd']:.6e} T/m")
    print(f"    rel err   = {res['rel_err']:.3e}")
    tests.append(res)

    # Test 2: half_z derivative
    print("\n  Test 2: dBz/d(half_z)")
    res = compare_autodiff_vs_fd(Bz_function_of_half_z, 15e-9, 0.05e-9, "half_z")
    print(f"    half_z = {res['var_value']*1e9:.1f} nm,  FD step = {res['fd_step']*1e9:.2f} nm")
    print(f"    grad_auto = {res['grad_auto']:.6e}")
    print(f"    grad_fd   = {res['grad_fd']:.6e}")
    print(f"    rel err   = {res['rel_err']:.3e}")
    tests.append(res)

    # Test 3: period derivative
    # note: use a 0.05 nm FD step to balance truncation error O(h^2).
    # a 0.5 nm step pushes truncation error into the 1e-4 range (autodiff itself is exact).
    print("\n  Test 3: dBz/d(period)")
    res = compare_autodiff_vs_fd(Bz_function_of_period, 150e-9, 0.05e-9, "period")
    print(f"    period = {res['var_value']*1e9:.0f} nm,  FD step = {res['fd_step']*1e9:.3f} nm")
    print(f"    grad_auto = {res['grad_auto']:.6e}")
    print(f"    grad_fd   = {res['grad_fd']:.6e}")
    print(f"    rel err   = {res['rel_err']:.3e}")
    tests.append(res)

    # Test 4: depth derivative
    print("\n  Test 4: dBz/d(depth)")
    res = compare_autodiff_vs_fd(Bz_function_of_depth, -50e-9, 0.1e-9, "z_eval")
    print(f"    z_eval = {res['var_value']*1e9:.1f} nm,  FD step = {res['fd_step']*1e9:.2f} nm")
    print(f"    grad_auto = {res['grad_auto']:.6e}")
    print(f"    grad_fd   = {res['grad_fd']:.6e}")
    print(f"    rel err   = {res['rel_err']:.3e}")
    tests.append(res)

    # pass/fail
    threshold = 1e-4
    max_rel_err = max(t["rel_err"] for t in tests)
    all_pass = max_rel_err < threshold

    print(f"\n{'='*70}")
    print(f"  result -- pass criterion: |rel_err| < {threshold:.0e}")
    print(f"{'='*70}")
    for t in tests:
        status = "PASS" if t["rel_err"] < threshold else "FAIL"
        print(f"  {t['label']:<10}: rel err = {t['rel_err']:.3e}   {status}")
    print(f"\n  max relative error: {max_rel_err:.3e}")
    print(f"  overall PASS: {all_pass}")
    
    return {
        "tests": tests,
        "max_rel_err": float(max_rel_err),
        "passed": bool(all_pass),
    }


if __name__ == "__main__":
    res = run_geometry_autodiff_test()


def test_geometry_autodiff():
    """pytest entry point: geometry autodiff matches finite differences."""
    res = run_geometry_autodiff_test()
    assert res["passed"], f"max rel err {res['max_rel_err']:.3e} exceeds 1e-4"
