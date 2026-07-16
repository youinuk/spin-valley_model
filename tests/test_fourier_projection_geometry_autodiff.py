"""
End-to-end geometry-parameter autodiff test of the Fourier projection coefficients.

`fit_from_prism_array_jax` is structurally differentiable; this test
formally checks that the jax.grad of the geometry-parameter -> Fourier-coefficient
path matches finite differences.

Checks:
    ∂A_1 / ∂half_x
    ∂A_2 / ∂period
    ∂C_1 / ∂half_z
    ∂C_2 / ∂z_2DEG (depth)

Pass criterion: rel err < 1e-3 (accounting for FD truncation)
"""

import jax
import jax.numpy as jnp
import numpy as np

from constants import PROJECT_ROOT, FIG_DIR_PHASE3  # noqa: F401
from geometry.periodic_array import PeriodicPrismArray
from geometry.fourier_field import fit_from_prism_array_jax


def coefficient_of(geometry_kwargs, target_key, target_n, z_eval):
    """Build a PeriodicPrismArray from the geometry -> JAX projection -> return a coefficient.
    
    target_key: 'A' or 'C',  target_n: 1, 2, 3, ...
    """
    arr = PeriodicPrismArray(**geometry_kwargs)
    coef = fit_from_prism_array_jax(arr, z_eval, N_harm=3, n_samples=120)
    return coef[target_key][target_n - 1]   # 0-indexed in array


def compare_grad(label, fn, var_value, fd_step):
    """jax.grad vs central FD."""
    g_auto = float(jax.grad(fn)(float(var_value)))
    fp = float(fn(var_value + fd_step))
    fm = float(fn(var_value - fd_step))
    g_fd = (fp - fm) / (2 * fd_step)
    rel = abs(g_auto - g_fd) / max(abs(g_fd), 1e-30)
    return {"label": label, "auto": g_auto, "fd": g_fd, "rel_err": rel}


def run_tests():
    print("="*70)
    print("Fourier projection coefficient: geometry parameter autodiff test")
    print("="*70)

    # Baseline geometry (same as step 5)
    base = dict(
        period_a=150e-9, half_x=25e-9, half_y=25e-9, half_z=15e-9,
        cz=15e-9, N_periods_each_side=3, Ms=1.4e6,
    )
    z_eval = -50e-9

    tests = []

    # Test 1: ∂A_1 / ∂half_x
    def f1(half_x):
        kw = dict(base)
        kw["half_x"] = half_x
        return coefficient_of(kw, "A", 1, z_eval)
    print("\n  Test 1: ∂A_1 / ∂half_x  (half_x=25 nm, FD step=0.05 nm)")
    t = compare_grad("A_1/half_x", f1, 25e-9, 0.05e-9)
    print(f"    auto = {t['auto']:.6e},  fd = {t['fd']:.6e},  rel err = {t['rel_err']:.3e}")
    tests.append(t)

    # Test 2: ∂A_2 / ∂period
    def f2(period):
        kw = dict(base)
        kw["period_a"] = period
        return coefficient_of(kw, "A", 2, z_eval)
    print("\n  Test 2: ∂A_2 / ∂period  (period=150 nm, FD step=0.05 nm)")
    t = compare_grad("A_2/period", f2, 150e-9, 0.05e-9)
    print(f"    auto = {t['auto']:.6e},  fd = {t['fd']:.6e},  rel err = {t['rel_err']:.3e}")
    tests.append(t)

    # Test 3: ∂C_1 / ∂half_z
    def f3(half_z):
        kw = dict(base)
        kw["half_z"] = half_z
        # tie cz to half_z (magnet bottom always at z=0)
        kw["cz"] = half_z
        return coefficient_of(kw, "C", 1, z_eval)
    print("\n  Test 3: ∂C_1 / ∂half_z  (half_z=15 nm, FD step=0.05 nm)")
    t = compare_grad("C_1/half_z", f3, 15e-9, 0.05e-9)
    print(f"    auto = {t['auto']:.6e},  fd = {t['fd']:.6e},  rel err = {t['rel_err']:.3e}")
    tests.append(t)

    # Test 4: ∂C_2 / ∂z_2DEG
    def f4(z_2deg):
        return coefficient_of(base, "C", 2, z_2deg)
    print("\n  Test 4: ∂C_2 / ∂z_2DEG  (z=-50 nm, FD step=0.05 nm)")
    t = compare_grad("C_2/z_2DEG", f4, -50e-9, 0.05e-9)
    print(f"    auto = {t['auto']:.6e},  fd = {t['fd']:.6e},  rel err = {t['rel_err']:.3e}")
    tests.append(t)

    # pass/fail
    threshold = 1e-3
    max_rel = max(t["rel_err"] for t in tests)
    all_pass = max_rel < threshold

    print(f"\n{'='*70}")
    print(f"  result -- pass criterion |rel_err| < {threshold:.0e}")
    print(f"{'='*70}")
    for t in tests:
        status = "PASS" if t["rel_err"] < threshold else "FAIL"
        print(f"  {t['label']:<15}: rel err = {t['rel_err']:.3e}   {status}")
    print(f"\n  max relative error: {max_rel:.3e}")
    print(f"  overall PASS: {all_pass}")
    return {"tests": tests, "max_rel_err": max_rel, "passed": bool(all_pass)}


if __name__ == "__main__":
    run_tests()


def test_fourier_projection_geometry_autodiff():
    """pytest entry point: Fourier-projection geometry autodiff vs finite diff."""
    res = run_tests()
    assert res["passed"], f"max rel err {res['max_rel_err']:.3e} exceeds threshold"
