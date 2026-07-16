"""V0.3 -- float32 vs float64 precision check for the field-gradient pipeline.

The supplement claims that the analytic-vs-autodiff gradient comparison of
V0 agrees to ~1e-9 in float64 while float32 shows O(1) disagreement,
because the mixed nanometre and tesla-per-metre scales make single
precision insufficient for the derivative and cancellation steps.

This standalone script measures both numbers directly. JAX fixes its
default dtype at import time, so the script re-executes itself in a
subprocess for each precision setting.

Run:  PYTHONPATH=. python reproduce/v0p3_float_precision_check.py
"""

import os
import subprocess
import sys


def _measure() -> float:
    """Max relative error between analytic d_x B_z and jax.grad, current dtype."""
    import jax

    want64 = os.environ.get("V0P3_X64", "1") == "1"
    # constants.py force-enables x64 at import; override afterwards for the
    # float32 branch (before any computation is traced).
    import constants  # noqa: F401  (sets x64=True)
    jax.config.update("jax_enable_x64", want64)
    import jax.numpy as jnp
    from field_landscape import PeriodicField

    field = PeriodicField()
    dt = jnp.float64 if want64 else jnp.float32
    xs = jnp.linspace(0.0, 4 * field.period, 1001, dtype=dt)
    g_auto = jax.vmap(jax.grad(field.B_long))(xs)
    g_ana = field.dB_long_dx(xs)
    scale = jnp.max(jnp.abs(g_ana))
    abs_err = float(jnp.max(jnp.abs(g_auto - g_ana)))
    return abs_err, float(abs_err / scale)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("32", "64"):
        # child: measure with the requested precision (set via env by parent)
        a, r = _measure()
        print(f"{a:.6e} {r:.6e}")
        sys.exit(0)

    results = {}
    for bits, x64 in (("64", "1"), ("32", "0")):
        env = dict(os.environ)
        env["JAX_ENABLE_X64"] = x64
        env["V0P3_X64"] = x64
        env["JAX_PLATFORM_NAME"] = "cpu"
        out = subprocess.run(
            [sys.executable, os.path.abspath(__file__), bits],
            env=env, capture_output=True, text=True, check=True,
        )
        a, r = out.stdout.strip().splitlines()[-1].split()
        results[bits] = (float(a), float(r))

    print("V0.3 float-precision check (analytic vs jax.grad, d_x B_long)")
    print(f"  float64: max abs error {results['64'][0]:.3e} T/m  (rel {results['64'][1]:.1e})")
    print(f"  float32: max abs error {results['32'][0]:.3e} T/m  (rel {results['32'][1]:.1e})")
    gap = results["32"][0] / max(results["64"][0], 1e-300)
    ok = results["64"][1] < 1e-9 and gap > 1e3
    print(f"  float32/float64 error ratio: {gap:.1e}")
    print(f"  V0.3 PASS: {ok}")
