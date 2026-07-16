"""
Deterministic stray-field landscape.

Model 0 (periodic toy landscape):
    B_long(x) = B_ext + dB * cos(2*pi*x/a)
    B_trans(x) =  b   * sin(2*pi*x/a)

Phase 2 does not vary the magnet geometry (that starts in Phase 3).
Here only the *shape* of the landscape matters (periodic / monotonic / disordered).

All functions use jax.numpy so they are JAX-traceable.
Differentiability is used heavily in Phase 3 (geometry knob); in Phase 2,
B_long'(x) is also used directly in the motional-narrowing calculation.
"""

from __future__ import annotations
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Callable, Tuple

from constants import Defaults, FIG_DIR


# ============================================================
# 1. periodic landscape (core of Model 0)
# ============================================================
@dataclass(frozen=True)
class PeriodicField:
    """Orbit-averaged stray field of a periodic magnet array.
    
    B_long(x) = B_ext + dB_long * cos(2*pi*x/a)
    B_trans(x) =        b_trans  * sin(2*pi*x/a)
    """
    B_ext:   float = Defaults.B_ext_T
    dB_long: float = Defaults.dB_long
    b_trans: float = Defaults.b_trans
    period:  float = Defaults.period_lo_m

    def B_long(self, x):
        return self.B_ext + self.dB_long * jnp.cos(2 * jnp.pi * x / self.period)

    def B_trans(self, x):
        return self.b_trans * jnp.sin(2 * jnp.pi * x / self.period)

    def dB_long_dx(self, x):
        """x-derivative of B_long. Central to the motional-narrowing analysis."""
        k = 2 * jnp.pi / self.period
        return -self.dB_long * k * jnp.sin(k * x)


# ============================================================
# 2. monotonic-gradient landscape (Phase 2 baseline 1)
# ============================================================
@dataclass(frozen=True)
class LinearGradientField:
    """B_long(x) = B_ext + grad * x  (monotonic increase everywhere)"""
    B_ext: float = Defaults.B_ext_T
    grad:  float = Defaults.grad_lo
    b_trans_const: float = Defaults.b_trans

    def B_long(self, x):
        return self.B_ext + self.grad * x

    def B_trans(self, x):
        # simplification: constant transverse component
        return self.b_trans_const * jnp.ones_like(jnp.atleast_1d(x))

    def dB_long_dx(self, x):
        return self.grad * jnp.ones_like(jnp.atleast_1d(x))


# ============================================================
# 3. self-check
# ============================================================
if __name__ == "__main__":
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    field = PeriodicField()
    x = jnp.linspace(0, 4 * field.period, 1000)

    BL = field.B_long(x)
    BT = field.B_trans(x)
    dBL = field.dB_long_dx(x)

    # compare the analytic derivative against jax.grad
    grad_fn = jax.vmap(jax.grad(field.B_long))
    dBL_auto = grad_fn(x)

    err = float(jnp.max(jnp.abs(dBL - dBL_auto)))
    print(f"analytic derivative vs jax.grad max error: {err:.3e}  (should be 0)")

    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    axes[0].plot(np.asarray(x) * 1e9, np.asarray(BL) * 1e3, label=r"$B_\mathrm{long}$")
    axes[0].plot(np.asarray(x) * 1e9, np.asarray(BT) * 1e3, label=r"$B_\mathrm{trans}$")
    axes[0].set_ylabel("B [mT]")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title("Periodic landscape (PeriodicField)")
    axes[1].plot(np.asarray(x) * 1e9, np.asarray(dBL) * 1e-6, color="C2")
    axes[1].set_ylabel(r"$dB_\mathrm{long}/dx$ [mT/μm]")
    axes[1].set_xlabel("x [nm]")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = str(FIG_DIR / "sanity_field.png")
    fig.savefig(out, dpi=130)
    print(f"check figure saved: {out}")
