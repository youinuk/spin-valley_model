"""
Analytic stray field of a single uniformly magnetized prism.

Standard result (Engel-Herbert & Hesjedal, J. Appl. Phys. 97, 074504 (2005);
also Newell, Williams & Dunlop 1993):
    
For a rectangular prism (origin at center, half-dimensions a x b x c) with
uniform magnetization M = Ms z_hat, the external stray field is computed from
the surface magnetic charge density sigma_m = +/-Ms on the +/-z faces.

    B_alpha(x,y,z) = (μ_0 Ms / 4π) · Σ_{i,j,k} (-1)^(i+j+k) · g_alpha(X_i, Y_j, Z_k)

with X_i = x - (-1)^i a, Y_j = y - (-1)^j b, Z_k = z - (-1)^k c, 
and the kernel functions:

    g_x(X, Y, Z) = -log(R - Y)
    g_y(X, Y, Z) = -log(R - X)
    g_z(X, Y, Z) = -arctan(X Y / (Z R))
    R = sqrt(X^2 + Y^2 + Z^2)

[note] The formula above gives each component of B for z-magnetization via the
vector-potential route. An alternative direct closed form gives the same result.
Here we implement it via the **scalar magnetic potential** phi_m for stability:

    φ_m(r) = (Ms / 4π) ∫_{top face} (1/|r-r'|) dA  -  (similar for bottom face)

Each face integral is the Newell 1993 closed form:
    F(X, Y, Z) = (1/2)[ Y(Z^2 - X^2) asinh(Y/sqrt(X^2+Z^2))
                       + X(Z^2 - Y^2) asinh(X/sqrt(Y^2+Z^2))
                       - X Y Z atan(X Y / (Z R))
                       + (3/2) X Y R ]  ... (energy form)

This module uses the more direct, well-tested **direct B-field component form**:

    Bz(x, y, z) = (μ_0 Ms / 4π) · Σ_{i,j,k} (-1)^(i+j+k) · 
                   atan( (X_i Y_j) / (Z_k R_{ijk}) )

    Bx(x, y, z) = (μ_0 Ms / 4π) · Σ_{i,j,k} (-1)^(i+j+k) · 
                   log( (R_{ijk} - Y_j) / (R_{ijk} + Y_j) ) / 2

    By(x, y, z) = (μ_0 Ms / 4π) · Σ_{i,j,k} (-1)^(i+j+k) · 
                   log( (R_{ijk} - X_i) / (R_{ijk} + X_i) ) / 2

where the sum runs over the 8 prism corners (i,j,k) in {0,1}^3.
This is the standard textbook formula; unit checks:
- inside the prism B_z (the magnetization component) is ~mu_0 Ms; outside it decays fast as a fringe field
- far away it becomes a dipole field
- div B = 0 (trivially, free space)
"""

from __future__ import annotations
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Tuple

# uses the existing constants module (auto-enables JAX x64)
from constants import PROJECT_ROOT, FIG_DIR_PHASE3 as FIG_DIR  # noqa: F401  -- ensures jax x64

# magnetic constant
MU_0 = 4 * jnp.pi * 1e-7   # T·m/A


@dataclass(frozen=True)
class UniformMagPrism:
    """
    Uniformly z-magnetized rectangular prism.
    
    Parameters
    ----------
    cx, cy, cz : prism center coordinates [m]
    half_x, half_y, half_z : half-dimensions [m]  (prism spans [cx +/- half_x] x ...)
    Ms : saturation magnetization [A/m]. Co ~1.4e6, Fe ~1.7e6.
    """
    cx: float
    cy: float
    cz: float
    half_x: float
    half_y: float
    half_z: float
    Ms: float = 1.4e6   # Co default

    def _summands(self, x, y, z):
        """
        Returns (B_x_sum, B_y_sum, B_z_sum) before the μ_0 Ms / (4π) prefactor.
        
        Sum over the 8 corners at each (x,y,z). JAX-traceable.
        """
        # 8 corner coordinates
        # i, j, k in {0, 1}; the (-1)^i sign assigns +/-half
        # X_i = x - (cx + (-1)^i * half_x)  --  shift the prism corner to the origin
        # standard form: X_i = x - x_i_corner
        # where x_i_corner = cx - half_x (i=0) or cx + half_x (i=1)
        # by the sign convention: X_i = (x - cx) - (-1)^i * half_x
        #   i=0:  X_0 = (x-cx) - (+half_x)   <- "X_+"
        #   i=1:  X_1 = (x-cx) - (-half_x)   <- "X_−"
        # and the summand factor is (-1)^(i+j+k). Standard form, used as-is.
        u = x - self.cx
        v = y - self.cy
        w = z - self.cz
        # the two coordinates per axis:  [u - half_x, u + half_x]
        Xs = jnp.array([u - self.half_x, u + self.half_x])
        Ys = jnp.array([v - self.half_y, v + self.half_y])
        Zs = jnp.array([w - self.half_z, w + self.half_z])
        # signs: (-1)^(i+j+k) where indices 0->-half (X_0=u-half) and 1->+half (X_1=u+half)
        # standard sign: (-1)^(1+1+1) = -1 for the "all-plus" corner, etc.
        # signs[i,j,k] = (-1)^(i+j+k) with i,j,k in {0,1}, i=0 corresponding to -half.
        # here X_0 = u-half, X_1 = u+half, so index i=0 is the "-half corner", i=1 the "+half corner".
        # standard prism formula: sign factor = (-1)^(i+j+k)
        sign = jnp.array([[(-1.0)**(i + j + k)
                            for k in (0, 1)]
                           for j in (0, 1)
                           for i in (0, 1)]).reshape(2, 2, 2)
        # epsilon: avoid log/arctan singularities (directly on the prism surface)
        eps = 1e-30
        # sum over the 8 corners
        Bx_sum = 0.0
        By_sum = 0.0
        Bz_sum = 0.0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    X = Xs[i]; Y = Ys[j]; Z = Zs[k]
                    R = jnp.sqrt(X*X + Y*Y + Z*Z + eps)
                    s = sign[i, j, k]
                    # B_z component:  atan( (XY) / (Z R) )
                    Bz_sum = Bz_sum + s * jnp.arctan2(X * Y, Z * R + eps)
                    # B_x component:  0.5 * log( (R - Y) / (R + Y) )
                    Bx_sum = Bx_sum + s * 0.5 * jnp.log((R - Y + eps) / (R + Y + eps))
                    # B_y component:  0.5 * log( (R - X) / (R + X) )
                    By_sum = By_sum + s * 0.5 * jnp.log((R - X + eps) / (R + X + eps))
        return Bx_sum, By_sum, Bz_sum

    def B(self, x, y, z) -> Tuple:
        """Stray field components (Bx, By, Bz) at point (x, y, z) [T]."""
        Bx_s, By_s, Bz_s = self._summands(x, y, z)
        prefactor = MU_0 * self.Ms / (4 * jnp.pi)
        # standard sign convention: for M = Ms z_hat, the prism produces a Bz fringe field
        # that is *positive* above the prism (z > +half_z). The sum above matches that sign
        # in the standard textbook form.
        return prefactor * Bx_s, prefactor * By_s, prefactor * Bz_s


# ============================================================
# Self-checks
# ============================================================
def _sanity_checks():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("="*60)
    print("Single-prism stray field -- sanity check")
    print("="*60)

    # standard Co micromagnet: 200 nm x 200 nm x 100 nm thick (along z)
    # substrate surface at z=0, magnet sits *above* it (z>0)
    # placing the magnet center at z = +half_thickness puts its bottom at the z=0 surface.
    Lx = 200e-9   # half-size along x = 100 nm  -> total 200 nm
    Ly = 200e-9   # half-size along y = 100 nm
    Lz_thick = 100e-9  # thickness 100 nm
    prism = UniformMagPrism(
        cx=0.0, cy=0.0, cz=Lz_thick/2,
        half_x=Lx/2, half_y=Ly/2, half_z=Lz_thick/2,
        Ms=1.4e6,   # Co
    )
    
    # 1. unit check: far away (z >> magnet size) it is a dipole field
    # dipole moment m = Ms × V = 1.4e6 × (Lx × Ly × Lz)
    V = Lx * Ly * Lz_thick
    m_dipole = prism.Ms * V
    # axial dipole field at distance d on z-axis (above prism center):
    # B_z = μ_0 m / (2π d^3)
    d_far = 5e-6   # 5 μm above prism center
    z_test = prism.cz + d_far
    _, _, Bz_num = prism.B(0.0, 0.0, z_test)
    Bz_dipole = MU_0 * m_dipole / (2 * jnp.pi * d_far**3)
    rel_err = float(abs(Bz_num - Bz_dipole) / Bz_dipole)
    print(f"  dipole limit at d={d_far*1e6:.1f} μm:")
    print(f"    numerical Bz   = {float(Bz_num)*1e6:.2f} μT")
    print(f"    dipole formula = {float(Bz_dipole)*1e6:.2f} μT")
    print(f"    relative error = {rel_err*100:.2f}%   (< 5% if far enough)")

    # 2. fringe-field profile in the 2DEG plane (z = -50 nm, -100 nm)
    # magnet at z in [0, +100 nm], substrate surface at z = 0, 2DEG at z < 0 (substrate).
    # typical quantum-dot 2DEG depth ~50-100 nm below the surface -> z = -50 to -100 nm.
    x_line = jnp.linspace(-400e-9, 400e-9, 400)
    y_line = 0.0
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    # accumulate results per depth
    results_by_depth = {}
    for ax_row, z_2deg, label in [(axes[0], -50e-9, "z = -50 nm (2DEG)"),
                                   (axes[1], -100e-9, "z = -100 nm (2DEG)")]:
        Bx_line, By_line, Bz_line = prism.B(x_line, y_line, z_2deg)
        # gradient: d/dx
        import jax
        # vectorize over a list of x
        @jax.vmap
        def grad_Bz_at(xi):
            return jax.grad(lambda xx: prism.B(xx, y_line, z_2deg)[2])(xi)
        @jax.vmap
        def grad_Bx_at(xi):
            return jax.grad(lambda xx: prism.B(xx, y_line, z_2deg)[0])(xi)
        dBz_dx = grad_Bz_at(x_line)
        dBx_dx = grad_Bx_at(x_line)

        # report values
        Bz_peak = float(jnp.max(jnp.abs(Bz_line)))
        Bx_peak = float(jnp.max(jnp.abs(Bx_line)))
        dBz_peak = float(jnp.max(jnp.abs(dBz_dx)))
        dBx_peak = float(jnp.max(jnp.abs(dBx_dx)))
        # unit conversion: dBz_peak is in T/m. 1 T/m = 1e-6 mT/nm.
        print(f"\n  {label}:")
        print(f"    |B_z|_max = {Bz_peak*1e3:.2f} mT")
        print(f"    |B_x|_max = {Bx_peak*1e3:.2f} mT")
        print(f"    |dB_z/dx|_max = {dBz_peak*1e-6:.4f} mT/nm  (= {dBz_peak*1e-6:.4f} T/um -- same value)")
        print(f"    |dB_x/dx|_max = {dBx_peak*1e-6:.4f} mT/nm")

        depth_nm = int(z_2deg * 1e9)
        results_by_depth[f"z_{depth_nm}_nm"] = {
            "Bz_peak_mT": Bz_peak * 1e3,
            "Bx_peak_mT": Bx_peak * 1e3,
            "dBz_dx_peak_mT_per_nm": dBz_peak * 1e-6,
            "dBx_dx_peak_mT_per_nm": dBx_peak * 1e-6,
        }
        
        ax_row[0].plot(np.asarray(x_line)*1e9, np.asarray(Bz_line)*1e3, label="$B_z$")
        ax_row[0].plot(np.asarray(x_line)*1e9, np.asarray(Bx_line)*1e3, label="$B_x$")
        ax_row[0].set_xlabel("x [nm]"); ax_row[0].set_ylabel("B [mT]")
        ax_row[0].set_title(label + " — field")
        ax_row[0].legend(); ax_row[0].grid(alpha=0.3)
        ax_row[1].plot(np.asarray(x_line)*1e9, np.asarray(dBz_dx)*1e-6, label="$\\partial_x B_z$")
        ax_row[1].plot(np.asarray(x_line)*1e9, np.asarray(dBx_dx)*1e-6, label="$\\partial_x B_x$")
        ax_row[1].set_xlabel("x [nm]"); ax_row[1].set_ylabel("$\\partial_x B$ [T/μm]")
        ax_row[1].set_title(label + " — gradient (jax.grad)")
        ax_row[1].legend(); ax_row[1].grid(alpha=0.3)
    
    # show the magnet footprint
    for ax in axes.flat:
        ax.axvspan(-Lx/2*1e9, Lx/2*1e9, alpha=0.15, color="gray", label="_magnet")
    
    fig.tight_layout()
    out = str(FIG_DIR / "phase3_prism_fields.png")
    fig.savefig(out, dpi=130)
    print(f"\n  saved: {out}")
    
    # 3. JAX gradient vs finite difference
    print("\n  JAX autodiff vs finite difference (B_z at z = -50 nm, x scan):")
    eps_fd = 1e-12   # 1 pm
    x_pts = jnp.linspace(-200e-9, 200e-9, 11)
    z_2deg = -50e-9
    import jax
    grad_fn = jax.vmap(jax.grad(lambda xx: prism.B(xx, 0.0, z_2deg)[2]))
    grad_auto = grad_fn(x_pts)
    # finite diff (centered)
    Bz_plus, _ = jax.vmap(lambda xx: prism.B(xx + eps_fd, 0.0, z_2deg)[2:3])(x_pts), None
    # the call above is awkward, so use a direct list comprehension:
    grad_fd_list = []
    for xp in np.asarray(x_pts):
        _, _, bp = prism.B(float(xp) + eps_fd, 0.0, z_2deg)
        _, _, bm = prism.B(float(xp) - eps_fd, 0.0, z_2deg)
        grad_fd_list.append((float(bp) - float(bm)) / (2 * eps_fd))
    grad_fd = np.array(grad_fd_list)
    grad_auto_np = np.asarray(grad_auto)
    # relative error (normalized by dBz_peak)
    scale = np.max(np.abs(grad_auto_np))
    rel_errs = np.abs(grad_auto_np - grad_fd) / (scale + 1e-20)
    max_rel = float(np.max(rel_errs))
    print(f"    x points: {len(x_pts)},  scale = {scale*1e-6:.2f} T/μm")
    print(f"    max relative error (autodiff vs FD, normalized): {max_rel:.3e}")
    fd_pass = max_rel < 1e-4
    print(f"    PASS (rel err < 1e-4): {fd_pass}")
    
    return {
        "dipole_far_field_rel_err": rel_err,
        "by_depth": results_by_depth,
        "autodiff_vs_fd_max_rel_err": max_rel,
        "passed_dipole_limit": bool(rel_err < 0.05),
        "passed_autodiff_check": bool(fd_pass),
    }


if __name__ == "__main__":
    res = _sanity_checks()
    print(f"\nSummary: {res}")
