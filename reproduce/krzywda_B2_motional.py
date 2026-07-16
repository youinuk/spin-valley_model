"""
B2 — Motional narrowing.

A shuttling dot sees a position-dependent field B_long(x). The Larmor jitter from
charge noise -> dx(t) differs from the stationary dot:

    dw(t) = (g mu_B / hbar) * B_long'(x(t)) * dx(t)        [gradient varies with x]

On a periodic landscape, B_long'(x(t)) = -dB_long * k * sin(k v t) (constant-velocity shuttling).
This modulates the low-frequency part of dx(t), weakening dephasing (motional narrowing).

B2 pass criterion: T2*(v) > T2*(v=0). Improvement with increasing velocity (up to a sweet point).
The Krzywda absolute value (11.5 us) is device-specific and not directly compared here.
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constants import g_Si, mu_B, hbar, Defaults, FIG_DIR
from noise.charge_noise import OneOverFNoise
from field_landscape import PeriodicField
from reproduce.krzywda_B1_stationary import extract_T2


def simulate_FID_shuttling(
    v: float,
    field: PeriodicField,
    noise: OneOverFNoise,
    T_max: float,
    dt: float,
    n_realizations: int,
    rng: np.random.Generator,
    n_T_samples: int = 200,
):
    """
    FID of a shuttling dot.

    δφ(T) = (g μ_B / ℏ) · ∫_0^T B_long'(x(t)) δx(t) dt
    x(t) = v · t  (constant velocity)
    """
    # generate one long trace + window sampling (same pattern as B1)
    T_long = 50.0 / noise.f_low
    t_long, dx_long = noise.generate(T_long, dt, rng=rng)
    N_long = len(dx_long)
    N_win = int(np.round(T_max / dt))
    assert N_win < N_long

    T_grid = np.linspace(dt, T_max, n_T_samples)
    starts = rng.integers(0, N_long - N_win - 1, size=n_realizations)

    # time grid and position within the window
    t_win = np.arange(N_win) * dt
    x_t = v * t_win
    # compute B_long'(x(t)) once -- the windowed integrand is dB_long'/dx * dx(t)
    # (specifically B_long'(x) = -dB_long * k * sin(kx))
    k = 2 * np.pi / field.period
    dBdx_along_traj = -field.dB_long * k * np.sin(k * x_t)
    prefactor = g_Si * mu_B / hbar

    phi_at_T = np.zeros((n_realizations, n_T_samples))
    for r, s in enumerate(starts):
        dx_win = dx_long[s : s + N_win]
        integrand = prefactor * dBdx_along_traj * dx_win   # rad/s
        integ = np.concatenate([
            [0.0],
            np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dt)
        ])
        phi_at_T[r] = np.interp(T_grid, t_win, integ)

    C = np.abs(np.mean(np.exp(1j * phi_at_T), axis=0))
    return T_grid, C


def run_B2_check(seed: int = 11, n_real: int = 600) -> dict:
    print("="*60)
    print("B2 — Motional narrowing (T2* vs shuttling velocity)")
    print("="*60)

    from constants import Phase2Grid
    field = PeriodicField()
    noise = OneOverFNoise(
        sigma_total=Defaults.sigma_dx_m,
        alpha=1.0,
        f_low=Phase2Grid.f_low_Hz,    # 250 kHz
        f_high=Phase2Grid.f_high_Hz,  # 500 MHz - includes the shuttling modulation (v/a)
    )
    dt = Phase2Grid.dt_s              # 1 ns
    T_max = Phase2Grid.T_max_s        # 4 us
    rng = np.random.default_rng(seed)

    # v selection: f_drive = v/a must lie inside the noise bandwidth [f_low, f_high]
    # for a meaningful motional-narrowing measurement.
    # a = 100 nm => v = 25 m/s gives f_drive = 250 MHz (well resolved)
    # v = 50 m/s gives 500 MHz (near Nyquist -- risky)
    # v = 5 m/s gives 50 MHz (well resolved)
    # v = 1 m/s gives 10 MHz (well resolved)

    # baseline 1: v=0, a stationary dot seeing the mean gradient (same single-grad model as B1)
    avg_abs_grad = (2.0 / np.pi) * field.dB_long * (2 * np.pi / field.period)
    print(f"  magnet period a = {field.period*1e9:.0f} nm")
    print(f"  dB_long = {field.dB_long*1e3:.1f} mT")
    print(f"  mean |B'_long| of the periodic landscape = {avg_abs_grad*1e-6:.3f} T/um")
    print(f"  noise bandwidth: [{noise.f_low/1e3:.0f} kHz, {noise.f_high/1e6:.0f} MHz]")
    print(f"  realizations: {n_real}, T_max = {T_max*1e6:.1f} us, dt = {dt*1e9:.0f} ns")

    # v=0 baseline using the same monotonic model as B1
    from reproduce.krzywda_B1_stationary import simulate_FID_stationary
    T0_grid, C0 = simulate_FID_stationary(
        g_grad=avg_abs_grad,
        noise=noise,
        T_max=T_max,
        dt=dt,
        n_realizations=n_real,
        rng=rng,
        n_T_samples=300,
    )
    T2_stationary = extract_T2(T0_grid, C0)
    print(f"\n  v = 0 (stationary, avg grad):   T_2*  = {T2_stationary*1e9:.0f} ns")

    # shuttling: sweep v
    # keep f_drive = v/a well inside the noise bandwidth [250 kHz, 500 MHz]:
    # v in [0.05, 25] m/s gives f_drive in [0.5 MHz, 250 MHz] -- safe
    v_list = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]   # m/s
    results = {"v_ms": [0.0], "T2_ns": [T2_stationary * 1e9],
               "T_grid": [T0_grid], "C": [C0], "labels": ["v=0"]}

    for v in v_list:
        T_grid, C = simulate_FID_shuttling(
            v=v, field=field, noise=noise, T_max=T_max, dt=dt,
            n_realizations=n_real, rng=rng, n_T_samples=300,
        )
        T2 = extract_T2(T_grid, C)
        results["v_ms"].append(v)
        results["T2_ns"].append(T2 * 1e9 if not np.isnan(T2) else np.nan)
        results["T_grid"].append(T_grid); results["C"].append(C)
        results["labels"].append(f"v={v:g} m/s")
        if np.isnan(T2):
            print(f"  v = {v:5.1f} m/s :  T_2*  = no decay in {T_max*1e6:.1f} μs")
        else:
            print(f"  v = {v:5.1f} m/s :  T_2*  = {T2*1e9:.0f} ns  "
                  f"(improvement {T2/T2_stationary:.2f}x)")

    # pass: improvement over stationary at at least one velocity
    T2_vals = np.array([t for t in results["T2_ns"][1:] if not np.isnan(t)])
    if len(T2_vals) > 0:
        max_improvement = np.max(T2_vals) / (T2_stationary * 1e9)
    else:
        max_improvement = 0.0
    print(f"\n  max improvement: {max_improvement:.2f}x  (vs stationary)")
    passed = max_improvement > 1.2   # at least 20% improvement
    print(f"  B2 PASS (improvement > 1.2x): {passed}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # left: C(T) curves
    cmap = plt.get_cmap("viridis")
    for i, (label, T_grid, C) in enumerate(zip(results["labels"],
                                                results["T_grid"],
                                                results["C"])):
        color = "k" if i == 0 else cmap(i / len(results["labels"]))
        lw = 2 if i == 0 else 1.2
        axes[0].plot(T_grid * 1e9, C, lw=lw, color=color, label=label)
    axes[0].axhline(1/np.e, color="k", ls=":", alpha=0.5)
    axes[0].set_xlabel("T [ns]"); axes[0].set_ylabel("|C(T)|")
    axes[0].set_title("B2: FID with shuttling")
    axes[0].legend(fontsize=9, ncol=2); axes[0].grid(alpha=0.3)
    # right: T2*(v)
    v_arr = np.array(results["v_ms"])
    T2_arr = np.array(results["T2_ns"])
    axes[1].plot(v_arr, T2_arr, "o-", lw=1.5)
    axes[1].axhline(T2_stationary * 1e9, color="k", ls=":",
                    label=f"stationary {T2_stationary*1e9:.0f} ns")
    axes[1].set_xlabel("v [m/s]"); axes[1].set_ylabel("T_2* [ns]")
    axes[1].set_title("B2: Motional narrowing")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = str(FIG_DIR / "step2_B2_motional.png")
    fig.savefig(out, dpi=130)
    print(f"  saved: {out}")

    return {
        "T2_stationary_ns": float(T2_stationary * 1e9),
        "T2_shuttling_ns_by_v": dict(zip(
            [float(v) for v in v_list],
            [float(t) for t in results["T2_ns"][1:]]
        )),
        "max_improvement": float(max_improvement),
        "passed": bool(passed),
    }


if __name__ == "__main__":
    res = run_B2_check()
    print()
    print(f"  Result: {res}")
