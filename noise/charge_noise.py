"""
Charge noise generator.

Builds the position jitter delta_x(t) as a time-dependent stochastic process.
Primary model: 1/f power spectral density (PSD).

    S_dx(f) = A^2 / |f|^alpha   for f_low < |f| < f_high

Typically alpha = 1 (true 1/f). The Phase 1 quasi-static limit corresponds to
delta_x being constant per realization (all power at f -> 0).

Generation: standard frequency-domain synthesis.
    1) amplitude = sqrt(S_dx(f) * df) on the df grid
    2) uniform random phase at each frequency
    3) Hermitian symmetry to ensure a real signal
    4) IFFT -> time-domain trace

Self-check: the empirical PSD of the generated trace reproduces the input PSD.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Tuple, Optional

from constants import Defaults, FIG_DIR


@dataclass(frozen=True)
class OneOverFNoise:
    """
    1/f^alpha charge noise generator.
    
    Parameters
    ----------
    sigma_total : float
        RMS amplitude of delta_x integrated over [f_low, f_high], in metres.
        Specifically: sigma_total^2 = integral S_dx(f) df over [f_low, f_high].
    alpha : float
        PSD exponent.  1.0 = true 1/f.  0 = white.
    f_low, f_high : float
        Integration limits [Hz].
    """
    sigma_total: float = Defaults.sigma_dx_m
    alpha:       float = 1.0
    f_low:       float = Defaults.f_low_Hz
    f_high:      float = Defaults.f_high_Hz

    def psd(self, f: np.ndarray) -> np.ndarray:
        """
        One-sided PSD [m^2/Hz] at frequency f [Hz].
        Below f_low / above f_high is cut off to 0.

        Normalization: integral psd df = sigma_total^2 over [f_low, f_high].
        """
        f = np.asarray(f)
        out = np.zeros_like(f, dtype=float)
        mask = (f >= self.f_low) & (f <= self.f_high)
        # unnormalized 1/f^alpha
        if abs(self.alpha - 1.0) < 1e-9:
            # normalization constant: integral 1/f df = ln(f_high/f_low)
            norm = np.log(self.f_high / self.f_low)
            out[mask] = self.sigma_total**2 / (f[mask] * norm)
        else:
            # integral f^-alpha df = (f^(1-alpha))/(1-alpha)
            f1, f2 = self.f_low, self.f_high
            norm = (f2**(1 - self.alpha) - f1**(1 - self.alpha)) / (1 - self.alpha)
            out[mask] = self.sigma_total**2 * f[mask]**(-self.alpha) / norm
        return out

    def generate(self, T_total: float, dt: float,
                 rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a delta_x(t) trace sampled at dt over [0, T_total].

        Standard frequency-domain synthesis:
            X_k = sqrt(S(f_k) * df / 2) * sqrt(N) * exp(i * phi_k)   for k = 1..N/2
            x_n = Re[ ifft(X) ]    (numpy ifft includes 1/N)
            => <|x|^2> = sum_k S(f_k) * df   (Parseval, one-sided)

        Check: as dt -> 0, T_total -> infinity, std(x) -> sigma_total.

        Returns
        -------
        t : np.ndarray, shape (N,)
        dx_t : np.ndarray, shape (N,)
        """
        if rng is None:
            rng = np.random.default_rng()
        N = int(np.round(T_total / dt))
        if N % 2 == 1:
            N += 1
        df = 1.0 / (N * dt)
        f_pos = np.arange(1, N // 2 + 1) * df

        # target RMS = sigma_total * sqrt(fraction of variance inside [f_low, f_high])
        #                                that is covered by [f_low, min(f_high, Nyquist)])
        # psd() already normalizes to sigma_total^2 over [f_low, f_high], so
        # if the grid [df, Nyq] covers [f_low, f_high] this is automatically OK.
        nyq = 1.0 / (2 * dt)
        if nyq < self.f_high:
            # grid does not fully cover f_high -- warn
            import warnings
            warnings.warn(
                f"Nyquist {nyq:.2e} Hz < f_high {self.f_high:.2e}; "
                f"empirical RMS will be reduced.",
                stacklevel=2,
            )
        if df > self.f_low:
            import warnings
            warnings.warn(
                f"df {df:.2e} Hz > f_low {self.f_low:.2e}; "
                f"empirical RMS will be reduced.",
                stacklevel=2,
            )

        # 1-bin variance computation:
        # X_k = A exp(i phi),  X_{-k} = A exp(-i phi),  others 0
        # => x_n = (2A/N) cos(2 pi k n / N + phi)
        # => var(x) = 2 A^2 / N^2
        # desired: variance contribution of one bin = S_one(f_k) * df
        # => A = N * sqrt(S_one * df / 2)
        amp = N * np.sqrt(self.psd(f_pos) * df / 2.0)
        phases = rng.uniform(0.0, 2 * np.pi, size=len(f_pos))
        X_pos = amp * np.exp(1j * phases)

        # Hermitian-symmetric full spectrum, numpy packing [0, 1..N/2, -(N/2-1)..-1]
        X = np.zeros(N, dtype=complex)
        X[0] = 0.0
        X[1:N // 2] = X_pos[:-1]
        X[N // 2] = np.real(X_pos[-1])    # Nyquist is real
        X[N // 2 + 1:] = np.conj(X_pos[-2::-1])

        dx_t = np.real(np.fft.ifft(X))   # numpy ifft has 1/N normalization
        t = np.arange(N) * dt
        return t, dx_t


# ============================================================
# Self-checks: empirical PSD recovers input PSD, RMS matches sigma_total
# ============================================================
def empirical_psd(x: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided PSD by periodogram. units of x^2/Hz."""
    N = len(x)
    X = np.fft.fft(x) * dt
    psd_two = np.abs(X)**2 / (N * dt)
    f = np.fft.fftfreq(N, dt)
    # one-sided
    f_pos = f[1:N // 2]
    psd_one = 2 * psd_two[1:N // 2]
    return f_pos, psd_one


if __name__ == "__main__":
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="1/f charge noise generator self-check")
    parser.add_argument("--full", action="store_true",
                        help="full validation (5M samples, 20 realizations, ~slow)")
    args = parser.parse_args()
    mode_quick = not args.full

    print("="*60)
    print(f"Charge noise generator self-check  [{'QUICK' if mode_quick else 'FULL'}]")
    print("="*60)

    if mode_quick:
        # light check: narrower time range, fewer realizations. Enough for 1/f slope and RMS.
        noise = OneOverFNoise(
            sigma_total=0.3e-9, alpha=1.0,
            f_low=1e4, f_high=1e7,   # 3 decade
        )
        dt = 0.1 / noise.f_high       # 10 ns
        T_total = 5.0 / noise.f_low   # 0.5 ms
        n_real = 5
    else:
        # full check: wider bandwidth, more realizations
        noise = OneOverFNoise(
            sigma_total=0.3e-9, alpha=1.0,
            f_low=1e3, f_high=1e7,
        )
        dt = 0.1 / noise.f_high       # 10 ns
        T_total = 50.0 / noise.f_low  # 50 ms
        n_real = 20

    N = int(round(T_total / dt))
    print(f"  sigma_total = {noise.sigma_total*1e9:.3f} nm RMS")
    print(f"  alpha = {noise.alpha}")
    print(f"  f range = [{noise.f_low:.0e}, {noise.f_high:.0e}] Hz")
    print(f"  dt = {dt*1e9:.2f} ns,  T_total = {T_total*1e3:.3f} ms,  N = {N}")
    print(f"  realizations = {n_real}")

    rng = np.random.default_rng(42)
    psd_avg = None
    rms_list = []
    for k in range(n_real):
        t, dx = noise.generate(T_total, dt, rng=rng)
        rms_list.append(np.std(dx))
        f_emp, psd_emp = empirical_psd(dx, dt)
        if psd_avg is None:
            psd_avg = psd_emp.copy()
        else:
            psd_avg += psd_emp
    psd_avg /= n_real
    rms_mean = np.mean(rms_list)
    rms_std = np.std(rms_list)
    print(f"  empirical RMS: {rms_mean*1e9:.3f} +/- {rms_std*1e9:.3f} nm  "
          f"(target {noise.sigma_total*1e9:.3f})")

    # PSD log-log fit slope
    mask = (f_emp >= 3 * noise.f_low) & (f_emp <= noise.f_high / 3)
    log_f = np.log10(f_emp[mask])
    log_p = np.log10(psd_avg[mask])
    slope = np.polyfit(log_f, log_p, 1)[0]
    print(f"  empirical PSD slope: {slope:.3f}  (target {-noise.alpha:.3f})")

    # plots
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t * 1e6, dx * 1e9, lw=0.6)
    axes[0].set_xlabel("t [us]")
    axes[0].set_ylabel("dx(t) [nm]")
    axes[0].set_title("One realization, 1/f charge noise")
    axes[0].grid(alpha=0.3)
    axes[1].loglog(f_emp, psd_avg, label=f"empirical (avg of {n_real})", lw=0.8)
    axes[1].loglog(f_emp, noise.psd(f_emp), "--", label="input PSD", lw=1.2)
    axes[1].set_xlabel("f [Hz]")
    axes[1].set_ylabel("S_dx(f) [m^2/Hz]")
    axes[1].set_title(f"PSD recovery (alpha={noise.alpha})")
    axes[1].legend(); axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = str(FIG_DIR / "step2_charge_noise.png")
    fig.savefig(out, dpi=130)
    print(f"  saved: {out}")

    print()
    rms_ok = abs(rms_mean - noise.sigma_total) / noise.sigma_total < 0.10
    slope_ok = abs(slope - (-noise.alpha)) < 0.15
    print(f"  RMS within 10%:    {'PASS' if rms_ok else 'FAIL'}")
    print(f"  slope within 0.15: {'PASS' if slope_ok else 'FAIL'}")
