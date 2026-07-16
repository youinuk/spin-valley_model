"""
Physical constants and representative Si/SiGe shuttling parameters.
All units are SI. Frequency (Hz) and angular frequency (rad/s) are kept
clearly distinct.

This file is the single source of truth; no other module defines its own
constants.

Important: products such as nm-scale length times T/m gradient lose
precision in float32, which makes automatic differentiation drift
inconsistently. Importing this file forces JAX into 64-bit mode.
"""

# Force JAX x64 (must run before any other jax import)
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

# ============================================================
# Fundamental physical constants (CODATA 2018)
# ============================================================
hbar = 1.054571817e-34      # J*s
h    = 6.62607015e-34       # J*s
e    = 1.602176634e-19      # C
mu_B = 9.2740100783e-24     # J/T
k_B  = 1.380649e-23         # J/K

# ============================================================
# Si/SiGe platform parameters (representative, literature-based)
# ============================================================
g_Si = 2.0                  # electron g-factor (Si)
m_Si_t = 0.19 * 9.109e-31   # Si transverse effective mass (kg)

# frequency <-> field conversion helpers
def freq_to_B(f_Hz: float, g: float = g_Si) -> float:
    """Frequency f [Hz] -> field [T] producing that Zeeman splitting."""
    return h * f_Hz / (g * mu_B)

def B_to_freq(B_T: float, g: float = g_Si) -> float:
    """Field [T] -> Zeeman frequency [Hz]."""
    return g * mu_B * B_T / h

def B_to_omega(B_T: float, g: float = g_Si) -> float:
    """Field [T] -> Larmor angular frequency [rad/s]."""
    return g * mu_B * B_T / hbar

# ============================================================
# Representative shuttling parameters (Phase 1 regime map)
# ============================================================
class Defaults:
    """Representative values fixed in Phase 1."""
    B_ext_T        = 0.5             # external field
    g              = g_Si
    # micromagnet stray field
    grad_lo        = 0.1e-3 / 1e-9   # T/m (0.1 mT/nm)
    grad_hi        = 1.0e-3 / 1e-9   # T/m (1.0 mT/nm)
    dB_long        = 20e-3           # T, longitudinal modulation amplitude (~20 mT)
    b_trans        = 5e-3            # T, transverse (drive) amplitude
    # magnet-array period
    period_lo_m    = 100e-9
    period_hi_m    = 400e-9
    # shuttling velocity -- exploration range 20-60 m/s
    v_lo_ms        = 10.0
    v_hi_ms        = 100.0
    v_sweet_lo     = 20.0
    v_sweet_hi     = 60.0
    # orbital / quantum-dot size
    orbital_size_m = 20e-9
    # valley
    Ev_typical_J   = 100e-6 * e      # 100 ueV (center of the Volmer 2024 distribution)
    Ev_min_J       = 1.5e-6 * e      # 1.5 ueV (low-Ev pocket)
    Ev_max_J       = 200e-6 * e      # 200 ueV
    Delta_sv_J     = 0.5e-6 * e      # valley off-diagonal coupling 0.5 ueV
    Lc_valley_m    = 50e-9           # valley-landscape correlation length
    # charge noise
    sigma_dx_m     = 0.3e-9          # position-jitter amplitude (RMS), 0.3 nm
    f_low_Hz       = 1e3             # 1/f lower cutoff
    f_high_Hz      = 1e9             # 1/f upper cutoff
    # charge-noise strength for T2* (Krzywda before-mitigation reference)
    T2_before_us   = 4.4
    T2_after_us    = 8.5
    T2_ratio_target= 8.5 / 4.4       # ~1.93


# ============================================================
# Phase 2-B/C simulation grid (standard)
# ============================================================
# Rationale:
#   - The shuttling modulation frequency f_drive = v/a is 100-600 MHz for
#     v = 10-60 m/s and a = 100 nm.
#   - The noise generator's f_high must resolve this (>= 500 MHz).
#   - The noise generator's f_low must be at least the trace grid df = 1/T_max.
#   - dt < 1/(2 f_high), and T_max is ~5x the expected T2*.
class Phase2Grid:
    dt_s     = 1e-9      # 1 ns
    T_max_s  = 4e-6      # 4 us  -> df = 250 kHz
    f_low_Hz = 2.5e5     # 250 kHz  (exactly df = 1/T_max)
    # set f_high to 90% of Nyquist (= 0.5/dt = 500 MHz) to avoid boundary warnings
    f_high_Hz= 4.5e8     # 450 MHz (0.9x the 500 MHz Nyquist)
    # long trace for window sampling
    T_long_s = 5e-4      # 500 us  ~ 2000 windows


# ============================================================
# Project paths (avoid hard-coded absolute paths)
# ============================================================
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
FIG_DIR = PROJECT_ROOT / "figures" / "phase2"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR_PHASE3 = PROJECT_ROOT / "figures" / "phase3"
FIG_DIR_PHASE3.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # quick self-check: Zeeman frequency at representative values
    fz = B_to_freq(Defaults.B_ext_T)
    print(f"Zeeman frequency at B_ext={Defaults.B_ext_T} T:  {fz/1e9:.3f} GHz")
    print(f"Zeeman from 20 mT modulation:             {B_to_freq(Defaults.dB_long)/1e9:.3f} GHz")
    print(f"Valley splitting (typical 100 ueV):        {Defaults.Ev_typical_J/h/1e9:.3f} GHz")
    print(f"Valley splitting (low pocket 1.5 ueV):     {Defaults.Ev_min_J/h/1e9:.4f} GHz")
    print(f"T2 ratio target (Krzywda):                 {Defaults.T2_ratio_target:.3f}")
