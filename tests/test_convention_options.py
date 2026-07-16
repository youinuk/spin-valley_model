"""Unit tests for the ez-convention / profile-norm options (audit r5).

Fast tests only: kernel-level checks plus one short end-to-end run that
verifies the Zeeman convention reaches the Hamiltonian.
"""

import numpy as np
import pytest

from constants import Defaults, g_Si, mu_B
from reproduce.phase4p6_crossterm import lambda_sv_profile, run_one_condition
from reproduce.phase5_sensitivity_atlas import make_ff, GEOM0, dataset_tag
from reproduce.phase5_atlas_merge_validate import validate_raw_configs

e_C = 1.602176634e-19
PW = 30e-9


@pytest.fixture(scope="module")
def field_and_grid():
    ff = make_ff(**GEOM0)
    x = np.linspace(-4 * PW, 4 * PW, 1501)
    Bz = np.asarray(ff.B_z(x))
    Ev = 100e-6 * e_C - 95e-6 * e_C * np.exp(-x**2 / (2 * PW**2))
    return ff, x, Bz, Ev


def _profile(ff, x, Ev, EZ, model, norm="prefactor"):
    return np.asarray(lambda_sv_profile(
        x, ff, model, 1.0, pocket_x_center=0.0, sigma_lambda=PW,
        Ev_x=Ev, E_Z=EZ, sigma_E=10e-6 * e_C, profile_norm=norm))


def test_final_peak_normalizes_all_models(field_and_grid):
    """Audit r5 par.7: final-peak must apply to every ansatz, incl. A_pocket."""
    ff, x, Bz, Ev = field_and_grid
    EZ = g_Si * mu_B * (Defaults.B_ext_T + Bz)
    for m in ["A", "A_pocket", "B_z", "B_x"]:
        prof = _profile(ff, x, Ev, EZ, m, norm="final-peak")
        assert np.max(np.abs(prof)) == pytest.approx(1.0, rel=1e-12), m


def test_l2_norm_fixes_rms(field_and_grid):
    ff, x, Bz, Ev = field_and_grid
    EZ = g_Si * mu_B * (Defaults.B_ext_T + Bz)
    for m in ["A", "A_pocket", "B_z", "B_x"]:
        prof = _profile(ff, x, Ev, EZ, m, norm="l2")
        assert np.sqrt(np.mean(prof**2)) == pytest.approx(1.0, rel=1e-12), m


def test_prefactor_legacy_invariance(field_and_grid):
    """Regression pin: the legacy (prefactor) kernels are unchanged."""
    ff, x, Bz, Ev = field_and_grid
    ap = _profile(ff, x, Ev, 50e-6 * e_C, "A_pocket")
    assert np.max(np.abs(ap)) == pytest.approx(0.703, abs=5e-3)
    bx = np.abs(_profile(ff, x, Ev, 50e-6 * e_C, "B_x"))
    # twin flank peaks are exactly degenerate for the centred pocket, so pin
    # |x_peak|, not its sign (argmax side is a floating-point tie-break)
    assert abs(x[np.argmax(bx)]) * 1e9 == pytest.approx(38.8, abs=1.0)


def test_center_flank_peaks_are_symmetric(field_and_grid):
    """Centred pocket: both resonance ansaetze activate on BOTH flanks with
    equal strength; any 'opposite side' reading is a tie-break artefact."""
    ff, x, Bz, Ev = field_and_grid
    for EZ in [50e-6 * e_C, g_Si * mu_B * (Defaults.B_ext_T + Bz)]:
        for m in ["B_z", "B_x"]:
            prof = np.abs(_profile(ff, x, Ev, EZ, m))
            L = prof[x < 0].max(); R = prof[x > 0].max()
            assert R == pytest.approx(L, rel=1e-6), (m, "flank asymmetry")


def test_total_local_differs_from_total_mean(field_and_grid):
    ff, x, Bz, Ev = field_and_grid
    EZ_local = g_Si * mu_B * (Defaults.B_ext_T + Bz)
    EZ_mean = g_Si * mu_B * (Defaults.B_ext_T + float(Bz.mean()))
    a = _profile(ff, x, Ev, EZ_local, "B_z")
    b = _profile(ff, x, Ev, EZ_mean, "B_z")
    assert not np.allclose(a, b)


def test_dataset_tag_unique_and_legacy_empty():
    """Audit r5 par.2: non-legacy datasets must not collide with legacy names."""
    assert dataset_tag("stray-mean", "prefactor") == ""
    tags = {dataset_tag(ez, nm)
            for ez in ["stray-mean", "total-local", "total-mean"]
            for nm in ["prefactor", "final-peak", "l2"]}
    assert len(tags) == 9  # all distinct (legacy empty counted once)


def _cfg(**kw):
    base = dict(ez_convention="total-local", profile_norm="prefactor",
                B_ext_T=0.5, sigma_E_ueV=10.0, mode="validate", n_real=5,
                atlas_script_sha256_16="a" * 16,
                kernel_script_sha256_16="b" * 16,
                archive_version="phase5_atlas_v27")
    base.update(kw)
    return base


def test_merge_config_validation():
    """Audit r5 par.4: merge must reject inconsistent or unexpected configs."""
    ok = validate_raw_configs(
        {"c": _cfg(), "e": _cfg()}, "total-local", "prefactor")
    assert ok["ez_convention"] == "total-local"
    with pytest.raises(ValueError):
        validate_raw_configs({"c": _cfg(), "e": _cfg(ez_convention="stray-mean")},
                             "total-local", "prefactor")
    with pytest.raises(ValueError):  # config vs requested dataset mismatch
        validate_raw_configs({"c": _cfg(), "e": _cfg()}, "stray-mean", "prefactor")
    with pytest.raises(ValueError):  # legacy raw without --allow-legacy
        validate_raw_configs({"c": None, "e": _cfg()}, "stray-mean", "prefactor")
    assert validate_raw_configs({"c": None, "e": None}, "stray-mean", "prefactor",
                                allow_legacy=True) == "legacy-unrecorded"


def test_ez_convention_reaches_hamiltonian(field_and_grid):
    """Audit r5 test 1/2: the convention must change the simulated response."""
    ff, _, _, _ = field_and_grid
    from noise.charge_noise import OneOverFNoise
    noise = OneOverFNoise(sigma_total=Defaults.sigma_dx_m, alpha=1.0,
                          f_low=1e3, f_high=1e7)
    out = {}
    for conv in ["stray-mean", "total-local"]:
        R = run_one_condition(v=10.0, case_label="t", pocket_x_center=0.0,
                              lambda_0=1e-6 * e_C, coupling_model="B_z",
                              n_real=1, noise=noise, ff=ff,
                              Ev_baseline=100e-6 * e_C, Ev_min=5e-6 * e_C,
                              pocket_width=PW, Delta_v=0.5e-6 * e_C,
                              N_max=500, base_seed=41,
                              T_traj_for_noise=(8 * PW) / 10.0,
                              ez_convention=conv)
        out[conv] = R["M2"]["P_v_dia"]["mean"]
    assert out["stray-mean"] != pytest.approx(out["total-local"], rel=1e-3)


@pytest.mark.slow
def test_save_raw_e2e_tagged(tmp_path, monkeypatch):
    """Audit r6/r7 P0-1: the exact failing command path must work end to end
    (config block written inside an open file handle, tagged filename)."""
    import subprocess, sys, os, pickle, glob, shutil
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wt = tmp_path / "wt"
    shutil.copytree(root, wt, ignore=shutil.ignore_patterns(
        "__pycache__", ".pytest_cache", "*.pyc", ".git"))
    env = dict(os.environ, PYTHONPATH=str(wt), MPLBACKEND="Agg",
               JAX_PLATFORM_NAME="cpu", JAX_ENABLE_X64="1")
    r = subprocess.run(
        [sys.executable, "reproduce/phase5_sensitivity_atlas.py",
         "--mode", "preview", "--case", "case_i_center",
         "--ez-convention", "total-local", "--profile-norm", "final-peak",
         "--save-raw", "--no-plots"],
        cwd=wt, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-800:]
    f = glob.glob(str(wt / "figures/phase5/phase5_atlas_raw_preview_*"
                       "__ez-total-local__norm-final-peak.pkl"))
    assert len(f) == 1
    cfg = pickle.load(open(f[0], "rb"))["config"]
    assert cfg["ez_convention"] == "total-local"
    assert cfg["profile_norm"] == "final-peak"
    assert len(cfg["atlas_script_sha256"]) == 64
    # legacy preview outputs must remain untouched by the tagged run
    import json
    leg = json.load(open(wt / "figures/phase5/phase5_atlas_metadata_preview.json"))
    assert "ez_convention" not in leg


def test_paper_figure_total_local_runs(tmp_path, monkeypatch):
    """Audit r6/r7 P0-3: local E_Z(x) array must not crash the figure."""
    import matplotlib
    matplotlib.use("Agg")
    import reproduce.phase5_paper_figures as pf
    monkeypatch.setattr(pf, "FIG", tmp_path, raising=False)
    pf.make_coupling_profiles(ez_convention="total-local",
                              profile_norm="final-peak")
    assert list(tmp_path.glob("*__ez-total-local__norm-final-peak*"))


def test_absolute_wiring_exists():
    """Audit r6/r7 P0-4: absolute table must pass the convention into
    run_one_condition (source-level wiring check; the run itself is heavy)."""
    import inspect
    import reproduce.phase5_absolute_performance as ab
    src = inspect.getsource(ab)
    assert "ez_convention=EZ_CONVENTION" in src
    assert "profile_norm=PROFILE_NORM" in src
    assert '"--ez-convention"' in src and '"--profile-norm"' in src


def test_collect_figures_legacy_and_suffix_succeed(tmp_path):
    """Legacy collect must succeed; and because the released package now
    ships the full total-local validate outputs, a suffixed collect must
    also succeed and populate the manuscript figures."""
    import subprocess, os, shutil
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wt = tmp_path / "wt"
    shutil.copytree(root, wt, ignore=shutil.ignore_patterns(
        "__pycache__", ".pytest_cache", "*.pyc"))
    r = subprocess.run(["bash", "collect_figures.sh"], cwd=wt / "docs",
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (wt / "docs/figures/quadrant_schematic.pdf").exists()
    r2 = subprocess.run(["bash", "collect_figures.sh",
                         "__ez-total-local__norm-prefactor"],
                        cwd=wt / "docs", capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    assert (wt / "docs/figures/sensitivity_atlas.pdf").exists()


def test_collect_figures_suffix_fails_when_full_outputs_missing(tmp_path):
    """A suffixed collect must fail fast with a clear message when the
    full validate outputs for that suffix are absent (the quadrant
    schematic is convention-independent and untagged)."""
    import subprocess, os, shutil, glob
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wt = tmp_path / "wt"
    shutil.copytree(root, wt, ignore=shutil.ignore_patterns(
        "__pycache__", ".pytest_cache", "*.pyc"))
    # remove the total-local suffixed source figures so the collect misses
    for p in glob.glob(str(wt / "figures/phase5/*__ez-total-local__norm-prefactor.pdf")):
        os.remove(p)
    r = subprocess.run(["bash", "collect_figures.sh",
                        "__ez-total-local__norm-prefactor"],
                       cwd=wt / "docs", capture_output=True, text=True)
    assert r.returncode != 0
    assert "MISSING" in r.stderr and "full validate" in r.stderr.lower()


def test_robustness_clear_error_on_missing_dataset():
    import subprocess, sys, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=root, MPLBACKEND="Agg",
               JAX_PLATFORM_NAME="cpu")
    r = subprocess.run([sys.executable, "reproduce/phase5_supp_robustness.py",
                        "--dataset-suffix", "__ez-no-such__norm-x"],
                       cwd=root, env=env, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode != 0
    assert "run the full validate" in (r.stderr + r.stdout).lower()


def test_channel_balanced_distance_is_not_uniform_rescaling():
    """The channel-balanced model distance must use distinct per-channel
    scales (P vs chi effect-size thresholds), so it is a genuinely
    different metric from raw D -- not raw x const. Guards against the
    P_SCALE == CHI_SCALE regression that made the 'normalized' panel a
    duplicate of the raw panel."""
    from reproduce.phase5_sensitivity_atlas import P_SCALE, CHI_SCALE
    assert P_SCALE != CHI_SCALE, (
        "P_SCALE == CHI_SCALE makes the normalized distance a uniform "
        "rescaling of raw D (no extra information)")
    # a uniform rescaling would preserve the ranking exactly; distinct
    # scales must be able to reorder pairs. Check on a small synthetic set
    # where the two channels carry different information.
    import numpy as np
    # two pairs: pair1 differs mostly in P, pair2 mostly in chi
    raw1 = np.hypot(1e-3, 0.0);      raw2 = np.hypot(0.0, 2e-3)
    nrm1 = np.hypot(1e-3 / P_SCALE, 0.0)
    nrm2 = np.hypot(0.0, 2e-3 / CHI_SCALE)
    # raw ranks pair2 > pair1; channel-balanced must be able to flip this
    assert (raw2 > raw1) and (nrm1 > nrm2), (
        "channel-balanced distance should reweight P vs chi and can "
        "reorder pairs relative to raw D")
