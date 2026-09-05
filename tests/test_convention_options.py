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
        eps_v_x=Ev, E_Z=EZ, sigma_E=10e-6 * e_C, profile_norm=norm))


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
                              eps_v_baseline=100e-6 * e_C, eps_v_min=5e-6 * e_C,
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

    # untagged legacy artifact, in the shape an r31 preview run leaves behind
    legacy = wt / "figures" / "phase5" / "phase5_atlas_metadata_preview.json"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy_bytes = b'{"mode": "preview", "n_real": 3}\n'
    legacy.write_bytes(legacy_bytes)
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
    # The tagged run must not disturb an untagged legacy output. The fixture is
    # written here rather than assumed: this test used to read a
    # phase5_atlas_metadata_preview.json left in the working tree by some
    # earlier run, so it passed only while that artifact happened to exist and
    # failed the moment figures/ was cleaned. A test must create what it
    # depends on.
    assert legacy.read_bytes() == legacy_bytes, (
        "the tagged run overwrote the untagged legacy preview metadata")
    import json
    leg = json.loads(legacy.read_text())
    assert "ez_convention" not in leg


def test_paper_figure_total_local_runs(tmp_path, monkeypatch):
    """Audit r6/r7 P0-3: local E_Z(x) array must not crash the figure."""
    import matplotlib
    matplotlib.use("Agg")
    import reproduce.phase5_paper_figures as pf
    monkeypatch.setattr(pf, "FIG_PHASE5", tmp_path, raising=False)
    pf.make_fig2(ez_convention="total-local", profile_norm="final-peak")
    assert list(tmp_path.glob("*__ez-total-local__norm-final-peak*"))


def test_json_only_figures_need_no_simulator():
    """Figs. 3, 4 and S5 must regenerate on a machine that cannot import the
    simulator.

    "Reproducible from the shipped analysis outputs" is a claim about the
    dependency graph, not only about the data. Fig. 2 draws the real coupling
    profiles and needs geometry, the kernel module and constants, which pull in
    jax; the other three read JSON and CSV. This pins the simulator imports
    inside the Fig. 2 path.
    """
    import ast
    import inspect
    import reproduce.phase5_paper_figures as pf

    heavy = ("geometry", "constants", "reproduce.phase4p6_crossterm",
             "jax", "noise")
    for node in ast.parse(inspect.getsource(pf)).body:   # module scope only
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert not any(name == h or name.startswith(h + ".")
                           for h in heavy), (
                f"{name} is imported at module scope; it belongs inside the "
                "Fig. 2 path so --figures 4 runs without jax")


def test_classifier_is_imported_not_redefined():
    """The figure module must not carry its own floors or label rule. A second
    definition of the paper's central estimand living in plotting code is free
    to drift from the analysis, and nothing would report it."""
    import inspect
    import reproduce.phase5_paper_figures as pf
    import reproduce.t0.t0_step4_analysis as an

    assert pf.FLOOR_P is an.FLOOR_P
    assert pf.FLOOR_CHI is an.FLOOR_CHI
    assert pf.classify is an.classify
    src = inspect.getsource(pf)
    assert "FLOOR_P = " not in src and "FLOOR_CHI = " not in src


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
    """r32 contract: a bare (legacy/untagged) collect must be REJECTED, and a
    collect with the adopted suffix must succeed and populate the manuscript
    figures. The untagged r31 outputs must never be installed as manuscript
    figures, so the pre-r32 expectation that a bare collect succeeds no longer
    holds."""
    import subprocess, os, shutil
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wt = tmp_path / "wt"
    shutil.copytree(root, wt, ignore=shutil.ignore_patterns(
        "__pycache__", ".pytest_cache", "*.pyc"))

    r = subprocess.run(["bash", "collect_figures.sh"], cwd=wt / "docs",
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "Only the adopted dataset suffix is accepted" in r.stderr

    # This test covers the collector's routing and fail-closed contract, not
    # the figure generators, so nonempty fixtures stand in for the T0 outputs.
    t0 = wt / "figures" / "t0"
    t0.mkdir(parents=True, exist_ok=True)
    for _stem in ("sensitivity_atlas", "robustness_checks", "quadrant_data"):
        (t0 / f"{_stem}.pdf").write_bytes(b"%PDF-1.4\n% test fixture\n")

    r2 = subprocess.run(["bash", "collect_figures.sh",
                         "__ez-total-local__norm-prefactor"],
                        cwd=wt / "docs", capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    assert "collected 8 manuscript figures" in r2.stdout
    assert (wt / "docs/figures/sensitivity_atlas.pdf").exists()
    # Fig. 3 is now drawn from the shipped per-condition responses; the
    # schematic it replaced is no longer referenced by paper.tex.
    assert (wt / "docs/figures/quadrant_data.pdf").exists()
    assert not (wt / "docs/figures/quadrant_schematic.pdf").exists()


def test_collect_figures_suffix_fails_when_full_outputs_missing(tmp_path):
    """The adopted-suffix collect must fail fast when a required r32 source
    figure is absent, naming the missing file, and must not fall back to the
    r31 atlas outputs."""
    import subprocess, os, shutil
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wt = tmp_path / "wt"
    shutil.copytree(root, wt, ignore=shutil.ignore_patterns(
        "__pycache__", ".pytest_cache", "*.pyc"))

    # Keep the T0 sources present so the failure isolates the missing Fig. 2
    # source rather than tripping on something else first.
    t0 = wt / "figures" / "t0"
    t0.mkdir(parents=True, exist_ok=True)
    for _stem in ("sensitivity_atlas", "robustness_checks", "quadrant_data"):
        (t0 / f"{_stem}.pdf").write_bytes(b"%PDF-1.4\n% test fixture\n")

    missing = (wt / "figures" / "phase5"
               / "phase5_coupling_profiles__ez-total-local__norm-prefactor.pdf")
    assert missing.is_file()
    missing.unlink()

    r = subprocess.run(["bash", "collect_figures.sh",
                        "__ez-total-local__norm-prefactor"],
                       cwd=wt / "docs", capture_output=True, text=True)
    assert r.returncode != 0
    assert "MISSING or EMPTY" in r.stderr
    assert "phase5_coupling_profiles__ez-total-local__norm-prefactor.pdf" in r.stderr
    assert "required source figure(s) missing" in r.stderr


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
