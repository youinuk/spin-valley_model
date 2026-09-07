#!/usr/bin/env python3
"""Post-hoc sensitivity of the nine-category label to the operational floors.

This is a pure re-labelling analysis. It reads the corrected n_real=100 raw
pickles, reconstructs the registered nested prefixes from their stored raw
realization arrays, and varies only the two label floors. No simulation or
response value is recomputed beyond the same prefix aggregation used by the
registered T0 analysis.

Example
-------
python reproduce/t0/threshold_floor_sensitivity.py \
    ../repro-runs/t0c-step3b \
    data/t0c/analysis/threshold_floor_sensitivity.json

The registered floors remain the primary analysis. This script is post-hoc and
must be described as such if its output is reported.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
import pickle
import sys
from collections import Counter

import numpy as np

from t0_step4_analysis import (
    ANSATZE, BLOCK_OF_SEED, FLOOR_CHI as REGISTERED_FLOOR_CHI,
    FLOOR_P as REGISTERED_FLOOR_P, reconstruct,
)

FLOOR_P = REGISTERED_FLOOR_P
FLOOR_CHI = REGISTERED_FLOOR_CHI
if FLOOR_P != 1e-4 or FLOOR_CHI != 1e-3:
    raise RuntimeError("registered floor constants have changed")

FACTORS = (0.5, 0.75, 1.0, 1.5, 2.0)
PREFIXES = (5, 10, 20, 30, 40, 60, 80, 100)
DISCOVERY_BLOCKS = (1, 2, 3)
HOLDOUT_BLOCKS = (4, 5)
EXPECTED_CASES = {"case_i_center", "case_ii_edge"}
EXPECTED_ATLAS = "78daa7a9eabcac25"
EXPECTED_KERNEL = "a250a4343adb77af"
EXPECTED_META = {
    "ez_convention": "total-local",
    "profile_norm": "prefactor",
    "seed_scheme": "cross-ansatz",
    "archive_version": "phase5_atlas_v27",
}

# Exact registered trajectory from the corrected T0-C release. This is an input
# gate, not a fitted target: if it fails, the script is not reading the intended
# corrected layer or its reconstruction logic has drifted.
EXPECTED_REGISTERED = {
    5: (0.39351851851851855, 0.5418981481481482),
    10: (0.36882716049382713, 0.4766203703703704),
    20: (0.37746913580246916, 0.4268518518518518),
    30: (0.38765432098765434, 0.3851851851851852),
    40: (0.38580246913580246, 0.375462962962963),
    60: (0.3938271604938272, 0.34629629629629627),
    80: (0.38611111111111107, 0.31759259259259254),
    100: (0.3947530864197531, 0.29791666666666666),
}

HIGH_TIER = {("A", "A_pocket"), ("A_pocket", "B_z")}
MID_TIER = {("A", "B_x"), ("A_pocket", "B_x"), ("B_z", "B_x")}
LOW_TIER = ("A", "B_z")


def classify(dP: float, dchi: float, floor_p: float, floor_chi: float) -> str:
    p_active = abs(dP) >= floor_p
    c_active = abs(dchi) >= floor_chi
    if not p_active and not c_active:
        return "below_threshold"
    p_imp = dP < 0
    c_imp = dchi < 0
    if p_active and not c_active:
        return "P_only_improve" if p_imp else "P_only_worsen"
    if c_active and not p_active:
        return "chi_only_improve" if c_imp else "chi_only_worsen"
    if p_imp and c_imp:
        return "robust"
    if not p_imp and not c_imp:
        return "both_worsen"
    return "trade_Pimp_chiwor" if p_imp else "trade_Pwor_chiimp"


def load_prefixes(raw_dir: pathlib.Path):
    files = sorted(raw_dir.glob("*__n100.pkl"))
    if len(files) != 10:
        raise SystemExit(f"BLOCKED: expected 10 n=100 pickles in {raw_dir}, found {len(files)}")

    R = {n: {} for n in PREFIXES}
    topology = set()
    sha_pairs = set()
    for path in files:
        with path.open("rb") as fh:
            d = pickle.load(fh)
        cfg = d.get("config", {})
        if d.get("n_real") != 100:
            raise SystemExit(f"BLOCKED: {path.name} carries n_real={d.get('n_real')!r}")
        for k, want in EXPECTED_META.items():
            if cfg.get(k) != want:
                raise SystemExit(f"BLOCKED: {path.name} has {k}={cfg.get(k)!r}, expected {want!r}")
        pair = (cfg.get("atlas_script_sha256_16"), cfg.get("kernel_script_sha256_16"))
        sha_pairs.add(pair)
        block = BLOCK_OF_SEED.get(cfg.get("base_seed"))
        if block is None:
            raise SystemExit(f"BLOCKED: unmapped base_seed in {path.name}")
        case_cfg = cfg.get("case")
        topology.add((block, case_cfg))

        for key, val in d["data"].items():
            ansatz, case, v, lam, geom = key
            if case != case_cfg:
                raise SystemExit(f"BLOCKED: case mismatch in {path.name}: {case!r} vs {case_cfg!r}")
            cond = (case, float(v), float(lam), geom)
            raw = val.get("raw")
            if raw is None:
                raise SystemExit(f"BLOCKED: {path.name} condition {key!r} has no raw block")
            if len(raw["M2"]["phase"]) < max(PREFIXES):
                raise SystemExit(f"BLOCKED: {path.name} condition {key!r} has fewer than 100 realizations")
            for n in PREFIXES:
                k = (ansatz, cond, block)
                if k in R[n]:
                    raise SystemExit(f"BLOCKED: duplicate key at n={n}: {k!r}")
                R[n][k] = reconstruct(raw, n)

    if sha_pairs != {(EXPECTED_ATLAS, EXPECTED_KERNEL)}:
        raise SystemExit(f"BLOCKED: producer SHA pair(s) {sorted(sha_pairs)!r}")
    want_topology = {(b, c) for b in BLOCK_OF_SEED.values() for c in EXPECTED_CASES}
    if topology != want_topology:
        raise SystemExit(f"BLOCKED: block x case topology mismatch: {sorted(topology)!r}")
    for n in PREFIXES:
        if len(R[n]) != 2160:
            raise SystemExit(f"BLOCKED: n={n} has {len(R[n])} response cells, expected 2160")
    return R


def labels(Rn, floor_p, floor_chi):
    return {k: classify(v[0], v[1], floor_p, floor_chi) for k, v in Rn.items()}


def pairwise_metrics(L):
    conds = sorted({k[1] for k in L}, key=repr)
    blocks = sorted({k[2] for k in L})
    d_a = {}
    for a, a2 in itertools.combinations(ANSATZE, 2):
        hits = [L[(a, c, b)] != L[(a2, c, b)] for c in conds for b in blocks]
        d_a[(a, a2)] = float(np.mean(hits))
    d_s = {}
    for b, b2 in itertools.combinations(blocks, 2):
        hits = [L[(a, c, b)] != L[(a, c, b2)] for c in conds for a in ANSATZE]
        d_s[(b, b2)] = float(np.mean(hits))
    D_a = float(np.mean(list(d_a.values())))
    D_s = float(np.mean(list(d_s.values())))
    return D_a, D_s, d_a, d_s


def candidate_metrics(L):
    conds = sorted({k[1] for k in L}, key=repr)
    C = {
        a: {c for c in conds if all(L[(a, c, b)] == "robust" for b in DISCOVERY_BLOCKS)}
        for a in ANSATZE
    }
    U = sorted(set().union(*C.values()), key=repr)
    holdout_transport = Counter()
    for c in U:
        n_confirm = sum(
            all(L[(a, c, b)] == "robust" for b in HOLDOUT_BLOCKS)
            for a in ANSATZE
        )
        holdout_transport[n_confirm] += 1
    discovered_pairs = [(a, c) for a in ANSATZE for c in C[a]]
    confirmed = sum(
        all(L[(a, c, b)] == "robust" for b in HOLDOUT_BLOCKS)
        for a, c in discovered_pairs
    )
    return {
        "C_disc": {a: len(C[a]) for a in ANSATZE},
        "U_disc": len(U),
        "within_ansatz_replication": {"confirmed": confirmed, "total": len(discovered_pairs)},
        "holdout_transport": {str(k): holdout_transport.get(k, 0) for k in range(5)},
    }


def pair_key(pair):
    return f"{pair[0]}/{pair[1]}"


def tier_preserved(d_a) -> bool:
    return (
        min(d_a[p] for p in HIGH_TIER) >= max(d_a[p] for p in MID_TIER)
        and min(d_a[p] for p in MID_TIER) >= d_a[LOW_TIER]
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir", type=pathlib.Path)
    ap.add_argument("out_json", type=pathlib.Path)
    args = ap.parse_args(argv)

    R = load_prefixes(args.raw_dir)

    # Gate the registered point before doing any post-hoc sweep.
    baseline = {}
    for n in PREFIXES:
        L = labels(R[n], FLOOR_P, FLOOR_CHI)
        D_a, D_s, d_a, _ = pairwise_metrics(L)
        want_a, want_s = EXPECTED_REGISTERED[n]
        if D_a != want_a or D_s != want_s:
            raise SystemExit(
                f"BLOCKED: registered trajectory mismatch at n={n}: "
                f"{D_a:.16g}/{D_s:.16g} vs {want_a:.16g}/{want_s:.16g}"
            )
        baseline[str(n)] = {
            "D_ansatz": D_a,
            "D_seed": D_s,
            "R_D": D_a / D_s,
            "pairwise_ansatz": {pair_key(k): v for k, v in d_a.items()},
        }

    sweep = []
    for fp_factor in FACTORS:
        for fc_factor in FACTORS:
            by_n = {}
            for n in PREFIXES:
                floor_p = FLOOR_P * fp_factor
                floor_chi = FLOOR_CHI * fc_factor
                L = labels(R[n], floor_p, floor_chi)
                D_a, D_s, d_a, _ = pairwise_metrics(L)
                entry = {
                    "D_ansatz": D_a,
                    "D_seed": D_s,
                    "R_D": D_a / D_s,
                    "pairwise_ansatz": {pair_key(k): v for k, v in d_a.items()},
                    "tier_partition_preserved": tier_preserved(d_a),
                }
                if n in (30, 100):
                    entry["candidates"] = candidate_metrics(L)
                by_n[str(n)] = entry
            sweep.append({
                "floor_P_factor": fp_factor,
                "floor_chi_factor": fc_factor,
                "floor_P": FLOOR_P * fp_factor,
                "floor_chi": FLOOR_CHI * fc_factor,
                "by_n": by_n,
            })

    def vals(n, key):
        return [x["by_n"][str(n)][key] for x in sweep]

    seed_monotone = 0
    crossover = Counter()
    for x in sweep:
        series = [x["by_n"][str(n)] for n in PREFIXES]
        dseed = [e["D_seed"] for e in series]
        if all(b <= a for a, b in zip(dseed, dseed[1:])):
            seed_monotone += 1
        first = next((n for n, e in zip(PREFIXES, series) if e["R_D"] > 1.0), None)
        crossover[str(first) if first is not None else "none"] += 1

    def candidate_range(n, field, sub=None):
        out = []
        for x in sweep:
            c = x["by_n"][str(n)]["candidates"]
            v = c[field] if sub is None else c[field][sub]
            out.append(v)
        return [min(out), max(out)]

    summary = {
        "grid": {
            "factors": list(FACTORS),
            "n_combinations": len(sweep),
            "description": "independent multiplicative variation of both registered floors",
        },
        "registered_floors": {"dP_v": FLOOR_P, "dchi_phi": FLOOR_CHI},
        "n30": {
            "D_ansatz_range": [min(vals(30, "D_ansatz")), max(vals(30, "D_ansatz"))],
            "D_seed_range": [min(vals(30, "D_seed")), max(vals(30, "D_seed"))],
            "R_D_range": [min(vals(30, "R_D")), max(vals(30, "R_D"))],
            "D_ansatz_gt_D_seed": sum(v > 1.0 for v in vals(30, "R_D")),
            "tier_partition_preserved": sum(
                x["by_n"]["30"]["tier_partition_preserved"] for x in sweep
            ),
            "candidate_U_disc_range": candidate_range(30, "U_disc"),
            "candidate_transport_4of4_range": candidate_range(30, "holdout_transport", "4"),
        },
        "n100": {
            "D_ansatz_range": [min(vals(100, "D_ansatz")), max(vals(100, "D_ansatz"))],
            "D_seed_range": [min(vals(100, "D_seed")), max(vals(100, "D_seed"))],
            "R_D_range": [min(vals(100, "R_D")), max(vals(100, "R_D"))],
            "D_ansatz_gt_D_seed": sum(v > 1.0 for v in vals(100, "R_D")),
            "tier_partition_preserved": sum(
                x["by_n"]["100"]["tier_partition_preserved"] for x in sweep
            ),
            "candidate_U_disc_range": candidate_range(100, "U_disc"),
            "candidate_transport_4of4_range": candidate_range(100, "holdout_transport", "4"),
        },
        "trajectory": {
            "D_seed_monotone_decrease": seed_monotone,
            "crossover_first_R_D_gt_1": dict(sorted(crossover.items())),
        },
    }

    report = {
        "analysis": "post-hoc operational-floor sensitivity",
        "producer_provenance": {
            "atlas_sha256_16": EXPECTED_ATLAS,
            "kernel_sha256_16": EXPECTED_KERNEL,
            **EXPECTED_META,
        },
        "baseline": baseline,
        "summary": summary,
        "sweep": sweep,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"written: {args.out_json}")
    print("registered trajectory gate: PASS (8/8 exact)")
    print(f"floor grid: {len(sweep)} independent combinations, factors {FACTORS}")
    for n in (30, 100):
        s = summary[f"n{n}"]
        print(
            f"n={n}: D_ansatz {s['D_ansatz_range'][0]:.4f}-{s['D_ansatz_range'][1]:.4f}; "
            f"D_seed {s['D_seed_range'][0]:.4f}-{s['D_seed_range'][1]:.4f}; "
            f"R_D {s['R_D_range'][0]:.3f}-{s['R_D_range'][1]:.3f}; "
            f"D_ansatz>D_seed {s['D_ansatz_gt_D_seed']}/{len(sweep)}"
        )
        print(
            f"     pair-tier partition preserved {s['tier_partition_preserved']}/{len(sweep)}; "
            f"U_disc {s['candidate_U_disc_range'][0]}-{s['candidate_U_disc_range'][1]}; "
            f"4/4 holdout transport {s['candidate_transport_4of4_range'][0]}-"
            f"{s['candidate_transport_4of4_range'][1]} conditions"
        )
    print(
        f"D_seed decreases monotonically across all eight checkpoints in "
        f"{summary['trajectory']['D_seed_monotone_decrease']}/{len(sweep)} floor combinations"
    )
    print("first checkpoint with R_D>1: " + ", ".join(
        f"n={k}: {v}" for k, v in summary["trajectory"]["crossover_first_R_D_gt_1"].items()
    ))
    print("Interpretation: post-hoc only; the registered floors and their primary results are unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
