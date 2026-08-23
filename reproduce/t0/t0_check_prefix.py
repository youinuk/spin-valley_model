#!/usr/bin/env python3
"""Gate G-prefix -- the n=100 run must contain the n=30 run exactly.

Realization seeds are `stable_seed(case, v, lambda, r, base=base_seed)`: they
depend on the realization index and not on `n_real`. Each realization gets its
own `default_rng(seed_r)`, and the noise trace length is set by the trajectory
time, not by the ensemble size. Realization *r* is therefore bit-identical for
any `n_real > r`, and the n=100 production is a strict superset of the STEP 3
n=30 data.

That makes this an EXACT gate, not a tolerance check, and it does double duty:
it is also the outstanding environment verification of the n=30 analysis, since
it re-derives the n=30 numbers on the pinned machine. A separate n=30 re-run is
therefore not needed (protocol Sec. 8 entry 18).

  python t0_check_prefix.py --ref DIR_N30 --new DIR_N100

Any mismatch means the two datasets cannot be placed on one convergence
trajectory. Stop and diagnose rather than proceeding to analysis.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

CONFIG_MUST_MATCH = ("ez_convention", "profile_norm", "B_ext_T", "sigma_E_ueV",
                     "mode", "case", "base_seed", "seed_scheme",
                     "atlas_script_sha256_16", "kernel_script_sha256_16")
OBS = ("P_v_dia", "P_v_ad", "phase", "spin_purity", "S_s")


def index(directory: Path, n: int) -> dict:
    out = {}
    for p in sorted(directory.glob(f"*__n{n}.pkl")):
        with open(p, "rb") as fh:
            d = pickle.load(fh)
        cfg = d.get("config", {})
        out[(cfg.get("base_seed"), cfg.get("case"))] = (p, d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", type=Path, required=True, help="directory of n=30 pickles")
    ap.add_argument("--new", type=Path, required=True, help="directory of n=100 pickles")
    ap.add_argument("--n-ref", type=int, default=30)
    ap.add_argument("--n-new", type=int, default=100)
    args = ap.parse_args()

    ref, new = index(args.ref, args.n_ref), index(args.new, args.n_new)
    print(f"REF n={args.n_ref}: {len(ref)} files   NEW n={args.n_new}: {len(new)} files")
    problems: list[str] = []
    if set(ref) != set(new):
        problems.append(f"(base_seed, case) sets differ: "
                        f"only in REF {sorted(set(ref) - set(new))}, "
                        f"only in NEW {sorted(set(new) - set(ref))}")

    n_val = n_bad = 0
    for key in sorted(set(ref) & set(new)):
        (_, dr), (_, dn) = ref[key], new[key]
        for k in CONFIG_MUST_MATCH:
            if dr["config"].get(k) != dn["config"].get(k):
                problems.append(f"{key}: config '{k}' "
                                f"{dr['config'].get(k)!r} -> {dn['config'].get(k)!r}")
        if dn.get("n_real") != args.n_new:
            problems.append(f"{key}: n_real is {dn.get('n_real')}, expected {args.n_new}")
        if set(dr["data"]) != set(dn["data"]):
            problems.append(f"{key}: condition sets differ "
                            f"({len(dr['data'])} vs {len(dn['data'])})")
            continue
        for ck in dr["data"]:
            rr, rn = dr["data"][ck]["raw"], dn["data"][ck]["raw"]
            if list(rr["seeds"]) != list(rn["seeds"])[:args.n_ref]:
                problems.append(f"{key} {ck}: seed prefix differs")
            for m in ("M1", "M2"):
                for o in OBS:
                    a, b = list(rr[m][o]), list(rn[m][o])[:args.n_ref]
                    n_val += len(a)
                    d = sum(1 for x, y in zip(a, b) if x != y)
                    n_bad += d
                    if d:
                        problems.append(f"{key} {ck}: {m}.{o} prefix differs in "
                                        f"{d}/{len(a)} realizations")
            for o in OBS:
                n_val += 1
                if rr["M1V"][o] != rn["M1V"][o]:
                    n_bad += 1
                    problems.append(f"{key} {ck}: M1V.{o} differs "
                                    f"({rr['M1V'][o]!r} vs {rn['M1V'][o]!r})")

    print(f"realization values compared: {n_val}   mismatching: {n_bad}")
    if problems:
        print("\nGATE FAILED")
        for m in problems[:40]:
            print(f"  - {m}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        return 1
    print("\nGATE PASSED -- the n=100 data contains the n=30 data exactly.")
    print("This also verifies the n=30 analysis on the pinned environment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
