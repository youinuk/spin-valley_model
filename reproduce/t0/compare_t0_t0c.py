#!/usr/bin/env python3
"""Correction audit across the two shipped analysis layers.

    cd repo
    python reproduce/t0/compare_t0_t0c.py \
        --t0 data/t0/analysis --t0c data/t0c/analysis

Reads only. It recomputes nothing and writes nothing.

`data/t0c/analysis/` is the canonical layer and the source of every current
manuscript number. `data/t0/analysis/` is superseded historical provenance,
retained so that the valley-ordering correction has a public audit trail.

This is that audit. It is NOT a before-and-after on one estimand. The two
layers use opposite valley-energy orderings, so the same symbol names different
physical quantities in the two columns: under the superseded ordering the
electron began on the upper diabatic valley branch and `P_v` measured transfer
downward, while under the corrected ordering it begins on the lower branch and
`P_v` is the upper-branch population. Report the pair; never draw an arrow
between the columns, and do not express a difference between them as a
percentage change when the quantity is not ratio-scale.
"""
import argparse
import json
import pathlib
import sys

# (file stem, dotted path, label)
SCALARS = [
    ("attribution_n30", "by_n.30.D_ansatz",            "D_ansatz          n=30"),
    ("attribution_n30", "by_n.30.D_seed",              "D_seed            n=30"),
    ("attribution_n30", "by_n.30.R_D",                 "R_D = Da/Ds       n=30"),
    ("attribution_n30", "by_n.30.complete_separation", "complete separation"),
    ("attribution_n30", "by_n.30.p_all",               "p_all (inactive)"),
    ("attribution_n30", "by_n.30.continuous.dP_v.S_ansatz",     "S_ansatz  dP_v"),
    ("attribution_n30", "by_n.30.continuous.dP_v.S_seed",       "S_seed    dP_v"),
    ("attribution_n30", "by_n.30.continuous.dchi_phi.S_ansatz", "S_ansatz  dchi"),
    ("attribution_n30", "by_n.30.continuous.dchi_phi.S_seed",   "S_seed    dchi"),
    ("posthoc_n30",     "A_channel_three_state.dP_v.ratio",     "channel ratio dP_v"),
    ("posthoc_n30",     "A_channel_three_state.dchi_phi.ratio", "channel ratio dchi"),
    ("legacy_n30",      "mean_rho",                    "legacy rho    n30 b1-5"),
    ("legacy_n30",      "mean_classification_agreement","legacy agree  n30 b1-5"),
    # The n=5 block-1 slice is the seeding-only causal control: same n, same
    # block, only the seed matching changes. It is a DIFFERENT scope from the
    # n=30 aggregate above and the two must never be chained.
    ("legacy_n5_block1","mean_rho",                    "legacy rho    n5  b1"),
    ("legacy_n5_block1","mean_classification_agreement","legacy agree  n5  b1"),
]

# attribution has no --n, so attribution_n100.json carries the whole prefix
# trajectory. Its n=30 entries are the same objects the n=30 run produced,
# which is what the analysis-level prefix check verifies.
SCALARS_100 = [
    ("attribution_n100", "by_n.100.D_ansatz",            "D_ansatz        n=100"),
    ("attribution_n100", "by_n.100.D_seed",              "D_seed          n=100"),
    ("attribution_n100", "by_n.100.R_D",                 "R_D = Da/Ds     n=100"),
    ("attribution_n100", "by_n.100.complete_separation", "complete separation"),
    ("attribution_n100", "by_n.100.p_all",               "p_all (inactive)"),
    ("attribution_n100", "by_n.100.continuous.dP_v.S_ansatz",     "S_ansatz  dP_v"),
    ("attribution_n100", "by_n.100.continuous.dP_v.S_seed",       "S_seed    dP_v"),
    ("attribution_n100", "by_n.100.continuous.dchi_phi.S_ansatz", "S_ansatz  dchi"),
    ("attribution_n100", "by_n.100.continuous.dchi_phi.S_seed",   "S_seed    dchi"),
    ("posthoc_n100",     "A_channel_three_state.dP_v.ratio",     "channel ratio dP_v"),
    ("posthoc_n100",     "A_channel_three_state.dchi_phi.ratio", "channel ratio dchi"),
    ("legacy_n100",      "mean_rho",                     "legacy rho     n100"),
    ("legacy_n100",      "mean_classification_agreement","legacy agree   n100"),
]

SETS =     [("discovery_n30",  "U_disc", "U^disc  n=30")]
SETS_100 = [("discovery_n100", "U_disc", "U^disc  n=100")]


def dig(o, path):
    for k in path.split("."):
        if isinstance(o, list):
            return None
        if k not in o:
            return None
        o = o[k]
    return o


def load(d, stem):
    p = pathlib.Path(d) / f"{stem}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def check_dirs(t0, t0c):
    """Fail loudly, naming the side and the resolved path, before any table."""
    bad = False
    for side, d in (("--t0 ", t0), ("--t0c", t0c)):
        p = pathlib.Path(d)
        n = len(list(p.glob("*.json"))) if p.is_dir() else 0
        state = "OK" if p.is_dir() and n else ("empty" if p.is_dir() else "NOT FOUND")
        print(f"  {side}  {p.resolve()}   {state}"
              + (f", {n} json" if p.is_dir() else ""))
        if state != "OK":
            bad = True
    if bad:
        print("\nBoth paths are resolved against the current directory. The T0")
        print("layer lives inside repo/, so either cd into repo/ first or prefix")
        print("both arguments with repo/.")
    return not bad


def side_of(A, B):
    return ("T0" if A is None else "") + ("/" if A is None and B is None else "") \
           + ("T0-C" if B is None else "")


def table(get, rows, missing):
    print(f"{'estimand':<26}{'T0':>11}{'T0-C':>11}{'delta':>12}")
    for stem, path, label in rows:
        A, B = get("t0", stem), get("t0c", stem)
        if A is None or B is None:
            missing[0] += 1
            print(f"{label:<26}{'':>11}{'':>11}   {stem}.json absent on {side_of(A, B)}")
            continue
        va, vb = dig(A, path), dig(B, path)
        d = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) \
           and not isinstance(va, bool) and not isinstance(vb, bool):
            d = f"{vb - va:+12.6f}"
        elif va is not None and vb is not None and va != vb:
            d = "     changed"
        print(f"{label:<26}{fmt(va)}  {fmt(vb)}  {d}")
        if va is None or vb is None:
            print(f"{'':<26}  (path not found: {path})")


def sets_block(get, rows, missing):
    for stem, key, label in rows:
        A, B = get("t0", stem), get("t0c", stem)
        if A is None or B is None:
            print(f"\n{label}: SKIPPED, {stem}.json absent on {side_of(A, B)}")
            missing[0] += 1
            continue
        sa = {tuple(x) for x in A.get(key, [])}
        sb = {tuple(x) for x in B.get(key, [])}
        print(f"\n{label}: T0 {len(sa)}   T0-C {len(sb)}   shared {len(sa & sb)}")
        for tag, s in (("dropped", sa - sb), ("added", sb - sa)):
            for c in sorted(s, key=repr):
                print(f"   {tag:8s} {'/'.join(map(str, c))}")


def confirm_block(get, stem, label, missing):
    fa, fb = four_of_four(get("t0", stem)), four_of_four(get("t0c", stem))
    if fa is None or fb is None:
        print(f"\n{label}: SKIPPED, {stem}.json absent on "
              f"{side_of(fa, fb)}")
        missing[0] += 1
        return
    print(f"\n{label}: T0 {len(fa)}   T0-C {len(fb)}   shared {len(fa & fb)}")
    for tag, s in (("dropped", fa - fb), ("added", fb - fa)):
        for c in sorted(s, key=repr):
            print(f"   {tag:8s} {'/'.join(map(str, c))}")
    if len(fa) == len(fb) and fa != fb:
        print("   The COUNT is stable and the MEMBERSHIP is not. Reporting the")
        print("   count alone would hide a real effect of the correction.")


def four_of_four(doc):
    if doc is None or "rows" not in doc:
        return None
    out = set()
    for r in doc["rows"]:
        h = r.get("holdout", {})
        if h and all(v.get("confirmed_2of2") for v in h.values()):
            out.add(tuple(r["condition"]))
    return out


def fmt(v):
    if v is None:
        return "   --    "
    if isinstance(v, bool):
        return f"{str(v):>9}"
    if isinstance(v, (int, float)):
        return f"{v:9.6f}"
    return f"{v!s:>9}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", default="data/t0/analysis")
    ap.add_argument("--t0c", default="data/t0c/analysis")
    a = ap.parse_args()

    print("inputs")
    if not check_dirs(a.t0, a.t0c):
        return 2
    print()

    cache = {}

    def get(side, stem):
        key = (side, stem)
        if key not in cache:
            cache[key] = load(a.t0 if side == "t0" else a.t0c, stem)
        return cache[key]

    print("=" * 74)
    print("T0 (closed, +eps_v/2 tau_z)  vs  T0-C (corrected, -eps_v/2 tau_z)")
    print("Two conventions, not two measurements of one quantity.")
    print("=" * 74)
    missing = [0]
    print("n = 30 layer")
    table(get, SCALARS, missing)
    sets_block(get, SETS, missing)
    confirm_block(get, "holdout_n30", "4/4 extended confirmation  n=30", missing)

    print("\n" + "=" * 74)
    print("n = 100 layer")
    print("=" * 74)
    table(get, SCALARS_100, missing)
    sets_block(get, SETS_100, missing)
    confirm_block(get, "confirmation_n100",
                  "4/4 extended confirmation  n=100", missing)

    if missing[0]:
        print(f"\n{missing[0]} comparison(s) could not be made. Nothing above")
        print("is a complete picture until that count is zero.")

    print("\nA difference here is not an error. It is how much the valley-")
    print("ordering convention was carrying, recorded for audit. The canonical")
    print("column is T0-C.")
    return 1 if missing[0] else 0


if __name__ == "__main__":
    sys.exit(main())
