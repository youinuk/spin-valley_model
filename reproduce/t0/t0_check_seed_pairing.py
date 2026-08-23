#!/usr/bin/env python3
"""Record the actual seed-sharing structure of the patched producer.

EXPERIMENT_PROTOCOL_T0.md Sec. 3 states that the geometry-perturbation label is
deliberately absent from the seed, but shows `case_label` as a seed argument.
If the producer in fact passes a constant there, then centre and edge share
random numbers too, and the paired structure is wider than the protocol
documents. That is a legitimate variance-reduction design -- the only thing
that would be wrong is calling the resulting runs, cases or conditions
independent.

This is an experimental-design fact, not a provenance check. Run it once,
passing BOTH cases of one block so the case axis is testable, and write the
outcome into Sec. 3.

  python t0_check_seed_pairing.py CENTRE.pkl EDGE.pkl
"""
from __future__ import annotations

import pickle
import sys
from collections import defaultdict

AXES = (
    ("ansatz",   lambda k: (k[1], k[2], k[3], k[4])),
    ("case",     lambda k: (k[0], k[2], k[3], k[4])),
    ("geometry", lambda k: (k[0], k[1], k[2], k[3])),
)


def main(paths) -> int:
    seeds: dict[tuple, tuple] = {}
    for path in paths:
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        cfg = d.get("config", {})
        print(f"{path}\n  seed_scheme {cfg.get('seed_scheme')!r}   "
              f"base_seed {cfg.get('base_seed')}   n_real {d.get('n_real')}")
        for key, val in d["data"].items():
            raw = val.get("raw")
            if raw is None or "seeds" not in raw:
                raise SystemExit("  ! no raw['seeds'] -- run the patched atlas "
                                 "with realization-level raw enabled")
            seeds[key] = tuple(raw["seeds"])

    cases = sorted({k[1] for k in seeds}, key=repr)
    print(f"\npooled: {len(seeds)} (ansatz, case, v, lambda, geom) entries, "
          f"cases {cases}")
    if len(cases) < 2:
        print("  ! only one case present -- pass both cases of one block, "
              "or the case axis below is vacuous")

    print()
    for name, keep in AXES:
        if name == "case" and len(cases) < 2:
            print(f"  across {name:9s}: SKIPPED (single case in input)")
            continue
        groups: dict[tuple, set] = defaultdict(set)
        for k, s in seeds.items():
            groups[keep(k)].add(s)
        sizes = {len(g) for g in groups.values()}
        shared = sum(1 for g in groups.values() if len(g) == 1)
        total = len(groups)
        verdict = ("PAIRED (one seed list per group)" if shared == total
                   else f"NOT paired ({total - shared}/{total} groups differ)")
        print(f"  across {name:9s}: {shared}/{total} groups share one seed list  "
              f"-> {verdict}")
        if sizes and max(sizes) > 1:
            print(f"      distinct seed lists per group: {sorted(sizes)}")

    lens = {len(s) for s in seeds.values()}
    print(f"\n  seed-list lengths present: {sorted(lens)}")
    print("\nWrite the three verdicts into EXPERIMENT_PROTOCOL_T0.md Sec. 3. "
          "Wherever a verdict is PAIRED, that axis must never be described as "
          "independent, and conditions sharing a seed family must not be "
          "resampled as i.i.d. observations.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1:]))
