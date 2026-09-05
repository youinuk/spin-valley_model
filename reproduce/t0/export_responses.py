#!/usr/bin/env python3
"""Export the per-condition responses at n_real = 100 to a shippable CSV.

    python reproduce/t0/export_responses.py \
        ../repro-runs/t0c-step3b data/t0c/responses_n100.csv

Why this exists
---------------
Figure 3 shows where individual conditions land in the (dP_v, dchi_phi) plane
under two ans\"atze. Those per-condition values live in the raw pickles, which
ship in the Zenodo raw-data layer rather than in this source tree, so a figure
drawn straight from them would not regenerate from the public repository. That
would break the provenance chain the release gates on.

This lifts exactly the two response values per (ansatz, block, condition) out of
the raw layer into one small CSV inside `data/t0c/`, so Fig. 3 regenerates from
the shipped tree like every other figure. It adds no new quantity: `dP_v` and
`dchi_phi` are read verbatim from the raw records, not recomputed.

The file is a derived artifact of the corrected raw layer. It is regenerable by
this command from the Zenodo raw data, so `RELEASE_OPS.md` Sec. 1 is satisfied
in the ordinary way and needs no exception.
"""
import collections
import csv
import pathlib
import pickle
import sys

# The frozen producer closure. A CSV built by a different producer is not the
# corrected layer, however many files and rows it has.
EXPECTED_ATLAS = "78daa7a9eabcac25"
EXPECTED_KERNEL = "a250a4343adb77af"
EXPECTED_SEEDS = {41: 1, 100041: 2, 200041: 3, 300041: 4, 400041: 5}
EXPECTED_CASES = {"case_i_center", "case_ii_edge"}
# The same corrected producer can be driven under a different convention, so
# the SHA pair alone does not identify the shipped layer.
EXPECTED_META = {
    "ez_convention": "total-local",
    "profile_norm": "prefactor",
    "seed_scheme": "cross-ansatz",
    "archive_version": "phase5_atlas_v27",
}


def main(raw_dir, out_path):
    raw = pathlib.Path(raw_dir)
    files = sorted(raw.glob("*__n100.pkl"))
    if len(files) != 10:
        print(f"BLOCKED: expected 10 n=100 pickles in {raw}, found {len(files)}",
              file=sys.stderr)
        return 2

    rows = []
    seen_sha = set()
    for f in files:
        d = pickle.load(open(f, "rb"))
        cfg = d["config"]
        seen_sha.add((cfg["atlas_script_sha256_16"], cfg["kernel_script_sha256_16"]))
        if d.get("n_real") != 100:
            print(f"BLOCKED: {f.name} carries n_real={d.get('n_real')}",
                  file=sys.stderr)
            return 2
        for k, want in EXPECTED_META.items():
            got = cfg.get(k)
            if got != want:
                print(f"BLOCKED: {f.name} has {k}={got!r}, expected {want!r}",
                      file=sys.stderr)
                return 2
        block = EXPECTED_SEEDS.get(cfg["base_seed"])
        if block is None:
            print(f"BLOCKED: unmapped base_seed {cfg['base_seed']} in {f.name}",
                  file=sys.stderr)
            return 2
        for key, val in d["data"].items():
            ansatz, case, v, lam, geom = key
            rows.append({
                "ansatz": ansatz, "block": block, "case": case,
                "v_m_per_s": v, "lambda_ueV": lam, "geometry": geom,
                "dP_v": repr(float(val["dP_v"])),
                "dchi_phi": repr(float(val["dchi_phi"])),
            })

    if len(seen_sha) != 1:
        print(f"BLOCKED: the ten pickles carry {len(seen_sha)} distinct producer "
              f"SHA pairs; they are not one production layer", file=sys.stderr)
        return 2

    # Three guards the earlier version lacked. Each closes a way the export
    # could pass while describing something other than the corrected layer.

    # 1. One SHA pair is not enough: ten files can agree on the WRONG producer.
    atlas, kernel = next(iter(seen_sha))
    if (atlas, kernel) != (EXPECTED_ATLAS, EXPECTED_KERNEL):
        print(f"BLOCKED: producer SHAs are {atlas} / {kernel}; the corrected "
              f"layer is {EXPECTED_ATLAS} / {EXPECTED_KERNEL}", file=sys.stderr)
        return 2

    # 2. Ten files is not five blocks by two cases: one duplicate and one
    #    missing also gives ten.
    topo = collections.Counter((r["block"], r["case"]) for r in rows)
    want = {(b, c) for b in EXPECTED_SEEDS.values() for c in EXPECTED_CASES}
    if set(topo) != want:
        missing = sorted(want - set(topo))
        extra = sorted(set(topo) - want)
        print(f"BLOCKED: block x case coverage is wrong. missing={missing} "
              f"unexpected={extra}", file=sys.stderr)
        return 2

    # 3. And 2160 rows is not 2160 distinct conditions.
    keys = collections.Counter(
        (r["ansatz"], r["block"], r["case"], r["v_m_per_s"], r["lambda_ueV"],
         r["geometry"]) for r in rows)
    dupes = [k for k, c in keys.items() if c > 1]
    if dupes:
        print(f"BLOCKED: {len(dupes)} duplicate primary key(s), first "
              f"{dupes[0]}", file=sys.stderr)
        return 2
    if len(keys) != 2160:
        print(f"BLOCKED: {len(keys)} distinct keys, expected 2160",
              file=sys.stderr)
        return 2

    rows.sort(key=lambda r: (r["block"], r["case"], r["ansatz"],
                             r["v_m_per_s"], r["lambda_ueV"], r["geometry"]))
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # lineterminator="\n": csv.writer defaults to CRLF, and this repository
    # enforces LF. A CRLF file changes its own sha256 on disk while
    # .gitattributes normalizes the commit, which is the one way a manifest and
    # a tree can disagree with nothing to show for it.
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    print(f"written: {out}")
    print(f"  rows          : {len(rows)}  "
          f"(5 blocks x 4 ansaetze x 108 conditions = 2160)")
    print(f"  producer atlas : {atlas}")
    print(f"  producer kernel: {kernel}")
    print(f"  topology       : {len(EXPECTED_SEEDS)} blocks x "
          f"{len(EXPECTED_CASES)} cases, {len(keys)} distinct keys")
    print("  metadata       : " + ", ".join(f"{k}={v}"
                                            for k, v in EXPECTED_META.items()))
    print("  Values are copied verbatim from the raw records; nothing is "
          "recomputed here.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
