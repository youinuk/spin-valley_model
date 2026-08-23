#!/usr/bin/env python3
"""T0 STEP 4/5 analysis — matched attribution comparison and candidate screening.

Implements EXPERIMENT_PROTOCOL_T0.md Sec. 5 (STEP 4/5) exactly as pre-registered
on 2026-08-10. Nothing here may be changed after a science result has been seen;
if it must be, record a dated amendment in Sec. 8 and say so in the manuscript.

Subcommands, in the order the protocol requires:

  selftest      Reproduce the archived r31 headline from the STEP 1a pickles.
                Run 2026-08-10: sign convention and trade naming RESOLVED and
                frozen; the Spearman-rho quantity is still unresolved and is
                needed only by `legacy`.

  discovery     Blocks 1-3 only. Writes C_a^disc, U^disc and the block-count
                distribution to discovery.json. Blocks 4-5 are not opened.

  attribution   Blocks 1-5. D_ansatz / D_seed, the 6 and 10 pairwise values,
                the inactive diagnostics, the union-conditioned sensitivity,
                the per-channel pairwise RMS dispersions, and the nested-n
                convergence trajectory.

  holdout       Blocks 4-5, requires discovery.json. Evaluates every condition
                of U^disc under ALL FOUR ansaetze: within-ansatz replication
                and cross-ansatz transport.

  legacy        r31-comparability block: mean Spearman rho, mean classification
                agreement, Cohen's kappa, category counts, two-channel-restricted
                agreement, and the historical material-change descriptor.

Run `discovery` before `attribution` / `holdout`, per Sec. 5 execution order.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Frozen constants (AGENTS.md Sec. 1.1, EXPERIMENT_PROTOCOL_T0.md Sec. 3-5)
# --------------------------------------------------------------------------

FLOOR_P = 1e-4          # |dP_v|   effect-size floor -- operational criterion
FLOOR_CHI = 1e-3        # |dchi_phi| effect-size floor -- operational criterion

ANSATZE = ("A", "A_pocket", "B_z", "B_x")

BLOCK_OF_SEED = {41: 1, 100041: 2, 200041: 3, 300041: 4, 400041: 5}
DISCOVERY_BLOCKS = (1, 2, 3)
HOLDOUT_BLOCKS = (4, 5)

PREFIXES = (5, 10, 20, 30)
PRIMARY_N = 30

# r31 headline, "under revalidation" (AGENTS.md Sec. 1.3). Used only by
# `selftest` and by the historical material-change descriptor.
R31_MEAN_RHO = 0.8373806371
R31_MEAN_AGREEMENT = 0.3595679012
R31_CATEGORY_COUNTS = {
    "below_threshold": 129, "P_only_improve": 106, "P_only_worsen": 82,
    "spin_trade": 40, "both_worsen": 30, "robust": 30, "valley_trade": 15,
}
R31_PAIRWISE_RHO = {
    ("A", "A_pocket"): 0.785, ("A", "B_z"): 0.913, ("A", "B_x"): 0.789,
    ("A_pocket", "B_z"): 0.881, ("A_pocket", "B_x"): 0.855,
    ("B_z", "B_x"): 0.801,
}
MATERIAL_CHANGE_BAND = {"rho": 0.02, "agreement_pp": 3.0}

# ---- Conventions RESOLVED by `selftest` on 2026-08-10 -- do not change ----
#
# SIGN: VERIFIED. Reproduces the archived r31 category counts exactly
# (129 / 106 / 82 / 30 / 30 and the 40 / 15 trade split) and the archived mean
# classification agreement to all ten recorded digits. The opposite convention
# swaps P_only_improve and P_only_worsen (82 <-> 106), so the counts settle it.
IMPROVE_IS_NEGATIVE = True

# TRADE NAMING: VERIFIED, by two independent routes that agree.
#   counts   -- trade_Pimp_chiwor = 40 = archived spin_trade,
#               trade_Pwor_chiimp = 15 = archived valley_trade.
#   physics  -- P_v improving with chi_phi worsening means valley excitation is
#               bought at the cost of spin phase coherence, i.e. spin is the
#               channel traded away. The semantic reading matches the counts.
TRADE_NAMES = {
    "trade_Pimp_chiwor": "spin_trade",
    "trade_Pwor_chiimp": "valley_trade",
}

# The two chi-only categories are EMPTY in r31 (0 and 0), and the seven
# non-empty categories sum to exactly 432. The nine-category scheme is correct;
# two of its cells are simply unpopulated at this grid.

# RHO QUANTITY: RESOLVED 2026-08-10 from
# reproduce/phase5_atlas_merge_validate.py, metric 4:
#
#     r1.append(np.hypot(data[k1]["dP_v"], data[k1]["dchi_phi"]))
#
# i.e. the UNNORMALIZED two-channel magnitude -- no division by the floors.
# Verified against the shipped archive: recomputing Spearman on this quantity
# over the 108 conditions of
# figures/phase5/phase5_atlas_summary_validate__ez-total-local__norm-prefactor.csv
# returns 0.785380 / 0.913125 / 0.788762 / 0.881182 / 0.854897 / 0.800937,
# mean 0.837381, matching every archived pairwise value to all six recorded
# digits. The floor-normalized variant (`norm2`) returns 0.768845 and is NOT
# the released definition.
#
# CAUTION for STEP 7: rho and the classification agreement weight the two
# channels differently. In floor units the median |dP_v| is 2.8 floors and the
# median |dchi_phi| is 0.13 floors, so an unnormalized magnitude is dominated
# by the valley channel (rank correlation with |dP_v| 0.95, with |dchi_phi|
# 0.87), whereas the classifier treats the two channels symmetrically about
# their own floors. Any sentence contrasting "high rank agreement" with "low
# classification agreement" is therefore partly comparing two different channel
# weightings, not purely ranking versus classification.
RHO_QUANTITY = "hypot_raw"

RHO_CANDIDATES = ("hypot_raw", "norm2", "dP_v", "dchi_phi", "mean_of_channels",
                  "abs_dP_v", "abs_dchi_phi", "sum_abs", "max_abs")

NINE_CATEGORIES = (
    "below_threshold",
    "P_only_improve", "P_only_worsen",
    "chi_only_improve", "chi_only_worsen",
    "robust", "both_worsen",
    "trade_Pimp_chiwor", "trade_Pwor_chiimp",
)
TWO_CHANNEL = ("robust", "both_worsen", "trade_Pimp_chiwor", "trade_Pwor_chiimp")


# --------------------------------------------------------------------------
# Loading and nested-prefix reconstruction
# --------------------------------------------------------------------------

def chi(phases) -> float:
    """Circular phase variance, byte-for-byte the r31 aggregator."""
    vals = np.asarray(phases, dtype=float)
    R = float(abs(np.mean(np.exp(1j * vals))))
    return float(-2 * np.log(max(R, 1e-12)))


def reconstruct(raw: dict, n: int) -> tuple[float, float]:
    """(dP_v, dchi_phi) at nested prefix n, per protocol Sec. 4.

    dP_v is M2 - M1V and dchi_phi is M2 - M1. M1V is deterministic and stored
    once, but the r31 aggregator appends it n_real times and means over that
    list, so the baseline is mean([x]*n), not x -- they differ by up to one ULP
    for ~13 % of values at n=5 and ~75 % at n=30.
    """
    m2, m1, m1v = raw["M2"], raw["M1"], raw["M1V"]
    m2_mean = float(np.mean(np.asarray(m2["P_v_dia"][:n], dtype=float)))
    m1v_mean = float(np.mean(np.full(n, float(m1v["P_v_dia"]), dtype=float)))
    return m2_mean - m1v_mean, chi(m2["phase"][:n]) - chi(m1["phase"][:n])


def load_runs(paths, prefixes=PREFIXES, blocks=None, require_raw=True):
    """-> (R, meta) with R[n][(ansatz, cond)][block] = (dP_v, dchi_phi).

    `cond` is (case, v, lambda_0, geom_label): the atlas key WITHOUT the ansatz.
    """
    R = {n: defaultdict(dict) for n in prefixes}
    meta = {"files": [], "blocks": set(), "schemes": set(), "n_real": set()}

    for path in paths:
        path = Path(path)
        if not path.exists():
            hint = ""
            if any(ch in str(path) for ch in "*?["):
                hint = ("\n  This is an unexpanded shell glob: the pattern matched "
                        "nothing, so bash\n  passed it through literally. The likely "
                        "cause is a wrong working directory --\n  repro-runs/ sits "
                        "beside repo/, not inside it. Use absolute paths.")
            raise SystemExit(f"no such file: {path}{hint}")
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        cfg = d.get("config", {})
        base_seed = cfg.get("base_seed")
        if base_seed not in BLOCK_OF_SEED:
            raise SystemExit(f"{path}: base_seed {base_seed!r} is not one of the "
                             f"five pre-registered blocks {sorted(BLOCK_OF_SEED)}")
        b = BLOCK_OF_SEED[base_seed]
        if blocks is not None and b not in blocks:
            continue
        meta["files"].append(str(path))
        meta["blocks"].add(b)
        meta["schemes"].add(cfg.get("seed_scheme"))
        meta["n_real"].add(d.get("n_real"))

        for key, val in d["data"].items():
            ansatz, case, v, lam, geom = key
            cond = (case, v, lam, geom)
            if require_raw:
                if "raw" not in val:
                    raise SystemExit(f"{path}: condition {key} has no 'raw' block")
                for n in prefixes:
                    if n > len(val["raw"]["M2"]["phase"]):
                        continue
                    R[n][(ansatz, cond)][b] = reconstruct(val["raw"], n)
            else:
                R[max(prefixes)][(ansatz, cond)][b] = (
                    float(val["dP_v"]), float(val["dchi_phi"]))

    meta["blocks"] = sorted(meta["blocks"])
    meta["schemes"] = sorted(x for x in meta["schemes"] if x is not None)
    meta["n_real"] = sorted(x for x in meta["n_real"] if x is not None)
    meta["conditions"] = len({c for (_, c) in R[max(prefixes)]})
    meta["ansatz_conditions"] = len(R[max(prefixes)])
    return R, meta


# --------------------------------------------------------------------------
# Classification (nine categories)
# --------------------------------------------------------------------------

def classify(dP: float, dchi: float) -> str:
    p_active = abs(dP) >= FLOOR_P
    c_active = abs(dchi) >= FLOOR_CHI
    if not p_active and not c_active:
        return "below_threshold"
    sgn = (lambda x: x < 0) if IMPROVE_IS_NEGATIVE else (lambda x: x > 0)
    p_imp, c_imp = sgn(dP), sgn(dchi)
    if p_active and not c_active:
        return "P_only_improve" if p_imp else "P_only_worsen"
    if c_active and not p_active:
        return "chi_only_improve" if c_imp else "chi_only_worsen"
    if p_imp and c_imp:
        return "robust"
    if not p_imp and not c_imp:
        return "both_worsen"
    return "trade_Pimp_chiwor" if p_imp else "trade_Pwor_chiimp"


def label_table(R_n):
    """-> L[(ansatz, cond)][block] = label"""
    return {k: {b: classify(*v) for b, v in per_b.items()} for k, per_b in R_n.items()}


# --------------------------------------------------------------------------
# Primary estimands
# --------------------------------------------------------------------------

def _conditions(L):
    return sorted({c for (_, c) in L}, key=repr)


def pairwise_label_disagreement(L, blocks, ansatze=ANSATZE):
    """d_{aa'} over 6 ansatz pairs and d_{bb'} over the block pairs.

    d_{aa'} = < 1[L_cab != L_ca'b] >_{c,b};  d_{bb'} = < 1[L_cab != L_cab'] >_{c,a}
    Each is a plain mean over the (condition, block) or (condition, ansatz)
    cells in which both members of the pair exist.
    """
    conds = _conditions(L)
    d_a, d_b = {}, {}
    for a, a2 in itertools.combinations(ansatze, 2):
        hits = [L[(a, c)][b] != L[(a2, c)][b]
                for c in conds for b in blocks
                if (a, c) in L and (a2, c) in L
                and b in L[(a, c)] and b in L[(a2, c)]]
        d_a[(a, a2)] = float(np.mean(hits)) if hits else float("nan")
    for b, b2 in itertools.combinations(blocks, 2):
        hits = [L[(a, c)][b] != L[(a, c)][b2]
                for c in conds for a in ansatze
                if (a, c) in L and b in L[(a, c)] and b2 in L[(a, c)]]
        d_b[(b, b2)] = float(np.mean(hits)) if hits else float("nan")
    return d_a, d_b


def pairwise_shared_inactive(L, blocks, ansatze=ANSATZE):
    """B_ansatz, B_seed: how often an agreeing pair agrees only on below_threshold."""
    conds = _conditions(L)
    ba = [L[(a, c)][b] == L[(a2, c)][b] == "below_threshold"
          for a, a2 in itertools.combinations(ansatze, 2)
          for c in conds for b in blocks
          if (a, c) in L and (a2, c) in L and b in L[(a, c)] and b in L[(a2, c)]]
    bs = [L[(a, c)][b] == L[(a, c)][b2] == "below_threshold"
          for b, b2 in itertools.combinations(blocks, 2)
          for c in conds for a in ansatze
          if (a, c) in L and b in L[(a, c)] and b2 in L[(a, c)]]
    return float(np.mean(ba)), float(np.mean(bs))


def union_conditioned_disagreement(L, blocks, ansatze=ANSATZE):
    """Sensitivity only: restrict to pairs where AT LEAST ONE member is two-channel.

    Intersection conditioning is deliberately not implemented -- it conditions
    away the transitions the paper measures (protocol Sec. 5).
    """
    conds = _conditions(L)
    ua = [L[(a, c)][b] != L[(a2, c)][b]
          for a, a2 in itertools.combinations(ansatze, 2)
          for c in conds for b in blocks
          if (a, c) in L and (a2, c) in L and b in L[(a, c)] and b in L[(a2, c)]
          and (L[(a, c)][b] in TWO_CHANNEL or L[(a2, c)][b] in TWO_CHANNEL)]
    us = [L[(a, c)][b] != L[(a, c)][b2]
          for b, b2 in itertools.combinations(blocks, 2)
          for c in conds for a in ansatze
          if (a, c) in L and b in L[(a, c)] and b2 in L[(a, c)]
          and (L[(a, c)][b] in TWO_CHANNEL or L[(a, c)][b2] in TWO_CHANNEL)]
    return ({"D_ansatz": float(np.mean(ua)), "n_pairs": len(ua)},
            {"D_seed": float(np.mean(us)), "n_pairs": len(us)})


def pairwise_rms(R_n, blocks, channel: int, ansatze=ANSATZE):
    """S_ansatz, S_seed for one channel (0 = dP_v, 1 = dchi_phi).

    Pairwise, not centroid: sum_{i<j}(x_i-x_j)^2 = m*sum_i(x_i-xbar)^2 for any
    group size m, so the mean pairwise squared difference is exactly twice the
    unbiased variance and the 4-vs-5 divisor asymmetry cancels. S is therefore
    sqrt(2) times a standard deviation and must not be compared to the floors.
    """
    floor = FLOOR_P if channel == 0 else FLOOR_CHI
    conds = sorted({c for (_, c) in R_n}, key=repr)

    sq_a = [((R_n[(a, c)][b][channel] - R_n[(a2, c)][b][channel]) / floor) ** 2
            for a, a2 in itertools.combinations(ansatze, 2)
            for c in conds for b in blocks
            if (a, c) in R_n and (a2, c) in R_n
            and b in R_n[(a, c)] and b in R_n[(a2, c)]]
    sq_b = [((R_n[(a, c)][b][channel] - R_n[(a, c)][b2][channel]) / floor) ** 2
            for b, b2 in itertools.combinations(blocks, 2)
            for c in conds for a in ansatze
            if (a, c) in R_n and b in R_n[(a, c)] and b2 in R_n[(a, c)]]
    return float(np.sqrt(np.mean(sq_a))), float(np.sqrt(np.mean(sq_b)))


def p_all_inactive(L, blocks, ansatze=ANSATZE):
    conds = _conditions(L)
    n_all = 0
    for c in conds:
        cells = [L[(a, c)][b] for a in ansatze for b in blocks
                 if (a, c) in L and b in L[(a, c)]]
        if cells and all(x == "below_threshold" for x in cells):
            n_all += 1
    return n_all / len(conds), n_all, len(conds)


# --------------------------------------------------------------------------
# r31-comparability quantities
# --------------------------------------------------------------------------

def rho_series(R_n, ansatz, blocks, quantity):
    """The per-condition scalar the pairwise Spearman rho is computed on.

    `blocks` is averaged over. r31 computed this on a single dataset, so
    `legacy` passes one block at a time and reports the per-block values; this
    signature keeps `selftest` (single block) working unchanged.
    """
    conds = sorted({c for (_, c) in R_n}, key=repr)
    out = []
    for c in conds:
        vals = [R_n[(ansatz, c)][b] for b in blocks if b in R_n.get((ansatz, c), {})]
        if not vals:
            out.append(np.nan)
            continue
        dP = float(np.mean([v[0] for v in vals]))
        dc = float(np.mean([v[1] for v in vals]))
        p, q = dP / FLOOR_P, dc / FLOOR_CHI
        out.append({"hypot_raw": float(np.hypot(dP, dc)),
                    "norm2": float(np.hypot(p, q)),
                    "dP_v": dP,
                    "dchi_phi": dc,
                    "mean_of_channels": 0.5 * (p + q),
                    "abs_dP_v": abs(dP),
                    "abs_dchi_phi": abs(dc),
                    "sum_abs": abs(p) + abs(q),
                    "max_abs": max(abs(p), abs(q))}[quantity])
    return np.asarray(out, dtype=float)


def spearman(x, y) -> float:
    from scipy.stats import spearmanr
    m = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[m], y[m]).statistic)


def pairwise_rho(R_n, blocks, quantity, ansatze=ANSATZE, per_case=False):
    """One Spearman rho per ansatz pair.

    per_case=True computes rho within each case and averages the two, which is
    a different aggregation of the same quantity; it is offered because the r31
    definition is not recoverable from the governance documents and this is one
    of the plausible variants.
    """
    if not per_case:
        return {(a, a2): spearman(rho_series(R_n, a, blocks, quantity),
                                  rho_series(R_n, a2, blocks, quantity))
                for a, a2 in itertools.combinations(ansatze, 2)}
    cases = sorted({c[0] for (_, c) in R_n}, key=repr)
    out = {}
    for a, a2 in itertools.combinations(ansatze, 2):
        vals = []
        for case in cases:
            sub = {k: v for k, v in R_n.items() if k[1][0] == case}
            vals.append(spearman(rho_series(sub, a, blocks, quantity),
                                 rho_series(sub, a2, blocks, quantity)))
        out[(a, a2)] = float(np.mean(vals))
    return out


def cohen_kappa(l1, l2) -> float:
    cats = sorted(set(l1) | set(l2))
    idx = {c: i for i, c in enumerate(cats)}
    M = np.zeros((len(cats), len(cats)))
    for a, b in zip(l1, l2):
        M[idx[a], idx[b]] += 1
    M /= M.sum()
    po = float(np.trace(M))
    pe = float(M.sum(0) @ M.sum(1))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def classification_agreement(L, blocks, ansatze=ANSATZE):
    """Mean exact agreement of the nine-category label over the 6 ansatz pairs."""
    d_a, _ = pairwise_label_disagreement(L, blocks, ansatze)
    per_pair = {k: 1.0 - v for k, v in d_a.items()}
    return float(np.mean(list(per_pair.values()))), per_pair


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------

def _pk(d):
    return {"/".join(map(str, k)): v for k, v in d.items()}


def dump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))
    print(f"\nwritten: {path}")


def banner(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------

def cmd_selftest(args):
    """Resolve the two unrecoverable conventions against the archived r31 values."""
    banner("SELFTEST -- resolving conventions against the archived r31 headline")
    R, meta = load_runs(args.pickles, prefixes=(args.n,), require_raw=args.raw)
    blocks = meta["blocks"]
    print(f"files {len(meta['files'])}  blocks {blocks}  scheme {meta['schemes']}  "
          f"conditions {meta['conditions']}  ansatz-conditions {meta['ansatz_conditions']}")
    if meta["ansatz_conditions"] != 432:
        print(f"  ! expected 432 ansatz-conditions (108 conditions x 4), "
              f"got {meta['ansatz_conditions']}")

    for flip in (False, True):
        global IMPROVE_IS_NEGATIVE
        IMPROVE_IS_NEGATIVE = not flip
        L = label_table(R[args.n])
        counts = Counter(lab for per_b in L.values() for lab in per_b.values())
        agree, per_pair = classification_agreement(L, blocks)
        print(f"\n-- IMPROVE_IS_NEGATIVE = {IMPROVE_IS_NEGATIVE}")
        for k in NINE_CATEGORIES:
            arch = R31_CATEGORY_COUNTS.get(k)
            mark = ""
            if k in R31_CATEGORY_COUNTS:
                mark = "  <- matches" if counts[k] == arch else f"  <- archived {arch}"
            print(f"   {k:22s} {counts[k]:5d}{mark}")
        print(f"   trade split: Pimp/chiwor={counts['trade_Pimp_chiwor']}, "
              f"Pwor/chiimp={counts['trade_Pwor_chiimp']}  "
              f"(archived spin_trade=40, valley_trade=15)")
        print(f"   mean classification agreement {agree:.10f} "
              f"(archived {R31_MEAN_AGREEMENT:.10f})")

    print("\n-- Spearman rho candidates x aggregation "
          "(archived pairwise values in brackets)")
    best = None
    for per_case in (False, True):
        tag = "per-case mean" if per_case else "pooled"
        for q in RHO_CANDIDATES:
            rho = pairwise_rho(R[args.n], blocks, q, per_case=per_case)
            mean_rho = float(np.mean(list(rho.values())))
            devs = [abs(rho[k] - R31_PAIRWISE_RHO[k])
                    for k in R31_PAIRWISE_RHO if k in rho]
            worst = max(devs)
            if best is None or worst < best[0]:
                best = (worst, q, tag, rho, mean_rho)
            print(f"   {q:15s} [{tag:13s}] mean {mean_rho: .6f} "
                  f"(archived {R31_MEAN_RHO:.6f})  worst pair {worst:.4f}"
                  f"{'   <- REPRODUCES' if worst < 1e-3 else ''}")
    worst, q, tag, rho, mean_rho = best
    print(f"\n   closest: {q} [{tag}], worst pair deviation {worst:.4f}")
    for k in sorted(rho, key=repr):
        print(f"       {k[0]:10s}-{k[1]:10s} {rho[k]: .4f}  "
              f"[{R31_PAIRWISE_RHO.get(k, float('nan')):.3f}]")
    if worst >= 1e-3:
        print("\n   No candidate reproduces the archived rho. The exact "
              "agreement match above proves the data and classifier are right, "
              "so the rho DEFINITION is what is missing. Needed only by "
              "`legacy`; does not block STEP 3.")

    print("\nIMPROVE_IS_NEGATIVE and TRADE_NAMES are frozen from the "
          "2026-08-10 run of this command. Only RHO_QUANTITY remains open.")


def cmd_discovery(args):
    n = getattr(args, "n", PRIMARY_N)
    banner(f"DISCOVERY (n = {n}) -- blocks 1-3 only. Blocks 4-5 are NOT opened.")
    if n != PRIMARY_N:
        print(f"  NOTE: n = {n} is the registered sample-size-stability companion")
        print(f"        (protocol Sec. 8 entry 20). The pre-registered Table III is")
        print(f"        n = {PRIMARY_N} and is not replaced by this run.")
    R, meta = load_runs(args.pickles, prefixes=(n,), blocks=DISCOVERY_BLOCKS)
    if set(meta["blocks"]) != set(DISCOVERY_BLOCKS):
        raise SystemExit(f"need blocks {DISCOVERY_BLOCKS}, found {meta['blocks']}")
    if meta["n_real"] and n > max(meta["n_real"]):
        raise SystemExit(f"n = {n} exceeds the stored n_real {meta['n_real']}")
    L = label_table(R[n])

    robust_count = {k: sum(1 for b in DISCOVERY_BLOCKS
                           if per_b.get(b) == "robust")
                    for k, per_b in L.items()}
    dist = Counter(robust_count.values())
    print("\nblock-count distribution over all (ansatz, condition) pairs "
          "-- reported unconditionally:")
    for k in range(len(DISCOVERY_BLOCKS) + 1):
        print(f"   robust in {k}/3 discovery blocks : {dist.get(k, 0)}")

    C = {a: sorted((c for (aa, c) in L if aa == a and robust_count[(aa, c)] == 3),
                   key=repr) for a in ANSATZE}
    U = sorted({c for a in ANSATZE for c in C[a]}, key=repr)
    print("\nC_a^disc (robust in all three discovery blocks):")
    for a in ANSATZE:
        print(f"   {a:10s} {len(C[a])}")
    print(f"   U^disc  {len(U)}")
    if not U:
        print("   U^disc is empty. This is a legitimate result and is reported "
              "as one; the block-count distribution above carries the content.")

    overlap = Counter(sum(1 for a in ANSATZE if c in C[a]) for c in U)
    print("\ncross-ansatz transport on DISCOVERY blocks "
          "(an outcome, never a selection criterion):")
    for k in range(1, len(ANSATZE) + 1):
        print(f"   selected by {k}/4 ansaetze : {overlap.get(k, 0)}")

    dump({"n": n, "blocks": list(DISCOVERY_BLOCKS), "meta": meta,
          "block_count_distribution": {str(k): v for k, v in sorted(dist.items())},
          "C_disc": {a: [list(c) for c in C[a]] for a in ANSATZE},
          "U_disc": [list(c) for c in U],
          "discovery_overlap": {str(k): v for k, v in sorted(overlap.items())}},
         args.out)


def cmd_attribution(args):
    """ATTRIBUTION -- blocks 1-5, matched attribution comparison"""
    prefixes = tuple(getattr(args, "prefixes", None) or PREFIXES)
    banner("ATTRIBUTION -- blocks 1-5, matched attribution comparison")
    R, meta = load_runs(args.pickles, prefixes=prefixes)
    blocks = meta["blocks"]
    print(f"files {len(meta['files'])}  blocks {blocks}  scheme {meta['schemes']}  "
          f"conditions {meta['conditions']}")
    if meta["schemes"] != ["cross-ansatz"]:
        print(f"  ! seed_scheme is {meta['schemes']}, expected ['cross-ansatz']")

    report = {"meta": meta, "primary_n": PRIMARY_N, "by_n": {}}

    for n in prefixes:
        L = label_table(R[n])
        d_a, d_b = pairwise_label_disagreement(L, blocks)
        D_a = float(np.mean(list(d_a.values())))
        D_b = float(np.mean(list(d_b.values())))
        B_a, B_b = pairwise_shared_inactive(L, blocks)
        p_all, n_all, n_cond = p_all_inactive(L, blocks)
        ua, us = union_conditioned_disagreement(L, blocks)
        S = {q: dict(zip(("S_ansatz", "S_seed"), pairwise_rms(R[n], blocks, i)))
             for i, q in ((0, "dP_v"), (1, "dchi_phi"))}

        entry = {
            "D_ansatz": D_a, "D_seed": D_b,
            "Delta_D": D_a - D_b,
            "R_D": (D_a / D_b) if D_b else None,
            "pairwise_ansatz": _pk(d_a), "pairwise_seed": _pk(d_b),
            "complete_separation": bool(min(d_a.values()) > max(d_b.values())),
            "B_ansatz": B_a, "B_seed": B_b,
            "shared_inactive_share_of_agreement": {
                "ansatz": B_a / (1 - D_a) if D_a < 1 else None,
                "seed": B_b / (1 - D_b) if D_b < 1 else None},
            "p_all": p_all, "n_all_inactive": n_all, "n_conditions": n_cond,
            "union_conditioned_sensitivity": {"ansatz": ua, "seed": us},
            "continuous": S,
        }
        report["by_n"][str(n)] = entry

        if n == PRIMARY_N:
            banner(f"PRIMARY (n = {n})")
            print(f"  D_ansatz {D_a:.6f}      D_seed {D_b:.6f}")
            print(f"  Delta_D  {D_a - D_b:+.6f}     R_D    "
                  f"{(D_a / D_b) if D_b else float('nan'):.4f}   (companions only)")
            print("\n  6 ansatz-pair values:")
            for k in sorted(d_a, key=repr):
                print(f"     {k[0]:10s}-{k[1]:10s} {d_a[k]:.6f}")
            print("  10 seed-pair values:")
            for k in sorted(d_b):
                print(f"     block {k[0]}-{k[1]}          {d_b[k]:.6f}")
            print(f"\n  min ansatz-pair {min(d_a.values()):.6f}   "
                  f"max seed-pair {max(d_b.values()):.6f}   "
                  f"complete separation: {entry['complete_separation']}")
            print("    (separation is a sufficient condition for uniform "
                  "dominance, not a gate)")
            print(f"\n  p_all {p_all:.4f} ({n_all}/{n_cond} conditions inactive "
                  f"under every ansatz and block)")
            print(f"  B_ansatz {B_a:.6f}  -> {entry['shared_inactive_share_of_agreement']['ansatz']:.4f}"
                  f" of agreeing ansatz pairs are shared-inactive")
            print(f"  B_seed   {B_b:.6f}  -> {entry['shared_inactive_share_of_agreement']['seed']:.4f}"
                  f" of agreeing seed pairs are shared-inactive")
            print(f"\n  union-conditioned sensitivity (NOT a primary claim): "
                  f"D_ansatz {ua['D_ansatz']:.6f} over {ua['n_pairs']} pairs, "
                  f"D_seed {us['D_seed']:.6f} over {us['n_pairs']} pairs")
            print("\n  continuous pairwise RMS (floor-normalized, = sqrt(2) x SD):")
            for q in ("dP_v", "dchi_phi"):
                sa, sb = S[q]["S_ansatz"], S[q]["S_seed"]
                print(f"     {q:9s} S_ansatz {sa:.4f}   S_seed {sb:.4f}   "
                      f"ratio {sa / sb if sb else float('nan'):.4f}")

    banner("CONVERGENCE TRAJECTORY (nested prefixes -- not independent replicates)")
    print(f"  {'n':>4}  {'D_ansatz':>10} {'D_seed':>10} "
          f"{'S_a(dP)':>9} {'S_s(dP)':>9} {'S_a(dchi)':>10} {'S_s(dchi)':>10}")
    for n in prefixes:
        e = report["by_n"][str(n)]
        print(f"  {n:>4}  {e['D_ansatz']:>10.6f} {e['D_seed']:>10.6f} "
              f"{e['continuous']['dP_v']['S_ansatz']:>9.4f} "
              f"{e['continuous']['dP_v']['S_seed']:>9.4f} "
              f"{e['continuous']['dchi_phi']['S_ansatz']:>10.4f} "
              f"{e['continuous']['dchi_phi']['S_seed']:>10.4f}")

    # secondary: per-case breakdown at the primary n
    L = label_table(R[PRIMARY_N])
    per_case = {}
    for case in sorted({c[0] for (_, c) in L}, key=repr):
        Lc = {k: v for k, v in L.items() if k[1][0] == case}
        da, db = pairwise_label_disagreement(Lc, blocks)
        per_case[str(case)] = {"D_ansatz": float(np.mean(list(da.values()))),
                               "D_seed": float(np.mean(list(db.values())))}
    report["per_case_secondary"] = per_case
    banner("PER-CASE SECONDARY BREAKDOWN (n = 30)")
    for case, v in per_case.items():
        print(f"  {case:10s} D_ansatz {v['D_ansatz']:.6f}   D_seed {v['D_seed']:.6f}")

    dump(report, args.out)


def cmd_holdout(args):
    disc = json.loads(Path(args.discovery).read_text())
    n = getattr(args, "n", None) or disc.get("n", PRIMARY_N)
    if disc.get("n") not in (None, n):
        raise SystemExit(f"--n {n} contradicts the discovery file, which was "
                         f"built at n = {disc['n']}. Selection and confirmation "
                         f"must use the same prefix.")
    label = ("HOLDOUT" if n == PRIMARY_N else "EXTENDED CONFIRMATION")
    banner(f"{label} (n = {n}) -- every U^disc condition under ALL FOUR ansaetze")
    if n != PRIMARY_N:
        print("  This is the registered post-hoc sample-size-stability companion.")
        print("  Blocks 4-5 were already seen at n = 30 and the n > 30 aggregate")
        print("  contains those realizations, so this is NOT an independent")
        print("  holdout and must not be described as one (Sec. 8 entry 20).")
    U = [tuple(c) for c in disc["U_disc"]]
    C = {a: {tuple(c) for c in v} for a, v in disc["C_disc"].items()}
    if not U:
        print("U^disc is empty -- nothing to confirm. Reported as a result.")
        dump({"U_disc": [], "rows": []}, args.out)
        return

    R, meta = load_runs(args.pickles, prefixes=(n,), blocks=HOLDOUT_BLOCKS)
    if set(meta["blocks"]) != set(HOLDOUT_BLOCKS):
        raise SystemExit(f"need blocks {HOLDOUT_BLOCKS}, found {meta['blocks']}")
    L = label_table(R[n])

    rows = []
    for c in U:
        row = {"condition": list(c), "discovered_by": [], "holdout": {}}
        for a in ANSATZE:
            per_b = L.get((a, c), {})
            n_rob = sum(1 for b in HOLDOUT_BLOCKS if per_b.get(b) == "robust")
            row["holdout"][a] = {
                "robust_blocks": n_rob,
                "confirmed_2of2": n_rob == len(HOLDOUT_BLOCKS),
                "labels": {str(b): per_b.get(b) for b in HOLDOUT_BLOCKS}}
            if c in C[a]:
                row["discovered_by"].append(a)
        rows.append(row)

    print(f"\n{'condition':46s} {'disc':>5}  " +
          "  ".join(f"{a:>9s}" for a in ANSATZE))
    for row in rows:
        c = "/".join(map(str, row["condition"]))
        cells = []
        for a in ANSATZE:
            h = row["holdout"][a]
            tag = f"{h['robust_blocks']}/2"
            cells.append(f"{tag + ('*' if a in row['discovered_by'] else ''):>9s}")
        print(f"{c:46s} {len(row['discovered_by']):>5d}  " + "  ".join(cells))
    print("  * = this ansatz discovered the condition (within-ansatz replication);"
          "\n  unstarred columns are cross-ansatz transport.")

    repl = [(a, r) for r in rows for a in r["discovered_by"]]
    n_conf = sum(1 for a, r in repl if r["holdout"][a]["confirmed_2of2"])
    print(f"\nwithin-ansatz replication: {n_conf}/{len(repl)} discovered "
          f"(ansatz, condition) pairs robust in both holdout blocks")
    transport = Counter(sum(1 for a in ANSATZE if r["holdout"][a]["confirmed_2of2"])
                        for r in rows)
    print("cross-ansatz transport on holdout blocks:")
    for k in range(len(ANSATZE) + 1):
        print(f"   robust 2/2 under {k}/4 ansaetze : {transport.get(k, 0)}")

    dump({"n": n, "meta": meta, "U_disc": [list(c) for c in U], "rows": rows,
          "within_ansatz_replication": {"confirmed": n_conf, "total": len(repl)},
          "cross_ansatz_transport": {str(k): v for k, v in sorted(transport.items())}},
         args.out)


def cmd_legacy(args):
    banner("LEGACY / r31-COMPARABILITY -- historical material-change descriptor")
    n = getattr(args, "n", PRIMARY_N)
    R, meta = load_runs(args.pickles, prefixes=(n,))
    blocks = meta["blocks"]
    L = label_table(R[n])

    agree, per_pair = classification_agreement(L, blocks)
    # r31 computed rho on one dataset. Each block is a complete 108-condition
    # atlas, so rho is computed WITHIN each block and the per-block values are
    # reported; the mean over blocks is the r31-comparable summary. Averaging
    # the responses across blocks first would be a different quantity.
    rho_by_block = {b: pairwise_rho(R[n], [b], RHO_QUANTITY)
                    for b in blocks}
    block_means = {b: float(np.mean(list(v.values())))
                   for b, v in rho_by_block.items()}
    mean_rho = float(np.mean(list(block_means.values())))
    rho = {k: float(np.mean([rho_by_block[b][k] for b in blocks]))
           for k in rho_by_block[blocks[0]]}
    counts = Counter(lab for per_b in L.values() for lab in per_b.values())

    conds = _conditions(L)
    kap = {}
    for a, a2 in itertools.combinations(ANSATZE, 2):
        l1 = [L[(a, c)][b] for c in conds for b in blocks
              if (a, c) in L and b in L[(a, c)]]
        l2 = [L[(a2, c)][b] for c in conds for b in blocks
              if (a2, c) in L and b in L[(a2, c)]]
        kap[(a, a2)] = cohen_kappa(l1, l2)

    print(f"  mean Spearman rho ({RHO_QUANTITY})  {mean_rho:.10f}")
    print("    per block: " + "  ".join(f"b{b} {block_means[b]:.6f}" for b in blocks))
    print("    per ansatz pair (mean over blocks):")
    for k in sorted(rho, key=repr):
        print(f"       {k[0]:10s}-{k[1]:10s} {rho[k]: .6f}")
    print(f"  mean classification agreement       {agree:.10f}")
    print(f"  Cohen's kappa range                 "
          f"{min(kap.values()):.3f} - {max(kap.values()):.3f}")
    print("\n  category counts:")
    for k in NINE_CATEGORIES:
        print(f"     {k:22s} {counts[k]:5d}")

    d_rho = mean_rho - R31_MEAN_RHO
    d_agr = 100 * (agree - R31_MEAN_AGREEMENT)
    changed = (abs(d_rho) > MATERIAL_CHANGE_BAND["rho"]
               or abs(d_agr) > MATERIAL_CHANGE_BAND["agreement_pp"])
    banner("HISTORICAL MATERIAL-CHANGE DESCRIPTOR")
    print(f"  rho        {mean_rho:.6f} vs r31 {R31_MEAN_RHO:.6f}   "
          f"delta {d_rho:+.6f}  (band +/-{MATERIAL_CHANGE_BAND['rho']})")
    print(f"  agreement  {100*agree:.2f} % vs r31 {100*R31_MEAN_AGREEMENT:.2f} %   "
          f"delta {d_agr:+.2f} pp  (band +/-{MATERIAL_CHANGE_BAND['agreement_pp']})")
    print(f"  -> the seed correction {'materially altered' if changed else 'did not materially alter'}"
          " the historical r31 estimate.")
    print("  This descriptor does NOT choose the headline. The T0 values above "
          "are the successor primary unconditionally.")

    dump({"meta": meta, "rho_quantity": RHO_QUANTITY,
          "mean_rho": mean_rho, "pairwise_rho": _pk(rho),
          "rho_by_block": {str(b): _pk(v) for b, v in rho_by_block.items()},
          "rho_block_means": {str(b): v for b, v in block_means.items()},
          "mean_classification_agreement": agree,
          "pairwise_agreement": _pk(per_pair),
          "kappa": _pk(kap), "category_counts": dict(counts),
          "material_change": {"delta_rho": d_rho, "delta_agreement_pp": d_agr,
                              "materially_changed": changed}},
         args.out)



# --------------------------------------------------------------------------
# POST-HOC diagnostics (registered in protocol Sec. 8 entry 19)
#
# Everything below was specified AFTER the n=30 results were seen. It is
# reported as post-hoc throughout and never as a pre-registered estimand. No
# new threshold is introduced: the three-state channel labels are a
# marginalization of the existing nine-category classifier, and the family
# breakdown is a partition of the existing pairwise RMS.
# --------------------------------------------------------------------------

def channel_state(x: float, floor: float) -> str:
    """One channel's three-state label: the nine-category scheme marginalized."""
    if abs(x) < floor:
        return "inactive"
    improving = (x < 0) if IMPROVE_IS_NEGATIVE else (x > 0)
    return "improve" if improving else "worsen"


def channel_disagreement(R_n, blocks, channel, ansatze=ANSATZE):
    """D_ansatz, D_seed for a single channel's three-state label. POST-HOC."""
    floor = FLOOR_P if channel == 0 else FLOOR_CHI
    conds = sorted({c for (_, c) in R_n}, key=repr)
    S = {k: {b: channel_state(v[channel], floor) for b, v in per_b.items()}
         for k, per_b in R_n.items()}
    da = [S[(a, c)][b] != S[(a2, c)][b]
          for a, a2 in itertools.combinations(ansatze, 2)
          for c in conds for b in blocks
          if (a, c) in S and (a2, c) in S and b in S[(a, c)] and b in S[(a2, c)]]
    ds = [S[(a, c)][b] != S[(a, c)][b2]
          for b, b2 in itertools.combinations(blocks, 2)
          for c in conds for a in ansatze
          if (a, c) in S and b in S[(a, c)] and b2 in S[(a, c)]]
    return float(np.mean(da)), float(np.mean(ds))


def family_rms(R_n, blocks, channel, ansatze=ANSATZE):
    """Pairwise RMS split by (v, lambda) seed family. POST-HOC.

    Within a block the noise is organised into six (v, lambda) families, each
    a distinct 30-realization ensemble shared across geometry, case and ansatz.
    A heavy-tailed pooled RMS can therefore be carried by one family.
    """
    floor = FLOOR_P if channel == 0 else FLOOR_CHI
    fams = sorted({(c[1], c[2]) for (_, c) in R_n})
    out = {}
    for f in fams:
        sub = {k: v for k, v in R_n.items() if (k[1][1], k[1][2]) == f}
        conds = sorted({c for (_, c) in sub}, key=repr)
        qa = [((sub[(a, c)][b][channel] - sub[(a2, c)][b][channel]) / floor) ** 2
              for a, a2 in itertools.combinations(ansatze, 2)
              for c in conds for b in blocks
              if (a, c) in sub and (a2, c) in sub]
        qs = [((sub[(a, c)][b][channel] - sub[(a, c)][b2][channel]) / floor) ** 2
              for b, b2 in itertools.combinations(blocks, 2)
              for c in conds for a in ansatze if (a, c) in sub]
        out[f] = (float(np.sqrt(np.mean(qa))), float(np.sqrt(np.mean(qs))),
                  float(np.sum(qa)))
    return out


def median_companion(R_n, blocks, channel, ansatze=ANSATZE):
    """Median over cells of the per-cell pairwise RMS. POST-HOC."""
    floor = FLOOR_P if channel == 0 else FLOOR_CHI
    conds = sorted({c for (_, c) in R_n}, key=repr)
    per_cell_seed, per_cell_ans = [], []
    for c in conds:
        for a in ansatze:
            v = [R_n[(a, c)][b][channel] / floor for b in blocks
                 if (a, c) in R_n and b in R_n[(a, c)]]
            if len(v) > 1:
                per_cell_seed.append(np.sqrt(np.mean(
                    [(v[i] - v[j]) ** 2 for i, j in itertools.combinations(range(len(v)), 2)])))
        for b in blocks:
            v = [R_n[(a, c)][b][channel] / floor for a in ansatze
                 if (a, c) in R_n and b in R_n[(a, c)]]
            if len(v) > 1:
                per_cell_ans.append(np.sqrt(np.mean(
                    [(v[i] - v[j]) ** 2 for i, j in itertools.combinations(range(len(v)), 2)])))
    return float(np.median(per_cell_ans)), float(np.median(per_cell_seed))


def channel_identity(R_n, blocks, axis, ansatze=ANSATZE):
    """joint, D_P, D_chi, D_both along one axis. POST-HOC.

    The nine-category label is an exact bijection of the (P-state, chi-state)
    pair, so joint disagreement = D_P + D_chi - D_both identically. The residual
    printed by the caller is a consistency check on the implementation.
    """
    conds = sorted({c for (_, c) in R_n}, key=repr)
    if axis == "ansatz":
        pairs = [((a, c), (a2, c), (b, b))
                 for a, a2 in itertools.combinations(ansatze, 2)
                 for c in conds for b in blocks]
    else:
        pairs = [((a, c), (a, c), (b, b2)) for a in ansatze for c in conds
                 for b, b2 in itertools.combinations(blocks, 2)]
    j = p = q = both = 0
    n = 0
    for k1, k2, (b, b2) in pairs:
        if k1 not in R_n or k2 not in R_n:
            continue
        if b not in R_n[k1] or b2 not in R_n[k2]:
            continue
        x, y = R_n[k1][b], R_n[k2][b2]
        dp = channel_state(x[0], FLOOR_P) != channel_state(y[0], FLOOR_P)
        dc = channel_state(x[1], FLOOR_CHI) != channel_state(y[1], FLOOR_CHI)
        n += 1
        j += classify(*x) != classify(*y)
        p += dp
        q += dc
        both += dp and dc
    return j / n, p / n, q / n, both / n


def family_three_state(R_n, blocks, channel, ansatze=ANSATZE):
    """Three-state disagreement split by (v, lambda) family. POST-HOC."""
    floor = FLOOR_P if channel == 0 else FLOOR_CHI
    out = {}
    for f in sorted({(c[1], c[2]) for (_, c) in R_n}):
        sub = {k: v for k, v in R_n.items() if (k[1][1], k[1][2]) == f}
        conds = sorted({c for (_, c) in sub}, key=repr)
        da = [channel_state(sub[(a, c)][b][channel], floor)
              != channel_state(sub[(a2, c)][b][channel], floor)
              for a, a2 in itertools.combinations(ansatze, 2)
              for c in conds for b in blocks if (a, c) in sub and (a2, c) in sub]
        ds = [channel_state(sub[(a, c)][b][channel], floor)
              != channel_state(sub[(a, c)][b2][channel], floor)
              for a in ansatze for b, b2 in itertools.combinations(blocks, 2)
              for c in conds if (a, c) in sub]
        out[f] = (float(np.mean(da)), float(np.mean(ds)), int(sum(da)))
    return out


def cmd_posthoc(args):
    """POST-HOC diagnostics: channel-wise 3-state disagreement, family split."""
    banner("POST-HOC DIAGNOSTICS -- specified after the n=30 results were seen")
    print("Reported as post-hoc throughout. No new threshold is introduced.")
    R, meta = load_runs(args.pickles, prefixes=tuple(args.prefixes))
    blocks = meta["blocks"]
    n = args.n
    report = {"meta": meta, "n": n, "status": "POST-HOC"}

    banner(f"A. Channel-wise three-state disagreement (n = {n})")
    print("   Tests directly whether the label-level equality is carried by one")
    print("   channel. The nine-category label is a product of these two.")
    chan = {}
    for ch, name in ((0, "dP_v"), (1, "dchi_phi")):
        da, ds = channel_disagreement(R[n], blocks, ch)
        chan[name] = {"D_ansatz": da, "D_seed": ds,
                      "Delta": da - ds, "ratio": da / ds if ds else None}
        print(f"   {name:9s} D_ansatz {da:.6f}   D_seed {ds:.6f}   "
              f"Delta {da - ds:+.6f}   ratio {da / ds if ds else float('nan'):.4f}")
    report["A_channel_three_state"] = chan

    banner(f"A2. Channel decomposition identity (n = {n})")
    print("   The nine-category label is an exact bijection of the two channel")
    print("   states, so joint = D_P + D_chi - D_both. Reported because the joint")
    print("   near-equality is a CANCELLATION of opposing channel contrasts, not")
    print("   one channel dominating.")
    ident = {}
    for axis in ("ansatz", "seed"):
        j, dp, dc, both = channel_identity(R[n], blocks, axis)
        ident[axis] = {"joint": j, "D_P": dp, "D_chi": dc, "D_both": both}
        print(f"   {axis:7s} joint {j:.6f} = P {dp:.6f} + chi {dc:.6f} "
              f"- both {both:.6f}   (residual {j - (dp + dc - both):+.1e})")
    report["A2_identity"] = ident

    banner(f"B. Continuous contrast by (v, lambda) seed family (n = {n})")
    fam = {}
    for ch, name in ((0, "dP_v"), (1, "dchi_phi")):
        f = family_rms(R[n], blocks, ch)
        tot = sum(v[2] for v in f.values())
        print(f"\n   {name}")
        print(f"   {'(v, lambda)':>16}  {'S_ansatz':>10} {'S_seed':>10} "
              f"{'ratio':>7} {'share of sq mass':>18}")
        for k in sorted(f):
            sa, ss, mass = f[k]
            print(f"   {str(k):>16}  {sa:>10.3f} {ss:>10.3f} "
                  f"{sa / ss if ss else float('nan'):>7.3f} {100 * mass / tot:>17.1f}%")
        fam[name] = {str(k): {"S_ansatz": v[0], "S_seed": v[1],
                              "sq_mass_share": v[2] / tot} for k, v in f.items()}
    report["B_family_split"] = fam

    banner(f"B2. Three-state disagreement by (v, lambda) family (n = {n})")
    print("   A disagreement rate is bounded in [0,1], so unlike the RMS it cannot")
    print("   be inflated by a single large-amplitude cell. Splitting it by family")
    print("   tests the remaining concentration worry directly: whether the flips")
    print("   themselves pile up in one corner of the grid.")
    fam3 = {}
    for ch, name in ((0, "dP_v"), (1, "dchi_phi")):
        rows = family_three_state(R[n], blocks, ch)
        tot = sum(v[2] for v in rows.values())
        print(f"\n   {name}")
        print(f"   {'(v, lambda)':>16} {'D_ansatz':>9} {'D_seed':>8} {'ratio':>7} "
              f"{'share of flips':>16}")
        for k in sorted(rows):
            da, ds, nf = rows[k]
            print(f"   {str(k):>16} {da:>9.4f} {ds:>8.4f} "
                  f"{da / ds if ds else float('nan'):>7.2f} {100 * nf / tot:>15.1f}%")
        fam3[name] = {str(k): {"D_ansatz": v[0], "D_seed": v[1],
                               "flip_share": v[2] / tot} for k, v in rows.items()}
    report["B2_family_three_state"] = fam3

    banner(f"C. Median companion to the pre-registered RMS (n = {n})")
    med = {}
    for ch, name in ((0, "dP_v"), (1, "dchi_phi")):
        ma, ms = median_companion(R[n], blocks, ch)
        med[name] = {"median_ansatz": ma, "median_seed": ms,
                     "ratio": ma / ms if ms else None}
        print(f"   {name:9s} median cell RMS: ansatz {ma:8.3f}  seed {ms:8.3f}  "
              f"ratio {ma / ms if ms else float('nan'):.3f}")
    report["C_median_companion"] = med

    dump(report, args.out)



def cmd_ranking(args):
    """POST-HOC: is the ranking/classification decoupling a metric artifact?

    The released rho ranks conditions by the UNNORMALIZED hypot of the two
    responses, while the classifier judges each channel against its own floor.
    That mismatch is the leading alternative explanation for the decoupling, so
    it is tested directly rather than conceded in a caveat: the same six
    pairwise Spearman values are recomputed under floor-normalized and
    single-channel rankings, signed and unsigned.
    """
    from scipy.stats import spearmanr
    n = args.n
    banner(f"POST-HOC: ranking metric sensitivity (n = {n})")
    print("Specified after the results were seen. No new simulation, no new")
    print("threshold: only the ranked quantity changes.")
    R, meta = load_runs(args.pickles, prefixes=(n,))
    blocks = meta["blocks"]
    L = label_table(R[n])
    _, ag = classification_agreement(L, blocks)
    ks = sorted(ag, key=repr)

    quantities = ("hypot_raw", "norm2", "abs_dP_v", "abs_dchi_phi",
                  "dP_v", "dchi_phi")
    table = {}
    for q in quantities:
        acc = {}
        for b in blocks:
            for k, v in pairwise_rho(R[n], [b], q).items():
                acc.setdefault(k, []).append(v)
        table[q] = {k: float(np.mean(v)) for k, v in acc.items()}

    print(f"\n{'pair':24}{'agree':>8}" + "".join(f"{q:>14}" for q in quantities))
    for k in ks:
        print(f"{k[0] + '/' + k[1]:24}{ag[k]:8.4f}"
              + "".join(f"{table[q][k]:14.4f}" for q in quantities))

    ref = ("A", "A_pocket")
    alt = ("A", "B_z")
    ag_gap = abs(ag[ref] - ag[alt])
    print(f"\nThe counterexample: A/A_pocket and A/B_z differ by {ag_gap:.4f} in "
          f"classification agreement.\nUnder each ranking metric they differ by:")
    out = {}
    for q in quantities:
        s_ = float(spearmanr([table[q][k] for k in ks],
                             [ag[k] for k in ks]).statistic)
        gap = abs(table[q][ref] - table[q][alt])
        out[q] = {"spearman_vs_agreement": s_, "counterexample_gap": gap,
                  "pairwise": _pk(table[q])}
        print(f"  {q:14} gap {gap:.4f}   Spearman(metric, agreement) = {s_:+.4f}")

    print("\nRead it this way: a ranking metric that reproduces the "
          "classification\nordering would show a large gap here and a Spearman "
          "near +1.")
    dump({"n": n, "meta": meta, "agreement": _pk(ag), "by_quantity": out},
         args.out)


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        p = sub.add_parser(name, help=fn.__doc__)
        p.add_argument("--pickles", type=Path, nargs="+", required=True)
        p.set_defaults(func=fn, **kw)
        return p

    p = add("selftest", cmd_selftest)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--raw", action="store_true",
                   help="reconstruct from realization-level raw (STEP 1b pickles); "
                        "omit for STEP 1a pickles, which store aggregates only")

    p = add("discovery", cmd_discovery)
    p.add_argument("--out", default="discovery.json")
    p.add_argument("--n", type=int, default=PRIMARY_N,
                   help="nested prefix to classify at. 30 is the pre-registered "
                        "Table III; 100 is the registered stability companion.")
    p = add("attribution", cmd_attribution)
    p.add_argument("--out", default="attribution.json")
    p.add_argument("--prefixes", type=int, nargs="+", default=list(PREFIXES),
                   help="nested prefixes to report; pass 5 10 20 30 40 60 80 100 "
                        "for the post-hoc extension data")
    p = add("holdout", cmd_holdout)
    p.add_argument("--discovery", required=True)
    p.add_argument("--n", type=int, default=None,
                   help="defaults to the prefix recorded in the discovery file")
    p.add_argument("--out", default="holdout.json")
    p = add("ranking", cmd_ranking)
    p.add_argument("--n", type=int, default=PRIMARY_N)
    p.add_argument("--out", default="ranking.json")

    p = add("legacy", cmd_legacy)
    p.add_argument("--out", default="legacy.json")
    p.add_argument("--n", type=int, default=PRIMARY_N)
    p = add("posthoc", cmd_posthoc)
    p.add_argument("--n", type=int, default=PRIMARY_N)
    p.add_argument("--prefixes", type=int, nargs="+", default=list(PREFIXES))
    p.add_argument("--out", default="posthoc.json")

    args = ap.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
