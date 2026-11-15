"""Prototype: MILP feasibility of a kept-page set, cross-checked by brute force."""
import itertools
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".pydeps"))
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from source import EvidenceError, load, replay

BIG = None  # set per model


def build_chains(pack):
    """id -> list of version dicts in revision order.

    An ID first written by a transaction (absent from initial) gets a virtual
    unborn state at index 0 (writer == its first writer, sentinel unborn=True).
    """
    chains = {}
    for r in pack["initial"]:
        chains.setdefault(r[0], []).append(
            {"rev": r[1], "rank": r[2], "value": r[3], "del": r[4],
             "writer": None, "unborn": False})
    # transactions already dependency-sorted by validate; version chain order
    # is guaranteed by BAD_VERSION_CHAIN checks.
    for tx in pack["transactions"]:
        for r in tx["writes"]:
            if r[0] not in chains:
                chains[r[0]] = [{
                    "rev": 0, "rank": None, "value": None, "del": True,
                    "writer": tx["id"], "unborn": True}]
            chains[r[0]].append(
                {"rev": r[1], "rank": r[2], "value": r[3], "del": r[4],
                 "writer": tx["id"], "unborn": False})
    return chains


def model_feasible(pack, kept_ids):
    txs = pack["transactions"]
    n = len(txs)
    tx_idx = {t["id"]: i for i, t in enumerate(txs)}
    kept = [p for p in pack["pages"] if p["id"] in kept_ids]
    H = max([t["window"][1] for t in txs]
            + [p["window"][1] for p in kept] + [1])
    M = H + 1
    chains = build_chains(pack)
    ids = sorted(chains)
    rows = []  # (lb, ub, integrality)
    # 0..n-1 commit times
    for t in txs:
        rows.append((t["window"][0], t["window"][1], 1))
    # n..n+q-1 read times
    for p in kept:
        rows.append((p["window"][0], p["window"][1], 1))
    zmap = {}  # (id, version_index, page_index) -> var index
    for pi, p in enumerate(kept):
        for rid in ids:
            for k in range(len(chains[rid])):
                zmap[(rid, k, pi)] = len(rows)
                rows.append((0, 1, 1))
    V = len(rows)
    lb = np.array([r[0] for r in rows], dtype=float)
    ub = np.array([r[1] for r in rows], dtype=float)
    integrality = np.ones(V)
    A, lo, hi = [], [], []

    def add(coeffs, low, high):
        v = np.zeros(V)
        for j, c in coeffs.items():
            v[j] = c
        A.append(v)
        lo.append(low)
        hi.append(high)

    # precedence: C_child - C_parent >= 1
    for t in txs:
        c = tx_idx[t["id"]]
        for par in t["after"]:
            pp = tx_idx[par]
            add({c: 1, pp: -1}, 1, np.inf)

    for pi, page in enumerate(kept):
        R = n + pi
        forced = {}  # id -> version index required by page rows
        for r in page["rows"]:
            rid, rev = r[0], r[1]
            if rid not in chains:
                return None
            ks = [k for k, v in enumerate(chains[rid]) if v["rev"] == rev]
            if not ks:
                return None
            k = ks[0]
            v = chains[rid][k]
            if (v["rank"], v["value"], v["del"]) != (r[2], r[3], r[4]):
                return None  # fields contradict the immutable chain version
            if rid in forced and forced[rid] != k:
                return None
            forced[rid] = k
        # exactly one active version per id per page
        for rid in ids:
            zs = [zmap[(rid, k, pi)] for k in range(len(chains[rid]))]
            add({j: 1 for j in zs}, 1, 1)
        for rid in ids:
            chain = chains[rid]
            for k, v in enumerate(chain):
                z = zmap[(rid, k, pi)]
                w = tx_idx[v["writer"]] if v["writer"] is not None else None
                # Version k is current at read R iff its birth commit is <= R
                # and its superseding commit (if any) is > R. One-way big-M
                # implications plus the exactly-one-active sum suffice;
                # M = H+1 makes every inactive branch slack over integer times.
                if v.get("unborn"):
                    # virtual state: active only while first writer uncommitted
                    add({w: 1, R: -1, z: -M}, 1 - M, np.inf)
                elif w is None:
                    # initial version exists since time 0: birth is automatic.
                    pass
                else:
                    # z=1 => C_w - R <= 0 (this version already committed)
                    add({w: 1, R: -1, z: M}, -np.inf, M)
                if k + 1 < len(chain):
                    nxt = tx_idx[chain[k + 1]["writer"]]
                    # z=1 => C_next - R >= 1 (version k still current)
                    add({nxt: 1, R: -1, z: -M}, 1 - M, np.inf)
        for rid, k in forced.items():
            add({zmap[(rid, k, pi)]: 1}, 1, 1)  # force exact chain version
        # ordering / slicing constraints
        o, L, m = page["offset"], page["limit"], len(page["rows"])
        boundaries = [(r[2], r[0]) for r in page["rows"]]
        for b, key in enumerate(boundaries):
            coeffs = {}
            for rid in ids:
                for k, v in enumerate(chains[rid]):
                    if v["del"]:
                        continue
                    if (v["rank"], rid) < key:
                        coeffs[zmap[(rid, k, pi)]] = 1
            add(coeffs, o + b, o + b)
        if 0 < m < L:  # short page: exact total visible count
            coeffs = {}
            for rid in ids:
                for k, v in enumerate(chains[rid]):
                    if not v["del"]:
                        coeffs[zmap[(rid, k, pi)]] = 1
            add(coeffs, o + m, o + m)
        if m == 0:
            coeffs = {}
            for rid in ids:
                for k, v in enumerate(chains[rid]):
                    if not v["del"]:
                        coeffs[zmap[(rid, k, pi)]] = 1
            add(coeffs, -np.inf, o)

    constraints = (LinearConstraint(np.array(A), np.array(lo), np.array(hi))
                   if A else None)
    res = milp(c=np.zeros(V), integrality=integrality,
               bounds=Bounds(lb, ub),
               constraints=constraints,
               options={"time_limit": 60})
    if res.x is None:
        return None
    x = np.rint(res.x).astype(int)
    commits = {t["id"]: int(x[tx_idx[t["id"]]]) for t in txs}
    reads = {p["id"]: int(x[n + pi]) for pi, p in enumerate(kept)}
    return {"commits": commits, "reads": reads}


def brute_force(pack, kept_ids):
    """Independent enumeration using only original replay semantics."""
    txs = pack["transactions"]
    kept = [p for p in pack["pages"] if p["id"] in kept_ids]
    commit_ranges = [range(t["window"][0], t["window"][1] + 1) for t in txs]
    read_ranges = [range(p["window"][0], p["window"][1] + 1) for p in kept]
    for cvals in itertools.product(*commit_ranges):
        commits = {t["id"]: v for t, v in zip(txs, cvals)}
        if any(commits[par] >= commits[t["id"]]
               for t in txs for par in t["after"]):
            continue
        for rvals in itertools.product(*read_ranges):
            reads = {p["id"]: v for p, v in zip(kept, rvals)}
            reduced = dict(pack)
            reduced["pages"] = kept
            try:
                replay(reduced, {"commits": commits, "reads": reads})
            except EvidenceError:
                continue
            return {"commits": commits, "reads": reads}
    return None


def cross_check(name):
    pack = load(f"/workspace/data/{name}")
    pids = [p["id"] for p in pack["pages"]]
    allok = True
    for r in range(len(pids) + 1):
        for iso in itertools.combinations(pids, r):
            kept = frozenset(pids) - frozenset(iso)
            bf = brute_force(pack, kept)
            ms = model_feasible(pack, kept)
            ok = (bf is not None) == (ms is not None)
            if not ok:
                allok = False
                print("MISMATCH", name, "kept", kept, "bf", bf, "milp", ms)
            if ms is not None:
                # the MILP witness must itself replay
                reduced = dict(pack)
                reduced["pages"] = [p for p in pack["pages"]
                                    if p["id"] in kept]
                replay(reduced, ms)
    print(name, "cross-check", "OK" if allok else "FAILED")
    return allok


if __name__ == "__main__":
    ok = True
    for name in ["tiny", "conflict", "joint"]:
        ok &= cross_check(name)
    # audit_conflict: no pages; scheduling itself impossible
    audit = load("/workspace/data/audit_conflict")
    w = model_feasible(audit, frozenset())
    b = brute_force(audit, frozenset())
    print("audit_conflict milp:", w, "brute:", b)
    assert w is None and b is None
    print("PROTOTYPE_OK" if ok else "PROTOTYPE_FAILED")
