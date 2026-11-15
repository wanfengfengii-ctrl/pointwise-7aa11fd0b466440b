"""Synthetic edge-case packs: MILP verdict must match brute-force replay."""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".pydeps"))

from source import EvidenceError, validate, replay
from proto import brute_force, model_feasible

PACKS = {}

# 1) withdraw then resurrection of one id, plus a never-born id.
PACKS["resurrect_unborn"] = {
    "target": 5,
    "initial": [["a", 1, 10, 100, False], ["b", 1, 20, 200, False]],
    "transactions": [
        {"id": "u", "window": [1, 4], "after": [],
         "writes": [["a", 2, 10, 101, True], ["c", 1, 5, 300, False]]},
        {"id": "v", "window": [1, 5], "after": ["u"],
         "writes": [["a", 3, 15, 102, False]]},
    ],
    "pages": [
        # before u: a,b visible, c unborn
        {"id": "g1", "window": [1, 1], "offset": 0, "limit": 4,
         "rows": [["a", 1, 10, 100, False], ["b", 1, 20, 200, False]]},
        # after u before v: a tombstoned, c born
        {"id": "g2", "window": [2, 3], "offset": 0, "limit": 4,
         "rows": [["c", 1, 5, 300, False], ["b", 1, 20, 200, False]]},
        # after v: a resurrected with rank 15 -> order c,a,b
        {"id": "g3", "window": [4, 5], "offset": 0, "limit": 4,
         "rows": [["c", 1, 5, 300, False], ["a", 3, 15, 102, False],
                  ["b", 1, 20, 200, False]]},
        # short page (m < limit): total visible count pinned
        {"id": "g4", "window": [2, 3], "offset": 0, "limit": 5,
         "rows": [["c", 1, 5, 300, False], ["b", 1, 20, 200, False]]},
        # empty-rows page with offset beyond size: only satisfiable when
        # visible count <= offset
        {"id": "g5", "window": [1, 5], "offset": 3, "limit": 2, "rows": []},
    ],
}

# 2) same-time commit-first: read at commit time must see the new version.
PACKS["same_tick"] = {
    "target": 3,
    "initial": [["a", 1, 10, 0, False]],
    "transactions": [
        {"id": "u", "window": [2, 4], "after": [],
         "writes": [["a", 2, 10, 1, False]]}],
    "pages": [
        {"id": "h1", "window": [2, 2], "offset": 0, "limit": 1,
         "rows": [["a", 2, 10, 1, False]]},   # commit at 2, read at 2
        {"id": "h2", "window": [1, 1], "offset": 0, "limit": 1,
         "rows": [["a", 1, 10, 0, False]]},
    ],
}

# 3) atomic multi-row move: a single tx shifts two rows across an anchor.
PACKS["atomic_move"] = {
    "target": 4,
    "initial": [["a", 1, 10, 1, False], ["b", 1, 20, 2, False],
                ["c", 1, 30, 3, False]],
    "transactions": [
        {"id": "u", "window": [2, 4], "after": [],
         "writes": [["a", 2, 25, 1, False], ["c", 2, 5, 3, False]]}],
    "pages": [
        {"id": "k1", "window": [1, 1], "offset": 0, "limit": 3,
         "rows": [["a", 1, 10, 1, False], ["b", 1, 20, 2, False],
                  ["c", 1, 30, 3, False]]},
        {"id": "k2", "window": [2, 4], "offset": 0, "limit": 3,
         "rows": [["c", 2, 5, 3, False], ["b", 1, 20, 2, False],
                  ["a", 2, 25, 1, False]]},
        # offset window into the post-move ordering
        {"id": "k3", "window": [2, 4], "offset": 1, "limit": 1,
         "rows": [["b", 1, 20, 2, False]]},
    ],
}


def main():
    all_ok = True
    for name, raw in PACKS.items():
        pack = validate(raw)
        pids = [p["id"] for p in pack["pages"]]
        for r in range(len(pids) + 1):
            for iso in itertools.combinations(pids, r):
                kept = frozenset(pids) - frozenset(iso)
                bf = brute_force(pack, kept)
                ms = model_feasible(pack, kept)
                if (bf is None) != (ms is None):
                    all_ok = False
                    print("MISMATCH", name, "kept", sorted(kept),
                          "bf", bf, "milp", ms)
                if ms is not None:
                    reduced = dict(pack)
                    reduced["pages"] = [p for p in pack["pages"]
                                        if p["id"] in kept]
                    replay(reduced, ms)
        print(name, "OK")
    print("SYNTHETIC_" + ("OK" if all_ok else "FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
