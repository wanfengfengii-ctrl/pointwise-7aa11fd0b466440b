import copy
import json
import itertools
import sys
from pathlib import Path

import checks
from evidence import replay_kept
from source import EvidenceError, load


def reject(pack, witness, marker):
    try:
        replay_kept(pack, witness)
    except EvidenceError as exc:
        assert str(exc) == marker, (str(exc), marker)
    else:
        raise AssertionError("accepted invalid witness: " + marker)


def normal():
    checks.normal()
    tiny = load("data/tiny")
    witness = json.loads(Path("data/tiny/witness.json").read_text())
    before = copy.deepcopy(tiny)
    shown = replay_kept(tiny, witness)
    assert shown["isolated"] == []
    assert shown["pages"]["p2"] == [["a", 2, 40, 11, False]]
    assert tiny == before

    # A discarded response supplies no rows or read-time constraints.
    conflict = load("data/conflict")
    keep_first = {"isolated": ["p2"],
                  "schedule": {"commits": {"u": 1}, "reads": {"p1": 2}}}
    keep_second = {"isolated": ["p1"],
                   "schedule": {"commits": {"u": 6}, "reads": {"p2": 4}}}
    assert replay_kept(conflict, keep_first)["kept"] == ["p1"]
    assert replay_kept(conflict, keep_second)["kept"] == ["p2"]
    invalid = copy.deepcopy(keep_first)
    invalid["isolated"] = ["p2", "p2"]
    reject(conflict, invalid, "BAD_ISOLATED")
    invalid = copy.deepcopy(keep_first)
    invalid["schedule"]["reads"]["p2"] = 4
    reject(conflict, invalid, "SCHEDULE_KEYS:reads")
    invalid = copy.deepcopy(keep_first)
    invalid["schedule"]["commits"]["u"] = 6
    reject(conflict, invalid, "PAGE_MISMATCH:p1")

    # Isolating all pages cannot excuse impossible trusted audit ordering.
    audit = load("data/audit_conflict")
    reject(audit, {"isolated": [],
                  "schedule": {"commits": {"u": 2, "v": 2}, "reads": {}}},
           "CAUSAL_ORDER:v")
    joint = load("data/joint")
    page_ids = [p["id"] for p in joint["pages"]]
    for size in [2, 3]:
        for kept in itertools.combinations(page_ids, size):
            possible = False
            for times in itertools.product(range(1, 4), repeat=3):
                trial = {"isolated": [p for p in page_ids if p not in kept],
                         "schedule": {"commits": dict(zip(["A", "B", "C"],
                                                         times)),
                                      "reads": {p: 2 for p in kept}}}
                try:
                    replay_kept(joint, trial)
                    possible = True
                    break
                except EvidenceError:
                    pass
            assert possible == (size == 2), kept
    incident = load("data/incident")
    assert len(incident["transactions"]) == 18
    assert len(incident["pages"]) == 8
    print("EVIDENCE_REGRESSION_OK: retained-page replay, strict audit, "
          "atomic endpoint and short page")


if __name__ == "__main__":
    if sys.argv[1:] != ["normal"]:
        raise SystemExit("usage: evidence_checks.py normal")
    normal()
