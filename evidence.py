import json
import sys
from pathlib import Path

from source import EvidenceError, load, replay, validate


def replay_kept(pack, witness):
    """Replay an already chosen quarantine; does not select or score it."""
    validate(pack)
    isolated = witness["isolated"]
    page_ids = {p["id"] for p in pack["pages"]}
    if (not isinstance(isolated, list)
            or not all(isinstance(x, str) for x in isolated)
            or len(set(isolated)) != len(isolated)
            or not set(isolated) <= page_ids):
        raise EvidenceError("BAD_ISOLATED")
    reduced = dict(pack)
    reduced["pages"] = [p for p in pack["pages"]
                        if p["id"] not in set(isolated)]
    result = replay(reduced, witness["schedule"])
    return {"isolated": sorted(isolated),
            "kept": [p["id"] for p in reduced["pages"]],
            "pages": result["pages"]}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit("usage: evidence.py INPUT_DIR WITNESS.json")
    witness = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    print(json.dumps(replay_kept(load(argv[0]), witness),
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
