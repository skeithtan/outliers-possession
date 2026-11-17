"""Round 8 arm DK: where the kickoff rule K(a) changes labels against E2 (external matches). Not part of the submission.

    python -m tools.learned_spike.r8_scripts.dk_kcheck --codes ~/mtah2027/r8/dk/lopo/codes \
        ~/mtah2027/r8/dk/pff/codes --out ~/mtah2027/r8/dk/kcheck.json

For each match: the frame ranges where K's codes differ from E2's, with the old and new codes.
A range is inside its intended window when it lies in [kick, end) of a half's kickoff frame
kick (games.py, data.kickoff_frames), with end the latest of kick + the floor (1 s) and the
ends of the first run from kick in E2's codes and in K's codes. Kickoffs are the provider's half starts
recorded in games.py. Python 3.9.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from engine import data  # noqa: E402
from tools.learned_spike import decoders  # noqa: E402
from tools.learned_spike.r8_scripts import dk_decoders as dd  # noqa: E402

FLOOR = int(round(dd.KICK_FLOOR_S * dd.FPS))


def run_end(codes, start):
    seg = codes[start:]
    br = np.flatnonzero(seg != seg[0])
    return start + (int(br[0]) if br.size else len(seg))


def check(game, e2, k):
    kicks = data.kickoff_frames(game)
    rows, inside = [], True
    for a, b in decoders.runs_of(e2 != k):
        half = next((h for h, f in kicks.items()
                     if f <= a and b <= max(f + FLOOR, run_end(e2, f), run_end(k, f))), None)
        ok = half is not None
        inside &= ok
        rows.append({"range": [int(a), int(b)], "frames": int(b - a), "half": half,
                     "old": "".join(sorted(set("HADX"[c] for c in e2[a:b]))),
                     "new": "".join(sorted(set("HADX"[c] for c in k[a:b]))), "inside": ok})
    return {"kickoffs": {str(h): int(f) for h, f in kicks.items()},
            "e2_at_kick": {str(h): "HADX"[int(e2[f])] for h, f in kicks.items()},
            "k_at_kick": {str(h): "HADX"[int(k[f])] for h, f in kicks.items()},
            "changes": rows, "inside": bool(inside)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--codes", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    res = {}
    for folder in args.codes:
        for path in sorted(glob.glob(os.path.join(folder, "*.npz"))):
            game = os.path.basename(path)[:-4]
            with np.load(path) as z:
                res[game] = check(game, z["E2"], z["K"])
    n_ch = sum(1 for r in res.values() if r["changes"])
    print("matches", len(res), "with changes", n_ch, "all inside",
          all(r["inside"] for r in res.values()),
          "frames changed", sum(c["frames"] for r in res.values() for c in r["changes"]))
    for g, r in res.items():
        for c in r["changes"]:
            print(g, c)
    with open(args.out, "w") as h:
        json.dump(res, h, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
