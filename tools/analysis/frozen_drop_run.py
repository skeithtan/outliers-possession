"""P1.3 counterfactual: the engine with frozen-exact slots blanked before tracks.prepare.

    python tools/analysis/frozen_drop_run.py --out X.csv [--min-s 2.0]

Wraps engine.data.load_tactical so that every player-frame inside a frozen-exact run of
--min-s or longer (frozen_slots_probe.frozen_exact) becomes NaN, then runs the default engine
and prints the summary and md5. The engine code is not changed. Reads no ground truth.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (ROOT, os.path.join(ROOT, "tools", "analysis")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from engine import data, possession, submission  # noqa: E402
from frozen_slots_probe import frozen_exact  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-s", type=float, default=2.0)
    parser.add_argument("--game", default=None)
    args = parser.parse_args(argv)
    game = data.resolve_game(args.game)
    original = data.load_tactical

    def patched(game=None, data_dir=None, cache_dir=None, use_cache=True):
        tactical = original(game, data_dir, cache_dir, use_cache)
        fz = frozen_exact(tactical["px"], tactical["py"], int(round(args.min_s * data.fps(game))))
        for key in ("px", "py", "ps"):
            arr = np.array(tactical[key], copy=True)
            arr[fz] = np.nan
            tactical[key] = arr
        print("blanked {} frozen-exact player-frames".format(int(fz.sum())))
        return tactical

    data.load_tactical = patched
    labels = possession.run(game=game)
    path = submission.write_submission(labels, args.out, game=game)
    print(submission.format_summary(submission.summarize(labels, game=game)))
    print("md5   {}".format(submission.md5_of(path)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
