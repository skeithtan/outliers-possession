"""Write the md5 of the 5 fps feature matrix per match, from the engine and from the spike.

    python -m tools.learned_spike.feature_md5 --out FILE.json [--games G ...]
        [--engine-cache DIR]

For each match the tool computes the 5 fps feature matrix of the raw view twice: with
engine/learned_features.py and with tools/learned_spike/features.py. It writes
{game: {"bins", "engine", "spike", "same"}} as JSON, with the md5 of
engine.learned_features.matrix_md5(). The two must agree byte for byte
(LEARNED_STAGE_PROTOCOL.md Section 2). The default matches are the eight open references.

The tool refuses the competition match and the sealed DFL matches (build.check). It reads no
reference. Python 3.9, numpy only.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from engine import learned_features as lf  # noqa: E402
from tools.learned_spike import build, features  # noqa: E402


def match_md5s(tactical, game):
    """{"bins", "engine", "spike", "same"} of one tactical dict."""
    eng = lf.downsample(lf.frame_features(tactical, game))
    spk = features.downsample(features.frame_features(tactical, game))
    a, b = lf.matrix_md5(eng), lf.matrix_md5(spk)
    return {"bins": int(eng.shape[0]), "engine": a, "spike": b, "same": a == b}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--games", nargs="+", default=list(build.OPEN_GAMES))
    parser.add_argument("--engine-cache", default=os.path.join("/tmp", "spike_engine"))
    args = parser.parse_args(argv)
    out = {}
    for game in sorted(args.games):
        build.check(game)
        out[game] = match_md5s(build.load_view(game, "raw", args.engine_cache), game)
        print("{:<12} bins {:>6}  {}  {}".format(game, out[game]["bins"], out[game]["engine"],
                                                "same" if out[game]["same"] else "DIFF"),
              flush=True)
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0 if all(row["same"] for row in out.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
