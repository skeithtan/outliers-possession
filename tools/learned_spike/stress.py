"""A label-free ball-gap stress view for the model and for c22. Not part of the submission.

    python -m tools.learned_spike.stress engine --game G --out PRED.csv --engine-cache DIR
    python -m tools.learned_spike.stress model --game G --lomo DIR --out PRED.csv

The view hides the ball of the raw view in extra gaps. The gaps lie inside the graded blocks
of games.py. Each gap has a log-uniform length from GAP_S[0] to GAP_S[1] seconds and starts on
a frame drawn uniformly. Gaps are added until HIDE_SHARE of the visible graded ball frames are
hidden. The seed is fixed per match. The mask reads no label, so both systems see the same
input and neither is favoured by the placement. BAR-ATH misses the ball on about 41 % of
graded frames in long gaps (PLAN.md 1.1), so the view is closer to it than the raw view.

`engine` runs engine.run on the masked view with the default config, in this process, after
it replaces engine.data.load_tactical with a masking wrapper (the pattern of tools/gapmask.py).
engine/ does not change. `model` predicts with the fold model of the match and decodes with
the fold's costs. Python 3.9.
"""

import argparse
import json
import os
import sys
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GAP_S = (2.0, 30.0)
HIDE_SHARE = 0.30
FPS = 25
FORBIDDEN = {"BAR-ATH", "DFL-J03WQQ", "DFL-J03WR9"}


def gap_mask(game, bx, blocks):
    """bool (N,): the frames whose ball the view hides."""
    rng = np.random.RandomState((zlib.crc32(game.encode("utf-8")) + 1) & 0x7FFFFFFF)
    n = len(bx)
    graded = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    visible = graded & np.isfinite(bx)
    target = int(HIDE_SHARE * visible.sum())
    hide = np.zeros(n, dtype=bool)
    idx = np.flatnonzero(graded)
    hidden = 0
    for _ in range(100000):
        if hidden >= target:
            break
        length = int(np.exp(rng.uniform(np.log(GAP_S[0]), np.log(GAP_S[1]))) * FPS)
        s = int(idx[rng.randint(idx.size)])
        e = min(n, s + length)
        seg = graded[s:e]
        hidden += int((visible[s:e] & ~hide[s:e] & seg).sum())
        hide[s:e] |= seg
    return hide


def masked_loader(original):
    from engine import data

    def load(game=None, *args, **kwargs):
        tac = dict(original(game, *args, **kwargs))
        name = data.resolve_game(game)
        hide = gap_mask(name, tac["bx"], data.graded_blocks(name))
        tac["bx"] = np.where(hide, np.nan, tac["bx"]).astype(np.float32)
        tac["by"] = np.where(hide, np.nan, tac["by"]).astype(np.float32)
        return tac
    return load


def run_engine(game, out, engine_cache):
    from engine import data
    from engine import run as engine_run
    data.DEFAULT_CACHE_DIR = engine_cache
    data.load_tactical = masked_loader(data.load_tactical)
    return engine_run.main(["--game", game, "--out", out])


def run_model(game, lomo, out, threads=2):
    import torch
    from engine import data
    from tools.learned_spike import build, features, model
    from tools.learned_spike.lomo import write_csv
    torch.set_num_threads(threads)
    fold = os.path.join(lomo, game)
    with open(os.path.join(fold, "fold.json")) as handle:
        rec = json.load(handle)
    net = model.Net()
    net.load_state_dict(torch.load(os.path.join(fold, "net.pt")))
    tac = build.load_view(game, "raw", None)
    blocks = data.graded_blocks(game)
    hide = gap_mask(game, tac["bx"], blocks)
    tac = dict(tac)
    tac["bx"] = np.where(hide, np.nan, tac["bx"]).astype(np.float32)
    tac["by"] = np.where(hide, np.nan, tac["by"]).astype(np.float32)
    f5 = features.downsample(features.frame_features(tac, game))
    probs = model.upsample(model.predict_symmetric(net, f5), tac["n_frames"])
    write_csv(out, model.decode(probs, blocks, rec["cost_ha"], rec["cost_ad"]))
    print("hidden share of graded frames {:.3f}".format(hide.mean()), file=sys.stderr)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("engine", "model"))
    parser.add_argument("--game", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--engine-cache")
    parser.add_argument("--lomo")
    args = parser.parse_args(argv)
    if args.game in FORBIDDEN:
        raise SystemExit("{} is forbidden here".format(args.game))
    if args.mode == "engine":
        return run_engine(args.game, args.out, args.engine_cache)
    return run_model(args.game, args.lomo, args.out)


if __name__ == "__main__":
    sys.exit(main())
