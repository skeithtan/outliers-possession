"""Label-free sanity numbers of the learned spike on the competition match. Not part of the submission.

    python -m tools.learned_spike.bar_sanity --lomo DIR [--out PRED.csv]

The tool loads the eight fold models of a LOMO run (DIR/<game>/net.pt), averages their
symmetric class probabilities on the competition match and decodes them with the median of
the switch costs that the folds chose on their validation matches. It prints only label-free
numbers: the possession count, the D share, the D segment count and the graded X count, on
the graded blocks of games.py, and the run time. It never reads evaluation/gt and never
scores. Nothing here is tuned on this match. Python 3.9.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import games  # noqa: E402
import score_game as sg  # noqa: E402
from engine import data  # noqa: E402
from tools.learned_spike import features, model  # noqa: E402
from tools.learned_spike.lomo import write_csv  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lomo", required=True)
    parser.add_argument("--out")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    folds = sorted(g for g in os.listdir(args.lomo)
                   if os.path.isfile(os.path.join(args.lomo, g, "net.pt")))
    costs = []
    nets = []
    for g in folds:
        with open(os.path.join(args.lomo, g, "fold.json")) as handle:
            rec = json.load(handle)
        costs.append((rec["cost_ha"], rec["cost_ad"]))
        net = model.Net()
        net.load_state_dict(torch.load(os.path.join(args.lomo, g, "net.pt")))
        nets.append(net)
    cha = float(np.median([c[0] for c in costs]))
    cad = float(np.median([c[1] for c in costs]))
    game = games.GAME
    t0 = time.time()
    tac = data.load_tactical(game)
    f5 = features.downsample(features.frame_features(tac, game))
    t_feat = time.time() - t0
    p5 = np.mean([model.predict_symmetric(net, f5) for net in nets], axis=0)
    t_net = time.time() - t0 - t_feat
    probs = model.upsample(p5, tac["n_frames"])
    blocks = data.graded_blocks(game)
    pred = model.decode(probs, blocks, cha, cad)
    t_all = time.time() - t0
    runs = [(first, last + 1) for _, first, last in blocks]
    stats = sg.label_stats(pred, runs)
    graded_x = int(sum(np.sum(pred[a:b] == 3) for a, b in runs))
    report = {"folds": len(nets), "cost_ha": cha, "cost_ad": cad,
              "possessions": stats["possessions"], "d_share": round(stats["d_share"], 4),
              "d_segments": stats["d_segments"], "graded_x": graded_x,
              "mean_possession_s": round(stats["mean_duration_s"], 2),
              "seconds_features": round(t_feat, 1), "seconds_networks": round(t_net, 1),
              "seconds_total": round(t_all, 1)}
    print(json.dumps(report, sort_keys=True))
    if args.out:
        write_csv(args.out, pred)
    return 0


if __name__ == "__main__":
    sys.exit(main())
