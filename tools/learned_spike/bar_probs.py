"""Probability dumps of the spike models on the competition match, for bar_report.py.

    python -m tools.learned_spike.bar_probs --out DIR --member NAME=FOLD_DIR [--member ...]

Each FOLD_DIR holds net.pt (a model of tools/learned_spike/model.py). The tool computes the
features of the competition match once, from the ATD that the engine reads
(engine.possession.engine_tactical(), with the C9 goalkeeper guard). For each member it
writes DIR/member_<NAME>.npz with `p5`, the mean probability over the four symmetric copies
(5 fps, H A D). For each symmetric copy it writes DIR/copy_<k>.npz with `p5`, the mean over
the members of that copy alone (swapped copies with H and A put back). The tool reads no
label, never reads evaluation/gt, never decodes and never scores. Nothing here is tuned on the
match. Run it with the torch venv. Python 3.9.
"""

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import games  # noqa: E402
from engine.possession import engine_tactical  # noqa: E402
from tools.learned_spike import features, model  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--member", action="append", required=True, help="NAME=FOLD_DIR")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    game = games.GAME
    tac = engine_tactical(None, game)     # the ATD of the engine, with the C9 guard (R2 9, D1)
    f5 = features.downsample(features.frame_features(tac, game))
    copies = [np.zeros((f5.shape[0], model.N_CLASSES)) for _ in model.SYMMETRIES]
    members = [item.split("=", 1) for item in args.member]
    for name, fold in members:
        net = model.Net()
        net.load_state_dict(torch.load(os.path.join(fold, "net.pt")))
        per_copy = [model.predict_symmetric(net, f5, symmetries=(sym,))
                    for sym in model.SYMMETRIES]
        for k, p in enumerate(per_copy):
            copies[k] += p / len(members)
        np.savez(os.path.join(args.out, "member_{}.npz".format(name)),
                 p5=np.mean(per_copy, axis=0).astype(np.float32))
        print(name, file=sys.stderr, flush=True)
    for k, p in enumerate(copies):
        np.savez(os.path.join(args.out, "copy_{}.npz".format(k)), p5=p.astype(np.float32))
    return 0


if __name__ == "__main__":
    sys.exit(main())
