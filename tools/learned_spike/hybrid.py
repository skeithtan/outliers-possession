"""Two cheap hybrids of the learned spike and c22. Not part of the submission.

    python -m tools.learned_spike.hybrid --lomo DIR --cache DIR --c22-runs DIR --c22-keys SWEEP.json

The tool writes two files per test match next to pred.csv. Both use the fold's own costs,
chosen on its validation match for the plain model. Nothing is chosen on the test match.

- hyb_c22dead.csv: the alive and dead frames of c22. On each alive run of c22 the team comes
  from a two-state Viterbi (H, A) over the model probabilities with the fold's H-A cost.
- hyb_modeldead.csv: the alive and dead frames of the model. On a frame where both are alive,
  the team of c22. Elsewhere the model.

The first asks whether the model owns the ball better. The second asks whether the model
finds the dead ball better. Python 3.9.
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import score_game as sg  # noqa: E402
from tools.learned_spike import model  # noqa: E402
from tools.learned_spike.lomo import write_csv  # noqa: E402


def runs_of(mask):
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return zip(edges[0::2], edges[1::2])


def team_viterbi(logp2, cost):
    """Two-state path (0 H, 1 A) for (N, 2) log-probabilities."""
    lp = np.column_stack([logp2, np.full(len(logp2), -1e9)])
    return model.viterbi(lp, cost, 1e9)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lomo", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--c22-runs", required=True)
    parser.add_argument("--c22-keys", required=True)
    args = parser.parse_args(argv)
    with open(args.c22_keys) as handle:
        keys = json.load(handle)["run_keys"]["base"]
    for game in sorted(os.listdir(args.lomo)):
        fold = os.path.join(args.lomo, game)
        if not os.path.isfile(os.path.join(fold, "fold.json")):
            continue
        with open(os.path.join(fold, "fold.json")) as handle:
            rec = json.load(handle)
        n = len(np.load(os.path.join(args.cache, "labels", game + ".npz"))["raw"])
        base = sg.read_codes(os.path.join(args.c22_runs, game, keys[game] + ".csv"), n)
        pred = sg.read_codes(os.path.join(fold, "pred.csv"), n)
        p5 = np.load(os.path.join(fold, "probs.npz"))["p5"].astype(np.float64)
        probs = model.upsample(p5, n)
        team = probs[:, :2] / np.maximum(probs[:, :2].sum(axis=1, keepdims=True), 1e-9)
        logt = np.log(np.clip(team, 1e-4, 1.0))
        hyb1 = base.copy()
        for a, b in runs_of(np.isin(base, (0, 1))):
            hyb1[a:b] = team_viterbi(logt[a:b], rec["cost_ha"])
        hyb2 = pred.copy()
        both = np.isin(pred, (0, 1)) & np.isin(base, (0, 1))
        hyb2[both] = base[both]
        write_csv(os.path.join(fold, "hyb_c22dead.csv"), hyb1)
        write_csv(os.path.join(fold, "hyb_modeldead.csv"), hyb2)
        print(game, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
