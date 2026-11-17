"""Frame-level difference of two submission CSVs of one match. Not part of the submission.

    python tools/analysis/label_diff.py A.csv B.csv [--game BAR-ATH] [--list 20]

Prints the graded frames that differ, the changed stretches (maximal runs of differing frames)
with their length distribution, the label-pair matrix, the per-half counts, and the first
--list stretches as frame spans. Reads no ground truth. Python 3.9, numpy only.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from engine import data  # noqa: E402

LABELS = "HADX"


def read_labels(path, n_frames):
    codes = np.full(n_frames, 3, dtype=np.int8)
    with open(path) as handle:
        next(handle)
        for line in handle:
            frame, label = line.strip().split(",")
            codes[int(frame)] = LABELS.index(label)
    return codes


def runs_of(mask):
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def spells(codes, graded):
    """(possessions, d_segments): maximal runs of one H/A label, and of D, on graded frames."""
    poss, dseg = 0, 0
    for start, stop in runs_of(graded):
        seg = codes[start:stop]
        change = np.flatnonzero(np.diff(seg) != 0) + 1
        starts = np.concatenate([[0], change])
        poss += int(np.isin(seg[starts], (0, 1)).sum())
        dseg += int((seg[starts] == 2).sum())
    return poss, dseg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--game", default=None)
    parser.add_argument("--list", type=int, default=20)
    args = parser.parse_args(argv)
    game = data.resolve_game(args.game)
    n = data.frame_count(game)
    fps = data.fps(game)
    a, b = read_labels(args.a, n), read_labels(args.b, n)
    graded = np.zeros(n, dtype=bool)
    half_of = np.zeros(n, dtype=np.int8)
    for h, lo, hi in data.graded_blocks(game):
        graded[lo:hi + 1] = True
        half_of[lo:hi + 1] = h
    diff = (a != b) & graded
    print("{} vs {}".format(os.path.basename(args.a), os.path.basename(args.b)))
    print("graded frames {} | changed {} ({:.3f}%)".format(graded.sum(), diff.sum(),
                                                            100.0 * diff.sum() / graded.sum()))
    runs = runs_of(diff)
    lens = np.array([e - s for s, e in runs])
    print("changed stretches {}".format(len(runs)))
    if len(runs):
        print("  length (frames): min {} p25 {:.0f} median {:.0f} p75 {:.0f} max {} | sum {}".format(
            lens.min(), np.quantile(lens, 0.25), np.median(lens), np.quantile(lens, 0.75),
            lens.max(), lens.sum()))
        print("  stretches >= 1 s: {} | >= 5 s: {}".format((lens >= fps).sum(), (lens >= 5 * fps).sum()))
    for h in (1, 2):
        sel = diff & (half_of == h)
        print("  half {}: changed frames {} stretches {}".format(h, sel.sum(), len(runs_of(sel))))
    print("label pairs (a -> b) on changed frames:")
    for i, la in enumerate(LABELS):
        for j, lb in enumerate(LABELS):
            c = int((diff & (a == i) & (b == j)).sum())
            if c:
                print("  {} -> {}: {}".format(la, lb, c))
    pa, da = spells(a, graded)
    pb, db = spells(b, graded)
    print("possessions a {} b {} | D segments a {} b {} | D share a {:.4f} b {:.4f}".format(
        pa, pb, da, db, (a[graded] == 2).mean(), (b[graded] == 2).mean()))
    for s, e in runs[:args.list]:
        print("  stretch {}-{} ({} fr, half {}) a={} b={}".format(
            s, e - 1, e - s, half_of[s], "".join(LABELS[k] for k in np.unique(a[s:e])),
            "".join(LABELS[k] for k in np.unique(b[s:e]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
