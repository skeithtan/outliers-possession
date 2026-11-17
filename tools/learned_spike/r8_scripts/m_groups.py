"""Arm M (round 8): training rows of member spec F6 = F5T's columns + the group columns of
engine.learned_pf_groups. Not part of the submission.

    # pf venv, OMP_NUM_THREADS=1
    python -m tools.learned_spike.r8_scripts.m_groups build --f5t SP/r6_pf/feats_F5T \
        --labels CA --engine-cache SP/r4_work/engine_data_raw --out ~/mtah2027/r8/m/feats_F6 \
        [--games G ...]

For each external training match the rows of F5T's training file (feats_F5T/train/<g>.npz:
every 8th graded frame from offset crc32(game) % 8, view 0 = the tracking as provided, DFL input
c_train_j; view 1 = defect copy 0 of augment.augment(version r3) of that tracking) are kept as
they are, and the group columns are computed on the same two views (pf_features.train_base and
pf_features.aug_copy, the code that built F5T's rows) and appended. The rows, labels and indices
are checked against F5T's file, and the 50 base columns recomputed on both views must equal the
stored ones (max |diff| recorded in timing/<g>.json), so the appended columns belong to the same
tracking. --scale robust (member spec F6c, round 8 note 6.3): the group columns in robust
units of each view (engine.learned_pf_groups.robust_units, statistics from that view's own
tracking: the tracking as provided and the defect copy each get their own). Output:
OUT/train/<g>.npz (X, y, idx, view) and OUT/spec.json (names, kinds).
Never reads a competition, sealed or PFF match (pf_features.check). Python 3.9.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__)))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from engine import learned_features as lf  # noqa: E402
from engine import learned_pf_groups as lpg  # noqa: E402
from tools.learned_spike import folds  # noqa: E402
from tools.learned_spike import pf_features as pff  # noqa: E402

BASE = "stage50-v2t"


def build_game(game, args):
    path = os.path.join(args.out, "train", game + ".npz")
    if os.path.isfile(path) and not args.force:
        return None
    t0 = time.time()
    with np.load(os.path.join(args.f5t, "train", game + ".npz")) as z:
        X, y, idx, view = z["X"], z["y"], z["idx"], z["view"]
    codes = pff.labels(args.labels, game)
    rows = pff.train_rows(game, codes)
    assert np.array_equal(idx[view == 0], rows) and np.array_equal(idx[view == 1], rows), game
    assert np.array_equal(y[view == 0], codes[rows]), game
    tac0 = pff.train_base(game, args.engine_cache)
    G0 = lpg.group_features(tac0, game, BASE, scale=args.scale)[rows]
    b0 = lf.frame_features(tac0, game, version=BASE)[rows]
    tac1 = pff.aug_copy(tac0, codes, game)
    G1 = lpg.group_features(tac1, game, BASE, scale=args.scale)[rows]
    b1 = lf.frame_features(tac1, game, version=BASE)[rows]
    d0 = float(np.abs(b0 - X[view == 0][:, :50]).max())
    d1 = float(np.abs(b1 - X[view == 1][:, :50]).max())
    if max(d0, d1) > 1e-6:
        raise SystemExit("{}: base columns differ from F5T's rows ({:g}, {:g})".format(game, d0,
                                                                                     d1))
    G = np.zeros((len(y), G0.shape[1]), np.float32)
    G[view == 0] = G0
    G[view == 1] = G1
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path + ".tmp.npz", X=np.concatenate([X, G], axis=1), y=y, idx=idx, view=view)
    os.replace(path + ".tmp.npz", path)
    tm = {"seconds": time.time() - t0, "base_diff_view0": d0, "base_diff_view1": d1,
          "rows": int(len(y))}
    with open(os.path.join(args.out, "timing", game + ".json"), "w") as h:
        json.dump(tm, h, indent=1)
    return tm


def cmd_build_eval(args):
    """Group columns of the evaluation views (pf_features.eval_view: raw; DFL input "c", the
    view of F5T's feats_F5T/eval) into OUT/eval/<g>.npy, float32 (N, 36)."""
    os.makedirs(os.path.join(args.out, "eval"), exist_ok=True)
    for game in args.games or list(folds.ALL):
        pff.check(game)
        assert not game.startswith("PFF-")
        path = os.path.join(args.out, "eval", game + ".npy")
        if os.path.isfile(path) and not args.force:
            continue
        t0 = time.time()
        G = lpg.group_features(pff.eval_view(game, args.engine_cache), game, BASE,
                               scale=args.scale)
        np.save(path + ".tmp.npy", G)
        os.replace(path + ".tmp.npy", path)
        print(game, G.shape, "{:.1f}s".format(time.time() - t0), file=sys.stderr, flush=True)
    return 0


def cmd_build(args):
    os.makedirs(os.path.join(args.out, "timing"), exist_ok=True)
    spec = json.load(open(os.path.join(args.f5t, "spec.json")))
    assert spec["base_version"] == BASE
    names = spec["names"] + lpg.GROUP_NAMES
    kinds = spec["kinds"] + [k for _, k in lpg.GROUP_SPEC]
    with open(os.path.join(args.out, "spec.json"), "w") as h:
        json.dump({"fset": "F6", "group_scale": args.scale, "base_version": BASE,
                   "names": names, "kinds": kinds,
                   "n_f5t": len(spec["names"]), "stride": spec["stride"]}, h)
    for game in args.games or list(folds.ALL):
        pff.check(game)
        assert not game.startswith("PFF-")
        tm = build_game(game, args)
        print(game, tm, file=sys.stderr, flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--f5t", required=True)
    b.add_argument("--labels", required=True)
    b.add_argument("--engine-cache", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--games", nargs="+", default=None)
    b.add_argument("--force", action="store_true")
    e = sub.add_parser("build-eval")
    e.add_argument("--engine-cache", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--games", nargs="+", default=None)
    e.add_argument("--force", action="store_true")
    for q in (b, e):
        q.add_argument("--scale", choices=lpg.SCALES, default=lpg.SCALE_RAW,
                       help="units of the group columns: raw (F6a/F6b) or robust per match "
                            "(F6c); the robust statistics come from each view's own tracking")
    args = p.parse_args(argv)
    return {"build": cmd_build, "build-eval": cmd_build_eval}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
